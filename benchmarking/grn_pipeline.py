from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("GRN")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


_TF_URLS = [
    "https://raw.githubusercontent.com/aertslab/pySCENIC/master/resources/hs_hgnc_tfs.txt",
    "https://raw.githubusercontent.com/saezlab/CollecTRI/main/data/human_TFs_Lambert2018.txt",
]

def _save_stats(stats: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump({k: v for k, v in stats.items() if isinstance(v, (int, float, str, list, dict))}, f, indent=2)
    log.info(f"  Stats            -> {path}")


def _basic_stats(df: pd.DataFrame) -> dict:
    """Compute basic edge-list statistics."""
    return {
        "n_edges":   int(len(df)),
        "n_tfs":     int(df["TF"].nunique()),
        "n_targets": int(df["target"].nunique()),
        "n_nodes":   int(pd.concat([df["TF"], df["target"]]).nunique()),
        "top_regulators": (
            df.groupby("TF")["target"].count()
            .sort_values(ascending=False).head(20)
            .to_dict()
        ),
    }


def fetch_tf_list(gene_names: List[str], cache_path: Optional[str] = None) -> List[str]:
    """Download Lambert et al. 2018 TF census and intersect with gene_names."""
    import requests

    cache_dir = Path.home() / ".cache" / "grn_pipeline"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_path) if cache_path else cache_dir / "human_tfs.txt"
    gene_set = set(gene_names)

    if cache_path.exists():
        log.info(f"  TF list from cache: {cache_path}")
        tfs_raw = [l.strip() for l in cache_path.read_text().splitlines() if l.strip()]
    else:
        tfs_raw = None
        for url in _TF_URLS:
            try:
                log.info(f"  Downloading TF list from {url} ...")
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                tfs_raw = [l.strip() for l in r.text.splitlines() if l.strip()]
                cache_path.write_text("\n".join(tfs_raw))
                log.info(f"  Cached -> {cache_path}")
                break
            except Exception as exc:
                log.warning(f"  Download failed ({url}): {exc}")
        if tfs_raw is None:
            log.warning("  Could not fetch TF list - using all HVGs as regulators.")
            return list(gene_names)

    found = [g for g in tfs_raw if g in gene_set]
    log.info(f"  TF census: {len(tfs_raw)} total | {len(found)} found in HVGs.")
    if not found:
        log.warning("  Zero TF/HVG overlap - using all HVGs. Check gene symbol format.")
        return list(gene_names)
    return found


def extract_rna(mdata, gene_symbol_col: Optional[str] = None):
    """Pull mdata.mod['rna'] as a copy. Optionally remap Ensembl -> HGNC."""
    if not hasattr(mdata, "mod") or "rna" not in mdata.mod:
        raise ValueError('mdata must have mdata.mod["rna"].')
    adata = mdata.mod["rna"].copy()
    adata.var_names_make_unique()

    if gene_symbol_col is not None:
        if gene_symbol_col not in adata.var.columns:
            raise ValueError(
                f"gene_symbol_col='{gene_symbol_col}' not in adata.var.columns. "
                f"Available: {list(adata.var.columns)}"
            )
        symbols = adata.var[gene_symbol_col].astype(str)
        adata.var["ensembl_id"] = adata.var_names
        unique_mask = ~symbols.duplicated(keep="first")
        adata = adata[:, unique_mask].copy()
        adata.var_names = symbols[unique_mask].values
        adata.var_names_make_unique()
        log.info(f"  Remapped var_names via '{gene_symbol_col}'. Genes: {adata.n_vars:,}.")

    log.info(f"  RNA: {adata.n_obs:,} cells x {adata.n_vars:,} genes.")
    log.info(f"  Sample var_names: {list(adata.var_names[:5])}")
    return adata


def _looks_prenormalised(X) -> bool:
    sample = X[:min(50, X.shape[0]), :min(50, X.shape[1])]
    if sp.issparse(sample):
        sample = sample.toarray()
    return not np.all(sample == sample.astype(int))


def preprocess(
    adata,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    n_hvg: int = 2000,
    random_state: int = 42,
    skip_qc: bool = False,
):
    """QC -> normalise -> HVG -> PCA -> UMAP -> leiden. Returns (adata, expr_df)."""
    import scanpy as sc

    log.info("-- Preprocessing ---------------------------------------------")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    log.info(
        f"  Input: {adata.n_obs:,} cells, {adata.n_vars:,} genes. "
        f"Median genes/cell: {np.median(adata.obs['n_genes_by_counts']):.0f}."
    )

    if not skip_qc:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
        log.info(f"  After QC: {adata.n_obs:,} cells, {adata.n_vars:,} genes.")
    else:
        log.info("  QC skipped (skip_qc=True).")

    if _looks_prenormalised(adata.X):
        log.info("  Pre-normalised matrix detected - skipping normalize_total.")
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()

    if "highly_variable" not in adata.var.columns:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3", span=0.3)
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
        log.info(f"  HVGs computed: {int(adata.var['highly_variable'].sum()):,}.")
    else:
        log.info(f"  Using existing HVG annotation: {int(adata.var['highly_variable'].sum()):,} genes.")

    adata_sc = adata.copy()
    sc.pp.scale(adata_sc, max_value=10)
    sc.tl.pca(adata_sc, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(adata_sc, n_neighbors=15, n_pcs=40, random_state=random_state)
    sc.tl.umap(adata_sc, random_state=random_state)
    sc.tl.leiden(adata_sc, resolution=0.5, random_state=random_state)
    adata.obsm["X_pca"]  = adata_sc.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_sc.obsm["X_umap"]
    adata.obs["leiden"]  = adata_sc.obs["leiden"]

    hvg_mask = adata.var["highly_variable"]
    X_hvg = adata[:, hvg_mask].layers["log_norm"]
    if sp.issparse(X_hvg):
        X_hvg = X_hvg.toarray()
    expr_df = pd.DataFrame(X_hvg, index=adata.obs_names, columns=adata.var_names[hvg_mask])
    log.info(f"  GRN input matrix: {expr_df.shape} (cells x HVGs).")
    return adata, expr_df


def run_grnboost2(
    expr_df: pd.DataFrame,
    tf_names: List[str],
    outdir: Path,
    top_k: int = 100_000,
    seed: int = 42,
    n_workers: int = 1,
    threads_per_worker: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    '''Run GRNBoost2 and save all outputs to outdir/grnboost2/.'''
    method_dir = outdir / "grnboost2"
    method_dir.mkdir(parents=True, exist_ok=True)

    adj_path   = method_dir / "grnboost2_adjacencies.csv"
    edges_path = method_dir / "grnboost2_edges.csv"
    stats_path = method_dir / "grnboost2_stats.json"

    if adj_path.exists() and not force:
        log.info(f"  GRNBoost2: output already exists, loading from {adj_path}")
        log.info("  (Delete grnboost2/grnboost2_adjacencies.csv to force rerun.)")
        return pd.read_csv(adj_path)

    try:
        from arboreto.algo import grnboost2
        from dask.distributed import Client, LocalCluster
    except ImportError:
        raise ImportError(
            "arboreto / dask not found.\n"
            "Install: pip install 'arboreto==0.1.5' 'dask==2022.2.0' 'distributed==2022.2.0'"
        )

    valid_tfs = [tf for tf in tf_names if tf in expr_df.columns]
    log.info(
        f"  GRNBoost2: {expr_df.shape[0]:,} cells | {expr_df.shape[1]:,} genes | "
        f"{len(valid_tfs):,} TFs"
    )

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit="auto")
    client  = Client(cluster)
    log.info(f"  Dask dashboard: {client.dashboard_link}")

    try:
        t0  = time.time()
        net = grnboost2(
            expression_data=expr_df,
            tf_names=valid_tfs,
            verbose=True,
            seed=seed,
            client_or_address=client,
        )
        elapsed = time.time() - t0
        log.info(f"  Done in {elapsed:.1f}s - {len(net):,} raw edges.")
    finally:
        client.close()
        cluster.close()

    net.columns = ["TF", "target", "importance"]
    net = net.sort_values("importance", ascending=False).reset_index(drop=True)

    net.to_csv(adj_path, index=False)
    log.info(f"  GRNBoost2 adjacencies ({len(net):,} edges) -> {adj_path}")

    top = net.head(top_k)
    top.to_csv(edges_path, index=False)
    log.info(f"  GRNBoost2 top-{top_k:,} edges              -> {edges_path}")

    stats = _basic_stats(top)
    stats["n_raw_edges"] = int(len(net))
    stats["runtime_seconds"] = round(elapsed, 1)
    _save_stats(stats, stats_path)

    return net


def _merge_and_save(
    networks: Dict[str, pd.DataFrame],
    outdir: Path,
    top_k: int = 100_000,
    skip_viz: bool = False,
) -> pd.DataFrame:
    """Rank-normalise, merge, compute consensus_score, save all merged outputs."""
    merge_dir = outdir / "merged"
    merge_dir.mkdir(parents=True, exist_ok=True)

    log.info("-- Merging networks ------------------------------------------")
    dfs = []
    for method, df in networks.items():
        if df is None or df.empty:
            log.warning(f"  {method}: empty - skipping in merge.")
            continue
        df = df[df["TF"] != df["target"]].copy()
        rng = df["importance"].max() - df["importance"].min()
        df["score_norm"] = (df["importance"] - df["importance"].min()) / (rng + 1e-12)
        df["method"] = method
        dfs.append(df.head(top_k))

    if not dfs:
        raise RuntimeError("All methods returned empty networks.")

    combined = pd.concat(dfs, ignore_index=True)
    consensus = (
        combined.groupby(["TF", "target"])["score_norm"]
        .mean().reset_index()
        .rename(columns={"score_norm": "consensus_score"})
    )
    combined = combined.merge(consensus, on=["TF", "target"], how="left")
    combined = combined.sort_values("consensus_score", ascending=False)

    n_unique = combined[["TF", "target"]].drop_duplicates().shape[0]
    log.info(f"  Unique edges: {n_unique:,}")

    p = merge_dir / "grn_edges.csv"
    combined.to_csv(p, index=False)
    log.info(f"  Merged edges     -> {p}")

    p = merge_dir / "grn_network.sif"
    (
        combined[["TF", "target"]].drop_duplicates()
        .assign(interaction="regulates")[["TF", "interaction", "target"]]
        .to_csv(p, sep="\t", index=False, header=False)
    )
    log.info(f"  Cytoscape SIF    -> {p}")

    methods = combined["method"].unique()
    if len(methods) > 1:
        sets = {m: set(zip(g["TF"], g["target"])) for m, g in combined.groupby("method")}
        rows = [{"method_A": a, "method_B": b, "shared_edges": len(sets[a] & sets[b])}
                for a in methods for b in methods]
        p = merge_dir / "grn_method_overlap.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        log.info(f"  Method overlap   -> {p}")

    if not skip_viz:
        _save_html(combined, merge_dir)
        _save_plots(combined, merge_dir)

    return combined


def _save_html(edge_df: pd.DataFrame, outdir: Path, max_edges: int = 500) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        log.warning("pyvis not installed - skipping HTML. pip install pyvis")
        return

    top = edge_df[["TF", "target", "consensus_score", "method"]].drop_duplicates(
        subset=["TF", "target"]
    ).head(max_edges)
    tf_set = set(top["TF"])
    method_colours = {"GRNBoost2": "#53c0f0", "SCENIC": "#f0a500", "Correlation": "#a0e0a0"}

    net = Network(height="850px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="#e0e0e0")
    net.barnes_hut(gravity=-12000, central_gravity=0.3, spring_length=150)

    added = set()
    for _, row in top.iterrows():
        for node in [row["TF"], row["target"]]:
            if node not in added:
                is_tf = node in tf_set
                net.add_node(node, label=node, size=28 if is_tf else 10,
                             color="#e94560" if is_tf else "#0f3460",
                             title=f"{'TF' if is_tf else 'Target'}: {node}")
                added.add(node)
        net.add_edge(row["TF"], row["target"],
                     value=float(row["consensus_score"]),
                     title=f"score: {row['consensus_score']:.3f} | {row.get('method','')}",
                     color=method_colours.get(row.get("method", ""), "#53c0f0"),
                     arrows="to")

    net.set_options('{"physics":{"barnesHut":{"gravitationalConstant":-12000},'
                    '"stabilization":{"iterations":200}},'
                    '"interaction":{"hover":true,"tooltipDelay":100}}')
    p = str(outdir / "grn_network.html")
    net.write_html(p)
    log.info(f"  HTML network     -> {p}")


def _save_plots(edge_df: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed - skipping plots.")
        return

    top = (
        edge_df.groupby("TF")["target"].count()
        .sort_values(ascending=False).head(15).sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot.barh(ax=ax, color="#e94560")
    ax.set_xlabel("Out-degree (# targets regulated)")
    ax.set_title("Top 15 hub TFs by out-degree")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "top_regulators.png", dpi=150)
    plt.close()
    log.info(f"  Regulator plot   -> {outdir / 'top_regulators.png'}")

    if edge_df["method"].nunique() > 1:
        counts = edge_df.groupby("method").size().sort_values(ascending=True)
        colours = ["#e94560", "#f0a500", "#53c0f0", "#a0e0a0"]
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot.barh(ax=ax, color=colours[:len(counts)])
        ax.set_xlabel("Number of edges")
        ax.set_title("Edges per GRN method (before merge)")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(outdir / "grn_method_comparison.png", dpi=150)
        plt.close()
        log.info(f"  Method comparison-> {outdir / 'grn_method_comparison.png'}")


def _store_in_adata(adata, edge_df: pd.DataFrame, scenic_results: Optional[Dict]) -> None:
    adata.uns["grn"] = {
        "edge_list":   edge_df[["TF", "target", "consensus_score", "method"]].to_dict("list"),
        "description": "GRN inferred by grn_pipeline.py",
        "methods":     list(edge_df["method"].unique()),
    }

    if scenic_results and "auc_matrix" in scenic_results:
        auc = scenic_results["auc_matrix"]
        shared = adata.obs_names.intersection(auc.index)
        adata.obsm["X_scenic_auc"] = auc.loc[shared].values
        adata.uns["grn"]["scenic_regulon_names"] = list(auc.columns)
        log.info(
            f"  SCENIC AUC stored in adata.obsm['X_scenic_auc'] "
            f"({len(shared)} cells x {auc.shape[1]} regulons)."
        )

    for col in ["grn_out_degree", "grn_in_degree"]:
        if col in adata.var.columns:
            adata.var.drop(columns=[col], inplace=True)
    out_deg = edge_df.groupby("TF")["target"].count().rename("grn_out_degree")
    in_deg  = edge_df.groupby("target")["TF"].count().rename("grn_in_degree")
    adata.var = adata.var.join(out_deg, how="left").join(in_deg, how="left")
    adata.var[["grn_out_degree", "grn_in_degree"]] = (
        adata.var[["grn_out_degree", "grn_in_degree"]].fillna(0).astype(int)
    )


def run_grn_pipeline(
    mdata,
    outdir: str = "grn_output",
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    n_hvg: int = 2000,
    skip_qc: bool = False,
    gene_symbol_col: Optional[str] = None,
    top_k: int = 100_000,
    n_workers: int = 1,
    threads_per_worker: int = 4,
    grnboost2_adjacencies=None,
    force_grnboost2: bool = False,
    tf_cache_path: Optional[str] = None,
    random_state: int = 42,
    skip_viz: bool = False,
) -> Dict:
    '''Run GRNBoost2 inference on mdata.mod["rna"]. The SCENIC baseline lives in grn_analysis.py.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"  Output root: {outdir.resolve()}")

    log.info("-- [1/4] Extracting RNA ---------------------------------------")
    adata = extract_rna(mdata, gene_symbol_col=gene_symbol_col)

    log.info("-- [2/4] Preprocessing ----------------------------------------")
    adata, expr_df = preprocess(
        adata, min_genes=min_genes, min_cells=min_cells,
        max_pct_mito=max_pct_mito, n_hvg=n_hvg,
        random_state=random_state, skip_qc=skip_qc,
    )

    log.info("-- [3/4] Fetching TF list -------------------------------------")
    tf_names = fetch_tf_list(expr_df.columns.tolist(), cache_path=tf_cache_path)
    log.info(
        f"  TF/HVG overlap: "
        f"{sum(1 for tf in tf_names if tf in expr_df.columns)}/{len(tf_names)} TFs."
    )

    log.info("-- [4/4] GRN inference (GRNBoost2) -----------------------------")
    grnboost2_adj = run_grnboost2(
        expr_df, tf_names, outdir,
        top_k=top_k, seed=random_state,
        n_workers=n_workers, threads_per_worker=threads_per_worker,
        force=force_grnboost2,
    )
    networks = {"GRNBoost2": grnboost2_adj.head(top_k)}

    log.info("-- Merging & saving merged outputs ----------------------------")
    edge_df = _merge_and_save(networks, outdir, top_k=top_k, skip_viz=skip_viz)

    import networkx as nx
    G = nx.from_pandas_edgelist(
        edge_df[["TF", "target", "consensus_score"]].drop_duplicates(subset=["TF","target"]),
        source="TF", target="target", edge_attr="consensus_score",
        create_using=nx.DiGraph(),
    )
    stats = {
        "n_nodes":   G.number_of_nodes(),
        "n_edges":   G.number_of_edges(),
        "n_tfs":     int(edge_df["TF"].nunique()),
        "n_targets": int(edge_df["target"].nunique()),
        "methods":   list(networks.keys()),
        "top_regulators": (
            pd.Series(dict(G.out_degree()))
            .sort_values(ascending=False).head(20).to_dict()
        ),
    }
    _save_stats(stats, outdir / "merged" / "merged_stats.json")

    _store_in_adata(adata, edge_df, None)
    h5ad_path = outdir / "adata_with_grn.h5ad"
    adata.write_h5ad(h5ad_path)
    log.info(f"  AnnData          -> {h5ad_path}")

    log.info("=" * 62)
    log.info("  GRN pipeline complete.")
    log.info(f"    Cells    : {adata.n_obs:,}")
    log.info(f"    HVGs     : {int(adata.var['highly_variable'].sum()):,}")
    log.info(f"    TFs      : {len(tf_names):,}")
    log.info(f"    Nodes    : {stats['n_nodes']:,}")
    log.info(f"    Edges    : {stats['n_edges']:,}")
    log.info(f"    Output   : {outdir.resolve()}")
    log.info("=" * 62)

    return {"edge_df": edge_df, "stats": stats, "adata": adata}