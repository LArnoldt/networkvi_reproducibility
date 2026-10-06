from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("STD_GRN")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


CELL_TYPES_OF_INTEREST = {
    "CD14+ Mono": "CD14+ Mono",
    "CD16+ Mono": "CD16+ Mono",
    "NK":        "NK",
    "pDC":       "pDC",
    "Naive CD20+ B IGKC+":   "Naive CD20+ B IGKC+",
}

NETWORKVI_GO_TERMS = {
    "CD14+ Mono": [
        ("GO:0002457", "T cell antigen processing and presentation"),
        ("GO:0002444", "myeloid leukocyte mediated immunity"),
        ("GO:0002200", "somatic diversification of immune receptors"),
    ],
    "CD16+ Mono": [
        ("GO:0002457", "T cell antigen processing and presentation"),
    ],
    "NK": [
        ("GO:0002519", "natural killer cell tolerance induction"),
    ],
    "pDC": [
        ("GO:0002270", "plasmacytoid dendritic cell activation"),
    ],
    "Naive CD20+ B IGKC+": [
        ("GO:0006959", "humoral immune response"),
        ("GO:0002208", "adaptive immune response based on somatic recombination"),
    ],
}

NETWORKVI_SUBSTATES = {
    "CD14+ Mono": {
        "ICAM1_inflammatory":        ["ICAM1"],
        "Transitional_inflammatory": ["FCGR3A", "CX3CR1"],
    },
    "CD16+ Mono": {
        "ICAM1_inflammatory":        ["ICAM1"],
    },
    "NK": {
        "Transitional_NK":           ["NCAM1", "TIGIT"],
        "Active_NK_I":               ["HAVCR2", "FCGR3A"],
    },
    "pDC": {
        "IRF8_high":                 ["IRF8", "BCL11A"],
    },
}

_CISTARGET_BASE = (
    "https://resources.aertslab.org/cistarget/databases/"
    "homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/"
)
_CISTARGET_DBS = {
    "500bp": "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    "10kbp": "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
}
_MOTIF_URL = (
    "https://resources.aertslab.org/cistarget/motif2tf/"
    "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
)


def _save_json(d: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(
            {k: v for k, v in d.items()
             if isinstance(v, (int, float, str, list, dict, bool))},
            f, indent=2,
        )
    log.info(f"  Stats -> {path}")


def _expr_matrix(adata_sub) -> pd.DataFrame:
    """Log-normalised cells x HVGs expression DataFrame."""
    X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()
    df = pd.DataFrame(X, columns=adata_sub.var_names)
    if "highly_variable" in adata_sub.var.columns:
        df = df.loc[:, adata_sub.var["highly_variable"]]
    return df


def _pseudo_labels(adata_sub, marker_genes: List[str]) -> Optional[np.ndarray]:
    """Top-25% mean expression of marker_genes = substate (1), rest = 0."""
    available = [g for g in marker_genes if g in adata_sub.var_names]
    if not available:
        return None
    X = adata_sub[:, available].X
    if sp.issparse(X):
        X = X.toarray()
    mean_expr = X.mean(axis=1)
    threshold = np.percentile(mean_expr, 75)
    labels = (mean_expr >= threshold).astype(int)
    return labels if labels.sum() >= 10 else None


def fetch_tf_list(gene_names: List[str], cache_path: Optional[str] = None) -> List[str]:
    import requests
    cache = Path(cache_path) if cache_path else \
        Path.home() / ".cache" / "grn_pipeline" / "human_tfs.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    gene_set = set(gene_names)
    if cache.exists():
        tfs = [l.strip() for l in cache.read_text().splitlines() if l.strip()]
        log.info(f"  TF list from cache: {cache}")
    else:
        tfs = []
        for url in [
            "https://raw.githubusercontent.com/aertslab/pySCENIC/master/resources/hs_hgnc_tfs.txt",
            "https://raw.githubusercontent.com/saezlab/CollecTRI/main/data/human_TFs_Lambert2018.txt",
        ]:
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                tfs = [l.strip() for l in r.text.splitlines() if l.strip()]
                cache.write_text("\n".join(tfs))
                log.info(f"  TF list downloaded and cached -> {cache}")
                break
            except Exception:
                pass
        if not tfs:
            log.warning("  Could not fetch TF list - using all genes.")
            return list(gene_names)
    found = [g for g in tfs if g in gene_set]
    log.info(f"  TF list: {len(tfs)} total | {len(found)} in data.")
    return found if found else list(gene_names)


def run_clustering(
        adata,
        cell_type_col: str,
        outdir: Path,
        resolutions: Tuple[float, ...] = (0.3, 0.5, 1.0, 1.5),
        random_state: int = 0,
        force: bool = False,
) -> dict:
    '''Run Leiden at multiple resolutions and compute Wilcoxon markers per cell type.'''
    import scanpy as sc

    clust_dir = outdir / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)

    if "highly_variable" not in adata.var.columns:
        sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor="seurat_v3")
    if "X_pca" not in adata.obsm:
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack", random_state=random_state)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, random_state=random_state)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=random_state)

    for res in resolutions:
        key = f"leiden_{res}"
        path = clust_dir / f"leiden_{res}.csv"
        if path.exists() and not force:
            log.info(f"  Leiden {res}: loading from {path}")
            saved = pd.read_csv(path, index_col=0)
            adata.obs[key] = saved["leiden"].astype(str).values
        else:
            sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
            pd.DataFrame({
                "cell": adata.obs_names,
                "leiden": adata.obs[key],
            }).to_csv(path, index=False)
            log.info(
                f"  Leiden {res}: {adata.obs[key].nunique()} clusters -> {path}"
            )

    markers_key = "rank_genes_cell_type"
    all_markers = {}
    any_missing = False

    for ct in adata.obs[cell_type_col].unique():
        safe_ct = ct.replace("/", "_").replace(" ", "_").replace("+", "pos")
        p = clust_dir / f"markers_{safe_ct}.csv"
        if p.exists() and not force:
            log.info(f"  Markers [{ct}]: loading from {p}")
            all_markers[ct] = pd.read_csv(p)
        else:
            any_missing = True

    if any_missing:
        log.info("  Computing Wilcoxon markers for all cell types ...")
        sc.tl.rank_genes_groups(
            adata, groupby=cell_type_col, method="wilcoxon",
            key_added=markers_key, use_raw=False,
        )
        for ct in adata.obs[cell_type_col].unique():
            safe_ct = ct.replace("/", "_").replace(" ", "_").replace("+", "pos")
            p = clust_dir / f"markers_{safe_ct}.csv"
            if p.exists() and not force:
                continue
            try:
                df = sc.get.rank_genes_groups_df(
                    adata, group=ct, key=markers_key,
                    pval_cutoff=0.05, log2fc_min=0.5,
                )
                df.to_csv(p, index=False)
                all_markers[ct] = df
                log.info(f"  Markers [{ct}]: {len(df)} genes -> {p}")
            except Exception as e:
                log.warning(f"  Markers [{ct}]: failed ({e})")

    return {"adata": adata, "markers": all_markers}


def run_marker_enrichment(
        markers_df: pd.DataFrame,
        cell_type_label: str,
        outdir: Path,
        force: bool = False,
) -> dict:
    try:
        import gseapy as gp
    except ImportError:
        log.error("gseapy not found. Install: pip install gseapy")
        return {}

    method_dir = outdir / "method_a"
    method_dir.mkdir(parents=True, exist_ok=True)

    ora_path = method_dir / "go_enrichment_markers_ora.csv"
    gsea_path = method_dir / "go_gsea_markers.csv"
    results = {}

    if ora_path.exists() and not force:
        log.info(f"  ORA [{cell_type_label}]: loading from {ora_path}")
        results["ora"] = pd.read_csv(ora_path)
    else:
        sig_up = markers_df[
            (markers_df["pvals_adj"] < 0.05) & (markers_df["logfoldchanges"] > 0.5)
            ]["names"].tolist()
        if len(sig_up) >= 5:
            try:
                ora = gp.enrichr(
                    gene_list=sig_up, gene_sets=["GO_Biological_Process_2023"],
                    organism="Human", outdir=None, verbose=False,
                )
                df = ora.results.sort_values("Adjusted P-value")
                df.to_csv(ora_path, index=False)
                results["ora"] = df
                n_sig = len(df[df["Adjusted P-value"] < 0.05])
                log.info(f"  ORA [{cell_type_label}]: {n_sig} sig GO terms -> {ora_path}")
            except Exception as e:
                log.warning(f"  ORA [{cell_type_label}]: failed ({e})")
        else:
            log.warning(f"  ORA [{cell_type_label}]: too few sig markers ({len(sig_up)}).")

    if gsea_path.exists() and not force:
        log.info(f"  GSEA [{cell_type_label}]: loading from {gsea_path}")
        results["gsea"] = pd.read_csv(gsea_path)
    else:
        ranked = pd.Series(
            markers_df["scores"].values, index=markers_df["names"].values
        ).sort_values(ascending=False)
        ranked = ranked[~ranked.index.duplicated()]
        if len(ranked) >= 10:
            try:
                gsea = gp.prerank(
                    rnk=ranked, gene_sets="GO_Biological_Process_2023",
                    organism="Human", outdir=None, verbose=False,
                    permutation_num=100, seed=42,
                )
                df = gsea.res2d.sort_values("NOM p-val")
                df.to_csv(gsea_path, index=False)
                results["gsea"] = df
                n_sig = len(df[df["FDR q-val"] < 0.25])
                log.info(f"  GSEA [{cell_type_label}]: {n_sig} GO terms (FDR<0.25) -> {gsea_path}")
            except Exception as e:
                log.warning(f"  GSEA [{cell_type_label}]: failed ({e})")

    return results


def check_leiden_substates(
        adata,
        cell_type: str,
        cell_type_col: str,
        substates: Dict[str, List[str]],
        outdir: Path,
        force: bool = False,
) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score

    p = outdir / "leiden_substate_ari.csv"
    if p.exists() and not force:
        log.info(f"  Leiden substate [{cell_type}]: loading from {p}")
        return pd.read_csv(p)

    mask = adata.obs[cell_type_col] == cell_type
    if mask.sum() < 50:
        return pd.DataFrame()
    adata_sub = adata[mask].copy()

    records = []
    for res in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        key = f"leiden_{res}"
        if key not in adata_sub.obs.columns:
            import scanpy as sc
            sc.tl.leiden(adata_sub, resolution=res, key_added=key, random_state=0)
        for substate_name, markers in substates.items():
            pseudo = _pseudo_labels(adata_sub, markers)
            if pseudo is None:
                continue
            ari = adjusted_rand_score(pseudo, adata_sub.obs[key].astype(str))
            records.append({
                "cell_type": cell_type,
                "method": "Leiden",
                "substate": substate_name,
                "markers": ";".join(markers),
                "resolution": res,
                "ari": round(ari, 4),
                "n_substate": int(pseudo.sum()),
                "n_total": int(len(pseudo)),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(p, index=False)
        log.info(
            f"  Leiden substates [{cell_type}]: "
            f"max ARI = {df.groupby('substate')['ari'].max().to_dict()} -> {p}"
        )
    return df


def run_grnboost2_celltype(
        adata,
        cell_type: str,
        cell_type_col: str,
        outdir: Path,
        tf_names: List[str],
        n_cells_max: int = 5000,
        seed: int = 42,
        force: bool = False,
) -> Optional[pd.DataFrame]:
    method_dir = outdir / "method_b"
    method_dir.mkdir(parents=True, exist_ok=True)
    adj_path = method_dir / "grn_adjacencies.csv"

    if adj_path.exists() and not force:
        log.info(f"  GRNBoost2 [{cell_type}]: loading from {adj_path}")
        return pd.read_csv(adj_path)

    mask = adata.obs[cell_type_col] == cell_type
    n_cells = mask.sum()
    if n_cells < 50:
        log.warning(f"  GRNBoost2 [{cell_type}]: too few cells ({n_cells}).")
        return None

    adata_sub = adata[mask].copy()
    if n_cells > n_cells_max:
        np.random.seed(seed)
        idx = np.random.choice(n_cells, n_cells_max, replace=False)
        adata_sub = adata_sub[idx].copy()
        log.info(f"  GRNBoost2 [{cell_type}]: subsampled to {n_cells_max:,} cells.")

    expr_df = _expr_matrix(adata_sub)
    valid_tfs = [tf for tf in tf_names if tf in expr_df.columns]
    log.info(
        f"  GRNBoost2 [{cell_type}]: {expr_df.shape[0]:,} cells | "
        f"{expr_df.shape[1]:,} genes | {len(valid_tfs):,} TFs"
    )
    if not valid_tfs:
        log.warning(f"  GRNBoost2 [{cell_type}]: no TFs found.")
        return None

    try:
        from arboreto.algo import grnboost2
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(n_workers=1, threads_per_worker=4, memory_limit="auto")
        client = Client(cluster)
        try:
            net = grnboost2(
                expression_data=expr_df, tf_names=valid_tfs,
                verbose=True, seed=seed, client_or_address=client,
            )
        finally:
            client.close()
            cluster.close()
    except ImportError:
        log.error("arboreto not found. Install: pip install arboreto==0.1.5")
        return None

    net.columns = ["TF", "target", "importance"]
    net = net.sort_values("importance", ascending=False).reset_index(drop=True)
    net.to_csv(adj_path, index=False)
    log.info(f"  GRNBoost2 [{cell_type}]: {len(net):,} edges -> {adj_path}")
    return net


def run_hub_tf_enrichment(
        adjacencies: pd.DataFrame,
        cell_type_label: str,
        outdir: Path,
        top_tfs: int = 10,
        min_targets: int = 10,
        force: bool = False,
) -> dict:
    try:
        import gseapy as gp
    except ImportError:
        log.error("gseapy not found.")
        return {}

    method_dir = outdir / "method_b"
    method_dir.mkdir(parents=True, exist_ok=True)

    hub_path = method_dir / "grn_hub_tfs.csv"
    tf_path = method_dir / "go_enrichment_per_tf.csv"
    agg_path = method_dir / "go_enrichment_targets_aggregated.csv"

    if agg_path.exists() and not force:
        log.info(f"  Hub TF enrichment [{cell_type_label}]: loading from {agg_path}")
        return {
            "aggregated": pd.read_csv(agg_path, index_col=0),
            "per_tf": pd.read_csv(tf_path) if tf_path.exists() else pd.DataFrame(),
        }

    hub_tfs = (
        adjacencies.groupby("TF")["target"].count()
        .sort_values(ascending=False).head(top_tfs)
    )
    hub_tfs.to_csv(hub_path, header=["n_targets"])
    log.info(f"  Hub TFs [{cell_type_label}]: {list(hub_tfs.index[:5])} -> {hub_path}")

    per_tf = []
    for tf in hub_tfs.index:
        targets = adjacencies[adjacencies["TF"] == tf]["target"].tolist()
        if len(targets) < min_targets:
            continue
        try:
            ora = gp.enrichr(
                gene_list=targets, gene_sets=["GO_Biological_Process_2023"],
                organism="Human", outdir=None, verbose=False,
            )
            sig = ora.results[ora.results["Adjusted P-value"] < 0.05].copy()
            sig["TF"] = tf
            per_tf.append(sig)
        except Exception as e:
            log.warning(f"    ORA for TF {tf}: {e}")

    results = {}
    if per_tf:
        df = pd.concat(per_tf, ignore_index=True)
        df.to_csv(tf_path, index=False)
        agg = (
            df.groupby("Term")
            .agg(
                n_tfs=("TF", "nunique"),
                min_padj=("Adjusted P-value", "min"),
                tfs=("TF", lambda x: ";".join(sorted(set(x)))),
            )
            .sort_values(["n_tfs", "min_padj"], ascending=[False, True])
        )
        agg.to_csv(agg_path)
        results["per_tf"] = df
        results["aggregated"] = agg
        log.info(
            f"  Hub TF enrichment [{cell_type_label}]: "
            f"{len(agg[agg['n_tfs'] >= 2])} GO terms in >=2 hub TFs -> {agg_path}"
        )
    return results


def _fetch_scenic_assets(cache_dir: Path) -> tuple:
    '''Download cisTarget databases and motif annotation table.'''
    import requests

    _BASE = (
        "https://resources.aertslab.org/cistarget/databases/"
        "homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/"
    )
    _DBS = {
        "500bp": "hg38_500bp_up_100bp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
        "10kbp": "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
    }
    _MOTIF_URL = (
        "https://resources.aertslab.org/cistarget/motif2tf/"
        "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
    )

    db_dir = cache_dir / "scenic"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_paths = []

    for label, fname in _DBS.items():
        dest = db_dir / fname
        if not dest.exists():
            url = _BASE + fname
            log.info(f"  Downloading cisTarget {label} DB (~1-4 GB): {url}")
            log.info("  This will take several minutes ...")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
            log.info(f"  Saved -> {dest}")
        else:
            log.info(f"  cisTarget {label} cached: {dest}")
        db_paths.append(dest)

    motif_path = db_dir / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
    if not motif_path.exists():
        log.info("  Downloading motif annotation table ...")
        r = requests.get(_MOTIF_URL, timeout=120)
        r.raise_for_status()
        motif_path.write_bytes(r.content)
        log.info(f"  Saved -> {motif_path}")
    else:
        log.info(f"  Motif annotation cached: {motif_path}")

    return db_paths, motif_path


def _fetch_tf_list_file(cache_dir: Path) -> Path:
    """Download the human TF list as a plain text file for the pySCENIC CLI."""
    import requests
    tf_path = cache_dir / "scenic" / "allTFs_hg38.txt"
    tf_path.parent.mkdir(parents=True, exist_ok=True)
    if not tf_path.exists():
        url = "https://raw.githubusercontent.com/aertslab/pySCENIC/master/resources/hs_hgnc_tfs.txt"
        log.info(f"  Downloading TF list -> {tf_path}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        tf_path.write_text(r.text)
    return tf_path


def run_scenic_celltype(
        adata,
        cell_type: str,
        cell_type_col: str,
        outdir: Path,
        cache_dir: Path,
        tf_names: list,
        sif_path: str,
        precomputed_adjacencies=None,
        n_cells_max: int = 5000,
        n_workers: int = 4,
        seed: int = 42,
        force_step1: bool = False,
        force_step2: bool = False,
        force_step3: bool = False,
) -> dict:
    '''Full pySCENIC 0.12.1 pipeline via Singularity CLI.'''
    import subprocess
    import shutil

    method_dir = outdir / "method_c"
    method_dir.mkdir(parents=True, exist_ok=True)

    expr_csv_path = method_dir / "expression_matrix.csv"
    adj_path = method_dir / "scenic_adjacencies.tsv"
    reg_path = method_dir / "scenic_regulons.csv"
    auc_path = method_dir / "scenic_auc_matrix.csv"
    stats_path = method_dir / "scenic_stats.json"

    if not shutil.which("singularity"):
        raise RuntimeError(
            "singularity not found on PATH. "
            "Make sure Singularity/Apptainer is loaded, e.g.: module load singularity"
        )
    if not Path(sif_path).exists():
        raise RuntimeError(
            f"Singularity image not found: {sif_path}\n"
            "Pull it with: singularity pull pyscenic_0.12.1.sif docker://aertslab/pyscenic:0.12.1"
        )

    stats = {}

    mask = adata.obs[cell_type_col] == cell_type
    n_cells = mask.sum()
    if n_cells < 50:
        log.warning(f"  SCENIC [{cell_type}]: too few cells ({n_cells}), skipping.")
        return {}

    adata_sub = adata[mask].copy()
    if n_cells > n_cells_max:
        import numpy as np
        np.random.seed(seed)
        idx = np.random.choice(n_cells, n_cells_max, replace=False)
        adata_sub = adata_sub[idx].copy()
        log.info(f"  SCENIC [{cell_type}]: subsampled to {n_cells_max:,} cells.")

    db_paths, motif_path = _fetch_scenic_assets(cache_dir)
    tf_list_path = _fetch_tf_list_file(cache_dir)

    bind_arg = f"{method_dir.resolve()}:/scenic_work,{cache_dir.resolve()}/scenic:/scenic_db"

    def _container_path(host_path: Path) -> str:
        """Convert a host path to the path inside the container."""
        host_path = Path(host_path).resolve()
        if str(host_path).startswith(str(method_dir.resolve())):
            return "/scenic_work/" + host_path.name
        if str(host_path).startswith(str((cache_dir / "scenic").resolve())):
            return "/scenic_db/" + host_path.name
        raise ValueError(f"Path {host_path} is not under method_dir or cache_dir/scenic")

    def _run(cmd: list, step_name: str) -> None:
        """Run a subprocess command, logging output and raising on failure."""
        log.info(f"  SCENIC [{cell_type}] {step_name}: running ...")
        log.info(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            log.error(
                f"  SCENIC [{cell_type}] {step_name} FAILED (exit {result.returncode}).\n"
                f"  Output:\n{result.stdout[-3000:]}"
            )
            raise RuntimeError(
                f"pyscenic {step_name} failed for cell type '{cell_type}'. "
                f"See log above for details."
            )
        log.info(f"  SCENIC [{cell_type}] {step_name}: done.")
        last_lines = [l for l in result.stdout.strip().splitlines() if l][-5:]
        for line in last_lines:
            log.info(f"    {line}")

    if adj_path.exists() and not force_step1:
        log.info(f"  SCENIC [{cell_type}] step 1: loading {adj_path}")
        adjacencies = pd.read_csv(adj_path, sep="	")
    else:
        if precomputed_adjacencies is not None:
            log.info(
                f"  SCENIC [{cell_type}] step 1: reusing GRNBoost2 adjacencies "
                f"({len(precomputed_adjacencies):,} edges)."
            )
            adjacencies = precomputed_adjacencies.copy()
            if list(adjacencies.columns[:3]) != ["TF", "target", "importance"]:
                adjacencies.columns = ["TF", "target", "importance"] + list(adjacencies.columns[3:])
            adjacencies.to_csv(adj_path, sep="	", index=False)
            log.info(f"  SCENIC [{cell_type}] step 1: adjacencies -> {adj_path}")
        else:
            if not expr_csv_path.exists():
                log.info(f"  SCENIC [{cell_type}]: writing expression matrix -> {expr_csv_path}")
                expr_df = _expr_matrix(adata_sub)
                expr_df.to_csv(expr_csv_path)
                log.info(f"  Expression matrix: {expr_df.shape}")

            cmd = [
                "singularity", "exec",
                "--bind", bind_arg,
                sif_path,
                "pyscenic", "grn",
                _container_path(expr_csv_path),
                _container_path(tf_list_path),
                "--method", "grnboost2",
                "--num_workers", str(n_workers),
                "--seed", str(seed),
                "-o", _container_path(adj_path),
            ]
            _run(cmd, "grn")
            adjacencies = pd.read_csv(adj_path, sep="	")

    stats["n_adjacencies"] = int(len(adjacencies))
    log.info(f"  SCENIC [{cell_type}] step 1: {len(adjacencies):,} adjacencies.")

    parsed_reg_path = reg_path.with_suffix(".parsed.csv")
    if (reg_path.exists() or parsed_reg_path.exists()) and not force_step2:
        if parsed_reg_path.exists():
            log.info(f"  SCENIC [{cell_type}] step 2: loading parsed regulons from {parsed_reg_path}")
            regulon_df = pd.read_csv(parsed_reg_path)
        else:
            log.info(f"  SCENIC [{cell_type}] step 2: loading and parsing {reg_path}")
            regulon_df = _parse_regulon_csv(reg_path)
            regulon_df.to_csv(parsed_reg_path, index=False)
    else:
        if not expr_csv_path.exists():
            log.info(f"  SCENIC [{cell_type}]: writing expression matrix -> {expr_csv_path}")
            expr_df = _expr_matrix(adata_sub)
            expr_df.to_csv(expr_csv_path)

        db_container_paths = [_container_path(p) for p in db_paths]

        cmd = [
                  "singularity", "exec",
                  "--bind", bind_arg,
                  sif_path,
                  "pyscenic", "ctx",
                  _container_path(adj_path),
              ] + db_container_paths + [
                  "--annotations_fname", _container_path(motif_path),
                  "--expression_mtx_fname", _container_path(expr_csv_path),
                  "--mode", "custom_multiprocessing",
                  "--output", _container_path(reg_path),
                  "--num_workers", str(n_workers),
              ]
        _run(cmd, "ctx")

        regulon_df = _parse_regulon_csv(reg_path)
        regulon_df.to_csv(reg_path.with_suffix(".parsed.csv"), index=False)
        log.info(
            f"  SCENIC [{cell_type}] step 2: {regulon_df['regulon'].nunique()} regulons, "
            f"{len(regulon_df):,} TF-target pairs -> {reg_path}"
        )

    stats["n_regulons"] = int(regulon_df["regulon"].nunique()) if not regulon_df.empty else 0
    stats["n_regulon_edges"] = int(len(regulon_df))

    if auc_path.exists() and not force_step3:
        log.info(f"  SCENIC [{cell_type}] step 3: loading {auc_path}")
        auc_matrix = pd.read_csv(auc_path, index_col=0)
    else:
        if not expr_csv_path.exists():
            log.info(f"  SCENIC [{cell_type}]: writing expression matrix -> {expr_csv_path}")
            expr_df = _expr_matrix(adata_sub)
            expr_df.to_csv(expr_csv_path)

        cmd = [
            "singularity", "exec",
            "--bind", bind_arg,
            sif_path,
            "pyscenic", "aucell",
            _container_path(expr_csv_path),
            _container_path(reg_path),
            "-o", _container_path(auc_path),
            "--num_workers", str(n_workers),
            "--seed", str(seed),
        ]
        _run(cmd, "aucell")
        auc_matrix = pd.read_csv(auc_path, index_col=0)
        log.info(f"  SCENIC [{cell_type}] step 3: AUC matrix {auc_matrix.shape} -> {auc_path}")

    stats["auc_shape"] = list(auc_matrix.shape)
    _save_json(stats, stats_path)

    return {
        "adjacencies": adjacencies,
        "regulon_df": regulon_df,
        "auc_matrix": auc_matrix,
        "edge_df": regulon_df[["TF", "target", "importance"]].copy()
        if not regulon_df.empty
        else pd.DataFrame(columns=["TF", "target", "importance"]),
    }


def _parse_regulon_csv(reg_path: Path) -> pd.DataFrame:
    '''Parse the regulon CSV output from pyscenic ctx into a long-form DataFrame'''
    try:
        df_raw = pd.read_csv(reg_path, index_col=0)
        if "TF" in df_raw.columns and "target" in df_raw.columns:
            if "regulon" not in df_raw.columns:
                df_raw["regulon"] = df_raw["TF"]
            return df_raw[["TF", "target", "importance", "regulon"]]
    except Exception:
        pass

    records = []
    try:
        df_raw = pd.read_csv(reg_path)
        if "TF" in df_raw.columns:
            for _, row in df_raw.iterrows():
                tf = row.get("TF", "")
                regulon_name = f"{tf}({int(row.get('NofTargetGenes', 0))}g)"
                targets_raw = row.get("TargetGenes", "[]")
                try:
                    import ast
                    targets = ast.literal_eval(str(targets_raw))
                    for gene, weight in targets:
                        records.append({
                            "TF": tf,
                            "target": gene,
                            "importance": float(weight),
                            "regulon": regulon_name,
                        })
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"  Could not parse regulon CSV ({e}). Returning empty DataFrame.")

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["TF", "target", "importance", "regulon"])


def run_scenic_regulon_go_enrichment(
        regulon_df: pd.DataFrame,
        cell_type_label: str,
        outdir: Path,
        min_targets: int = 10,
        force: bool = False,
) -> pd.DataFrame:
    '''GO ORA on every SCENIC regulon's targets.'''
    try:
        import gseapy as gp
    except ImportError:
        log.error("gseapy not found.")
        return pd.DataFrame()

    method_dir = outdir / "method_c"
    method_dir.mkdir(parents=True, exist_ok=True)
    p = method_dir / "scenic_regulon_go_enrichment.csv"

    if p.exists() and not force:
        log.info(f"  SCENIC regulon GO enrichment [{cell_type_label}]: loading {p}")
        return pd.read_csv(p)

    log.info(
        f"  SCENIC regulon GO enrichment [{cell_type_label}]: "
        f"{regulon_df['regulon'].nunique()} regulons ..."
    )
    records = []
    for regulon_name, grp in regulon_df.groupby("regulon"):
        targets = grp["target"].tolist()
        if len(targets) < min_targets:
            continue
        try:
            ora = gp.enrichr(
                gene_list=targets, gene_sets=["GO_Biological_Process_2023"],
                organism="Human", outdir=None, verbose=False,
            )
            sig = ora.results[ora.results["Adjusted P-value"] < 0.05].copy()
            sig["regulon"] = regulon_name
            sig["TF"] = grp["TF"].iloc[0]
            sig["n_targets"] = len(targets)
            records.append(sig)
        except Exception as e:
            log.warning(f"    {regulon_name}: {e}")

    if records:
        df = pd.concat(records, ignore_index=True)
        df.to_csv(p, index=False)
        log.info(
            f"  SCENIC regulon GO enrichment [{cell_type_label}]: "
            f"{df['Term'].nunique()} unique GO terms -> {p}"
        )
        return df
    return pd.DataFrame()


def run_scenic_substate_analysis(
        auc_matrix: pd.DataFrame,
        cell_type: str,
        outdir: Path,
        networkvi_substates: Dict[str, List[str]],
        adata_sub=None,
        force: bool = False,
) -> pd.DataFrame:
    '''HDBSCAN on SCENIC's AUCell space, then ARI vs NetworkVI substate pseudo-labels.'''
    method_dir = outdir / "method_c"
    method_dir.mkdir(parents=True, exist_ok=True)
    p = method_dir / "scenic_substate_hdbscan.csv"

    if p.exists() and not force:
        log.info(f"  SCENIC substate [{cell_type}]: loading {p}")
        return pd.read_csv(p)

    try:
        import hdbscan
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        log.error("hdbscan / sklearn not found. Install: pip install hdbscan scikit-learn")
        return pd.DataFrame()

    log.info(f"  SCENIC substate [{cell_type}]: HDBSCAN on AUC {auc_matrix.shape}")
    X = auc_matrix.values
    if X.shape[1] < 2:
        log.warning(f"  SCENIC [{cell_type}]: too few regulons for PCA.")
        return pd.DataFrame()

    X_2d = PCA(n_components=min(2, X.shape[1]), random_state=0).fit_transform(X)
    labels = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(X_2d)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_2d[labels != -1], labels[labels != -1]) \
        if n_cl > 1 else 0.0

    records = []
    if adata_sub is not None:
        auc_cell_names = auc_matrix.index.tolist()
        adata_auc = None
        if hasattr(adata_sub, "obs_names"):
            common = [c for c in auc_cell_names if c in set(adata_sub.obs_names)]
            if len(common) > 0:
                adata_auc = adata_sub[common].copy()

        for substate_name, markers in networkvi_substates.items():
            src = adata_auc if adata_auc is not None else adata_sub
            pseudo = _pseudo_labels(src, markers)
            if pseudo is None:
                continue

            if len(pseudo) != len(labels):
                log.warning(
                    f"  SCENIC substate [{cell_type}] {substate_name}: "
                    f"shape mismatch pseudo={len(pseudo)} labels={len(labels)}, "
                    f"truncating to min length."
                )
                min_len = min(len(pseudo), len(labels))
                pseudo = pseudo[:min_len]
                lbl = labels[:min_len]
            else:
                lbl = labels

            non_noise = lbl != -1
            if non_noise.sum() >= 20:
                ari = adjusted_rand_score(pseudo[non_noise], lbl[non_noise])
            else:
                ari = float("nan")

            records.append({
                "cell_type": cell_type,
                "method": "SCENIC_AUCell",
                "substate": substate_name,
                "markers": ";".join(markers),
                "n_hdbscan_clusters": n_cl,
                "ari_vs_pseudolabel": round(ari, 4) if not np.isnan(ari) else float("nan"),
                "n_substate_cells": int(pseudo.sum()),
                "n_total_cells": int(len(lbl)),
                "auc_silhouette": round(sil, 4),
            })

    records.append({
        "cell_type": cell_type,
        "method": "SCENIC_AUCell",
        "substate": "OVERALL",
        "markers": "",
        "n_hdbscan_clusters": n_cl,
        "ari_vs_pseudolabel": float("nan"),
        "n_substate_cells": int((labels != -1).sum()),
        "n_total_cells": int(len(labels)),
        "auc_silhouette": round(sil, 4),
    })

    df = pd.DataFrame(records)
    df.to_csv(p, index=False)
    log.info(
        f"  SCENIC substate [{cell_type}]: {n_cl} clusters, "
        f"silhouette={sil:.3f} -> {p}"
    )
    return df


def build_comparison_table(
        method_a_results: dict,
        method_b_results: dict,
        method_c_results: dict,
        networkvi_go_terms: dict,
        outdir: Path,
        force: bool = False,
) -> pd.DataFrame:
    p = outdir / "go_overlap_networkvi.csv"
    if p.exists() and not force:
        log.info(f"  Comparison table: loading {p}")
        return pd.read_csv(p)

    records = []
    for ct_key, go_list in networkvi_go_terms.items():
        a_res = method_a_results.get(ct_key, {})
        b_res = method_b_results.get(ct_key, {})
        c_res = method_c_results.get(ct_key, {})

        for go_id, go_name in go_list:
            term_frag = go_name[:30]

            ora_a = a_res.get("ora", pd.DataFrame())
            gsea_a = a_res.get("gsea", pd.DataFrame())
            agg_b = b_res.get("aggregated", pd.DataFrame())
            sc_go = c_res.get("regulon_go", pd.DataFrame())

            found_ora_a = (not ora_a.empty and "Term" in ora_a.columns and
                           ora_a[ora_a["Adjusted P-value"] < 0.05]["Term"]
                           .str.contains(term_frag, case=False, na=False).any())
            found_gsea_a = (not gsea_a.empty and "Term" in gsea_a.columns and
                            gsea_a[gsea_a["FDR q-val"] < 0.25]["Term"]
                            .str.contains(term_frag, case=False, na=False).any())
            found_b = (not agg_b.empty and
                       agg_b.index.str.contains(term_frag, case=False, na=False).any())
            found_c = (not sc_go.empty and "Term" in sc_go.columns and
                       sc_go[sc_go["Adjusted P-value"] < 0.05]["Term"]
                       .str.contains(term_frag, case=False, na=False).any())

            records.append({
                "cell_type": ct_key,
                "go_id": go_id,
                "go_name": go_name,
                "networkvi_finds": True,
                "method_a_ora": found_ora_a,
                "method_a_gsea": found_gsea_a,
                "method_b_hub_tf_ora": found_b,
                "method_c_scenic_ora": found_c,
                "found_by_any_standard": any([found_ora_a, found_gsea_a, found_b, found_c]),
            })

    df = pd.DataFrame(records)
    df.to_csv(p, index=False)
    log.info(f"  Comparison table -> {p}")
    log.info(f"\n{df.to_string(index=False)}")
    return df


def build_summary_table(
        comparison_df: pd.DataFrame,
        leiden_ari_dfs: dict,
        scenic_substate_dfs: dict,
        outdir: Path,
        force: bool = False,
) -> pd.DataFrame:
    p = outdir / "summary_table.csv"
    if p.exists() and not force:
        log.info(f"  Summary table: loading {p}")
        return pd.read_csv(p)

    rows = []
    for ct in comparison_df["cell_type"].unique():
        ct_df = comparison_df[comparison_df["cell_type"] == ct]

        lei_df = leiden_ari_dfs.get(ct, pd.DataFrame())
        max_lei = lei_df["ari"].max() if not lei_df.empty else float("nan")

        sc_df = scenic_substate_dfs.get(ct, pd.DataFrame())
        sc_sub = sc_df[sc_df["substate"] != "OVERALL"] if not sc_df.empty else pd.DataFrame()
        max_sc = sc_sub["ari_vs_pseudolabel"].max() \
            if not sc_sub.empty and "ari_vs_pseudolabel" in sc_sub.columns \
            else float("nan")
        sc_sil = sc_df[sc_df["substate"] == "OVERALL"]["auc_silhouette"].values[0] \
            if not sc_df.empty and "OVERALL" in sc_df.get("substate", pd.Series()).values \
            else float("nan")

        rows.append({
            "cell_type": ct,
            "n_networkvi_go_terms": len(ct_df),
            "found_by_ora": int(ct_df["method_a_ora"].sum()),
            "found_by_gsea": int(ct_df["method_a_gsea"].sum()),
            "found_by_grnboost2_hub_tf_ora": int(ct_df["method_b_hub_tf_ora"].sum()),
            "found_by_scenic_regulon_ora": int(ct_df["method_c_scenic_ora"].sum()),
            "found_by_any_standard": int(ct_df["found_by_any_standard"].sum()),
            "leiden_max_ari_vs_substates": round(max_lei, 3) if not np.isnan(max_lei) else "N/A",
            "scenic_auc_max_ari_vs_substates": round(max_sc, 3) if not np.isnan(max_sc) else "N/A",
            "scenic_auc_silhouette": round(sc_sil, 3) if not np.isnan(sc_sil) else "N/A",
            "interpretation":
                "NetworkVI unique" if not ct_df["found_by_any_standard"].any()
                else "Also found by standard",
        })

    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    log.info(f"\n  Summary:\n{df.to_string(index=False)}\n  -> {p}")
    return df


def run_analysis(
        adata,
        cell_type_col: str,
        outdir: str,
        seed: int = 42,
        n_cells_max: int = 5000,
        scenic_workers: int = 4,
        scenic_cache_dir: Optional[str] = None,
        tf_cache_path: Optional[str] = None,
        sif_path: str = "pyscenic_0.12.1.sif",
        skip_grn: bool = False,
        skip_scenic: bool = False,
        force_clustering: bool = False,
        force_marker_enrichment: bool = False,
        force_leiden_substates: bool = False,
        force_grn: bool = False,
        force_hub_tf_enrichment: bool = False,
        force_scenic_step1: bool = False,
        force_scenic_step2: bool = False,
        force_scenic_step3: bool = False,
        force_scenic_regulon_go: bool = False,
        force_scenic_substates: bool = False,
        force_comparison: bool = False,
) -> dict:
    '''Full standard GRN analysis callable from main.py.'''
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(scenic_cache_dir) if scenic_cache_dir \
        else Path.home() / ".cache" / "grn_pipeline"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Standard GRN analysis | outdir: {outdir.resolve()}")
    log.info(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes | cell_type_col='{cell_type_col}'")

    tf_names = fetch_tf_list(adata.var_names.tolist(), cache_path=tf_cache_path)

    clust = run_clustering(
        adata, cell_type_col=cell_type_col,
        outdir=outdir, random_state=seed, force=force_clustering,
    )
    adata = clust["adata"]
    all_markers = clust["markers"]

    obs_cts = set(adata.obs[cell_type_col].unique())
    ct_label_map = {}
    for ct_key, ct_label in CELL_TYPES_OF_INTEREST.items():
        if ct_label in obs_cts:
            ct_label_map[ct_key] = ct_label
        else:
            matches = [c for c in obs_cts if ct_label.lower() in c.lower()]
            if matches:
                ct_label_map[ct_key] = matches[0]
                log.info(f"  Matched '{ct_label}' -> '{matches[0]}'")
            else:
                log.warning(f"  '{ct_label}' not found in data - skipping.")

    method_a_results = {}
    method_b_results = {}
    method_c_results = {}
    leiden_ari_dfs = {}
    scenic_substate_dfs = {}

    for ct_key, ct_label in ct_label_map.items():
        log.info(f"\n{'=' * 60}\n  {ct_label} [{ct_key}]\n{'=' * 60}")
        ct_outdir = outdir / "per_celltype" / ct_key
        ct_outdir.mkdir(parents=True, exist_ok=True)

        markers_df = all_markers.get(ct_label)
        if markers_df is not None and len(markers_df) > 0:
            method_a_results[ct_key] = run_marker_enrichment(
                markers_df=markers_df, cell_type_label=ct_label,
                outdir=ct_outdir, force=force_marker_enrichment,
            )

        substates = NETWORKVI_SUBSTATES.get(ct_key, {})
        if substates:
            leiden_ari_dfs[ct_key] = check_leiden_substates(
                adata=adata, cell_type=ct_label, cell_type_col=cell_type_col,
                substates=substates, outdir=ct_outdir, force=force_leiden_substates,
            )

        adjacencies = None
        if not skip_grn:
            adjacencies = run_grnboost2_celltype(
                adata=adata, cell_type=ct_label, cell_type_col=cell_type_col,
                outdir=ct_outdir, tf_names=tf_names,
                n_cells_max=n_cells_max, seed=seed, force=force_grn,
            )
            if adjacencies is not None and len(adjacencies) > 0:
                method_b_results[ct_key] = run_hub_tf_enrichment(
                    adjacencies=adjacencies, cell_type_label=ct_label,
                    outdir=ct_outdir, force=force_hub_tf_enrichment,
                )

        if not skip_scenic:
            scenic = run_scenic_celltype(
                adata=adata, cell_type=ct_label, cell_type_col=cell_type_col,
                outdir=ct_outdir, cache_dir=cache_dir, tf_names=tf_names,
                sif_path=sif_path,
                precomputed_adjacencies=adjacencies,
                n_cells_max=n_cells_max, n_workers=scenic_workers, seed=seed,
                force_step1=force_scenic_step1,
                force_step2=force_scenic_step2,
                force_step3=force_scenic_step3,
            )
            if scenic:
                if "regulon_df" in scenic and not scenic["regulon_df"].empty:
                    scenic["regulon_go"] = run_scenic_regulon_go_enrichment(
                        regulon_df=scenic["regulon_df"],
                        cell_type_label=ct_label, outdir=ct_outdir,
                        force=force_scenic_regulon_go,
                    )
                if "auc_matrix" in scenic and not scenic["auc_matrix"].empty:
                    mask = adata.obs[cell_type_col] == ct_label
                    adata_sub = adata[mask].copy()
                    scenic_substate_dfs[ct_key] = run_scenic_substate_analysis(
                        auc_matrix=scenic["auc_matrix"],
                        cell_type=ct_label, outdir=ct_outdir,
                        networkvi_substates=substates, adata_sub=adata_sub,
                        force=force_scenic_substates,
                    )
                method_c_results[ct_key] = scenic

    comp_dir = outdir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_table(
        method_a_results=method_a_results, method_b_results=method_b_results,
        method_c_results=method_c_results, networkvi_go_terms=NETWORKVI_GO_TERMS,
        outdir=comp_dir, force=force_comparison,
    )
    summary_df = build_summary_table(
        comparison_df=comparison_df, leiden_ari_dfs=leiden_ari_dfs,
        scenic_substate_dfs=scenic_substate_dfs,
        outdir=comp_dir, force=force_comparison,
    )

    log.info(f"\nAll outputs -> {outdir.resolve()}")
    return {
        "comparison_df": comparison_df,
        "summary_df": summary_df,
        "method_a_results": method_a_results,
        "method_b_results": method_b_results,
        "method_c_results": method_c_results,
        "leiden_ari_dfs": leiden_ari_dfs,
        "scenic_substate_dfs": scenic_substate_dfs,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Standard GRN baseline vs NetworkVI")
    p.add_argument("--input", required=True, help="Path to .h5ad file")
    p.add_argument("--cell_type_col", default="cell_type")
    p.add_argument("--outdir", default="standard_grn_output")
    p.add_argument("--tf_cache", default=None)
    p.add_argument("--scenic_cache", default=None)
    p.add_argument("--n_cells_max", type=int, default=5000)
    p.add_argument("--scenic_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sif_path", default="pyscenic_0.12.1.sif",
                   help="Path to pySCENIC Singularity .sif file")
    p.add_argument("--skip_grn", action="store_true")
    p.add_argument("--skip_scenic", action="store_true")
    p.add_argument("--force_clustering", action="store_true")
    p.add_argument("--force_marker_enrichment", action="store_true")
    p.add_argument("--force_leiden_substates", action="store_true")
    p.add_argument("--force_grn", action="store_true")
    p.add_argument("--force_hub_tf_enrichment", action="store_true")
    p.add_argument("--force_scenic_step1", action="store_true")
    p.add_argument("--force_scenic_step2", action="store_true")
    p.add_argument("--force_scenic_step3", action="store_true")
    p.add_argument("--force_scenic_regulon_go", action="store_true")
    p.add_argument("--force_scenic_substates", action="store_true")
    p.add_argument("--force_comparison", action="store_true")
    return p.parse_args()


def main():
    import scanpy as sc
    args = parse_args()
    adata = sc.read_h5ad(args.input)
    adata.var_names_make_unique()

    run_analysis(
        adata=adata,
        cell_type_col=args.cell_type_col,
        outdir=args.outdir,
        seed=args.seed,
        n_cells_max=args.n_cells_max,
        scenic_workers=args.scenic_workers,
        scenic_cache_dir=args.scenic_cache,
        tf_cache_path=args.tf_cache,
        sif_path=args.sif_path,
        skip_grn=args.skip_grn,
        skip_scenic=args.skip_scenic,
        force_clustering=args.force_clustering,
        force_marker_enrichment=args.force_marker_enrichment,
        force_leiden_substates=args.force_leiden_substates,
        force_grn=args.force_grn,
        force_hub_tf_enrichment=args.force_hub_tf_enrichment,
        force_scenic_step1=args.force_scenic_step1,
        force_scenic_step2=args.force_scenic_step2,
        force_scenic_step3=args.force_scenic_step3,
        force_scenic_regulon_go=args.force_scenic_regulon_go,
        force_scenic_substates=args.force_scenic_substates,
        force_comparison=args.force_comparison,
    )


if __name__ == "__main__":
    main()
