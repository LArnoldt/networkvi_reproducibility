import gseapy as gp
import scanpy as sc
from tqdm import tqdm
import regex as re
import os
import pandas as pd
import numpy as np


def _normalize_adata(adata):
    '''Normalize and log-transform an AnnData object in-place.'''
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def _base_dir(hyper_name, subsampled=None):
    """Return the root output directory, appending a subsampled suffix when relevant."""
    if subsampled and subsampled not in (False, 1.0, 1):
        return f"{hyper_name}_subsampled{subsampled}"
    return hyper_name


def _make_output_dir(hyper_name, subsampled=None):
    os.makedirs(os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis"), exist_ok=True)


def get_de_genes_scanpy(adata, group1, group2, labels_key,
                        padj_threshold=0.05, logfc_threshold=0.5):
    '''Get differentially expressed genes using Scanpy's rank_genes_groups (Wilcoxon).'''
    import scipy.sparse as sp
    import warnings
    X = adata.X
    sample = X[:200, :200] if not sp.issparse(X) else X[:200, :200].toarray()
    if sample.max() > 50:
        warnings.warn(
            "adata.X contains values > 50 -- data may not be log-normalized. "
            "Call _normalize_adata() on a raw-count copy before running DE.",
            UserWarning, stacklevel=2,
        )

    sc.tl.rank_genes_groups(
        adata,
        groupby=labels_key,
        groups=[group1],
        reference=group2,
        method='wilcoxon',
        use_raw=False,
    )
    result = sc.get.rank_genes_groups_df(adata, group=group1)
    de_genes = result[
        (result['pvals_adj'] < padj_threshold) &
        (abs(result['logfoldchanges']) > logfc_threshold)
    ]['names'].tolist()
    return de_genes, result


def run_go_enrichment_analysis(de_genes, organism='Human'):
    '''Run GO ORA via Enrichr on a list of DE genes.'''
    enr = gp.enrichr(
        gene_list=de_genes,
        gene_sets=[
            'GO_Biological_Process_2023',
            'GO_Molecular_Function_2023',
            'GO_Cellular_Component_2023',
        ],
        organism=organism,
        outdir=None,
        cutoff=0.05,
    )
    return enr.results


def save_go_enrichment_results(hyper_name, label, enr_results, subsampled=None):
    """Save full and top-20 ORA results for *label*."""
    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")

    enr_results.to_csv(
        os.path.join(base, f"go_enrichment_{label}_v2.csv"), index=False
    )
    enr_results[enr_results['Adjusted P-value'] < 0.05].head(20).to_csv(
        os.path.join(base, f"go_enrichment_{label}_top20_v2.csv"), index=False
    )


def _run_ora_for_condition(
    hyper_name, adata, labels_key, perturbation_gene, chunk,
    padj_threshold=0.05, logfc_threshold=0.5, subsampled=None
):
    '''Run ORA for one (perturbation_gene x perturbation_2) combination.'''
    label = f"{perturbation_gene}_{chunk}"

    de_genes, de_results = get_de_genes_scanpy(
        adata,
        group1=perturbation_gene,
        group2='control',
        labels_key=labels_key,
        padj_threshold=padj_threshold,
        logfc_threshold=logfc_threshold,
    )
    print(f"  [{label}] DE genes found: {len(de_genes)}")

    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    with open(os.path.join(base, f"de_genes_{label}.txt"), 'w') as f:
        f.write('\n'.join(de_genes))
    de_results.to_csv(os.path.join(base, f"de_results_{label}_v2.csv"), index=False)

    if not de_genes:
        print(f"  [{label}] No DE genes - skipping ORA.")
        return None

    print(f"  [{label}] Running GO ORA ...")
    enr_results = run_go_enrichment_analysis(de_genes, organism='Human')
    save_go_enrichment_results(hyper_name, label, enr_results, subsampled=subsampled)
    return enr_results


def write_data_for_enrichment_analysis(cell_values_path, mdata_select, train_indices, val_indices):
    adata = mdata_select.mod["rna"]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_select = adata.to_df().T
    adata_select = adata_select.rename_axis("Gene")
    adata_select.columns = [f"cell{i}" for i, _ in enumerate(adata_select.columns)]
    adata_select.insert(0, "NAME", list(adata_select.index))
    adata_select.to_csv(cell_values_path, sep="\t")


def write_phenotypes_enrichment_analysis(phenotype_values_path, mdata_select, labels_key,
                                         train_indices, val_indices):
    phenotypes = np.array(mdata_select.mod["rna"].obs[labels_key])
    unique = np.unique(phenotypes)
    with open(phenotype_values_path, "w") as f:
        f.write(f"{len(phenotypes)} {2} {1}\n")
        f.write(f"# {unique[0]} {unique[1]}\n")
        f.write(f"{' '.join(map(str, phenotypes))}")


def run_enrichment_analysis(cell_values_path, phenotype_values_path):
    return gp.gsea(
        data=cell_values_path,
        gene_sets="GO_Biological_Process_2023",
        cls=phenotype_values_path,
        permutation_type='phenotype',
        permutation_num=1000,
        outdir=None,
        method='signal_to_noise',
        threads=4,
        seed=7,
    )


def run_ss_enrichment_analysis(cell_values_path):
    return gp.ssgsea(
        data=cell_values_path,
        gene_sets="GO_Biological_Process_2023",
        outdir=None,
        sample_norm_method='rank',
        no_plot=True,
    )


def gsea_results_to_df(gs_res):
    '''Flatten a gseapy GSEA result object into a tidy DataFrame.'''
    rows = []
    for term, vals in gs_res.results.items():
        match = re.search(r'GO:\d+', term)
        go_id = match.group(0) if match else term
        rows.append({
            'go_id':   go_id,
            'go_term': term,
            'es':      float(vals.get('es',   float('nan'))),
            'nes':     float(vals.get('nes',  float('nan'))),
            'pval':    float(vals.get('pval', float('nan'))),
            'fdr':     float(vals.get('fdr',  float('nan'))),
        })
    return pd.DataFrame(rows).set_index('go_id')


def save_gsea_results(hyper_name, label, gs_res, subsampled=None):
    '''Persist a gseapy GSEA result as a tidy CSV (ES, NES, pval, FDR per GO term).'''
    df   = gsea_results_to_df(gs_res)
    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    path = os.path.join(base, f"gsea_full_{label}_v2.csv")
    df.to_csv(path)
    print(f"  Saved GSEA results → {path}  "
          f"({len(df)} GO terms, {(df['fdr'] < 0.05).sum()} significant at FDR<0.05)")
    return path


def get_enrichment_score_df(hyper_name, perturbation_2_list, gs_res_list, subsampled=None):
    go_terms_all = list({go_term for gs_res in gs_res_list for go_term in gs_res.results.keys()})
    es_results = pd.DataFrame(index=go_terms_all, columns=perturbation_2_list)
    nes_results = pd.DataFrame(index=go_terms_all, columns=perturbation_2_list)

    for chunk, gs_res in zip(perturbation_2_list, gs_res_list):
        for go_term in gs_res.results:
            es_results.loc[go_term, chunk] = gs_res.results[go_term]["es"]
            nes_results.loc[go_term, chunk] = gs_res.results[go_term]["nes"]

    _strip_go_index(es_results)
    _strip_go_index(nes_results)

    tag = '_'.join(perturbation_2_list)
    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    es_results.to_csv(os.path.join(base, f"es_results_{tag}_v2.csv"))
    nes_results.to_csv(os.path.join(base, f"nes_results_{tag}_v2.csv"))
    return es_results, nes_results


def get_ss_enrichment_score_df(hyper_name, chunk, ss, subsampled=None):
    first_key = list(ss.results.keys())[0]
    es_results = pd.DataFrame(index=ss.results[first_key].keys(), columns=ss.results.keys())
    nes_results = pd.DataFrame(index=ss.results[first_key].keys(), columns=ss.results.keys())

    for cell in tqdm(ss.results, desc="Collecting ssGSEA scores"):
        for go_term in ss.results[cell]:
            es_results.loc[go_term, cell] = ss.results[cell][go_term]["es"]
            nes_results.loc[go_term, cell] = ss.results[cell][go_term]["nes"]

    _strip_go_index(es_results)
    _strip_go_index(nes_results)

    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    es_results.to_csv(os.path.join(base, f"es_results_{chunk}_v2.csv"))
    nes_results.to_csv(os.path.join(base, f"nes_results_{chunk}_v2.csv"))
    return es_results, nes_results


def _strip_go_index(df):
    """Replace full GO term strings with bare GO IDs in-place."""
    df.index = [re.search(r'\(GO:\d+\)', term).group(0)[1:-1] for term in df.index]


def _run_gsea_for_condition(hyper_name, mdata_counts_copy, labels_key,
                            perturbation_gene, chunk, train_indices, val_indices,
                            subsampled=None):
    '''Run GSEA for one (perturbation_gene x perturbation_2) combination.'''
    label = f"{perturbation_gene}_{chunk}"
    print(f"  [{label}] Running GSEA ...")

    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    cell_values_path = os.path.join(base, f"counts_{label}.txt")
    phenotype_values_path = os.path.join(base, f"phenotypes_{label}.cls")

    write_data_for_enrichment_analysis(
        cell_values_path, mdata_counts_copy, train_indices, val_indices
    )
    write_phenotypes_enrichment_analysis(
        phenotype_values_path, mdata_counts_copy, labels_key, train_indices, val_indices
    )

    gs_res = run_enrichment_analysis(cell_values_path, phenotype_values_path)

    get_enrichment_score_df(hyper_name, [label], [gs_res], subsampled=subsampled)
    save_gsea_results(hyper_name, label, gs_res, subsampled=subsampled)
    return gs_res


def run_enrichment_for_frangieh2021(
    hyper_name,
    mdata_counts,
    labels_key,
    train_indices,
    val_indices,
    perturbation_genes=None,
    method='ORA',
    perturbation_2_list=None,
    padj_threshold=0.05,
    logfc_threshold=0.5,
    subsampled=None,
):
    '''Run GO enrichment analysis for the Frangieh2021 dataset.'''
    method = method.upper()
    if method not in ('ORA', 'GSEA'):
        raise ValueError(f"method must be 'ORA' or 'GSEA', got '{method}'")

    if perturbation_genes is None:
        perturbation_genes = ['CD58']
    if perturbation_2_list is None:
        perturbation_2_list = ['Control', 'Co-culture', 'IFNy']

    _make_output_dir(hyper_name, subsampled)

    mdata_counts.mod["rna"].obs["perturbation_2"] = (
        mdata_counts.mod["rna"].obs["perturbation_2"].replace('IFNγ', 'IFNy')
    )

    combo_key = f"{labels_key}_perturbation_2"
    mdata_counts.mod["rna"].obs[combo_key] = (
        mdata_counts.mod["rna"].obs[labels_key].astype(str)
        + "_"
        + mdata_counts.mod["rna"].obs["perturbation_2"].astype(str)
    )

    adata_full = mdata_counts.mod["rna"]

    all_results = {}

    for perturbation_gene in tqdm(perturbation_genes, desc="Perturbation genes"):
        print(f"\n=== Perturbation gene: {perturbation_gene} | method: {method} ===")
        gene_results = {}

        for chunk in tqdm(perturbation_2_list, desc=f"  Conditions [{perturbation_gene}]", leave=False):

            mask = (
                (adata_full.obs[combo_key] == f"control_{chunk}")
                | (adata_full.obs[combo_key] == f"{perturbation_gene}_{chunk}")
            )

            if method == 'ORA':
                adata_subset = adata_full[mask].copy()
                _normalize_adata(adata_subset)
                result = _run_ora_for_condition(
                    hyper_name, adata_subset, labels_key,
                    perturbation_gene, chunk,
                    padj_threshold=padj_threshold,
                    logfc_threshold=logfc_threshold,
                    subsampled=subsampled,
                )

            else:
                import mudata as md
                adata_subset = mdata_counts.mod["rna"][mask].copy()
                mdata_subset = md.MuData({"rna": adata_subset})
                result = _run_gsea_for_condition(
                    hyper_name, mdata_subset, labels_key,
                    perturbation_gene, chunk,
                    train_indices, val_indices,
                    subsampled=subsampled,
                )

            gene_results[chunk] = result

        all_results[perturbation_gene] = gene_results

    print(f"\nDone. Results in: {os.path.join(_base_dir(hyper_name, subsampled), 'enrichment_analysis')}")
    return all_results


def _make_binary_labels(adata, cell_type, labels_key, binary_key='_celltype_binary'):
    '''Add a binary obs column to *adata* in-place:'''
    adata.obs[binary_key] = np.where(
        adata.obs[labels_key] == cell_type, 'case', 'rest'
    )
    return binary_key


def _run_ora_celltype_vs_rest(
    hyper_name, adata_norm, cell_type, labels_key,
    padj_threshold=0.05, logfc_threshold=0.5, subsampled=None,
):
    '''ORA for one cell type vs. all others (pooled as 'rest').'''
    safe_label = cell_type.replace(' ', '_').replace('/', '-')

    binary_key = _make_binary_labels(adata_norm, cell_type, labels_key)

    de_genes, de_results = get_de_genes_scanpy(
        adata_norm,
        group1='case',
        group2='rest',
        labels_key=binary_key,
        padj_threshold=padj_threshold,
        logfc_threshold=logfc_threshold,
    )
    print(f"  [{safe_label} vs rest] DE genes found: {len(de_genes)}")

    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    with open(os.path.join(base, f"de_genes_{safe_label}_vs_rest.txt"), 'w') as f:
        f.write('\n'.join(de_genes))
    de_results.to_csv(
        os.path.join(base, f"de_results_{safe_label}_vs_rest_v2.csv"), index=False
    )

    if not de_genes:
        print(f"  [{safe_label} vs rest] No DE genes – skipping ORA.")
        return None

    print(f"  [{safe_label} vs rest] Running GO ORA ...")
    enr_results = run_go_enrichment_analysis(de_genes, organism='Human')
    save_go_enrichment_results(hyper_name, f"{safe_label}_vs_rest", enr_results, subsampled=subsampled)
    return enr_results


def _run_gsea_celltype_vs_rest(
    hyper_name, adata_raw, cell_type, labels_key,
    train_indices, val_indices, subsampled=None,
):
    '''GSEA for one cell type vs. all others (pooled as 'rest').'''
    import mudata as md

    safe_label = cell_type.replace(' ', '_').replace('/', '-')
    print(f"  [{safe_label} vs rest] Running GSEA ...")

    adata_subset = adata_raw.copy()
    binary_key = _make_binary_labels(adata_subset, cell_type, labels_key)

    mdata_subset = md.MuData({"rna": adata_subset})

    base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
    cell_values_path     = os.path.join(base, f"counts_{safe_label}_vs_rest.txt")
    phenotype_values_path = os.path.join(base, f"phenotypes_{safe_label}_vs_rest.cls")

    write_data_for_enrichment_analysis(
        cell_values_path, mdata_subset, train_indices, val_indices
    )
    write_phenotypes_enrichment_analysis(
        phenotype_values_path, mdata_subset, binary_key, train_indices, val_indices
    )

    gs_res = run_enrichment_analysis(cell_values_path, phenotype_values_path)

    get_enrichment_score_df(
        hyper_name, [f"{safe_label}_vs_rest"], [gs_res], subsampled=subsampled
    )
    save_gsea_results(hyper_name, f"{safe_label}_vs_rest", gs_res, subsampled=subsampled)
    return gs_res


def run_enrichment_for_neurips2021_cite(
    hyper_name,
    mdata_counts,
    labels_key,
    train_indices,
    val_indices,
    method='ORA',
    cell_types=None,
    padj_threshold=0.05,
    logfc_threshold=0.5,
    subsampled=None,
):
    '''Run one-vs-all GO enrichment for every cell type in the NeurIPS 2021 CITE-seq dataset.'''
    method = method.upper()
    if method not in ('ORA', 'GSEA'):
        raise ValueError(f"method must be 'ORA' or 'GSEA', got '{method}'")

    _make_output_dir(hyper_name, subsampled)

    adata_full = mdata_counts.mod["rna"]

    if cell_types is None:
        cell_types = sorted(adata_full.obs[labels_key].unique().tolist())

    print(f"\n=== NeurIPS 2021 CITE-seq | method: {method} | {len(cell_types)} cell types ===")

    all_results = {}

    for cell_type in tqdm(cell_types, desc="Cell types"):
        print(f"\n--- Cell type: {cell_type} ---")

        if method == 'ORA':
            adata_norm = adata_full.copy()
            _normalize_adata(adata_norm)
            result = _run_ora_celltype_vs_rest(
                hyper_name, adata_norm, cell_type, labels_key,
                padj_threshold=padj_threshold,
                logfc_threshold=logfc_threshold,
                subsampled=subsampled,
            )

        else:
            result = _run_gsea_celltype_vs_rest(
                hyper_name, adata_full, cell_type, labels_key,
                train_indices, val_indices,
                subsampled=subsampled,
            )

        all_results[cell_type] = result

    print(f"\nDone. Results in: {os.path.join(_base_dir(hyper_name, subsampled), 'enrichment_analysis')}")
    return all_results


def compare_gsea_vs_go_enrichment(gsea_results, go_results, top_n=20):
    '''Overlap analysis between GSEA and ORA top pathways.'''
    rows = []
    for condition in gsea_results:
        if condition not in go_results:
            continue
        gsea_top = set(gsea_results[condition].head(top_n)['Term'])
        go_top   = set(go_results[condition].head(top_n)['Term'])
        overlap  = gsea_top & go_top
        rows.append({
            'Condition':    condition,
            'GSEA_pathways': len(gsea_top),
            'GO_pathways':   len(go_top),
            'Overlap':       len(overlap),
            'Overlap_pct':   len(overlap) / min(len(gsea_top), len(go_top)) * 100,
        })
    return pd.DataFrame(rows)


def main(
    task, chunk, not_run_hyper, hyper_name, tool_name, dataset_name,
    subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant,
    utilize_de, spike_in, mdata, mdata_counts, adata,
    train_indices, val_indices, n_genes, n_regions, n_snps,
    labels_key, further_labels_keys, batch_key, patient_key,
    further_batch_keys, further_continuous_batch_keys,
    n_patient_covariates, protein_expression_obsm_key,
    enrichment_method='ORA',
    perturbation_genes=["ACSL3", "MYC", "ILF2", "CDK6", "DNMT1", 'CD58', "CD59"],
    neurips_cell_types=None,
):
    if paired_rate not in (1, 1.0):
        raise NotImplementedError("paired_rate != 1 is not supported.")

    _make_output_dir(hyper_name, subsampled)

    if dataset_name == "frangieh2021":
        all_indices = np.concatenate([train_indices, val_indices])
        mdata_counts = mdata_counts[all_indices, :]


        results = run_enrichment_for_frangieh2021(
            hyper_name=hyper_name,
            mdata_counts=mdata_counts,
            labels_key=labels_key,
            train_indices=all_indices,
            val_indices=all_indices,
            perturbation_genes=perturbation_genes,
            method=enrichment_method,
            perturbation_2_list=['Control', 'Co-culture', 'IFNy'],
            subsampled=subsampled,
        )
        return results

    elif dataset_name.lower() == "neurips2021_cite_bmmc":
        all_indices = np.concatenate([train_indices, val_indices])
        mdata_counts = mdata_counts[all_indices, :]

        return run_enrichment_for_neurips2021_cite(
            hyper_name=hyper_name,
            mdata_counts=mdata_counts,
            labels_key=labels_key,
            train_indices=all_indices,
            val_indices=all_indices,
            method=enrichment_method,
            cell_types=neurips_cell_types,
            subsampled=subsampled,
        )


    elif dataset_name == "stephenson2021":
        mdata = mdata[train_indices, :]
        mdata_counts = mdata_counts[train_indices, :]

        chunk_size = 90_000
        n_chunks = 6
        if chunk < 0 or chunk >= n_chunks:
            raise ValueError(f"chunk must be in [0, {n_chunks - 1}]")

        start = chunk * chunk_size
        end   = (chunk + 1) * chunk_size if chunk < n_chunks - 1 else None
        mdata        = mdata[start:end, :]
        mdata_counts = mdata_counts[start:end, :]

        base = os.path.join(_base_dir(hyper_name, subsampled), "enrichment_analysis")
        cell_values_path = os.path.join(base, f"counts_{chunk}.txt")
        write_data_for_enrichment_analysis(
            cell_values_path, mdata_counts, train_indices, val_indices
        )
        del mdata, mdata_counts

        ss = run_ss_enrichment_analysis(cell_values_path)
        return get_ss_enrichment_score_df(hyper_name, chunk, ss, subsampled=subsampled)

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not supported.")