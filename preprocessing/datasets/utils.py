import numpy as np
import scanpy as sc
import episcanpy as epi
import muon as mu
import pandas as pd
import requests
import os

ensembl_gene_id_chromosome_position = pd.read_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "ensembl_gene_id_chromosome_position_mapping", "ensembl_gene_id_chromosome_position_mapping.csv"),
    dtype={"Chromosome/scaffold name": str},
)
ensembl_genes_by_chromosome = {chromosome: genes for chromosome, genes in ensembl_gene_id_chromosome_position.groupby("Chromosome/scaffold name")}

def find_nearest_gene_atac_peaks(row, max_dist):
    genes = ensembl_genes_by_chromosome.get(str(row['chr']).replace('chr', ''))
    if genes is None:
        return np.nan

    peak_center = (row['start'] + row['end']) // 2

    dist_to_start = np.abs(genes['Gene start (bp)'].values - peak_center)
    dist_to_end = np.abs(genes['Gene end (bp)'].values - peak_center)

    min_dists = np.minimum(dist_to_start, dist_to_end)

    if min_dists.min() > max_dist:
        return np.nan

    return genes['Gene stable ID'].values[np.argmin(min_dists)]

def preprocessing_rna(rna, max_total_counts=None):

    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: x >= 200)
    mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)
    if max_total_counts is not None:
        mu.pp.filter_obs(rna, 'total_counts', lambda x: x < max_total_counts)

    rna.layers["counts"] = rna.X.copy()

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    sc.pp.highly_variable_genes(rna, n_top_genes=4000)

    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(rna)
    sc.tl.umap(rna)

    return rna

def preprocessing_atac(atac):

    epi.pp.filter_features(atac, min_cells=10)
    epi.pp.filter_cells(atac, min_features=10)

    atac.layers["counts"] = atac.X.copy()

    epi.pp.highly_variable(atac, n_features=20000)

    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)


    sc.pp.scale(atac)
    sc.tl.pca(atac, n_comps=100, svd_solver="auto")
    sc.pp.neighbors(atac)
    sc.tl.umap(atac)


    return atac

def preprocessing_prot(prot, k: int=100):

    prot.layers["counts"] = prot.X.copy()

    mu.prot.pp.clr(prot)
    sc.tl.pca(prot, n_comps=k, svd_solver="auto")
    sc.pp.neighbors(prot)
    sc.tl.umap(prot)

    return prot
