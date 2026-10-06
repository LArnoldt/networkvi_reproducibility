from pybiomart import Server
import numpy as np
import scanpy as sc
import episcanpy as epi
import muon as mu
import pandas as pd
import requests
import os

global ensembl_gene_id_start_position_end_position
server = Server(host="http://www.ensembl.org")
dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
ensembl_gene_id_start_position_end_position = dataset.query(attributes=["ensembl_gene_id", "start_position", "end_position"])

def find_nearest_gene_atac_peaks(row, max_dist):
    peak_center = (row['start'] + row['end']) // 2

    gene_starts = ensembl_gene_id_start_position_end_position['Gene start (bp)'].values
    gene_ends = ensembl_gene_id_start_position_end_position['Gene end (bp)'].values

    dist_to_start = np.abs(gene_starts - peak_center)
    dist_to_end = np.abs(gene_ends - peak_center)

    min_dists = np.minimum(dist_to_start, dist_to_end)

    valid_genes = min_dists <= max_dist

    if not np.any(valid_genes):
        return np.nan

    nearest_gene_idx = np.argmin(min_dists[valid_genes])
    nearest_gene = ensembl_gene_id_start_position_end_position[valid_genes].iloc[nearest_gene_idx]

    return nearest_gene['Gene stable ID']

def preprocessing_rna(rna):

    rna.var['mt'] = rna.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: x >= 200)
    mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)

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
