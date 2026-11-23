import numpy as np
import scanpy as sc
import anndata as ad
import subprocess
import pandas as pd
import pickle
import os
import muon as mu
import math
import re
import pickle

def load_dogma(paired_rate, subsampled, utilize_highly_variable):

    labels_key = 'majority_voting'
    further_labels_keys = []
    batch_key = None
    patient_key = None
    n_patient_covariates = 0
    further_batch_keys=None
    further_continuous_batch_keys=None
    protein_expression_obsm_key = "protein_expression"

    mdata = mu.read("./preprocessing/datasets/dogma/data/dogma.h5mu")
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
        mdata.mod["atac"] = mdata.mod["atac"].copy()[:, mdata.mod["atac"].var["highly_variable"]]
    mdata["rna"].var["modality"] = "expression"
    mdata["atac"].var["modality"] = "Peaks"
    mdata["prot"].var["modality"] = "Protein"
    mdata["rna"].obs["modality"] = "paired" 
    mdata["atac"].obs["modality"] = "paired"
    mdata["prot"].obs["modality"] = "paired"

    mdata.update()
    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = mdata["rna"].layers["counts"].todense()
    mdata_counts["atac"].X = mdata["atac"].layers["counts"].todense()
    mdata_counts["prot"].X = mdata["prot"].layers["counts"]

    adata = sc.concat([mdata_counts["rna"], mdata_counts["atac"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    for further_labels_key in further_labels_keys:
        if further_labels_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_labels_key] = mdata_counts["rna"].obs[further_labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]
    adata.var["modality"] = ["expression"] * len(mdata_counts["rna"].var) + ["Peaks"] * len(mdata_counts["atac"].var)

    n_genes = (adata.var.modality=='expression').sum()
    n_regions = (adata.var.modality=='Peaks').sum()

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs)*0.2)]
    train_indices = indices[math.floor(len(adata.obs)*0.2):]

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_hlca(paired_rate, subsampled, utilize_highly_variable):

    if paired_rate != 1.0 and paired_rate != 1:
        raise NotImplementedError("")

    labels_key = 'cell_type'
    further_labels_keys = []
    batch_key = "study"
    patient_key = "donor_id"
    further_batch_keys=None
    further_continuous_batch_keys=None
    protein_expression_obsm_key = None

    adata = ad.read_h5ad("./preprocessing/datasets/hlca/data/hlca_hvg.h5ad")
    mdata = mu.MuData({"rna": adata})
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    try:
        mdata["rna"].X = mdata["rna"].X.todense()
    except:
        print("No toarray().")
    mdata["rna"].var["modality"] = "expression"
    mdata["rna"].obs["modality"] = "paired"
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))

    mdata.update()
    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = mdata["rna"].layers["counts"]
    mdata_counts["rna"].X = mdata_counts["rna"].X.toarray()

    adata = mdata_counts["rna"]
    adata.var["modality"] = ["expression"] * len(mdata_counts["rna"].var)

    n_genes = (adata.var.modality=='expression').sum()
    n_regions = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs)*0.2)]
    train_indices = indices[math.floor(len(adata.obs)*0.2):]

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_hao2021(paired_rate, subsampled, utilize_highly_variable):

    labels_key = 'celltype.l2'
    further_labels_keys = ["Phase"]
    batch_key = "Batch"
    patient_key = "donor"
    further_batch_keys=None
    further_continuous_batch_keys=None
    protein_expression_obsm_key = "protein_expression"

    mdata = mu.read("./preprocessing/datasets/Hao2021/data/Hao2021.h5mu")
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
    mdata["rna"].var["modality"] = "expression"
    mdata["prot"].var["modality"] = "Protein"
    mdata["rna"].obs["modality"] = "paired"
    mdata["prot"].obs["modality"] = "paired"

    mdata.update()
    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = mdata["rna"].layers["counts"].todense()
    mdata_counts["prot"].X = mdata["prot"].layers["counts"].todense()

    adata = sc.concat([mdata_counts["rna"], mdata_counts["prot"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    for further_labels_key in further_labels_keys:
        if further_labels_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_labels_key] = mdata_counts["rna"].obs[further_labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]

    n_genes = (adata.var.modality == 'expression').sum()
    n_regions = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


def load_frangieh2021(paired_rate, subsampled, utilize_highly_variable, query_ref_selector=None):

    labels_key = 'perturbation'
    n_patient_covariates = 0
    further_labels_keys = []
    batch_key = None
    patient_key = None
    further_batch_keys = None
    further_continuous_batch_keys = None
    protein_expression_obsm_key = "protein_expression"

    mdata = mu.read("./preprocessing/datasets/Frangieh2021/data/Frangieh2021.h5mu")
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
    if query_ref_selector is not None:
        mdata = mdata[mdata.mod["rna"].obs[batch_key].isin(query_ref_selector)]
    mdata["rna"].var["modality"] = "expression"
    mdata["prot"].var["modality"] = "Protein"
    mdata["rna"].obs["modality"] = "paired"
    mdata["prot"].obs["modality"] = "paired"

    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = mdata["rna"].layers["counts"]
    mdata_counts["prot"].X = mdata["prot"].layers["counts"]

    adata = sc.concat([mdata_counts["rna"], mdata_counts["prot"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]

    n_genes = (adata.var.modality == 'expression').sum()
    n_regions = 0

    random_state = np.random.RandomState(seed=42)
    val_indices = np.where(mdata_counts.mod["rna"].obs[labels_key] != "control")[0]
    train_indices = np.where(mdata_counts.mod["rna"].obs[labels_key] == "control")[0]
    val_indices = random_state.permutation(val_indices)
    train_indices = random_state.permutation(train_indices)

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_neurips2021_cite_BMMC(paired_rate, subsampled, utilize_highly_variable, query_ref_selector=None):

    labels_key = 'cell_type'
    further_labels_keys = ['GEX_phase', 'DonorAge', 'DonorBMI', 'DonorBloodType', 'DonorRace', 'Ethnicity', 'DonorGender', 'DonorSmoker']
    batch_key = 'Site'
    patient_key = "DonorID"
    further_batch_keys=['DonorSmoker']
    further_continuous_batch_keys=['DonorBMI', 'DonorAge']
    protein_expression_obsm_key = "protein_expression"

    mdata = mu.read("./preprocessing/datasets/neurips2021_cite_BMMC/data/neurips2021_cite_BMMC.h5mu")
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
    if query_ref_selector is not None:
        mdata = mdata[mdata.mod["rna"].obs[batch_key].isin(query_ref_selector)]
    mdata["rna"].var["modality"] = "expression"
    mdata["prot"].var["modality"] = "Protein"
    mdata["rna"].obs["modality"] = "paired"
    mdata["prot"].obs["modality"] = "paired"

    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = np.array(mdata["rna"].layers["counts"].todense())
    mdata_counts["prot"].X = np.array(mdata["prot"].layers["counts"].todense())

    adata = sc.concat([mdata_counts["rna"], mdata_counts["prot"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    for further_labels_key in further_labels_keys:
        if further_labels_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_labels_key] = mdata_counts["rna"].obs[further_labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]
    for further_batch_key in further_batch_keys:
        if further_batch_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_batch_key] = mdata_counts["rna"].obs[further_batch_key]
    for further_continuous_batch_key in further_continuous_batch_keys:
        if further_continuous_batch_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_continuous_batch_key] = mdata_counts["rna"].obs[further_continuous_batch_key]

    n_genes = (adata.var.modality == "expression").sum()
    n_regions = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))
    
    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_neurips2021_multiome_BMMC(paired_rate, subsampled, utilize_highly_variable, query_ref_selector=None):

    labels_key = 'cell_type'
    further_labels_keys = ['GEX_phase', 'DonorAge', 'DonorBMI', 'DonorBloodType', 'DonorRace', 'Ethnicity', 'DonorGender', 'DonorSmoker']
    batch_key = 'Site'
    patient_key = "DonorID"
    further_batch_keys=None
    further_continuous_batch_keys=None
    protein_expression_obsm_key = None

    mdata = mu.read("./preprocessing/datasets/neurips2021_multiome_BMMC/data/neurips2021_multiome_BMMC.h5mu")
    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
        mdata.mod["atac"] = mdata.mod["atac"].copy()[:, mdata.mod["atac"].var["highly_variable"]]
    if query_ref_selector is not None:
        mdata = mdata[mdata.mod["rna"].obs[batch_key].isin(query_ref_selector)]
    mdata["rna"].var["modality"] = "expression"
    mdata["atac"].var["modality"] = "Peaks"
    mdata["rna"].obs["modality"] = "paired"
    mdata["atac"].obs["modality"] = "paired"

    mdata.update()
    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = mdata["rna"].layers["counts"].todense()
    mdata_counts["atac"].X = mdata["atac"].layers["counts"].todense()

    adata = sc.concat([mdata_counts["rna"], mdata_counts["atac"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    for further_labels_key in further_labels_keys:
        if further_labels_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_labels_key] = mdata_counts["rna"].obs[further_labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]

    n_genes = (adata.var.modality == 'expression').sum()
    n_regions = (adata.var.modality == 'Peaks').sum()

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))
    
    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys,batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


def load_tabula_sapiens(dataset_name, paired_rate, subsampled, utilize_highly_variable, query_ref_selector=None):
    labels_key = 'cell_type'
    further_labels_keys = []
    batch_key = 'batch'
    patient_key = None
    further_batch_keys = None
    further_continuous_batch_keys = None
    protein_expression_obsm_key = None

    if dataset_name == "ts_mammary":
        mdata = mu.read("./preprocessing/datasets/tabula_sapiens/data/tabula_sapiens_mammary.h5mu")
    elif dataset_name == "ts_intestine":
        mdata = mu.read("./preprocessing/datasets/tabula_sapiens/data/tabula_sapiens_large_intestine.h5mu")
    elif dataset_name == "ts_lung":
        mdata = mu.read("./preprocessing/datasets/tabula_sapiens/data/tabula_sapiens_lung.h5mu")
    elif dataset_name == "ts_endothelium":
        mdata = mu.read("./preprocessing/datasets/tabula_sapiens/data/tabula_sapiens_endothelium.h5mu")
    elif dataset_name == "ts_liver":
        mdata = mu.read("./preprocessing/datasets/tabula_sapiens/data/tabula_sapiens_liver.h5mu")
    else:
        raise ValueError("")
    mdata.mod["rna"].obs["batch"] = mdata.mod["rna"].obs["_scvi_batch"]

    if subsampled:
        np.random.seed(42)
        mdata = mu.pp.sample_obs(mdata, frac=subsampled)
    if utilize_highly_variable:
        mdata.mod["rna"] = mdata.mod["rna"].copy()[:, mdata.mod["rna"].var["highly_variable"]]
    if query_ref_selector is not None:
        mdata = mdata[mdata.mod["rna"].obs[batch_key].isin(query_ref_selector)]
    mdata["rna"].var["modality"] = "expression"
    mdata["rna"].obs["modality"] = "paired"
    mdata["rna"].var["gene_stable_id"] = mdata["rna"].var["ensembl_gene_id"]

    mdata_counts = mdata.copy()
    mdata_counts["rna"].X = np.array(mdata["rna"].layers["counts"].todense())

    adata = sc.concat([mdata_counts["rna"]], axis=1)
    if labels_key in mdata_counts["rna"].obs.columns:
        adata.obs[labels_key] = mdata_counts["rna"].obs[labels_key]
    for further_labels_key in further_labels_keys:
        if further_labels_key in mdata_counts["rna"].obs.columns:
            adata.obs[further_labels_key] = mdata_counts["rna"].obs[further_labels_key]
    if batch_key in mdata_counts["rna"].obs.columns:
        adata.obs[batch_key] = mdata_counts["rna"].obs[batch_key]
    if patient_key in mdata_counts["rna"].obs.columns:
        adata.obs[patient_key] = mdata_counts["rna"].obs[patient_key]

    n_genes = (adata.var.modality == 'expression').sum()
    n_regions = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = 0

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


