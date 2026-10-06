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
from goatools.obo_parser import GODag

global ensembl2go
global obodag
try:
    with open("_biological_process_go-basic_goa_human_ensembl_gene_mapping_2024-08-08_goa_human_isoform_ensembl_gene_mapping_2024-08-08_goa_human_rna_ensembl_gene_mapping_2024-08-08.v2_processed.pkl", "rb") as f:
        ensembl2go = pickle.load(f)
    obodag = GODag("preprocessing/resources/ebi_ensembl_go_mappings/go-basic.obo")
except:
    with open("../_biological_process_go-basic_goa_human_ensembl_gene_mapping_2024-08-08_goa_human_isoform_ensembl_gene_mapping_2024-08-08_goa_human_rna_ensembl_gene_mapping_2024-08-08.v2_processed.pkl", "rb") as f:
        ensembl2go = pickle.load(f)
    obodag = GODag("../preprocessing/resources/ebi_ensembl_go_mappings/go-basic.obo")

def parse_spike_in_token(token: str):
    token = token.strip().strip("()")
    parts = token.split(",")
    if len(parts) != 3:
        raise ValueError(f"Spike-in must be (ID,rate,celltype). Got: {token}")
    ID, rate, celltype = parts
    return ID.strip(), float(rate), celltype.strip()

def validate_spike_in_ID(ID, mdata, obodag):
    if ID.startswith("ENSG"):
        all_genes = set()
        for mod in mdata.mod.values():
            if "gene_stable_id" in mod.var.columns:
                all_genes.update(mod.var["gene_stable_id"].tolist())
            else:
                all_genes.update(mod.var_names.tolist())

        if ID not in all_genes:
            raise ValueError(f"Gene {ID} not found in dataset")
        return ("gene", ID)

    if ID.startswith("GO:"):
        if ID.endswith(".full"):
            go_id = ID.replace(".full", "")
            if go_id not in obodag:
                raise ValueError(f"GO term {go_id} not found")
            return ("go_full", go_id)
        else:
            if ID not in obodag:
                raise ValueError(f"GO term {ID} not found")
            return ("go_direct", ID)

    raise ValueError(f"Invalid spike-in ID: {ID}")

def apply_spike_in_to_mdata_all_modalities(mdata, spike_in_list, ensembl2go, obodag, labels_key):
    for ID, rate, celltype in spike_in_list:

        mode, key = validate_spike_in_ID(ID, mdata, obodag)

        if mode == "gene":
            genes_to_modify = {key}

        elif mode == "go_direct":
            genes_to_modify = {
                g for g, gos in ensembl2go.items() if key in gos
            }

        elif mode == "go_full":
            children = obodag[key].get_all_children()
            children.add(key)
            genes_to_modify = {
                g for g, gos in ensembl2go.items() if gos.intersection(children)
            }

        else:
            raise ValueError("Unknown spike-in mode")

        for mod_name, mod in mdata.mod.items():

            if "counts" not in mod.layers:
                raise NotImplementedError("")

            if "gene_stable_id" in mod.var.columns:
                var_ids = mod.var["gene_stable_id"].tolist()
            else:
                var_ids = mod.var_names.tolist()

            if genes_to_modify is None:
                selected_idx = list(range(len(var_ids)))
            else:
                selected_idx = [i for i, g in enumerate(var_ids) if g in genes_to_modify]

            if len(selected_idx) == 0:
                continue

            if celltype.lower() == "all":
                cell_mask = np.array([True]*len(mod.obs))
            else:
                cell_mask = mod.obs[labels_key] == celltype

            mod.layers["counts"][np.ix_(cell_mask, [0, 1])] *= rate

    return mdata


def load_dogma(paired_rate, subsampled, utilize_highly_variable, spike_in=None):

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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs)*0.2)]
    train_indices = indices[math.floor(len(adata.obs)*0.2):]

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_hlca(paired_rate, subsampled, utilize_highly_variable, spike_in=None):

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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs)*0.2)]
    train_indices = indices[math.floor(len(adata.obs)*0.2):]

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_hao2021(paired_rate, subsampled, utilize_highly_variable, spike_in=None):

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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


def load_frangieh2021(paired_rate, subsampled, utilize_highly_variable, spike_in=None, query_ref_selector=None):

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
    n_snps = 0

    random_state = np.random.RandomState(seed=42)
    val_indices = np.where(mdata_counts.mod["rna"].obs[labels_key] != "control")[0]
    train_indices = np.where(mdata_counts.mod["rna"].obs[labels_key] == "control")[0]
    val_indices = random_state.permutation(val_indices)
    train_indices = random_state.permutation(train_indices)

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_neurips2021_cite_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=None, query_ref_selector=None):

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
    if spike_in is not None:
        spike_in_list = [parse_spike_in_token(s) for s in spike_in]
        mdata["rna"].layers["counts"] = np.array(mdata["rna"].layers["counts"].todense())
        mdata["prot"].layers["counts"] = np.array(mdata["prot"].layers["counts"].todense())
        mdata = apply_spike_in_to_mdata_all_modalities(mdata, spike_in_list, ensembl2go, obodag, labels_key)
    mdata["rna"].var["modality"] = "expression"
    mdata["prot"].var["modality"] = "Protein"
    mdata["rna"].obs["modality"] = "paired"
    mdata["prot"].obs["modality"] = "paired"

    mdata_counts = mdata.copy()
    if spike_in is not None:
        mdata_counts["rna"].X = mdata["rna"].layers["counts"]
        mdata_counts["prot"].X = mdata["prot"].layers["counts"]
    else:
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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))
    
    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key

def load_neurips2021_multiome_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=None, query_ref_selector=None):

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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = len(np.unique(mdata.mod["rna"].obs[patient_key]))
    
    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys,batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


def load_tabula_sapiens(dataset_name, paired_rate, subsampled, utilize_highly_variable, spike_in=None, query_ref_selector=None):
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
    if spike_in is not None:
        spike_in_list = [parse_spike_in_token(s) for s in spike_in]
        mdata["rna"].layers["counts"] = np.array(mdata["rna"].layers["counts"].todense())
        mdata = apply_spike_in_to_mdata_all_modalities(mdata, spike_in_list, ensembl2go, obodag, labels_key)
    mdata["rna"].var["modality"] = "expression"
    mdata["rna"].obs["modality"] = "paired"
    mdata["rna"].var["gene_stable_id"] = mdata["rna"].var["ensembl_gene_id"]

    mdata_counts = mdata.copy()
    if spike_in is not None:
        mdata_counts["rna"].X = mdata["rna"].layers["counts"]
    else:
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
    n_snps = 0

    indices = np.arange(len(adata.obs))
    random_state = np.random.RandomState(seed=42)
    indices = random_state.permutation(indices)
    val_indices = indices[:math.floor(len(adata.obs) * 0.2)]
    train_indices = indices[math.floor(len(adata.obs) * 0.2):]
    n_patient_covariates = 0

    return mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key


