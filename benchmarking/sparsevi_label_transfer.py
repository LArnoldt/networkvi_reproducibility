import datasets
from sklearn.ensemble import RandomForestClassifier
import sklearn
from metrics import calculate_label_transfer_metrics
import metrics
import pickle
import os
import numpy as np
import math
import scanpy as sc
import sys
import anndata as ad
import torch.nn as nn

def get_query_dataset(perform_scarches_hao_query, adata_mvi, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in):
    if perform_scarches_hao_query:
        mdata_query, mdata_counts_query, adata_query, train_indices_query, val_indices_query, n_genes_query, n_regions_query, n_snps_query, labels_key_query, further_labels_keys_query, batch_key_query, patient_key_query, n_patient_covariates_query, protein_expression_obsm_key_query = datasets.load_hao2021(
            paired_rate, subsampled, utilize_highly_variable, utilize_highly_abundant, spike_in=spike_in)
        del mdata_counts_query.mod["rna"].varm["PCs"]
        del mdata_counts_query.mod["prot"].varm["PCs"]
        mdata_counts_query.mod["rna"].obs["Site"] = mdata_query.mod["rna"].obs[batch_key_query]
        batch_key_query = "Site"
        mdata_counts_query.mod["rna"].obs["DonorID"] = mdata_query.mod["rna"].obs[patient_key_query]
        patient_key_query = "DonorID"
        mdata_counts_query.mod["rna"].obs["cell_type"] = mdata_query.mod["rna"].obs[labels_key_query]
        labels_key_query = "cell_type"
        further_labels_keys_query = ['GEX_phase',
         'DonorAge',
         'DonorBMI',
         'DonorBloodType',
         'DonorRace',
         'Ethnicity',
         'DonorGender',
         'DonorSmoker']

        X_prot_new = []
        for surface_protein, row in adata_mvi.uns["protein_expression"]["var"].iterrows():
            index_surface_protein = np.where(mdata_counts_query.mod["prot"].var.index == surface_protein)[0]
            if len(index_surface_protein) > 0:
                X_prot_new.append(np.array(mdata_counts_query.mod["prot"].X[:, index_surface_protein]).flatten())
            else:
                X_prot_new.append(np.zeros(len(mdata_counts_query.mod["prot"].obs)))
        mdata_counts_query.mod["prot"] = ad.AnnData(obs=mdata_counts_query.mod["prot"].obs, var=adata_mvi.uns["protein_expression"]["var"], X=np.array(X_prot_new).T)

    elif dataset_name.lower() == "neurips2021_cite_bmmc":
        mdata_query, mdata_counts_query, adata_query, train_indices_query, val_indices_query, n_genes_query, n_regions_query, n_snps_query, labels_key_query, further_labels_keys_query, batch_key_query, patient_key_query, n_patient_covariates_query, protein_expression_obsm_key_query = datasets.load_neurips2021_cite_BMMC(
            paired_rate, subsampled, utilize_highly_variable, utilize_highly_abundant, spike_in=spike_in, query_ref_selector=["site1"])

    elif dataset_name.lower() == "neurips2021_multiome_bmmc":
        mdata_query, mdata_counts_query, adata_query, train_indices_query, val_indices_query, n_genes_query, n_regions_query, n_snps_query, labels_key_query, further_labels_keys_query, batch_key_query, patient_key_query, n_patient_covariates_query, protein_expression_obsm_key_query = datasets.load_neurips2021_multiome_BMMC(
            paired_rate, subsampled, utilize_highly_variable, utilize_highly_abundant, spike_in=spike_in, query_ref_selector=["site1"])

    return mdata_query, mdata_counts_query, adata_query, train_indices_query, val_indices_query, n_genes_query, n_regions_query, n_snps_query, labels_key_query, further_labels_keys_query, batch_key_query, patient_key_query, n_patient_covariates_query, protein_expression_obsm_key_query

def model_label_transfer(name, perform_scarches_hao_query, scarches_modality_mode, inject_batch_covariates, inject_patient_covariates, incorporate_batch_scib, incorporate_patient_scib, unfreeze_first_layers, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, adata_mvi=None, mdata_counts=None, model=None, labels_key=None, train_indices=None, val_indices=None):

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools/src/")
    import scvi

    labels = adata_mvi.obs[labels_key][train_indices]
    labels_valid = adata_mvi.obs[labels_key][val_indices]
    latent_representation = model.get_latent_representation(modality="joint", indices=train_indices)
    latent_representation_valid = model.get_latent_representation(modality="joint", indices=val_indices)


    clf = RandomForestClassifier(
        random_state=1,
        class_weight="balanced_subsample",
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(latent_representation, labels)
    model.latent_space_classifer_ = clf


    pred_valid = model.latent_space_classifer_.predict(latent_representation_valid)
    np.save(os.path.join(f"{name}", "labels_valid.npy"), labels_valid)
    np.save(os.path.join(f"{name}", "pred_valid.npy"), pred_valid)
    label_transfer_metrics_valid = calculate_label_transfer_metrics(labels_valid, pred_valid)

    with open(os.path.join(f"{name}", "label_transfer_metrics_valid.pickle"), 'wb') as handle:
        pickle.dump(label_transfer_metrics_valid, handle, protocol=pickle.DEFAULT_PROTOCOL)


    mdata_query, mdata_counts_query, adata_query, train_indices_query, val_indices_query, n_genes_query, n_regions_query, n_snps_query, labels_key_query, further_labels_keys_query, batch_key_query, patient_key_query, further_batch_keys_query, further_continuous_batch_keys_query, n_patient_covariates_query, protein_expression_obsm_key_query = get_query_dataset(perform_scarches_hao_query, adata_mvi, dataset_name, subsampled, 1.0, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in)

    if scarches_modality_mode is not None and scarches_modality_mode is not False and scarches_modality_mode != "all" and scarches_modality_mode[0] != "all":
        for mod in set(mdata_counts_query.mod)- set(scarches_modality_mode):
            mdata_counts_query.mod[mod].X = np.zeros(mdata_counts_query.mod[mod].X.shape)


    adata_mvi_query = sc.concat([mdata_counts_query.mod[mod] for mod in mdata_counts_query.mod if mod != "prot"], join="outer", merge="first", axis=1)
    adata_mvi_query.X = np.array(adata_mvi_query.X)
    if "prot" in mdata_counts_query.mod:
        adata_mvi_query.obsm["protein_expression"] = mdata_counts_query.mod["prot"].X

    categorical_covariate_keys_query = ([batch_key_query] if inject_batch_covariates else []) + ([patient_key_query] if inject_patient_covariates else []) if inject_batch_covariates or inject_patient_covariates or inject_further_batch_covariates else None
    scvi.model.SPARSEVI.prepare_query_anndata(adata_mvi_query, model)


    model_query = scvi.model.SPARSEVI.load_query_data(adata_mvi_query, model, freeze=True, unfreeze_first_layers=unfreeze_first_layers)
    model_query.train(shuffle_set_split=False, adversarial_mixing=True if (paired_rate != 1.0 and paired_rate != 1) else False, train_idx=train_indices_query, validation_indices=val_indices_query)
    latent_representation_query = model_query.get_latent_representation(adata_mvi_query, modality="joint", indices=train_indices_query)

    def print_frozen_layers(model: nn.Module):
        print("Model Layer Status:")
        print("---------------------")
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name: <30} : {status}")

    print_frozen_layers(model.module)
    print_frozen_layers(model_query.module)


    if perform_scarches_hao_query:
        query_to_reference_cell_type_mapping = {
            'CD14 Mono': 'CD14+ Mono',
            'CD4 Naive': 'CD4+ T naive',
            'NK': 'NK',
            'CD4 TCM': np.nan,
            'CD8 TEM': np.nan,
            'CD8 Naive': 'CD8+ T naive',
            'B naive': 'Naive CD20+ B IGKC+',
            'CD16 Mono': 'CD16+ Mono',
            'CD4 TEM': 'CD4+ T activated',
            'gdT': 'gdT TCRVD2+',
            'B memory': np.nan,
            'CD8 TCM': np.nan,
            'MAIT': 'MAIT',
            'Treg': 'T reg',
            'B intermediate': 'Transitional B',
            'Platelet': np.nan,
            'CD4 CTL': 'CD4+ T CD314+ CD45RA+',
            'NK_CD56bright': 'NK',
            'cDC2': 'cDC2',
            'pDC': 'pDC',
            'NK Proliferating': 'NK',
            'Doublet': np.nan,
            'dnT': 'dnT',
            'HSPC': 'HSC',
            'ILC': 'ILC',
            'Plasmablast': 'Plasmablast IGKC+',
            'CD8 Proliferating': np.nan,
            'Eryth': 'Erythroblast',
            'ASDC': np.nan,
            'cDC1': 'cDC1',
            'CD4 Proliferating': np.nan
        }
        adata_mvi_query.obs[labels_key_query] = adata_mvi_query.obs[labels_key_query].map(query_to_reference_cell_type_mapping).fillna(adata_mvi_query.obs[labels_key_query])

    labels_query = adata_mvi_query.obs[labels_key_query][train_indices_query]
    pred_query = model.latent_space_classifer_.predict(latent_representation_query)

    np.save(os.path.join(f"{name}", "labels_query.npy"), labels_query)
    np.save(os.path.join(f"{name}", "pred_query.npy"), pred_query)
    np.save(os.path.join(f"{name}", "labels_train_query.npy"), np.concatenate([labels, labels_query]))


    def generate_umap_save(name, latent_representation, suffix=""):

        latent_representation_anndata = ad.AnnData(latent_representation)
        sc.pp.neighbors(latent_representation_anndata)
        sc.tl.umap(latent_representation_anndata, n_components=2)
        np.save(f"{name}/X_umap{suffix}.npy", latent_representation_anndata.obsm["X_umap"])

        np.save(f"{name}/X_sparsevi{suffix}.npy", latent_representation)

    generate_umap_save(name, latent_representation_query, suffix="_query")
    generate_umap_save(name, np.concatenate([latent_representation, latent_representation_query], axis=0), suffix="_train_query")

    for name_query, key_query in zip(["labels", "modality", "batch", "patient"] + further_labels_keys_query, [labels_key_query, "modality", batch_key_query, patient_key_query] + further_labels_keys_query):
        if key_query in mdata_counts_query.mod["rna"][train_indices_query, :].obs.columns:
            np.save(f"{name}/{name_query}_query.npy", mdata_counts_query.mod["rna"][train_indices_query, :].obs[key_query])
            np.save(f"{name}/{name_query}_train_query.npy", np.concatenate([mdata_counts.mod["rna"][train_indices, :].obs[key_query], mdata_counts_query.mod["rna"][train_indices_query, :].obs[key_query]]))


    label_transfer_metrics_query = calculate_label_transfer_metrics(labels_query, pred_query)

    with open(os.path.join(f"{name}", "label_transfer_metrics_query.pickle"), 'wb') as handle:
        pickle.dump(label_transfer_metrics_query, handle, protocol=pickle.DEFAULT_PROTOCOL)


    scib = metrics.calculate_scib_metrics(f"{name}/", np.concatenate([latent_representation, latent_representation_query], axis=0), sc.concat([adata_mvi[train_indices, :], adata_mvi_query[train_indices_query, :]]), labels_key_query, batch_key_query, None, patient_key_query, incorporate_batch_scib, incorporate_patient_scib, query=True)

    sys.exit()