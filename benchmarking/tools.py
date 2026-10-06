from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
import numpy as np
import scanpy as sc
import anndata as ad
import os
import sys
import subprocess
import pandas as pd
import pickle
import os
from tqdm import tqdm
import torch
import muon as mu
import torch.nn as nn
from typing import Literal
import json
import math
import random

import metrics

def mofa(hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, inject_further_batch_covariates=None, inject_further_continuous_batch_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, default: bool = False, best: bool = False, hyper: bool = False, scale_views: bool = False, scale_groups: bool = False, center_groups: bool = True, ard_weights: bool = True, ard_factors: bool = True, spikeslab_weights: bool = True, spikeslab_factors: bool = False, svi_mode: bool = False, svi_batch_size: float = 0.5, svi_learning_rate: float = 1.0, svi_forgetting_rate: float = 0.5, svi_start_stochastic: int = 1, convergence_mode: Literal["fast", "medium", "slow"] = "fast", encode_covariates=False, seed=0):

    if seed is None:
        seed = 0

    hyperparameters = {
        "scale_views": scale_views,
        "scale_groups": scale_groups,
        "center_groups": center_groups,
        "ard_weights": ard_weights,
        "ard_factors": ard_factors,
        "spikeslab_weights": spikeslab_weights,
        "spikeslab_factors": spikeslab_factors,
        "svi_mode": svi_mode,
        "convergence_mode": convergence_mode,
        "incorporate_batch_scib": incorporate_batch_scib,
        "incorporate_patient_scib": incorporate_patient_scib,
    }

    with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


    modality_mapping = {
        "rna": "expression",
        "atac": "accessibility",
        "geno": "genotype",
        "prot": "protein_expression"
    }

    mdata.obs["modality"] = mdata.obs["rna:modality"]

    if paired_rate != 1 and paired_rate != 1.0:
        n_paired = math.ceil(len(mdata.obs) * paired_rate) + 1
        n_unpaired = math.floor((len(mdata.obs) - n_paired) / len(mdata.mod))
        n_paired += len(mdata.obs) - (n_paired + n_unpaired * len(mdata.mod))
        for mod_index, mod in enumerate(mdata.mod):
            mdata.obs.iloc[list(range(n_paired + n_unpaired * mod_index,
                                          n_paired + n_unpaired * (mod_index + 1))), mdata.obs.columns.get_loc(
                "modality")] = modality_mapping[mod]
            mdata.mod[mod].obs.iloc[list(range(n_paired + n_unpaired * mod_index,
                                          n_paired + n_unpaired * (mod_index + 1))), mdata.mod[mod].obs.columns.get_loc(
                "modality")] = modality_mapping[mod]
            mdata.mod[mod].obs.iloc[list(set(list(range(len(mdata.mod[mod].obs)))) - set(list(range(n_paired)) + list(range(n_paired + n_unpaired * mod_index,
                                          n_paired + n_unpaired * (mod_index + 1))))), mdata.mod[mod].obs.columns.get_loc(
                "modality")] = "missing"

    mdata = mdata.copy()[train_indices, :]
    mdata_rna = mdata.mod["rna"].copy()
    mdata_rna.obs = mdata.mod["rna"].obs

    if paired_rate != 1 and paired_rate != 1.0:
        for mod_index, mod in enumerate(mdata.mod):
            mdata.mod[mod] = mdata.mod[mod].copy()[np.where(mdata.mod[mod].obs["modality"] != "missing")[0], :]


    if encode_covariates:
        mu.tl.mofa(mdata, seed=seed, use_obs="union", groups_label=batch_key, n_factors=20, gpu_mode=True, scale_views=scale_views, scale_groups=scale_groups, center_groups=center_groups, ard_weights=ard_weights, ard_factors=ard_factors, spikeslab_weights=spikeslab_weights, spikeslab_factors=spikeslab_factors, svi_mode=svi_mode, svi_batch_size=svi_batch_size, svi_learning_rate=svi_learning_rate, svi_forgetting_rate=svi_forgetting_rate, svi_start_stochastic=svi_start_stochastic, convergence_mode=convergence_mode, save_data=False, save_parameters=False, save_metadata=False)
    else:
        mu.tl.mofa(mdata, seed=seed, use_obs="union", n_factors=20, gpu_mode=True, scale_views=scale_views, scale_groups=scale_groups, center_groups=center_groups, ard_weights=ard_weights, ard_factors=ard_factors, spikeslab_weights=spikeslab_weights, spikeslab_factors=spikeslab_factors, svi_mode=svi_mode, svi_batch_size=svi_batch_size, svi_learning_rate=svi_learning_rate, svi_forgetting_rate=svi_forgetting_rate, svi_start_stochastic=svi_start_stochastic, convergence_mode=convergence_mode, save_data=False, save_parameters=False, save_metadata=False)


    latent_representation_anndata = ad.AnnData(mdata.obsm["X_mofa"])
    sc.pp.neighbors(latent_representation_anndata)
    sc.tl.umap(latent_representation_anndata, n_components=2)
    np.save(f"{hyper_name}/X_umap.npy", latent_representation_anndata.obsm["X_umap"])

    np.save(f"{hyper_name}/X_mofa.npy", mdata.obsm["X_mofa"])
    
    for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
        if key in mdata_rna.obs.columns:
            np.save(f"{hyper_name}/{name}.npy", mdata_rna.obs[key])


    if paired_rate != 1 and paired_rate != 1.0:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", mdata.obsm["X_mofa"], mdata_rna, labels_key, "modality", batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)
    else:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", mdata.obsm["X_mofa"], mdata_rna, labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)


    return scib

def wnn(hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, inject_further_batch_covariates=None, inject_further_continuous_batch_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, default: bool = False, best: bool = False, hyper: bool = False, n_bandwidth_neighbors: int = 20, n_multineighbors: int = 200, metric: Literal['euclidean', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'] = 'euclidean', seed=42):

    if seed is None:
        seed = 42

    hyperparameters = {
        "n_bandwidth_neighbors": n_bandwidth_neighbors,
        "n_multineighbors": n_multineighbors,
        "metric": "metric",
        "incorporate_batch_scib": incorporate_batch_scib,
        "incorporate_patient_scib": incorporate_patient_scib,
    }

    with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


    mdata = mdata[train_indices, :]

    if paired_rate != 1 and paired_rate != 1.0:
        raise NotImplementedError("Paired rate is not 1.")


    mu.pp.neighbors(mdata, random_state=seed, n_bandwidth_neighbors=n_bandwidth_neighbors, n_multineighbors=n_multineighbors, metric=metric, add_weights_to_modalities=True)


    mu.tl.umap(mdata, n_components=2)
    np.save(f"{hyper_name}/X_umap.npy", mdata.obsm["X_umap"])

    mu.tl.umap(mdata, n_components=20)
    np.save(f"{hyper_name}/X_wnn.npy", mdata.obsm["X_umap"])

    for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
        if key in mdata.mod["rna"].obs.columns:
            np.save(f"{hyper_name}/{name}.npy", mdata.mod["rna"].obs[key])


    scib = metrics.calculate_scib_metrics(f"{hyper_name}/", mdata.obsm["X_umap"], mdata.mod["rna"], labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)


    return scib

def totalvi(hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, inject_further_batch_covariates=None, inject_further_continuous_batch_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, default: bool = False, best: bool = False, hyper: bool = False, lr=0.0001, weight_decay=1e-3, n_layers_encoder=2, n_layers_decoder=1, n_hidden=256, gene_dispersion="gene", protein_dispersion="protein", gene_likelihood="nb", latent_distribution="normal", dropout_rate_encoder= 0.2, dropout_rate_decoder=0.2, activation_fn="relu", encode_covariates=False, seed=None):

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools_2303/src/")
    import scvi

    if seed is not None:
        scvi.settings.seed = seed
    print("### Seed ###")
    print(scvi.settings.seed)


    hyperparameters = {
        "lr": lr,
        "n_layers_encoder": n_layers_encoder,
        "n_layers_decoder": n_layers_decoder,
        "n_hidden": n_hidden,
        "gene_dispersion": gene_dispersion,
        "protein_dispersion": protein_dispersion,
        "gene_likelihood": gene_likelihood,
        "latent_distribution": latent_distribution,
        "dropout_rate_encoder": dropout_rate_encoder,
        "dropout_rate_decoder": dropout_rate_decoder,
        "activation_fn": activation_fn,
        "encode_covariates": encode_covariates,
        "inject_batch_covariates": inject_batch_covariates,
        "inject_patient_covariates": inject_patient_covariates,
        "inject_further_batch_covariates": inject_further_batch_covariates,
        "inject_further_continuous_batch_covariates": inject_further_continuous_batch_covariates,
        "incorporate_batch_scib": incorporate_batch_scib,
        "incorporate_patient_scib": incorporate_patient_scib,
    }

    with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)

    if activation_fn == "relu":
        activation_fn = nn.ReLU
    if activation_fn == "leaky_relu":
        activation_fn = nn.LeakyReLU
    if activation_fn == "swish":
        activation_fn = nn.Hardswish


    if paired_rate != 1 and paired_rate != 1.0:
        raise NotImplementedError("Paired rate is not 1.")

    categorical_covariate_keys = ([batch_key] if inject_batch_covariates else []) + ([patient_key] if inject_patient_covariates else []) if inject_batch_covariates or inject_patient_covariates else None
    scvi.model.TOTALVI.setup_mudata(mdata_counts, modalities={"rna_layer": "rna", "protein_layer": "prot"}, categorical_covariate_keys=categorical_covariate_keys)

    model = scvi.model.TOTALVI(mdata_counts,
                               n_latent=20,
                               latent_distribution=latent_distribution,
                               dropout_rate_encoder=dropout_rate_encoder,
                               dropout_rate_decoder=dropout_rate_encoder,
                               n_layers_encoder=n_layers_encoder,
                               n_layers_decoder=n_layers_decoder,
                               n_hidden=n_hidden,
                               gene_dispersion=gene_dispersion,
                               protein_dispersion=protein_dispersion,
                               gene_likelihood=gene_likelihood,
                               encode_covariates=encode_covariates,
                               )


    model.train(lr=lr, datasplitter_kwargs={"train_idx": train_indices, "validation_idx": val_indices})
    latent_representation = model.get_latent_representation(indices=train_indices)


    latent_representation_anndata = ad.AnnData(latent_representation)
    sc.pp.neighbors(latent_representation_anndata)
    sc.tl.umap(latent_representation_anndata, n_components=2)
    np.save(f"{hyper_name}/X_umap.npy", latent_representation_anndata.obsm["X_umap"])

    np.save(f"{hyper_name}/X_totalvi.npy", latent_representation)
    
    for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
        if key in mdata.mod["rna"][train_indices, :].obs.columns:
            np.save(f"{hyper_name}/{name}.npy", mdata.mod["rna"][train_indices, :].obs[key])


    scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, mdata.mod["rna"][train_indices, :], labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)


    if optimization_metric.lower() == "scib":
        return scib
    elif optimization_metric.lower() == "elbo":
        return min(list(model.history["elbo_validation"]["elbo_validation"]))

def midas(hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, default: bool = False, best: bool = False, hyper: bool = False, lr=1e-4, drop=0.2, disc_train=3, loss_s_recon=1000.0, loss_mod_alignment=50.0, loss_disc = 30.0, encode_covariates=False, seed=None):

    from scmidas.datasets import GenDataFromPath
    from scmidas.models import MIDAS
    from scmidas.datasets import GetDataInfo


    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("### Seed ###")
    print(seed)

    hyperparameters = {
        "lr": lr,
        "drop": drop,
        "disc_train": disc_train,
        "loss_s_recon": loss_s_recon,
        "loss_mod_alignment": loss_mod_alignment,
        "loss_disc": loss_disc,
    }
    
    with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)


    midas_dataset_name = hyper_name.rstrip("/default").rstrip("/hyper_best").rsplit('/hyper/', 1)[0]
    save_dir = os.path.join(midas_dataset_name, "data_prepared")

    if not os.path.isdir(os.path.join(midas_dataset_name, "data")):
        os.makedirs(os.path.join(midas_dataset_name, "data"), exist_ok=True)

        if paired_rate != 1 and paired_rate != 1.0:
            n_groups = 1 + len(mdata_counts.mod)
        else:
            n_groups = 1

        if encode_covariates and inject_batch_covariates and batch_key:
            n_batches = len(np.unique(mdata_counts.mod["rna"].obs[batch_key]))
        else:
            n_batches = 1

        if encode_covariates and inject_patient_covariates and patient_key:
            n_patients = len(np.unique(mdata_counts.mod["rna"].obs[patient_key]))
        else:
            n_patients = 1

        for n_subset in range(n_groups * n_batches * n_patients):
            os.makedirs(os.path.join(midas_dataset_name, "data", f"subset_{n_subset}"), exist_ok=True)


        n_paired = math.ceil(len(mdata_counts.obs) * paired_rate)+1
        n_unpaired = math.floor((len(mdata_counts.obs) - n_paired) / len(mdata_counts.mod))
        n_paired += len(mdata_counts.obs) - (n_paired + n_unpaired * len(mdata_counts.mod))

        modality_mapping = {
            "rna": "rna",
            "atac": "atac",
            "prot": "adt"
        }

        if paired_rate != 1 and paired_rate != 1.0:
            for mod_index, mod in enumerate(mdata_counts.mod):
                mdata_counts.mod[mod].obs.iloc[list(range(n_paired + n_unpaired * mod_index,
                                              n_paired + n_unpaired * (mod_index + 1))), mdata_counts.mod[mod].obs.columns.get_loc("modality")] = modality_mapping[mod]
                mdata_counts.mod[mod].obs.iloc[list(set(list(range(len(mdata_counts.mod[mod].obs)))) - set(list(range(n_paired)) + list(range(n_paired + n_unpaired * mod_index,
                                              n_paired + n_unpaired * (mod_index + 1))))), mdata_counts.mod[mod].obs.columns.get_loc("modality")] = "missing"

        mdata_counts = mdata_counts.copy()[train_indices, :]
        mdata_counts_rna = mdata_counts.mod["rna"].copy()
        mdata_counts_rna.obs = mdata_counts.mod["rna"].obs


        data_path = [{} for _ in range(n_batches * n_patients)]

        for mod_index, mod in enumerate(mdata_counts.mod):
            if paired_rate != 1 and paired_rate != 1.0:
                adata_paired = mdata_counts.mod[mod].copy()[np.where(mdata_counts.mod[mod].obs["modality"] == "paired")[0], :]
                adata_unpaired = mdata_counts.mod[mod].copy()[np.where(mdata_counts.mod[mod].obs["modality"] == modality_mapping[mod])[0], :]
                adata_unpaired_batches_patients = []
            else:
                adata_paired = mdata_counts.mod[mod].copy()
            adata_paired_batches_patients = []

            if encode_covariates and inject_batch_covariates and batch_key and encode_covariates and inject_patient_covariates and patient_key:
                raise NotImplementedError("")
            elif encode_covariates and inject_batch_covariates and batch_key:
                for batch_sub in np.unique(mdata_counts.mod["rna"].obs[batch_key]):
                    adata_paired_batches_patients.append(adata_paired.copy()[np.where(adata_paired.obs[batch_key] == batch_sub)[0], :])
                    if paired_rate != 1 and paired_rate != 1.0:
                        adata_unpaired_batches_patients.append(adata_unpaired.copy()[np.where(adata_unpaired.obs[batch_key] == batch_sub)[0], :])
            elif encode_covariates and inject_patient_covariates and patient_key:
                for patient_sub in np.unique(mdata_counts.mod["rna"].obs[patient_key]):
                    adata_paired_batches_patients.append(adata_paired.copy()[np.where(adata_paired.obs[patient_key] == patient_sub)[0], :])
                    if paired_rate != 1 and paired_rate != 1.0:
                        adata_unpaired_batches_patients.append(adata_unpaired.copy()[np.where(adata_unpaired.obs[patient_key] == patient_sub)[0], :])
            else:
                adata_paired_batches_patients = [adata_paired]
                if paired_rate != 1 and paired_rate != 1.0:
                    adata_unpaired_batches_patients = [adata_unpaired]

            for adata_paired_sub_index, adata_paired_sub in enumerate(adata_paired_batches_patients):
                try:
                    adata_paired_sub_csv = pd.DataFrame(adata_paired_sub.X.astype("int"), columns=list(adata_paired_sub.var.index), index=list(adata_paired_sub.obs.index))
                except ValueError:
                    adata_paired_sub_csv = pd.DataFrame(adata_paired_sub.X.todense().astype("int"), columns=list(adata_paired_sub.var.index), index=list(adata_paired_sub.obs.index))
                n_subset = adata_paired_sub_index
                if len(data_path) < n_subset:
                    data_path.append({modality_mapping[mod]: os.path.join(midas_dataset_name, "data", f"subset_{n_subset}", f"{modality_mapping[mod]}.csv")})
                else:
                    data_path[n_subset][modality_mapping[mod]] = os.path.join(midas_dataset_name, "data", f"subset_{n_subset}", f"{modality_mapping[mod]}.csv")
                adata_paired_sub_csv.to_csv(os.path.join(midas_dataset_name, "data", f"subset_{n_subset}", f"{modality_mapping[mod]}.csv"))

            if paired_rate != 1 and paired_rate != 1.0:
                for adata_unpaired_sub_index, adata_unpaired_sub in enumerate(adata_unpaired_batches_patients):
                    try:
                        adata_unpaired_sub_csv = pd.DataFrame(adata_unpaired_sub.X.astype("int"), columns=list(adata_unpaired_sub.var.index), index=list(adata_unpaired_sub.obs.index))
                    except ValueError:
                        adata_unpaired_sub_csv = pd.DataFrame(adata_unpaired_sub.X.todense().astype("int"), columns=list(adata_unpaired_sub.var.index), index=list(adata_unpaired_sub.obs.index))
                    n_subset = mod_index+1 * (n_batches * n_patients) + adata_unpaired_sub_index
                    data_path.append({modality_mapping[mod]: os.path.join(midas_dataset_name, "data", f"subset_{n_subset}", f"{modality_mapping[mod]}.csv")})
                    adata_unpaired_sub_csv.to_csv(os.path.join(midas_dataset_name, "data", f"subset_{n_subset}", f"{modality_mapping[mod]}.csv"))

        GenDataFromPath(data_path, save_dir, True)


    data = [GetDataInfo(save_dir)]

    model = MIDAS(data)

    model.init_model(lr=lr,
                dim_c=18,
                dim_b=2,
                drop=drop,
                disc_train=disc_train,
                loss_s_recon=loss_s_recon,
                loss_mod_alignment=loss_mod_alignment,
                loss_disc=loss_disc)

    model.train(n_epoch=100, shuffle=False, save_path=f'{hyper_name}')


    model.predict(joint_latent=True, mod_latent=True, save_dir=f'{hyper_name}')

    emb = model.read_preds(joint_latent=True, mod_latent=True)

    latent_representation = emb["z"]["joint"]

    def generate_umap_save(hyper_name, latent_representation, suffix=""):

        latent_representation_anndata = ad.AnnData(latent_representation)
        sc.pp.neighbors(latent_representation_anndata)
        sc.tl.umap(latent_representation_anndata, n_components=2)
        np.save(f"{hyper_name}/X_umap{suffix}.npy", latent_representation_anndata.obsm["X_umap"])

        np.save(f"{hyper_name}/X_midas{suffix}.npy", latent_representation)

    generate_umap_save(hyper_name, latent_representation)

    for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
        if key in mdata_counts.mod["rna"].obs.columns:
            np.save(f"{hyper_name}/{name}.npy", mdata_counts.mod["rna"].obs[key])

    if "rna" in mdata_counts.mod:
        generate_umap_save(hyper_name, emb["z"]["rna"], suffix="_expression")
    if "atac" in mdata_counts.mod:
        generate_umap_save(hyper_name, emb["z"]["atac"], suffix="_accessibility")
    if "prot" in mdata_counts.mod:
        generate_umap_save(hyper_name, emb["z"]["adt"], suffix="_protein")


    os.makedirs(hyper_name + "_only_label_part_test", exist_ok=True)
    if paired_rate != 1 and paired_rate != 1.0:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}_only_label_part_test/", latent_representation[:, :18], mdata_counts.mod["rna"],
                                              labels_key, "modality", batch_key, patient_key, incorporate_batch_scib,
                                              incorporate_patient_scib)
    else:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}_only_label_part_test/", latent_representation[:, :18], mdata_counts.mod["rna"],
                                              labels_key, None, batch_key, patient_key, incorporate_batch_scib,
                                              incorporate_patient_scib)

    os.makedirs(hyper_name + "_only_batch_part_test", exist_ok=True)
    if paired_rate != 1 and paired_rate != 1.0:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}_only_batch_part_test/", latent_representation[:, 18:], mdata_counts.mod["rna"],
                                              labels_key, "modality", batch_key, patient_key, incorporate_batch_scib,
                                              incorporate_patient_scib)
    else:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}_only_batch_part_test/", latent_representation[:, 18:], mdata_counts.mod["rna"],
                                              labels_key, None, batch_key, patient_key, incorporate_batch_scib,
                                              incorporate_patient_scib)


    if paired_rate != 1 and paired_rate != 1.0:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, mdata_counts.mod["rna"], labels_key, "modality", batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)
    else:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, mdata_counts.mod["rna"], labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)


    if optimization_metric.lower() == "scib":
        return scib

def multimil(hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, inject_further_batch_covariates=None, inject_further_continuous_batch_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, default: bool = False, best: bool = False, hyper: bool = False, lr=0.0001, weight_decay=1e-3, dropout=0.2, kernel_type="gaussian", n_layers_encoders=None, n_layers_decoders=None, n_hidden_encoders=None, n_hidden_decoders=None, mmd="latent", normalization="layer", activation="leaky_relu", condition_encoders=False, seed=None):

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/multimil/src/")
    import multimil as mtm
    import scvi

    if seed is not None:
        scvi.settings.seed = seed
    else:
        scvi.settings.seed = np.random.randint(0, 2 ** 32 - 1)

    print("### Seed ###")
    print(scvi.settings.seed)


    hyperparameters = {
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "kernel_type": kernel_type,
        "n_layers_encoders": n_layers_encoders,
        "n_layers_decoders": n_layers_decoders,
        "n_hidden_encoders": n_hidden_encoders,
        "n_hidden_decoders": n_hidden_decoders,
        "mmd": mmd,
        "normalization": normalization,
        "activation": activation,
        "condition_encoders": condition_encoders,
        "inject_batch_covariates": inject_batch_covariates,
        "inject_patient_covariates": inject_patient_covariates,
        "inject_further_batch_covariates": inject_further_batch_covariates,
        "inject_further_continuous_batch_covariates": inject_further_continuous_batch_covariates,
        "incorporate_batch_scib": incorporate_batch_scib,
        "incorporate_patient_scib": incorporate_patient_scib,
    }

    with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)

    if n_layers_encoders == -1:
        n_layers_encoders = None
    if n_layers_decoders == -1:
        n_layers_decoders = None
    if n_hidden_encoders == -1:
        n_hidden_encoders = None
    if n_hidden_decoders == -1:
        n_hidden_decoders = None
    if kernel_type == -1:
        kernel_type = None
    if normalization == "none":
        normalization = None


    other_mod = list(set(mdata.mod.keys()) - set(["rna"]))
    mdata.mod["rna"].X = mdata_counts.mod["rna"].X

    if paired_rate != 1 and paired_rate != 1.0:
        n_paired = math.ceil(len(mdata.obs) * paired_rate)+1
        n_unpaired = math.floor((len(mdata.obs) - n_paired) / len(mdata.mod))
        n_paired += len(mdata.obs) - (n_paired + n_unpaired * len(mdata.mod))

        adatas = []
        layers = []
        for mod_index, mod in enumerate(["rna"] + other_mod):
            adata_paired = mdata.mod[mod][list(range(n_paired)), :].copy()
            adata_paired.obs["modality"] = "paired"
            mdata.mod["rna"].obs["modality"][list(range(n_paired))] = "paired"
            adata_unpaired = mdata.mod[mod][list(range(n_paired + n_unpaired * mod_index, n_paired + n_unpaired * (mod_index + 1))), :].copy()
            adata_unpaired.obs["modality"] = mod
            mdata.mod["rna"].obs["modality"][list(range(n_paired + n_unpaired * mod_index, n_paired + n_unpaired * (mod_index + 1)))] = mod
            adatas.append([adata_paired] + [None] * mod_index + [adata_unpaired] + [None] * (len(other_mod) - mod_index))
            if mod == "rna":
                layers.append(["counts"] * 2 + [None] * (len(mdata.mod)-1))
            else:
                layers.append([None] * (len(mdata.mod)+1))

        adata_multivae = mtm.data.organize_multimodal_anndatas(
            adatas=adatas,
            layers=layers,
        )

        rna_indices_end = mdata.mod["rna"].shape[1]

        mtm.model.MultiVAE.setup_anndata(
            adata_multivae,
            categorical_covariate_keys=["modality"] + ([batch_key] if inject_batch_covariates else []) + ([patient_key] if inject_patient_covariates else []),
            rna_indices_end=rna_indices_end,
        )

        model = mtm.model.MultiVAE(
            adata_multivae,
            z_dim=20,
            losses=["nb"] + ["mse"]*len(other_mod),
            loss_coefs={
                "integ": 500,
            },
            integrate_on="modality",
            dropout=dropout,
            kernel_type = kernel_type,
            n_layers_encoders=[n_layers_encoders]*(len(other_mod)+1) if n_layers_encoders is not None else n_layers_encoders,
            n_layers_decoders=[n_layers_decoders]*(len(other_mod)+1) if n_layers_decoders is not None else n_layers_decoders,
            n_hidden_encoders=[n_hidden_encoders]*(len(other_mod)+1) if n_hidden_encoders is not None else n_hidden_encoders,
            n_hidden_decoders=[n_hidden_decoders]*(len(other_mod)+1) if n_hidden_decoders is not None else n_hidden_decoders,
            mmd=mmd,
            normalization=normalization,
            activation=activation,
            condition_encoders=condition_encoders,
        )

    else:
        adata_multivae = mtm.data.organize_multimodal_anndatas(
            adatas=[[mdata.mod["rna"]]] + [[mdata.mod[mod]] for mod in other_mod],
            layers=[["counts"]] + [[None]] * len(other_mod),
        )

        rna_indices_end = mdata.mod["rna"].shape[1]

        mtm.model.MultiVAE.setup_anndata(
            adata_multivae,
            rna_indices_end=rna_indices_end,
        )

        model = mtm.model.MultiVAE(
            adata_multivae,
            losses=["nb"]+["mse"]*len(other_mod),
            dropout=dropout,
            kernel_type=kernel_type,
            n_layers_encoders=[n_layers_encoders]*(len(other_mod)+1) if n_layers_encoders is not None else n_layers_encoders,
            n_layers_decoders=[n_layers_decoders]*(len(other_mod)+1) if n_layers_decoders is not None else n_layers_decoders,
            n_hidden_encoders=[n_hidden_encoders]*(len(other_mod)+1) if n_hidden_encoders is not None else n_hidden_encoders,
            n_hidden_decoders=[n_hidden_decoders]*(len(other_mod)+1) if n_hidden_decoders is not None else n_hidden_decoders,
            mmd=mmd,
            normalization=normalization,
            activation=activation,
        )

    model.train(lr=lr, weight_decay=weight_decay, train_idx=train_indices, validation_idx=val_indices)
    latent_representation = model.get_model_output(indices=train_indices, give_mean=True)


    model.save(f"{hyper_name}", overwrite=True)

    history = model.history
    with open(os.path.join(f"{hyper_name}", "history.pickle"), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.DEFAULT_PROTOCOL)


    latent_representation_anndata = ad.AnnData(latent_representation)
    sc.pp.neighbors(latent_representation_anndata)
    sc.tl.umap(latent_representation_anndata, n_components=2)
    np.save(f"{hyper_name}/X_umap.npy", latent_representation_anndata.obsm["X_umap"])

    np.save(f"{hyper_name}/X_multimil.npy", latent_representation)

    for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
        if key in mdata.mod["rna"][train_indices, :].obs.columns:
            np.save(f"{hyper_name}/{name}.npy", mdata.mod["rna"][train_indices, :].obs[key])


    if paired_rate != 1 and paired_rate != 1.0:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, mdata.mod["rna"][train_indices, :], labels_key, "modality", batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)
    else:
        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, mdata.mod["rna"][train_indices, :], labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)


    if optimization_metric.lower() == "scib":
        return scib
    elif optimization_metric.lower() == "elbo":
        return min(list(model.history["elbo_validation"]["elbo_validation"]))


_FRANGIEH_PERTURBATION_CANDIDATES = [
    "CD58", "CD59", "CDK6", "MYC", "ILF2", "DNMT1", "ACSL3", "MYC",
    "CDK4/6", "CDK4", "B2M", "HLA-A", "JAK1", "JAK2", "STAT1", "IFNGR1",
    "IFNGR2", "CD47", "CDH19", "PD-L1", "CD274", "CXCR4", "CD184",
    "c-KIT", "CD117", "KDR", "CD309", "CD47", "IFNGR1", "CD119",
    "PSMB4", "PSMB8", "PSMB9", "PSMA4", "HLA-A,B,C", "HLA-E", "HLA-F",
    "HLAD-DPB", "CXCL1", "CXCL2", "CXCL8", "CXCL10", "CXCL11", "STAT3",
    "IL1B", "IL6", "STAT1", "IRF1", "IRF3", "IFITM3", "IFIT6", "CD274",
    "CD47", "SOX4", "ITGA3", "ITGA1", "SERPINE2", "TGFB2", "CD9",
]


def _run_go_importance(model, hyper_name, labels_key, train_indices, val_indices, labels_column=None, **kwargs):
    '''Shared entry point for every one-off GO/gene-importance interpretability experiment.'''
    labels_column = labels_column or labels_key
    results_dir = kwargs.pop("results_dir_subdir", None)
    results_dir = f"./{hyper_name}/{results_dir}" if results_dir else f"./{hyper_name}/calc_go_terms_{labels_column}"
    os.makedirs(f"./{hyper_name}", exist_ok=True)
    kwargs.setdefault("calc_go_terms", False)
    kwargs.setdefault("calc_gene_groups", False)
    model.get_go_gene_importances(
        labels_column=labels_column,
        results_dir=results_dir,
        shuffle_set_split=False,
        batch_size=512,
        train_idx=train_indices,
        validation_idx=val_indices,
        **kwargs,
    )


def _run_perturbation_importance(model, hyper_name, labels_key, train_indices, val_indices, labels_column=None, **kwargs):
    '''Shared entry point for every one-off perturbation/spike-in interpretability experiment.'''
    labels_column = labels_column or labels_key
    results_dir = kwargs.pop("results_dir_subdir", None)
    results_dir = f"./{hyper_name}/{results_dir}" if results_dir else f"./{hyper_name}/calc_go_terms_{labels_column}"
    os.makedirs(f"./{hyper_name}", exist_ok=True)
    kwargs.setdefault("modality_key", "modality")
    kwargs.setdefault("gene_stable_id_key", "gene_stable_id")
    kwargs.setdefault("calc_go_terms", True)
    kwargs.setdefault("calc_gene_groups", False)
    if kwargs.get("perturbation_mode") == "spikein":
        kwargs.setdefault("spikein_cells_column", labels_column)
    return model.get_perturbation_go_gene_importances(
        labels_column=labels_column,
        results_dir=results_dir,
        shuffle_set_split=False,
        batch_size=512,
        train_idx=train_indices,
        validation_idx=val_indices,
        **kwargs,
    )


_GO_IMPORTANCE_TASKS = {
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments": [
        dict(
            calc_go_terms=True,
            calc_gene_groups=False,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=[],
        ),
    ],
    "neurips2021_multiome_bmmc_sparsevi_interpretability_experiments": [
        dict(
            calc_go_terms=True,
            calc_gene_groups=False,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=[],
        ),
    ],
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments2": [
        dict(
            calc_go_terms=False,
            calc_gene_groups=False,
            calc_genes=True,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=['CD14+ Mono', 'CD16+ Mono', 'Naive CD20+ B IGKC+', 'NK', 'pDC', 'CD4+ T naive', 'CD8+ T naive'],
        ),
    ],
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments4": [
        dict(
            calc_go_terms=False,
            calc_gene_groups=False,
            calc_genes=True,
            save_fit=False,
            overwrite_save_fit=False,
            restrict_modalities='expression',
        ),
    ],
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments6": [
        dict(
            calc_go_terms=True,
            calc_gene_groups=False,
            calc_genes=False,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=['CD14+ Mono', 'CD16+ Mono', 'Naive CD20+ B IGKC+', 'pDC'],
        ),
    ],
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments7": [
        dict(
            calc_go_terms=False,
            calc_gene_groups='./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy',
            calc_genes=False,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=['CD14+ Mono', 'CD16+ Mono', 'Naive CD20+ B IGKC+', 'NK', 'pDC', 'CD4+ T naive', 'CD8+ T naive'],
        ),
    ],
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments8": [
        dict(
            calc_go_terms=True,
            calc_gene_groups=False,
            calc_genes=False,
            save_fit=False,
            overwrite_save_fit=False,
            labels_selection=['CD14+ Mono', 'CD16+ Mono', 'Naive CD20+ B IGKC+', 'NK', 'pDC', 'CD4+ T naive', 'CD8+ T naive'],
        ),
    ],
}

_PERTURBATION_IMPORTANCE_TASKS = {
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments5": dict(
        perturbation_gene_stable_ids=['ENSG00000177455'],
    ),
    "neurips2021_cite_bmmc_sparsevi_interpretability_experiments9": dict(
        perturbation_gene_stable_ids=['ENSG00000177455'],
    ),
    "spike_in_exp_01_CD19_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000177455'],
        spikein_gene_stable_id='ENSG00000177455',
        spikein_cells='pDC',
        spikein_value_mode=('mean', 'Naive CD20+ B IGKC+'),
        labels_selection=['pDC', 'Naive CD20+ B IGKC+', 'Naive CD20+ B IGKC-', 'B1 B IGKC+', 'Transitional B', 'CD8+ T naive', 'CD4+ T naive'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_02_MS4A1_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000156738'],
        spikein_gene_stable_id='ENSG00000156738',
        spikein_cells='pDC',
        spikein_value_mode=('mean', 'Naive CD20+ B IGKC+'),
        labels_selection=['pDC', 'Naive CD20+ B IGKC+', 'Naive CD20+ B IGKC-', 'B1 B IGKC+', 'Transitional B', 'CD8+ T naive'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_03_PAX5_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000196092'],
        spikein_gene_stable_id='ENSG00000196092',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'B1 B IGKC+'),
        labels_selection=['CD8+ T naive', 'CD4+ T naive', 'B1 B IGKC+', 'Naive CD20+ B IGKC+', 'Transitional B', 'Lymph prog'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_04_DNTT_into_matureB": dict(
        perturbation_gene_stable_ids=['ENSG00000107447'],
        spikein_gene_stable_id='ENSG00000107447',
        spikein_cells='Naive CD20+ B IGKC+',
        spikein_value_mode=('mean', 'Lymph prog'),
        labels_selection=['Naive CD20+ B IGKC+', 'Naive CD20+ B IGKC-', 'B1 B IGKC+', 'Transitional B', 'Lymph prog', 'HSC', 'T prog cycling'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_05_CD38_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000004468'],
        spikein_gene_stable_id='ENSG00000004468',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'Plasma cell IGKC+'),
        labels_selection=['CD8+ T naive', 'CD4+ T naive', 'Plasma cell IGKC+', 'Plasma cell IGKC-', 'Plasmablast IGKC+', 'NK'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_06_GADD45A_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000116717'],
        spikein_gene_stable_id='ENSG00000116717',
        spikein_cells='pDC',
        spikein_value_mode='multiplier',
        spikein_strength=5.0,
        labels_selection=['pDC', 'CD14+ Mono', 'CD16+ Mono', 'CD8+ T naive', 'CD4+ T naive', 'NK'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_07_CDKN1A_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000124762'],
        spikein_gene_stable_id='ENSG00000124762',
        spikein_cells='pDC',
        spikein_value_mode='multiplier',
        spikein_strength=5.0,
        labels_selection=['pDC', 'CD14+ Mono', 'CD8+ T naive', 'CD4+ T naive', 'NK', 'CD8+ T TIGIT+ CD45RO+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_08_BCL2_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000171791'],
        spikein_gene_stable_id='ENSG00000171791',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'B1 B IGKC+'),
        labels_selection=['CD8+ T naive', 'CD4+ T naive', 'B1 B IGKC+', 'Naive CD20+ B IGKC+', 'CD8+ T CD57+ CD45RO+', 'Plasma cell IGKC+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_09_XBP1_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000100219'],
        spikein_gene_stable_id='ENSG00000100219',
        spikein_cells='pDC',
        spikein_value_mode=('mean', 'Plasma cell IGKC+'),
        labels_selection=['pDC', 'Plasma cell IGKC+', 'Plasmablast IGKC+', 'CD14+ Mono', 'CD8+ T naive'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_10_ATF6_into_pDC": dict(
        perturbation_gene_stable_ids=['ENSG00000118217'],
        spikein_gene_stable_id='ENSG00000118217',
        spikein_cells='pDC',
        spikein_value_mode='multiplier',
        spikein_strength=5.0,
        labels_selection=['pDC', 'Plasma cell IGKC+', 'Plasmablast IGKC+', 'CD14+ Mono', 'CD8+ T naive'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_13_SIGLEC6_into_NK": dict(
        perturbation_gene_stable_ids=['ENSG00000105492'],
        spikein_gene_stable_id='ENSG00000105492',
        spikein_cells='NK',
        spikein_value_mode=('mean', 'pDC'),
        labels_selection=['NK', 'NK CD158e1+', 'pDC', 'CD8+ T naive', 'B1 B IGKC+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_15_CCL4_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000275302'],
        spikein_gene_stable_id='ENSG00000275302',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'CD8+ T CD57+ CD45RO+'),
        labels_selection=['CD8+ T naive', 'CD4+ T naive', 'CD8+ T CD57+ CD45RO+', 'CD8+ T CD57+ CD45RA+', 'NK', 'NK CD158e1+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_16_CCL5_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000271503'],
        spikein_gene_stable_id='ENSG00000271503',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'CD8+ T CD57+ CD45RO+'),
        labels_selection=['CD8+ T naive', 'CD4+ T naive', 'CD8+ T CD57+ CD45RO+', 'CD8+ T CD57+ CD45RA+', 'NK', 'NK CD158e1+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_17_TOX_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000198846'],
        spikein_gene_stable_id='ENSG00000198846',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'CD8+ T TIGIT+ CD45RO+'),
        labels_selection=['CD8+ T naive', 'CD8+ T TIGIT+ CD45RO+', 'CD8+ T TIGIT+ CD45RA+', 'CD8+ T CD69+ CD45RO+', 'CD4+ T naive', 'CD4+ T activated', 'NK'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_22_LAG3_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000089692'],
        spikein_gene_stable_id='ENSG00000089692',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'CD8+ T TIGIT+ CD45RO+'),
        labels_selection=['CD8+ T naive', 'CD8+ T TIGIT+ CD45RO+', 'CD8+ T TIGIT+ CD45RA+', 'T reg', 'NK', 'CD4+ T activated'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_25_GZMB_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000100453'],
        spikein_gene_stable_id='ENSG00000100453',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'NK CD158e1+'),
        labels_selection=['CD8+ T naive', 'CD8+ T CD57+ CD45RO+', 'NK', 'NK CD158e1+', 'CD4+ T naive', 'CD8+ T TIGIT+ CD45RO+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
    "spike_in_exp_26_PRF1_into_CD8Tnaive": dict(
        perturbation_gene_stable_ids=['ENSG00000180644'],
        spikein_gene_stable_id='ENSG00000180644',
        spikein_cells='CD8+ T naive',
        spikein_value_mode=('mean', 'NK CD158e1+'),
        labels_selection=['CD8+ T naive', 'CD8+ T CD57+ CD45RO+', 'NK', 'NK CD158e1+', 'CD4+ T naive', 'CD8+ T TIGIT+ CD45RO+'],
        perturbation_mode='spikein',
        results_dir_subdir='spike_in_experiments',
    ),
}

def sparsevi(task=None, hyper_name=None, inject_batch_covariates=None, inject_patient_covariates=None, inject_further_batch_covariates=None, inject_further_continuous_batch_covariates=None, incorporate_batch_scib=None, incorporate_patient_scib=None, optimization_metric=None, dataset_name=None, subsampled=None, paired_rate=None, utilize_highly_variable=None, utilize_highly_abundant=None, utilize_de=None, spike_in=None, mdata=None, mdata_counts=None, adata=None, train_indices=None, val_indices=None, n_genes=None, n_regions=None, n_snps=None, labels_key=None, further_labels_keys=None, batch_key=None, patient_key=None, further_batch_keys=None, further_continuous_batch_keys=None, n_patient_covariates=None, protein_expression_obsm_key=None, use_mean_mixing=None, use_transformer=None, use_transformer_latents=None, use_product_of_experts=None, use_mixture_of_experts=None, use_rectified_flows=None, use_gating_network=None, default: bool = False, best: bool = False, hyper: bool = False, lr=0.0001, weight_decay=1e-3, modality_penalty="Jeffreys", n_layers_encoder=2, n_layers_decoder=2, n_hidden=None, gene_dispersion="gene", protein_dispersion="protein", gene_likelihood="zinb", latent_distribution="normal", dropout_rate=0.1, activation_fn="relu", standard_gene_size=2, standard_go_size=3, layers_encoder_type=None, expression_gene_layer_type=None, accessibility_gene_layer_type=None, genotype_gene_layer_type=None, protein_gene_layer_type=None, sparsities=None, dynamic=None, dynamic_tend_epoch_num=10, gene_interaction_layer_dynamic=False, gene_layer_interaction_source=None, gene_interaction_layer_nlayers=1, gene_groups=None, group_loss_modality_penalty=None, group_loss_max_iter=5, map_ensembl_go=None, filter_namespace=True, calculate_model_interpretability=False, keep_activations=False, perform_perturbation_analysis=False, calculate_covariate_attention=False, perform_scarches=False, perform_scarches_hao_query=False, scarches_modality_mode=False, encode_covariates=False, deeply_inject_covariates=False, first_layer_inject_covariates=False, last_layer_inject_covariates=False, decoder_deeply_inject_covariates=False, unfreeze_first_layers=False, best_optimization_metric=0, obo_file=os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "go-basic.obo"), seed=None, perform_scib_calculation=True, perform_umap_plots=True, perform_model_downstream=True, model_path=None):

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools-original/src/")
    import scvi
    from sparsevi_evaluation import sparsevi_model_introspection
    from sparsevi_evaluation import model_interpretability, model_downstream
    from sparsevi_label_transfer import model_label_transfer

    if model_path is not None:

        hp_path = os.path.join(model_path, "hyperparameters.json")
        if not os.path.exists(hp_path):
            raise FileNotFoundError(
                f"hyperparameters.json not found in model_path: {model_path}. "
                "Cannot reconstruct model settings."
            )
        with open(hp_path, "r") as f:
            saved_hp = json.load(f)

        lr = saved_hp.get("lr", lr)
        weight_decay = saved_hp.get("weight_decay", weight_decay)
        modality_penalty = saved_hp.get("modality_penalty", modality_penalty)
        n_layers_encoder = saved_hp.get("n_layers_encoder", n_layers_encoder)
        n_layers_decoder = saved_hp.get("n_layers_decoder", n_layers_decoder)
        n_hidden = saved_hp.get("n_hidden", n_hidden)
        gene_dispersion = saved_hp.get("gene_dispersion", gene_dispersion)
        protein_dispersion = saved_hp.get("protein_dispersion", protein_dispersion)
        gene_likelihood = saved_hp.get("gene_likelihood", gene_likelihood)
        latent_distribution = saved_hp.get("latent_distribution", latent_distribution)
        dropout_rate = saved_hp.get("dropout_rate", dropout_rate)
        activation_fn = saved_hp.get("activation_fn", activation_fn)
        standard_gene_size = saved_hp.get("standard_gene_size", standard_gene_size)
        standard_go_size = saved_hp.get("standard_go_size", standard_go_size)
        layers_encoder_type = saved_hp.get("layers_encoder_type", layers_encoder_type)
        expression_gene_layer_type = saved_hp.get("expression_gene_layer_type", expression_gene_layer_type)
        accessibility_gene_layer_type = saved_hp.get("accessibility_gene_layer_type", accessibility_gene_layer_type)
        genotype_gene_layer_type = saved_hp.get("genotype_gene_layer_type", genotype_gene_layer_type)
        protein_gene_layer_type = saved_hp.get("protein_gene_layer_type", protein_gene_layer_type)
        sparsities = saved_hp.get("sparsities", sparsities)
        dynamic = saved_hp.get("dynamic", dynamic)
        dynamic_tend_epoch_num = saved_hp.get("dynamic_tend_epoch_num", dynamic_tend_epoch_num)
        gene_interaction_layer_dynamic = saved_hp.get("gene_interaction_layer_dynamic", gene_interaction_layer_dynamic)
        gene_layer_interaction_source = saved_hp.get("gene_layer_interaction_source", gene_layer_interaction_source)
        gene_interaction_layer_nlayers = saved_hp.get("gene_interaction_layer_nlayers", gene_interaction_layer_nlayers)
        gene_groups = saved_hp.get("gene_groups", gene_groups)
        group_loss_modality_penalty = saved_hp.get("group_loss_modality_penalty", group_loss_modality_penalty)
        group_loss_max_iter = saved_hp.get("group_loss_max_iter", group_loss_max_iter)
        map_ensembl_go = saved_hp.get("map_ensembl_go", map_ensembl_go)
        filter_namespace = saved_hp.get("filter_namespace", filter_namespace)
        encode_covariates = saved_hp.get("encode_covariates", encode_covariates)
        deeply_inject_covariates = saved_hp.get("deeply_inject_covariates", deeply_inject_covariates)
        first_layer_inject_covariates = saved_hp.get("first_layer_inject_covariates", first_layer_inject_covariates)
        last_layer_inject_covariates = saved_hp.get("last_layer_inject_covariates", last_layer_inject_covariates)
        decoder_deeply_inject_covariates = saved_hp.get("decoder_deeply_inject_covariates",
                                                        decoder_deeply_inject_covariates)
        inject_batch_covariates = saved_hp.get("inject_batch_covariates", inject_batch_covariates)
        inject_patient_covariates = saved_hp.get("inject_patient_covariates", inject_patient_covariates)
        inject_further_batch_covariates = saved_hp.get("inject_further_batch_covariates",
                                                       inject_further_batch_covariates)
        inject_further_continuous_batch_covariates = saved_hp.get("inject_further_continuous_batch_covariates",
                                                                  inject_further_continuous_batch_covariates)
        incorporate_batch_scib = saved_hp.get("incorporate_batch_scib", incorporate_batch_scib)
        incorporate_patient_scib = saved_hp.get("incorporate_patient_scib", incorporate_patient_scib)
        use_mean_mixing = saved_hp.get("use_mean_mixing", use_mean_mixing)
        use_transformer = saved_hp.get("use_transformer", use_transformer)
        use_transformer_latents = saved_hp.get("use_transformer_latents", use_transformer_latents)
        use_product_of_experts = saved_hp.get("use_product_of_experts", use_product_of_experts)
        use_mixture_of_experts = saved_hp.get("use_mixture_of_experts", use_mixture_of_experts)
        use_rectified_flows = saved_hp.get("use_rectified_flows", use_rectified_flows)
        use_gating_network = saved_hp.get("use_gating_network", use_gating_network)

        hyper_name = os.path.join(model_path, "reload_interpretability")
        os.makedirs(hyper_name, exist_ok=True)


        default = True
        best = False
        perform_scib_calculation = False
        perform_umap_plots = False
        perform_model_downstream = False

        if activation_fn == "relu":
            activation_fn = nn.ReLU
        elif activation_fn == "leaky_relu":
            activation_fn = nn.LeakyReLU
        elif activation_fn == "swish":
            activation_fn = nn.Hardswish
        if n_hidden == -1:
            n_hidden = None

        if paired_rate != 1 and paired_rate != 1.0:
            n_paired = math.ceil(len(mdata_counts.obs) * paired_rate) + 1
            n_unpaired = math.floor((len(mdata_counts.obs) - n_paired) / len(mdata_counts.mod))
            n_paired += len(mdata.obs) - (n_paired + n_unpaired * len(mdata.mod))
            modality_mapping = {
                "rna": "expression", "atac": "accessibility",
                "geno": "genotype", "prot": "protein_expression"
            }
            for mod_index, mod in enumerate(mdata_counts.mod):
                for _, mod_sub in enumerate(mdata_counts.mod):
                    mdata_counts.mod[mod_sub].obs.iloc[
                        list(range(n_paired + n_unpaired * mod_index,
                                   n_paired + n_unpaired * (mod_index + 1))),
                        mdata_counts.mod[mod_sub].obs.columns.get_loc("modality")
                    ] = modality_mapping[mod]
                for mod_index_not_paired in set(range(len(mdata_counts.mod))) - {mod_index}:
                    mdata_counts.mod[mod].X[
                    list(range(n_paired + n_unpaired * mod_index_not_paired,
                               n_paired + n_unpaired * (mod_index_not_paired + 1))), :
                    ] = 0

        adata_mvi = sc.concat(
            [mdata_counts.mod[mod] for mod in mdata_counts.mod if mod != "prot"],
            join="outer", merge="first", axis=1
        )
        adata_mvi.X = np.array(adata_mvi.X)
        if "prot" in mdata_counts.mod:
            adata_mvi.obsm["protein_expression"] = mdata_counts.mod["prot"].X
            adata_mvi.uns["protein_expression"] = {
                "var": mdata_counts.mod["prot"].var,
                "obs": mdata_counts.mod["prot"].obs,
            }

        categorical_covariate_keys = (
                ([batch_key] if inject_batch_covariates else [])
                + ([patient_key] if inject_patient_covariates else [])
                + (further_batch_keys if inject_further_batch_covariates else [])
        ) if inject_batch_covariates or inject_patient_covariates or inject_further_batch_covariates else None
        continuous_covariate_keys = further_continuous_batch_keys if inject_further_continuous_batch_covariates else None

        if perform_scarches or perform_scarches_hao_query:
            if paired_rate != 1 and paired_rate != 1.0:
                scvi.model.SPARSEVI.setup_anndata(
                    adata_mvi, batch_key=categorical_covariate_keys[0],
                    patient_key=patient_key, labels_key=labels_key,
                    protein_expression_obsm_key=protein_expression_obsm_key
                )
            else:
                scvi.model.SPARSEVI.setup_anndata(
                    adata_mvi, batch_key=categorical_covariate_keys[0],
                    patient_key=patient_key,
                    protein_expression_obsm_key=protein_expression_obsm_key
                )
        else:
            if paired_rate != 1 and paired_rate != 1.0:
                scvi.model.SPARSEVI.setup_anndata(
                    adata_mvi, batch_key="modality",
                    categorical_covariate_keys=categorical_covariate_keys,
                    continuous_covariate_keys=continuous_covariate_keys,
                    patient_key=patient_key, labels_key=labels_key,
                    protein_expression_obsm_key=protein_expression_obsm_key
                )
            else:
                scvi.model.SPARSEVI.setup_anndata(
                    adata_mvi,
                    categorical_covariate_keys=categorical_covariate_keys,
                    continuous_covariate_keys=continuous_covariate_keys,
                    patient_key=patient_key,
                    protein_expression_obsm_key=protein_expression_obsm_key
                )

        model = scvi.model.SPARSEVI.load(model_path, adata=adata_mvi)
        print(f"### Reloaded model from {model_path} ###")
        print(f"### Results will be written to {hyper_name} ###")

        _reload_done = True

    else:
        _reload_done = False

    if not _reload_done:

        if seed is not None:
            scvi.settings.seed = seed
        print("### Seed ###")
        print(scvi.settings.seed)

        if obo_file is None:
            obo_file = os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "go-basic.obo")


        if task == "dogma_sparsevi_interpretability_experiments":
            hyper_name = "/".join(hyper_name.split("/")[:-1]) +"_20" + "/" + hyper_name.split("/")[-1]

        hyperparameters = {
            "lr": lr,
            "weight_decay": weight_decay,
            "modality_penalty": modality_penalty,
            "n_layers_encoder": n_layers_encoder,
            "n_layers_decoder": n_layers_decoder,
            "n_hidden": n_hidden,
            "gene_dispersion": gene_dispersion,
            "protein_dispersion": protein_dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            "dropout_rate": dropout_rate,
            "activation_fn": activation_fn,
            "standard_gene_size": standard_gene_size,
            "standard_go_size": standard_go_size,
            "layers_encoder_type": layers_encoder_type,
            "expression_gene_layer_type": expression_gene_layer_type,
            "accessibility_gene_layer_type": accessibility_gene_layer_type,
            "genotype_gene_layer_type": genotype_gene_layer_type,
            "protein_gene_layer_type": protein_gene_layer_type,
            "sparsities": sparsities,
            "dynamic": dynamic,
            "dynamic_tend_epoch_num": dynamic_tend_epoch_num,
            "gene_interaction_layer_dynamic": gene_interaction_layer_dynamic,
            "gene_layer_interaction_source": gene_layer_interaction_source,
            "gene_interaction_layer_nlayers": gene_interaction_layer_nlayers,
            "gene_groups": gene_groups,
            "group_loss_modality_penalty": group_loss_modality_penalty,
            "group_loss_max_iter": group_loss_max_iter,
            "map_ensembl_go": map_ensembl_go,
            "filter_namespace": filter_namespace,
            "encode_covariates": encode_covariates,
            "deeply_inject_covariates": deeply_inject_covariates,
            "first_layer_inject_covariates": first_layer_inject_covariates,
            "last_layer_inject_covariates": last_layer_inject_covariates,
            "decoder_deeply_inject_covariates": decoder_deeply_inject_covariates,
            "inject_batch_covariates": inject_batch_covariates,
            "inject_patient_covariates": inject_patient_covariates,
            "inject_further_batch_covariates": inject_further_batch_covariates,
            "inject_further_continuous_batch_covariates": inject_further_continuous_batch_covariates,
            "incorporate_batch_scib": incorporate_batch_scib,
            "incorporate_patient_scib": incorporate_patient_scib,
            "use_mean_mixing": use_mean_mixing,
            "use_transformer": use_transformer,
            "use_transformer_latents": use_transformer_latents,
            "use_product_of_experts": use_product_of_experts,
            "use_mixture_of_experts": use_mixture_of_experts,
            "use_rectified_flows": use_rectified_flows,
            "use_gating_network": use_gating_network,
        }

        with open(f"{hyper_name}/hyperparameters.json", 'w') as json_file:
            json.dump(hyperparameters, json_file, indent=4)

        if n_hidden == -1:
            n_hidden = None
        if activation_fn == "relu":
            activation_fn = nn.ReLU
        elif activation_fn == "leaky_relu":
            activation_fn = nn.LeakyReLU
        elif activation_fn == "swish":
            activation_fn = nn.Hardswish


        if paired_rate != 1 and paired_rate != 1.0:

            n_paired = math.ceil(len(mdata_counts.obs)*paired_rate)+1
            n_unpaired = math.floor((len(mdata_counts.obs)-n_paired)/len(mdata_counts.mod))
            n_paired += len(mdata.obs) - (n_paired + n_unpaired * len(mdata.mod))

            modality_mapping = {
                "rna": "expression",
                "atac": "accessibility",
                "geno": "genotype",
                "prot": "protein_expression"
            }

            for mod_index, mod in enumerate(mdata_counts.mod):
                for _, mod_sub in enumerate(mdata_counts.mod):
                    mdata_counts.mod[mod_sub].obs.iloc[list(range(n_paired + n_unpaired * mod_index, n_paired + n_unpaired * (mod_index + 1))), mdata_counts.mod[mod_sub].obs.columns.get_loc("modality")] = modality_mapping[mod]
                for mod_index_not_paired in set(range(len(mdata_counts.mod))) - set([mod_index]):
                    mdata_counts.mod[mod].X[list(range(n_paired + n_unpaired * mod_index_not_paired, n_paired + n_unpaired * (mod_index_not_paired + 1))), :] = 0

        adata_mvi = sc.concat([mdata_counts.mod[mod] for mod in mdata_counts.mod if mod != "prot"], join="outer", merge="first", axis=1)
        adata_mvi.X = np.array(adata_mvi.X)
        if "prot" in mdata_counts.mod:
            adata_mvi.obsm["protein_expression"] = mdata_counts.mod["prot"].X
            adata_mvi.uns['protein_expression'] = {
                'var': mdata_counts.mod["prot"].var,
                'obs': mdata_counts.mod["prot"].obs
            }

        categorical_covariate_keys = ([batch_key] if inject_batch_covariates else []) + ([patient_key] if inject_patient_covariates else []) + (further_batch_keys if inject_further_batch_covariates else []) if inject_batch_covariates or inject_patient_covariates or inject_further_batch_covariates else None
        continuous_covariate_keys = further_continuous_batch_keys if inject_further_continuous_batch_covariates else None
        if perform_scarches or perform_scarches_hao_query:
            if paired_rate != 1 and paired_rate != 1.0:
                scvi.model.SPARSEVI.setup_anndata(adata_mvi, batch_key=categorical_covariate_keys[0], patient_key=patient_key, labels_key=labels_key, protein_expression_obsm_key=protein_expression_obsm_key)
            else:
                scvi.model.SPARSEVI.setup_anndata(adata_mvi, batch_key=categorical_covariate_keys[0], patient_key=patient_key, protein_expression_obsm_key=protein_expression_obsm_key)
        else:
            if paired_rate != 1 and paired_rate != 1.0:
                scvi.model.SPARSEVI.setup_anndata(adata_mvi, batch_key="modality", categorical_covariate_keys=categorical_covariate_keys, continuous_covariate_keys=continuous_covariate_keys, patient_key=patient_key, labels_key=labels_key, protein_expression_obsm_key=protein_expression_obsm_key)
            else:
                scvi.model.SPARSEVI.setup_anndata(adata_mvi, categorical_covariate_keys=categorical_covariate_keys, continuous_covariate_keys=continuous_covariate_keys, patient_key=patient_key, protein_expression_obsm_key=protein_expression_obsm_key)

        if best:
            model = scvi.model.SPARSEVI.load(hyper_name, adata=adata_mvi)
        else:
            model = scvi.model.SPARSEVI(
                adata_mvi,
                n_latent=20,
                n_genes=n_genes,
                ensembl_ids_genes=adata_mvi.var[adata_mvi.var["modality"] == "expression"]["gene_stable_id"].to_numpy(),
                n_regions=n_regions,
                ensembl_ids_regions=adata_mvi.var[adata_mvi.var["modality"] == "Peaks"]["gene_stable_id"].to_numpy(),
                n_snps=n_snps,
                ensembl_ids_snps=adata_mvi.var[adata_mvi.var["modality"] == "SNP"]["gene_stable_id"].to_numpy(),
                ensembl_ids_proteins=mdata_counts.mod["prot"].var["gene_stable_id"].to_numpy() if "prot" in mdata_counts.mod else None,
                n_patient_covariates=n_patient_covariates,
                layers_encoder_type=layers_encoder_type,
                expression_gene_layer_type=expression_gene_layer_type,
                accessibility_gene_layer_type=accessibility_gene_layer_type,
                genotype_gene_layer_type=genotype_gene_layer_type,
                protein_gene_layer_type=protein_gene_layer_type,
                layers_decoder_type="standard",
                library_size_layers_type="dense",
                gene_interaction_layer_dynamic=gene_interaction_layer_dynamic,
                gene_interaction_layer_pruning_frac=0.1,
                gene_interaction_layer_dynamic_update_rate=int(np.round(train_indices.shape[0]/scvi.settings._batch_size))+1,
                gene_interaction_layer_dynamic_end_update_rate=int(np.round(train_indices.shape[0]/scvi.settings._batch_size))*10,
                gene_interaction_layer_dynamic_save_path=hyper_name,
                gene_layer_interaction_source=gene_layer_interaction_source,
                gene_interaction_layer_nlayers=gene_interaction_layer_nlayers,
                sparsities=sparsities,
                dynamic=dynamic,
                dynamic_update_rate=int(np.round(train_indices.shape[0]/scvi.settings._batch_size))+1,
                dynamic_end_update_rate=int(np.round(train_indices.shape[0]/scvi.settings._batch_size))*dynamic_tend_epoch_num,
                keep_activations=False,
                fully_paired=False if (paired_rate != 1.0 and paired_rate != 1) else True,
                dropout_rate=dropout_rate,
                n_layers_encoder=n_layers_encoder,
                n_layers_decoder=n_layers_decoder,
                n_hidden=n_hidden,
                activation_fn=activation_fn,
                standard_gene_size=standard_gene_size,
                standard_go_size=standard_go_size,
                obo_file=obo_file,
                map_ensembl_go=map_ensembl_go,
                filter_namespace=filter_namespace,
                encode_covariates=encode_covariates,
                deeply_inject_covariates=deeply_inject_covariates,
                first_layer_inject_covariates=first_layer_inject_covariates,
                last_layer_inject_covariates=last_layer_inject_covariates,
                decoder_deeply_inject_covariates=decoder_deeply_inject_covariates,
                group_loss_modality_penalty=group_loss_modality_penalty,
                group_loss_max_iter=group_loss_max_iter,
                use_mean_mixing=use_mean_mixing,
                use_transformer=use_transformer,
                use_transformer_latents=use_transformer_latents,
                use_product_of_experts=use_product_of_experts,
                use_mixture_of_experts=use_mixture_of_experts,
                use_rectified_flows=use_rectified_flows,
                modality_penalty=modality_penalty,
                modality_weights="moe" if use_gating_network else "equal",
            )

            try:
                sparsevi_model_introspection(model, adata_mvi, categorical_covariate_keys)
            except Exception as e:
                print(f"Model introspection failed due to exception: {str(e)}")

            max_epochs=500
            model.train(max_epochs=max_epochs, shuffle_set_split=False, adversarial_mixing=True if (paired_rate != 1.0 and paired_rate != 1) else False, train_idx=train_indices, validation_indices=val_indices)


            history = model.history
            with open(os.path.join(f"{hyper_name}", "history.pickle"), 'wb') as handle:
                pickle.dump(history, handle, protocol=pickle.DEFAULT_PROTOCOL)


            if perform_umap_plots:
                latent_representation = model.get_latent_representation(modality="joint", indices=train_indices)

                def generate_umap_save(hyper_name, latent_representation, suffix=""):

                    latent_representation_anndata = ad.AnnData(latent_representation)
                    sc.pp.neighbors(latent_representation_anndata)
                    sc.tl.umap(latent_representation_anndata, n_components=2)
                    np.save(f"{hyper_name}/X_umap{suffix}.npy", latent_representation_anndata.obsm["X_umap"])

                    np.save(f"{hyper_name}/X_sparsevi{suffix}.npy", latent_representation)

                generate_umap_save(hyper_name, latent_representation)

                for name, key in zip(["labels", "modality", "batch", "patient"] + further_labels_keys, [labels_key, "modality", batch_key, patient_key] + further_labels_keys):
                    if key in mdata_counts.mod["rna"][train_indices, :].obs.columns:
                        np.save(f"{hyper_name}/{name}.npy", mdata_counts.mod["rna"][train_indices, :].obs[key])

                if paired_rate != 1 and paired_rate != 1.0:
                    if "rna" in mdata.mod:
                        latent_representation_expression = model.get_latent_representation(modality="expression", indices=train_indices)
                        generate_umap_save(hyper_name, latent_representation_expression, suffix="_expression")
                    if "atac" in mdata.mod:
                        latent_representation_accessibility = model.get_latent_representation(modality="accessibility", indices=train_indices)
                        generate_umap_save(hyper_name, latent_representation_accessibility, suffix="_accessibility")
                    if "geno" in mdata.mod:
                        latent_representation_genotype = model.get_latent_representation(modality="genotype", indices=train_indices)
                        generate_umap_save(hyper_name, latent_representation_genotype, suffix="_genotype")
                    if "prot" in mdata.mod:
                        latent_representation_protein = model.get_latent_representation(modality="protein", indices=train_indices)
                        generate_umap_save(hyper_name, latent_representation_protein, suffix="_protein")

            if perform_scarches or perform_scarches_hao_query:
                latent_representation = model.get_latent_representation(modality="joint", indices=train_indices)
                scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, adata_mvi[train_indices, :],
                                                      labels_key, batch_key, None, patient_key, incorporate_batch_scib,
                                                      incorporate_patient_scib)
            else:
                if perform_scib_calculation:
                    latent_representation = model.get_latent_representation(modality="joint", indices=train_indices)
                    if paired_rate != 1 and paired_rate != 1.0:
                        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, adata_mvi[train_indices, :], labels_key, "modality", batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)
                    else:
                        scib = metrics.calculate_scib_metrics(f"{hyper_name}/", latent_representation, adata_mvi[train_indices, :], labels_key, None, batch_key, patient_key, incorporate_batch_scib, incorporate_patient_scib)
                else:
                    scib = 0

            if default or scib > best_optimization_metric:
                model.save(f"{hyper_name}", overwrite=True)


    if _reload_done:
        model.get_go_gene_importances(labels_column=labels_key,
                                      labels_selection=labels_selection,
                                      results_dir=f"{hyper_name}/calc_go_terms_{labels_key}",
                                      shuffle_set_split=False,
                                      batch_size=512, train_idx=train_indices,
                                      validation_idx=val_indices, calc_go_terms=True,
                                      calc_gene_groups=False, save_fit=False,
                                      overwrite_save_fit=False)

    if default or best or _reload_done:

        if calculate_covariate_attention:
            model.calculate_covariate_attention(labels_column=labels_key,
                                          results_dir=f"{hyper_name}/covariate_attention_{labels_key}",
                                          shuffle_set_split=False,
                                          batch_size=512, train_idx=train_indices,
                                          validation_idx=val_indices,
                                          modality_categorical_covariate_keys=(["modality"] if paired_rate != 1 and paired_rate != 1.0 else []) + categorical_covariate_keys,
                                          continuous_covariate_keys=continuous_covariate_keys)

        calculate_attention_rollout = False
        if calculate_attention_rollout and (use_transformer or use_transformer_latents):
            model.calculate_attention_rollout(labels_column=labels_key,
                                          results_dir=f"{hyper_name}/calc_go_terms_{labels_key}",
                                          shuffle_set_split=False,
                                          batch_size=512, train_idx=train_indices,
                                          validation_idx=val_indices)


        if calculate_model_interpretability:
            model_interpretability(hyper_name, adata_mvi=adata_mvi, mdata=mdata, model=model, labels_key=labels_key,
                                   train_indices=train_indices, val_indices=val_indices,
                                   calc_go_terms=True, calc_gene_groups=gene_groups,
                                   perform_perturbation_analysis=perform_perturbation_analysis,
                                   keep_activations=keep_activations)
            if patient_key is not None:
                model_interpretability(hyper_name, adata_mvi=adata_mvi, mdata=mdata, model=model, labels_key=patient_key,
                                       train_indices=train_indices, val_indices=val_indices,
                                       calc_go_terms=True if layers_encoder_type == "go" else False,
                                       calc_gene_groups=gene_groups,
                                       perform_perturbation_analysis=perform_perturbation_analysis,
                                       keep_activations=keep_activations)

        if task in _GO_IMPORTANCE_TASKS:
            for cfg in _GO_IMPORTANCE_TASKS[task]:
                _run_go_importance(model, hyper_name, labels_key, train_indices, val_indices, **cfg)

        elif task in _PERTURBATION_IMPORTANCE_TASKS:
            _run_perturbation_importance(model, hyper_name, labels_key, train_indices, val_indices,
                                         **_PERTURBATION_IMPORTANCE_TASKS[task])

        elif task == "neurips2021_cite_bmmc_sparsevi_interpretability_experiments3":
            t_cells = [
                "CD4+ T activated", "CD4+ T naive", "CD8+ T naive", "CD8+ T CD57+ CD45RO+",
                "CD8+ T CD57+ CD45RA+", "CD8+ T TIGIT+ CD45RO+", "CD8+ T TIGIT+ CD45RA+",
                "CD8+ T CD49f+", "CD8+ T CD69+ CD45RO+", "CD8+ T CD69+ CD45RA+",
                "CD4+ T activated integrinB7+", "CD4+ T CD314+ CD45RA+", "T reg", "MAIT", "dnT",
                "T prog cycling", "gdT CD158b+", "gdT TCRVD2+",
            ]
            other_t_cells = [l for l in t_cells if l not in ("gdT CD158b+", "gdT TCRVD2+")]
            for gdt_subset in ["gdT CD158b+", "gdT TCRVD2+"]:
                _run_go_importance(model, hyper_name, labels_key, train_indices, val_indices,
                                   calc_go_terms=True, calc_gene_groups=False, calc_genes=True,
                                   save_fit=False, overwrite_save_fit=False,
                                   labels_selection=[gdt_subset],
                                   comparison=other_t_cells, comparison_alt_des="other_t_cells")

        elif task == "frangieh2021_sparsevi_interpretability_experiments":
            os.makedirs(f"./{hyper_name}", exist_ok=True)
            all_indices = np.concatenate([train_indices, val_indices])
            latent_representation = model.get_latent_representation(modality="joint", indices=all_indices)

            latent_ad = ad.AnnData(latent_representation)
            sc.pp.neighbors(latent_ad)
            sc.tl.umap(latent_ad, n_components=2)
            np.save(f"{hyper_name}/X_umap_with_perturbation.npy", latent_ad.obsm["X_umap"])
            np.save(f"{hyper_name}/X_sparsevi_with_perturbation.npy", latent_representation)

            for name, key in zip(["labels", "perturbation_2", "batch"], [labels_key, "perturbation_2", batch_key]):
                if key in mdata_counts.mod["rna"][all_indices, :].obs.columns:
                    np.save(f"{hyper_name}/{name}_with_perturbation.npy",
                            mdata_counts.mod["rna"][all_indices, :].obs[key])

            combo_key = labels_key + "_perturbation_2"
            adata_mvi.obs[combo_key] = (
                adata_mvi.obs[labels_key].astype(str) + "_" + adata_mvi.obs["perturbation_2"].astype(str)
            )

            for perturbation_gene in _FRANGIEH_PERTURBATION_CANDIDATES:
                try:
                    _run_go_importance(model, hyper_name, labels_key, train_indices, val_indices,
                                       comparison=perturbation_gene, comparison_alt_des=perturbation_gene,
                                       restrict_by_column_key_activations_mask=labels_key,
                                       restrict_by_column_values_activations_mask=["control", perturbation_gene],
                                       restrict_by_column_values_alt_des_activations_mask="")
                except Exception:
                    print(f"{perturbation_gene} not working.")

                for perturbation_2 in ["Control", "Co-culture", "IFNγ"]:
                    try:
                        _run_go_importance(model, hyper_name, labels_key, train_indices, val_indices,
                                           labels_column=combo_key,
                                           comparison=f"{perturbation_gene}_{perturbation_2}",
                                           comparison_alt_des=f"{perturbation_gene}_{perturbation_2}",
                                           restrict_by_column_key_activations_mask=combo_key,
                                           restrict_by_column_values_activations_mask=[
                                               f"control_{perturbation_2}", f"{perturbation_gene}_{perturbation_2}"],
                                           restrict_by_column_values_alt_des_activations_mask="")
                    except Exception:
                        print(f"{perturbation_gene} with {perturbation_2} not working.")

        elif task == "dogma_sparsevi_interpretability_experiments":
            perturbation_gene_stable_ids = list(mdata_counts.mod["prot"].var["gene_stable_id"])
            _run_perturbation_importance(model, hyper_name, labels_key, train_indices, val_indices,
                                         labels_column="Status_on_day_collection_summary",
                                         perturbation_gene_stable_ids=perturbation_gene_stable_ids)


        if perform_scarches or perform_scarches_hao_query:
            model_label_transfer(hyper_name, perform_scarches_hao_query, scarches_modality_mode, inject_batch_covariates, inject_patient_covariates, incorporate_batch_scib, incorporate_patient_scib, unfreeze_first_layers, dataset_name,
                                 subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, adata_mvi=adata_mvi, mdata_counts=mdata_counts, model=model,
                                 labels_key=labels_key, train_indices=train_indices, val_indices=val_indices)

        if perform_model_downstream:
            model_downstream(hyper_name, adata_mvi=adata_mvi, mdata=mdata, model=model, labels_key=labels_key,
                             train_indices=train_indices, val_indices=val_indices)
            if patient_key is not None:
                model_downstream(hyper_name, adata_mvi=adata_mvi, mdata=mdata, model=model, labels_key=patient_key,
                                 train_indices=train_indices, val_indices=val_indices)

    if not best:
        if optimization_metric.lower() == "scib":
            return scib
        elif optimization_metric.lower() == "elbo":
            return min(list(model.history["elbo_validation"]["elbo_validation"]))
