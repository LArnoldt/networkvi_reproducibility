from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
import numpy as np
import scanpy as sc
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
import tools


def main(task, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, sparsevi_config_settings, default_option, seed=None):

    os.makedirs(f"./experiments_new/", exist_ok=True)
    if seed is not None:
        os.makedirs(f"./experiments_new/go_gene_stability_s{seed}", exist_ok=True)
    else:
        os.makedirs(f"./experiments_new/go_gene_stability", exist_ok=True)

    if tool_name.lower() != "sparsevi":
        raise NotImplementedError(f"Tool must be sparsevi.")

    import sparsevi_evaluation
    model_func = tools.sparsevi

    for i in tqdm(range(10)):
        if seed is not None:
            os.makedirs(f"./experiments_new/go_gene_stability_s{seed}/default{i}", exist_ok=True)
        else:
            os.makedirs(f"./experiments_new/go_gene_stability/default{i}", exist_ok=True)

        sparsevi_config = sparsevi_evaluation.get_sparsevi_config(**{"n_layers_encoder": 2}, **sparsevi_config_settings)
        if seed is not None:
            metric = model_func(task, f"./experiments_new/go_gene_stability_s{seed}/default{i}", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, default=True, best=False, hyper=False, calculate_model_interpretability=True, keep_activations=False, perform_perturbation_analysis=True, calculate_covariate_attention=False, perform_scarches=False, perform_scarches_hao_query=False, scarches_modality_mode=False, **covariates_config, **sparsevi_config)
        else:
            metric = model_func(task, f"./experiments_new/go_gene_stability/default{i}", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, default=True, best=False, hyper=False, calculate_model_interpretability=True, keep_activations=False, perform_perturbation_analysis=True, calculate_covariate_attention=False, perform_scarches=False, perform_scarches_hao_query=False, scarches_modality_mode=False, **covariates_config, **sparsevi_config)

        print(metric)
