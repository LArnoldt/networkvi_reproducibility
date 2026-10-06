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
import json


def main(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option, different_seeds=None, model_parameters_json=None):

    os.makedirs(f"./experiments_new/", exist_ok=True)

    if tool_name.lower() == "wnn":
        model_func = tools.wnn
    elif tool_name.lower() == "mofa" or tool_name.lower() == "mofa+":
        model_func = tools.mofa
    elif tool_name.lower() == "midas":
        model_func  = tools.midas
    elif tool_name.lower() == "multimil":
        model_func  = tools.multimil
    elif tool_name.lower() == "totalvi":
        model_func = tools.totalvi
    elif tool_name.lower() == "sparsevi":
        model_func = tools.sparsevi
    else:
        raise NotImplementedError(f"Tool {tool_name} not available.")

    if model_parameters_json is not None:
        with open(model_parameters_json, 'r') as json_file:
            parameters = json.load(json_file)
    else:
        import sparsevi_evaluation
        from tools_hyper import sparsevi_hyperparameter_setup
        ax_client = sparsevi_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric,
                                                  sparsevi_config_settings=sparsevi_config_settings,
                                                  default_option=default_option)
        parameters = ax_client.get_trial_parameters(trial_index=0)

    for seed in tqdm(different_seeds):

        os.makedirs(f"./experiments_new/{hyper_name}_s{seed}", exist_ok=True)
        os.makedirs(f"./experiments_new/{hyper_name}_s{seed}/default", exist_ok=True)

        if tool_name.lower() == "sparsevi":
            sparsevi_config = sparsevi_evaluation.get_sparsevi_config(**parameters, **sparsevi_config_settings)
            metric = model_func(task=task, hyper_name=f"./experiments_new/{hyper_name}_s{seed}/default", optimization_metric=optimization_metric, dataset_name=dataset_name, subsampled=subsampled, paired_rate=paired_rate, utilize_highly_variable=utilize_highly_variable, utilize_highly_abundant=utilize_highly_abundant, utilize_de=utilize_de, spike_in=spike_in, mdata=mdata, mdata_counts=mdata_counts, adata=adata, train_indices=train_indices, val_indices=val_indices, n_genes=n_genes, n_regions=n_regions,
                       n_snps=n_snps, labels_key=labels_key, further_labels_keys=further_labels_keys, batch_key=batch_key, patient_key=patient_key, further_batch_keys=further_batch_keys, further_continuous_batch_keys=further_continuous_batch_keys, n_patient_covariates=n_patient_covariates, protein_expression_obsm_key=protein_expression_obsm_key,
                       default=True, best=False, hyper=False, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations, perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode, **sparsevi_config, seed=seed)
        else:
            metric = model_func(hyper_name=f"./experiments_new/{hyper_name}_s{seed}/default", optimization_metric=optimization_metric, dataset_name=dataset_name, subsampled=subsampled, paired_rate=paired_rate, utilize_highly_variable=utilize_highly_variable, utilize_highly_abundant=utilize_highly_abundant, utilize_de=utilize_de, spike_in=spike_in, mdata=mdata, mdata_counts=mdata_counts, adata=adata, train_indices=train_indices, val_indices=val_indices, n_genes=n_genes, n_regions=n_regions,
                       n_snps=n_snps, labels_key=labels_key, further_labels_keys=further_labels_keys, batch_key=batch_key, patient_key=patient_key, further_batch_keys=further_batch_keys, further_continuous_batch_keys=further_continuous_batch_keys, n_patient_covariates=n_patient_covariates, protein_expression_obsm_key=protein_expression_obsm_key,
                       default=True, best=False, hyper=False, **parameters, seed=seed)

        print(metric)
