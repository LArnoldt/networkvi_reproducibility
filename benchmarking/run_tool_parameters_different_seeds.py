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


def main(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, networkvi_config_settings, different_seeds=None, model_parameters_json=None):

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
    elif tool_name.lower() == "networkvi":
        model_func = tools.networkvi
    else:
        raise NotImplementedError(f"Tool {tool_name} not available.")

    if model_parameters_json is not None:
        with open(model_parameters_json, 'r') as json_file:
            parameters = json.load(json_file)
    else:
        from tools_hyper import networkvi_hyperparameter_setup
        ax_client = networkvi_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric,
                                                  networkvi_config_settings=networkvi_config_settings)
        parameters = ax_client.get_trial_parameters(trial_index=0)

    if tool_name.lower() == "networkvi":
        import networkvi_evaluation
        if networkvi_config_settings.get("modality_penalty") is not None:
            parameters["modality_penalty"] = networkvi_config_settings["modality_penalty"]
        networkvi_config_settings = {key: value for key, value in networkvi_config_settings.items() if key != "modality_penalty"}
        networkvi_config = networkvi_evaluation.get_networkvi_config(**parameters, **networkvi_config_settings)
        if "map_ensembl_go" in networkvi_config.keys():
            parameters["map_ensembl_go"] = networkvi_config.pop("map_ensembl_go")

    for seed in tqdm(different_seeds):

        os.makedirs(f"./experiments_new/{hyper_name}_s{seed}", exist_ok=True)
        os.makedirs(f"./experiments_new/{hyper_name}_s{seed}/default", exist_ok=True)

        if tool_name.lower() == "networkvi":
            metric = model_func(task, f"./experiments_new/{hyper_name}_s{seed}/default", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network,
                       default=True, best=False, hyper=False, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations, perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode, **covariates_config, **parameters, **networkvi_config, best_optimization_metric=0, seed=seed)
        else:
            metric = model_func(hyper_name=f"./experiments_new/{hyper_name}_s{seed}/default", optimization_metric=optimization_metric, dataset_name=dataset_name, subsampled=subsampled, paired_rate=paired_rate, utilize_highly_variable=utilize_highly_variable, utilize_highly_abundant=utilize_highly_abundant, utilize_de=utilize_de, mdata=mdata, mdata_counts=mdata_counts, adata=adata, train_indices=train_indices, val_indices=val_indices, n_genes=n_genes, n_regions=n_regions,
                       default=True, best=False, hyper=False, **parameters, seed=seed)

        print(metric)
