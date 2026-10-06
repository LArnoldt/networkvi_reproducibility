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
from prettytable import PrettyTable
import time
import run_tool_parameters_different_seeds
import shutil

def main(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option, load_previous_ax_client=False, seed=None, model_path=None):

    if tool_name.lower() == "sparsevi":
        model_func = tools.sparsevi
        import sparsevi_evaluation
        from tools_hyper import sparsevi_hyperparameter_setup
        ax_client = sparsevi_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric, sparsevi_config_settings=sparsevi_config_settings, default_option=default_option)
    else:
        raise NotImplementedError(f"Tool {tool_name} not available.")

    default_parameters = ax_client.get_trial_parameters(trial_index=0)

    if sparsevi_config_settings["modality_penalty"] is not None:
        baseline_parameters["modality_penalty"] = sparsevi_config_settings["modality_penalty"]
    del sparsevi_config_settings["modality_penalty"]
    sparsevi_config = sparsevi_evaluation.get_sparsevi_config(**default_parameters, **sparsevi_config_settings)
    if "map_ensembl_go" in sparsevi_config.keys():
        default_parameters["map_ensembl_go"] = sparsevi_config["map_ensembl_go"]
        del sparsevi_config["map_ensembl_go"]

    metric = model_func(task, f"./experiments_new/{hyper_name}/default", inject_batch_covariates,
                        inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib,
                        optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts,
                        adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys,
                        batch_key,
                        patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing=use_mean_mixing, use_transformer=use_transformer, use_transformer_latents=use_transformer_latents, use_product_of_experts=use_product_of_experts, use_mixture_of_experts=use_mixture_of_experts, use_rectified_flows=use_rectified_flows, use_gating_network=use_gating_network, default=False, best=True,
                        hyper=False, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations,
                        perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode,
                        **covariates_config, **default_parameters, **sparsevi_config, seed=seed, model_path=model_path)
