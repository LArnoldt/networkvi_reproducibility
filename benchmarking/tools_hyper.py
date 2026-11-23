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

def fixed_configuration_setup(hyper_name="", parameters=None):

    ax_client = AxClient()
    ax_client.create_experiment(
        name=hyper_name,
        parameters=[{"name": name, "type": "fixed", "value": value} for name, value in parameters.items()],
        objectives={"scib": ObjectiveProperties(minimize=False)},
    )
    ax_client.attach_trial(parameters=dict(parameters))

    return ax_client


def mofa_hyperparameter_setup(hyper_name=""):

    return fixed_configuration_setup(hyper_name, {"scale_views": False, "scale_groups": False, "center_groups": True, "ard_weights": True, "ard_factors": True, "spikeslab_weights": True, "spikeslab_factors": False, "convergence_mode": "fast"})


def wnn_hyperparameter_setup(hyper_name=""):

    return fixed_configuration_setup(hyper_name, {"n_bandwidth_neighbors": 20, "n_multineighbors": 200, "metric": "euclidean"})


def totalvi_hyperparameter_setup(hyper_name="", optimization_metric="elbo"):

    ax_client = AxClient()
    ax_client.create_experiment(
        name=hyper_name,
        parameters=[
            {
                "name": "lr",
                "type": "choice",
                "values": [1e-6, 1e-5, 1e-4, 1e-3],
                "value_type": "float",
            },
            {
                "name": "n_layers_encoder",
                "type": "choice",
                "values": [2, 3, 4, 5],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_layers_decoder",
                "type": "choice",
                "values": [1, 2, 3, 4, 5],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_hidden",
                "type": "choice",
                "values": [128, 256, 384, 512],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "gene_likelihood",
                "type": "choice",
                "values": ["zinb", "nb"],
                "value_type": "str",
                "sort_values": False,
                "is_ordered": False,
            },
            {
                "name": "dropout_rate_encoder",
                "type": "choice",
                "values": [0.0, 0.1, 0.2, 0.3],
                "value_type": "float",
            },
        ],
        objectives={"elbo_validation": ObjectiveProperties(minimize=True)} if optimization_metric == "elbo" else {"scib": ObjectiveProperties(minimize=False)},
    )
    ax_client.attach_trial(
        parameters={"lr": 0.0001, "n_layers_encoder": 2, "n_layers_decoder": 1, "n_hidden": 256, "gene_likelihood": "nb", "dropout_rate_encoder": 0.2}
    )

    return ax_client

def midas_hyperparameter_setup(hyper_name="", optimization_metric="scib"):

    return fixed_configuration_setup(hyper_name, {"lr": 1e-4, "drop": 0.2, "disc_train": 3, "loss_s_recon": 1000.0, "loss_mod_alignment": 50.0, "loss_disc": 30.0})

def multimil_hyperparameter_setup(hyper_name="", optimization_metric="elbo"):

    ax_client = AxClient()
    ax_client.create_experiment(
        name=hyper_name,
        parameters=[
            {
                "name": "lr",
                "type": "choice",
                "values": [1e-6, 1e-5, 1e-4, 1e-3],
                "value_type": "float",
            },
            {
                "name": "dropout",
                "type": "choice",
                "values": [0.0, 0.1, 0.2, 0.3],
                "value_type": "float",
            },
            {
                "name": "n_layers_encoders",
                "type": "choice",
                "values": [2, 3, 4, 5, -1],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_layers_decoders",
                "type": "choice",
                "values": [2, 3, 4, 5, -1],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_hidden_encoders",
                "type": "choice",
                "values": [128, 256, 384, 512, -1],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_hidden_decoders",
                "type": "choice",
                "values": [128, 256, 384, 512, -1],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "activation",
                "type": "choice",
                "values": ["leaky_relu", "tanh"],
                "value_type": "str",
                "sort_values": False,
                "is_ordered": False,
            },
        ],
        objectives={"elbo_validation": ObjectiveProperties(minimize=True)} if optimization_metric == "elbo" else {"scib": ObjectiveProperties(minimize=False)},
    )
    ax_client.attach_trial(
        parameters={"lr": 0.0001, "dropout": 0.2, "n_layers_encoders": -1, "n_layers_decoders": -1, "n_hidden_encoders": -1, "n_hidden_decoders": -1, "activation": "leaky_relu"}
    )

    return ax_client

def networkvi_hyperparameter_setup(hyper_name="", optimization_metric="elbo", networkvi_config_settings=None):

    modality_penalty = (networkvi_config_settings or {}).get("modality_penalty") or "Jeffreys"
    go_baseline_parameters = {"standard_go_size": 2} if "go" in networkvi_config_settings['model_type'] else {}
    gene_baseline_parameters = {"standard_gene_size": 5} if "gene" in networkvi_config_settings['model_type'] else {}
    if "go" in networkvi_config_settings['model_type']:
        default_parameters = {"modality_penalty": modality_penalty, "n_layers_encoder": 4, "n_layers_decoder": 2, "n_hidden": 384, "dropout_rate": 0.3, "activation_fn": "swish"}
    else:
        default_parameters = {"modality_penalty": modality_penalty, "n_layers_encoder": 2, "n_layers_decoder": 2, "n_hidden": -1, "dropout_rate": 0.1, "activation_fn": "relu"}

    ax_client = AxClient()
    ax_client.create_experiment(
        name=hyper_name,
        parameters=([
            {
                "name": "modality_penalty",
                "type": "fixed",
                "value": modality_penalty,
                "value_type": "str",
            },
            {
                "name": "n_layers_encoder",
                "type": "choice",
                "values": [2,3,4,5],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_layers_decoder",
                "type": "choice",
                "values": [2,3,4],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "n_hidden",
                "type": "choice",
                "values": [128, 256, 384, -1] if "go" in networkvi_config_settings['model_type'] else [128, 256, 384, 512, 640, 768, -1],
                "value_type": "int",
                "sort_values": True,
                "is_ordered": True,
            },
            {
                "name": "dropout_rate",
                "type": "choice",
                "values": [0.0, 0.1, 0.2, 0.3],
                "value_type": "float",
            },
            {
                "name": "activation_fn",
                "type": "choice",
                "values": ["relu", "leaky_relu", "swish"],
                "sort_values": False,
                "is_ordered": False,
            }]
            ) + (
            [
                {
                    "name": "standard_go_size",
                    "type": "choice",
                    "values": [2, 3, 4, 5],
                    "value_type": "int",
                    "sort_values": True,
                    "is_ordered": True,
                },
            ] if "go" in networkvi_config_settings['model_type'] else []
            ) + (
            [
                {
                    "name": "standard_gene_size",
                    "type": "choice",
                    "values": [2, 3, 4, 5],
                    "value_type": "int",
                    "sort_values": True,
                    "is_ordered": True,
                }
            ] if "gene" in networkvi_config_settings['model_type'] else []),
        objectives={"elbo_validation": ObjectiveProperties(minimize=True)} if optimization_metric == "elbo" else {"scib": ObjectiveProperties(minimize=False)},
    )
    ax_client.attach_trial(
        parameters={**default_parameters, **go_baseline_parameters, **gene_baseline_parameters}
    )

    return ax_client

def hyper_wrapper(task, not_run_hyper, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, networkvi_config_settings, load_previous_ax_client=False, obo_file=None, filter_namespace=True, seed=None):

    os.makedirs(f"./experiments_new/", exist_ok=True)
    os.makedirs(f"./experiments_new/{hyper_name}", exist_ok=True)
    os.makedirs(f"./experiments_new/{hyper_name}/default", exist_ok=True)
    os.makedirs(f"./experiments_new/{hyper_name}/hyper", exist_ok=True)

    x = PrettyTable()
    x.field_names = ["run", "running time", f"{optimization_metric}"]

    if tool_name.lower() == "wnn":
        model_func = tools.wnn
        ax_client = wnn_hyperparameter_setup(hyper_name=hyper_name)
    elif tool_name.lower() == "mofa" or tool_name.lower() == "mofa+":
        model_func = tools.mofa
        ax_client = mofa_hyperparameter_setup(hyper_name=hyper_name)
    elif tool_name.lower() == "multimil":
        model_func  = tools.multimil
        ax_client = multimil_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric)
    elif tool_name.lower() == "midas":
        model_func = tools.midas
        ax_client = midas_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric)
    elif tool_name.lower() == "totalvi":
        model_func = tools.totalvi
        ax_client = totalvi_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric)
    elif tool_name.lower() == "networkvi":
        model_func = tools.networkvi
        import networkvi_evaluation
        ax_client = networkvi_hyperparameter_setup(hyper_name=hyper_name, optimization_metric=optimization_metric, networkvi_config_settings=networkvi_config_settings)
    else:
        raise NotImplementedError(f"Tool {tool_name} not available.")

    if load_previous_ax_client:
        ax_client = AxClient().load_from_json_file(os.path.join(f"./experiments_new/{hyper_name}/hyper", "ax_client.json"))
    else:
        start_default = time.time()
        baseline_parameters = ax_client.get_trial_parameters(trial_index=0)

        if tool_name.lower() == "networkvi":
            if networkvi_config_settings["modality_penalty"] is not None:
                baseline_parameters["modality_penalty"] = networkvi_config_settings["modality_penalty"]
            del networkvi_config_settings["modality_penalty"]

            networkvi_config = networkvi_evaluation.get_networkvi_config(**baseline_parameters, **networkvi_config_settings)
            if "map_ensembl_go" in networkvi_config.keys():
                baseline_parameters["map_ensembl_go"] = networkvi_config["map_ensembl_go"]
                del networkvi_config["map_ensembl_go"]
            try:
                metric = model_func(task, f"./experiments_new/{hyper_name}/default", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network,
                       default=True, best=False, hyper=False, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations, perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode, **covariates_config, **baseline_parameters, **networkvi_config, best_optimization_metric=0, obo_file=obo_file, filter_namespace=filter_namespace, seed=seed)
            except Exception as e:
                metric = 0
                print(f"Trial failed due to exception: {str(e)}")
        else:
            try:
                metric = model_func(f"./experiments_new/{hyper_name}/default", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key,
                       default=True, best=False, hyper=False, **covariates_config, **baseline_parameters, seed=seed)
            except Exception as e:
                metric = 0
                print(f"Trial failed due to exception: {str(e)}")
        ax_client.complete_trial(trial_index=0, raw_data=metric)

        end_default = time.time()
        x.add_row(["default", end_default-start_default, metric])
        print(x)

    trials_data_frame = ax_client.get_trials_data_frame()
    trials_data_frame.to_csv(os.path.join(f"./experiments_new/{hyper_name}/hyper", "trials_data_frame.csv"))
    previous_runs = len(ax_client.get_trials_data_frame())
    best_optimization_metric = ax_client.get_trials_data_frame()[optimization_metric].max()

    if not not_run_hyper and tool_name.lower() not in ("wnn", "midas", "mofa", "mofa+"):

        for i in tqdm(range(50-previous_runs)):
            i=i+previous_runs-1
            os.makedirs(f"./experiments_new/{hyper_name}/hyper/hyper{i}", exist_ok=True)
            start_hyper = time.time()
            parameters, trial_index = ax_client.get_next_trial()

            try:
                if tool_name.lower() == "networkvi":
                    if "modality_penalty" in networkvi_config_settings.keys():
                        del networkvi_config_settings["modality_penalty"]
                    networkvi_config = networkvi_evaluation.get_networkvi_config(**parameters, **networkvi_config_settings)
                    if "map_ensembl_go" in networkvi_config.keys():
                        parameters["map_ensembl_go"] = networkvi_config.pop("map_ensembl_go")
                    metric = model_func(task, f"./experiments_new/{hyper_name}/hyper/hyper{i}", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network, default=False, best=False, hyper=True, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations, perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode, **covariates_config, **parameters, **networkvi_config, best_optimization_metric=best_optimization_metric, obo_file=obo_file,filter_namespace=filter_namespace,  seed=seed)
                else:
                    metric = model_func(f"./experiments_new/{hyper_name}/hyper/hyper{i}", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, default=False, best=False, hyper=True, **covariates_config, **parameters, seed=seed)
                ax_client.complete_trial(trial_index=trial_index, raw_data=metric)
            except Exception as e:
                metric = 0
                print(f"Trial failed due to exception: {str(e)}")
                ax_client.log_trial_failure(trial_index)

            end_hyper = time.time()
            x.add_row([f"hyper{i}", end_hyper - start_hyper, metric])

            trials_data_frame = ax_client.get_trials_data_frame()
            shutil.copyfile(os.path.join(f"./experiments_new/{hyper_name}/hyper", "trials_data_frame.csv"), os.path.join(f"./experiments_new/{hyper_name}/hyper", "trials_data_frame_COPY.csv"))
            trials_data_frame.to_csv(os.path.join(f"./experiments_new/{hyper_name}/hyper", "trials_data_frame.csv"))
            ax_client.save_to_json_file(os.path.join(f"./experiments_new/{hyper_name}/hyper", "ax_client.json"))

        print(x)

        best_parameters, values = ax_client.get_best_parameters()
        if tool_name.lower() == "networkvi":
            if "modality_penalty" in networkvi_config_settings.keys():
                del networkvi_config_settings["modality_penalty"]
            networkvi_config = networkvi_evaluation.get_networkvi_config(**best_parameters, **networkvi_config_settings)
            if "map_ensembl_go" in networkvi_config.keys():
                best_parameters["map_ensembl_go"] = networkvi_config.pop("map_ensembl_go")
            hyperi = np.where(ax_client.get_trials_data_frame()[optimization_metric] == ax_client.get_trials_data_frame()[optimization_metric].max())[0][0]-1
            if hyperi != -1:
                metric = model_func(task, f"./experiments_new/{hyper_name}/hyper/hyper{hyperi}", inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, mdata, mdata_counts,
                                    adata, train_indices, val_indices, n_genes, n_regions, labels_key, further_labels_keys, batch_key,
                                    patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_product_of_experts, use_mixture_of_experts, use_gating_network, default=False, best=True, hyper=False, calculate_model_interpretability=calculate_model_interpretability, keep_activations=keep_activations, perform_perturbation_analysis=perform_perturbation_analysis, calculate_covariate_attention=calculate_covariate_attention, perform_scarches=perform_scarches, perform_scarches_hao_query=perform_scarches_hao_query, scarches_modality_mode=scarches_modality_mode,
                                    **covariates_config, **best_parameters, **networkvi_config, obo_file=obo_file, filter_namespace=filter_namespace, seed=seed)

    trials_data_frame = ax_client.get_trials_data_frame()
    trials_data_frame.to_csv(os.path.join(f"./experiments_new/{hyper_name}/hyper", "trials_data_frame.csv"))
    ax_client.save_to_json_file(os.path.join(f"./experiments_new/{hyper_name}/hyper", "ax_client.json"))

