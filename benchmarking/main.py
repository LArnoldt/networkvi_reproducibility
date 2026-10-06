import numpy as np
import scanpy as sc
import os
import sys
from pathlib import Path
import subprocess
import pandas as pd
import pickle
import os
import muon as mu
from tqdm import tqdm
import argparse

import datasets

def get_covariates_config(tool_name, midas_encode_covariates=None, multimil_condition_encoders=None, mofa_encode_covariates=None, totalvi_encode_covariates=None, sparsevi_inject_covariates_mode=None):

    if tool_name.lower() == "midas" and (mofa_encode_covariates is True or totalvi_encode_covariates is True or multimil_condition_encoders is True or sparsevi_inject_covariates_mode is not None):
        raise NotImplementedError("")
    if tool_name.lower() == "multimil" and (mofa_encode_covariates is True or totalvi_encode_covariates is True or midas_encode_covariates is True or sparsevi_inject_covariates_mode is not None):
        raise NotImplementedError("")
    if tool_name.lower() == "totalvi" and (mofa_encode_covariates is True or midas_encode_covariates is True or multimil_condition_encoders is True or sparsevi_inject_covariates_mode is not None):
        raise NotImplementedError("")
    if (tool_name.lower() == "mofa" or tool_name.lower() == "mofa+") and (totalvi_encode_covariates is True or midas_encode_covariates is True or multimil_condition_encoders is True or sparsevi_inject_covariates_mode is not None):
        raise NotImplementedError("")
    if tool_name.lower() == "sparsevi" and (mofa_encode_covariates is True or totalvi_encode_covariates is True or midas_encode_covariates is True or multimil_condition_encoders is True ):
        raise NotImplementedError("")

    covariates_config = {}

    if tool_name.lower() == "multimil":
        if multimil_condition_encoders is True:
            covariates_config["condition_encoders"] = True
        else:
            covariates_config["condition_encoders"] = False
    if tool_name.lower() == "midas":
        if midas_encode_covariates is True:
            covariates_config["encode_covariates"] = True
        else:
            covariates_config["encode_covariates"] = False
    elif tool_name.lower() == "mofa" or tool_name.lower() == "mofa+":
        if mofa_encode_covariates is True:
            covariates_config["encode_covariates"] = True
        else:
            covariates_config["encode_covariates"] = False
    elif tool_name.lower() == "totalvi":
        if totalvi_encode_covariates is True:
            covariates_config["encode_covariates"] = True
        else:
            covariates_config["encode_covariates"] = False
    elif tool_name.lower() == "sparsevi":
        if sparsevi_inject_covariates_mode.lower() == "encode_covariates":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = False
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = False
            covariates_config["unfreeze_first_layers"] = False
        elif sparsevi_inject_covariates_mode.lower() == "encode_covariates_unfreeze_first_layers":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = False
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = False
            covariates_config["unfreeze_first_layers"] = True
        elif sparsevi_inject_covariates_mode.lower() == "deeply_inject_covariates":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = True
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = True
        elif sparsevi_inject_covariates_mode.lower() == "deeply_inject_covariates_encoder_only":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = True
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = False
        elif sparsevi_inject_covariates_mode.lower() == "deeply_inject_covariates_encoder_only_unfreeze_first_layers":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = True
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = False
            covariates_config["unfreeze_first_layers"] = True
        elif sparsevi_inject_covariates_mode.lower() == "first_layer_inject_covariates":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = False
            covariates_config["first_layer_inject_covariates"] = True
            covariates_config["last_layer_inject_covariates"] = False
            covariates_config["decoder_deeply_inject_covariates"] = False
            covariates_config["unfreeze_first_layers"] = False
        elif sparsevi_inject_covariates_mode.lower() == "last_layer_inject_covariates":
            covariates_config["encode_covariates"] = True
            covariates_config["deeply_inject_covariates"] = False
            covariates_config["first_layer_inject_covariates"] = False
            covariates_config["last_layer_inject_covariates"] = True
            covariates_config["decoder_deeply_inject_covariates"] = False
            covariates_config["unfreeze_first_layers"] = False
        else:
            covariates_config["encode_covariates"] = False

    return covariates_config

if __name__ == "__main__":

    file_path_parser = argparse.ArgumentParser(description="args")
    file_path_parser.add_argument("--task", type=str, default="hyper", help="task")
    file_path_parser.add_argument("--chunk", type=int, help="chunk")
    file_path_parser.add_argument("--not_run_hyper", action='store_true', help="not run hyper")
    file_path_parser.add_argument("--dataset_name", type=str, help="dataset name")
    file_path_parser.add_argument("--subsampled", type=float, default=None, help="susbsampled")
    file_path_parser.add_argument("--enrichment_method", type=str, default="GSEA", help="enrichment method (GSEA or ORA), used by --task=enrichment_analysis")
    file_path_parser.add_argument("--tool_name", type=str, help="tool name")
    file_path_parser.add_argument("--paired_rate", type=float, help="paired rate")
    file_path_parser.add_argument("--optimization_metric", type=str, help="optimization metric")
    file_path_parser.add_argument("--utilize_highly_variable", action='store_true', help="utilize highly variable")
    file_path_parser.add_argument("--utilize_highly_abundant", action='store_true', help="utilize highly abundant")
    file_path_parser.add_argument("--utilize_de", action='store_true', help="utilize DE genes")
    file_path_parser.add_argument("--spike_in", nargs='+', type=str, help="Spike In")
    file_path_parser.add_argument("--calculate_model_interpretability", action='store_true', help="calculate model interpretability")
    file_path_parser.add_argument("--keep_activations", action='store_true', help="keep activations")
    file_path_parser.add_argument("--perform_perturbation_analysis", action='store_true', help="perform perturbation analysis")
    file_path_parser.add_argument("--calculate_covariate_attention", action='store_true', help="calculate covariate attention")
    file_path_parser.add_argument("--perform_scarches", action='store_true', help="perform scarches")
    file_path_parser.add_argument("--perform_scarches_hao_query", action='store_true', help="perform scarches")
    file_path_parser.add_argument("--scarches_modality_mode", nargs='+', type=str, help="scarches_modality_mode")
    file_path_parser.add_argument("--model_type", type=str, help="model type")
    file_path_parser.add_argument("--gene_interaction_layer_dynamic", help="gene layer interaction layer dynamic", action='store_true')
    file_path_parser.add_argument("--gene_layer_interaction_source", type=str, help="gene layer interaction source")
    file_path_parser.add_argument("--gene_interaction_layer_nlayers", type=int, help="gene layer interaction layer nlayers", default=1)
    file_path_parser.add_argument("--modality_penalty", type=str, help="modality penalty")
    file_path_parser.add_argument("--gaf_source", type=str, help="gaf source")
    file_path_parser.add_argument("--midas_encode_covariates", action='store_true', help="midas encode covariates")
    file_path_parser.add_argument("--multimil_condition_encoders", action='store_true', help="multimil encode covariates")
    file_path_parser.add_argument("--mofa_encode_covariates", action='store_true', help="mofa encode covariates")
    file_path_parser.add_argument("--totalvi_encode_covariates", action='store_true', help="totalvi encode covariates")
    file_path_parser.add_argument("--sparsevi_inject_covariates_mode", type=str, help="sparsevi inject covariates mode")
    file_path_parser.add_argument("--inject_batch_covariates", action='store_true', help="inject batch covariates")
    file_path_parser.add_argument("--inject_patient_covariates", action='store_true', help="inject patient covariates")
    file_path_parser.add_argument("--inject_further_batch_covariates", action='store_true', help="inject further batch covariates")
    file_path_parser.add_argument("--inject_further_continuous_batch_covariates", action='store_true', help="inject further continuous batch covariates")
    file_path_parser.add_argument("--incorporate_batch_scib", action='store_true', help="incorporate batch scib")
    file_path_parser.add_argument("--incorporate_patient_scib", action='store_true', help="incorporate patient scib")
    file_path_parser.add_argument("--use_mean_mixing", action='store_true', help="use mean mixing")
    file_path_parser.add_argument("--use_product_of_experts", action='store_true', help="use product of experts")
    file_path_parser.add_argument("--use_mixture_of_experts", action='store_false', help="use mixture of experts")
    file_path_parser.add_argument("--use_gating_network", action='store_false', help="use gating network")
    file_path_parser.add_argument("--default_option", type=int, help="default option")
    file_path_parser.add_argument("--repeat_option", type=int, help="repeat option")
    file_path_parser.add_argument("--load_previous_ax_client", action='store_true', help="load previous ax client")
    file_path_parser.add_argument("--obo_file", default=None, type=str, help="obo_file")
    file_path_parser.add_argument("--filter_namespace", default=True, type=str, help="filter_namespace")
    file_path_parser.add_argument("--seed", type=int, help="seed")
    file_path_parser.add_argument("--different_seeds", nargs='+', type=int, help="different seeds")
    file_path_parser.add_argument("--n_different_seeds", type=int, default=20, help="n different seeds")
    file_path_parser.add_argument("--model_parameters_json", type=str, help="model_parameters_json")
    file_path_parser.add_argument("--ablation_group", type=str, default=None, help="ablation group: depth | go_size | gene_size")
    file_path_parser.add_argument("--ablation_value", type=int, default=None, help="the value to test for the ablation group")
    file_path_parser.add_argument("--model_path", type=str, help="path to existing model to reload for interpretability")
    file_path_parser.add_argument("--benchmarking_analysis_skip_training",  action='store_true',  help="benchmarking_analysis_skip_training")

    args = file_path_parser.parse_args()

    task = args.task
    chunk = args.chunk
    not_run_hyper = args.not_run_hyper
    dataset_name = args.dataset_name
    subsampled = args.subsampled
    enrichment_method = args.enrichment_method
    tool_name = args.tool_name
    paired_rate = args.paired_rate
    optimization_metric = args.optimization_metric
    utilize_highly_variable = args.utilize_highly_variable
    utilize_highly_abundant = args.utilize_highly_abundant
    utilize_de = args.utilize_de
    spike_in = args.spike_in

    calculate_model_interpretability = args.calculate_model_interpretability
    keep_activations = args.keep_activations
    perform_perturbation_analysis = args.perform_perturbation_analysis
    calculate_covariate_attention = args.calculate_covariate_attention
    perform_scarches = args.perform_scarches
    perform_scarches_hao_query = args.perform_scarches_hao_query
    scarches_modality_mode = args.scarches_modality_mode

    model_type = args.model_type
    gene_interaction_layer_dynamic = args.gene_interaction_layer_dynamic
    gene_layer_interaction_source = args.gene_layer_interaction_source
    gene_interaction_layer_nlayers = args.gene_interaction_layer_nlayers
    modality_penalty = args.modality_penalty
    gaf_source = args.gaf_source

    midas_encode_covariates = args.midas_encode_covariates
    multimil_condition_encoders = args.multimil_condition_encoders
    mofa_encode_covariates = args.mofa_encode_covariates
    totalvi_encode_covariates = args.totalvi_encode_covariates
    sparsevi_inject_covariates_mode = args.sparsevi_inject_covariates_mode
    inject_batch_covariates = args.inject_batch_covariates
    inject_patient_covariates = args.inject_patient_covariates
    inject_further_batch_covariates = args.inject_further_batch_covariates
    inject_further_continuous_batch_covariates = args.inject_further_continuous_batch_covariates
    incorporate_batch_scib = args.incorporate_batch_scib
    incorporate_patient_scib = args.incorporate_patient_scib
    use_mean_mixing = args.use_mean_mixing
    use_transformer = False
    use_transformer_latents = False
    use_product_of_experts = args.use_product_of_experts
    use_mixture_of_experts = args.use_mixture_of_experts
    use_rectified_flows = False
    use_gating_network = args.use_gating_network

    default_option = args.default_option
    repeat_option = args.repeat_option

    load_previous_ax_client = args.load_previous_ax_client

    obo_file = args.obo_file
    filter_namespace = args.filter_namespace
    if filter_namespace == "True":
        filter_namespace = True
    elif filter_namespace == "False":
        filter_namespace = False
    if filter_namespace is not True and filter_namespace is not False and filter_namespace is not None:
        filter_namespace = filter_namespace.split(",")

    seed = args.seed
    different_seeds = args.different_seeds
    n_different_seeds = args.n_different_seeds
    model_parameters_json = args.model_parameters_json

    ablation_group = args.ablation_group
    ablation_value = args.ablation_value

    model_path = args.model_path

    benchmarking_analysis_skip_training = args.benchmarking_analysis_skip_training

    os.makedirs("experiments_new", exist_ok=True)
    os.makedirs("./scvi_log/lightning_logs/version_1", exist_ok=True)
    os.makedirs("./scvi_log/lightning_logs/version_1/checkpoints", exist_ok=True)
    
    
    covariates_config_settings = {
        "midas_encode_covariates": midas_encode_covariates,
        "multimil_condition_encoders": multimil_condition_encoders,
        "mofa_encode_covariates": mofa_encode_covariates,
        "totalvi_encode_covariates": totalvi_encode_covariates,
        "sparsevi_inject_covariates_mode": sparsevi_inject_covariates_mode,
    }

    covariates_config = get_covariates_config(tool_name, **covariates_config_settings)


    sparsevi_config_settings = {
        "model_type": model_type,
        "gene_interaction_layer_dynamic": gene_interaction_layer_dynamic,
        "gene_layer_interaction_source": gene_layer_interaction_source,
        "gene_interaction_layer_nlayers": gene_interaction_layer_nlayers,
        "modality_penalty": modality_penalty,
        "gaf_source": gaf_source,
    }
    
    
    hyper_name = f"{tool_name}_{dataset_name}_{paired_rate}_{optimization_metric}_hvar_{utilize_highly_variable}_batscib_{incorporate_batch_scib}_patscib_{incorporate_patient_scib}"
    if subsampled:
        hyper_name += f"_subsam_{subsampled}"
    if utilize_highly_abundant:
        hyper_name += f"_abund_{utilize_highly_abundant}"
    if utilize_de:
        hyper_name += f"_de_{utilize_de}"
    if spike_in:
        hyper_name += f"_{'_'.join(spike_in)}"
    if default_option is not None:
        hyper_name += f"_d{default_option}"
    if repeat_option is not None:
        hyper_name += f"_r{repeat_option}"

    if use_mean_mixing is True:
        hyper_name += f"_mmix{use_mean_mixing}"
    if use_product_of_experts is True:
        hyper_name += f"_poe{use_product_of_experts}"
    if use_mixture_of_experts is True:
        hyper_name += f"_moe{use_mixture_of_experts}"
    if use_gating_network is not True:
        hyper_name += f"_no_gating"

    if (dataset_name.lower() == "neurips2021_cite_bmmc" or dataset_name.lower() == "neurips2021_multiome_bmmc"):
        if perform_scarches_hao_query:
            hyper_name += f"_scarches_hao_query_{perform_scarches_hao_query}"
        else:
            hyper_name +=  f"_scarches_{perform_scarches}"
        if scarches_modality_mode is not None:
            hyper_name += f"_mode_{'_'.join(scarches_modality_mode)}"
    if tool_name.lower() == "midas" or tool_name.lower() == "multimil" or tool_name.lower() == "totalvi" or tool_name.lower() == "sparsevi":
        for key, element in covariates_config.items():
            key_mod = ''.join([attr[0] for attr in key.split('_')])
            hyper_name += f"_{key_mod}_{element}" if element is True else ""
        hyper_name += f"_batcov_{inject_batch_covariates}_patcov_{inject_patient_covariates}"
        if inject_further_batch_covariates:
            hyper_name += f"_fbatcov"
        if inject_further_continuous_batch_covariates:
            hyper_name += f"_fcbatcov"
    if tool_name.lower() == "sparsevi":
        if sparsevi_config_settings['model_type'] == "go_with_interaction_gene_layer":
            hyper_name += f"_modtype_go"
        else:
            hyper_name += f"_modtype_{sparsevi_config_settings['model_type']}"
        if sparsevi_config_settings['gene_interaction_layer_dynamic']:
            hyper_name += f"_gint_dynamic"
        if sparsevi_config_settings['gene_layer_interaction_source'] is not None:
            hyper_name += f"_genelayerint_{sparsevi_config_settings['gene_layer_interaction_source']}"
        if sparsevi_config_settings["gene_interaction_layer_nlayers"] != 1:
            hyper_name += f"_nlay_{sparsevi_config_settings['gene_interaction_layer_nlayers']}"
        if sparsevi_config_settings['modality_penalty'] is not None:
            hyper_name += f"_modpen_{sparsevi_config_settings['modality_penalty']}"
        if sparsevi_config_settings['gaf_source'] is not None:
            if "/" in sparsevi_config_settings['gaf_source']:
                hyper_name += f"_gaf_{Path(sparsevi_config_settings['gaf_source']).stem}"
            else:
                hyper_name += f"_gaf_{sparsevi_config_settings['gaf_source']}"
    if filter_namespace is not True and filter_namespace is not None and filter_namespace != "biological_process":
        if filter_namespace is False:
            hyper_name += f"_filtont_False"
        else:
            hyper_name += f"_filtont_{'_'.join(filter_namespace)}"
    if seed is not None:
        hyper_name += f"_s{seed}"
        

    if dataset_name.lower() == "dogma":
        mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_dogma(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
    elif dataset_name.lower() == "hlca":
        mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_hlca(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
    elif dataset_name.lower() == "hao2021":
        mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_hao2021(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
    elif dataset_name.lower() == "frangieh2021":
        mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_frangieh2021(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
    elif dataset_name.lower() == "neurips2021_cite_bmmc":
        if not perform_scarches:
            mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_neurips2021_cite_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
        else:
            mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_neurips2021_cite_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in, query_ref_selector=["site2", "site3", "site4"])
    elif dataset_name.lower() == "neurips2021_multiome_bmmc":
        if not perform_scarches:
            mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_neurips2021_multiome_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
        else:
            mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_neurips2021_multiome_BMMC(paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in, query_ref_selector=["site2", "site3", "site4"])
    elif "ts" in dataset_name.lower():
        mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key = datasets.load_tabula_sapiens(
            dataset_name, paired_rate, subsampled, utilize_highly_variable, spike_in=spike_in)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not available.")

    if spike_in:
        spike_in = '_'.join(spike_in)

    if (tool_name.lower() == "wnn" or tool_name.lower() == "mofa" or tool_name.lower() == "mofa+" or tool_name.lower() == "midas") and optimization_metric.lower() != "scib":
         raise NotImplementedError("")
    if tool_name.lower() == "totalvi" and (dataset_name.lower() == "dogma" or dataset_name.lower() == "neurips2021_multiome_BMMC"):
        raise NotImplementedError("")
    if tool_name.lower() == "wnn" and (paired_rate != 1 and paired_rate != 1.0):
        raise NotImplementedError("")
    if perform_scarches and (dataset_name.lower() != "neurips2021_cite_bmmc" and dataset_name.lower() != "neurips2021_multiome_bmmc") and tool_name.lower != "sparsevi":
        raise NotImplementedError("")
    if perform_scarches_hao_query and dataset_name.lower() != "neurips2021_cite_bmmc" and tool_name.lower != "sparsevi":
        raise NotImplementedError("")
    if tool_name.lower() != "sparsevi" and (calculate_model_interpretability or keep_activations or perform_perturbation_analysis or calculate_covariate_attention):
        raise NotImplementedError("")

    if task == "go_gene_stability":
        import go_gene_stability
        go_gene_stability.main(task, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, sparsevi_config_settings, default_option, seed=seed)
    elif task == "run_tool_parameters_different_seeds":
        import run_tool_parameters_different_seeds
        if different_seeds is None:
            different_seeds = [np.random.randint(0, 2 ** 32 - 1) for _ in range(n_different_seeds)]
        print(different_seeds)
        run_tool_parameters_different_seeds.main(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option, different_seeds=different_seeds, model_parameters_json=model_parameters_json)
    elif "spike_in" in task or "sparsevi_interpretability_experiments" in task or "sparsevi_att_interpretability_experiments" in task:
        import sparsevi_interpretability_experiments
        sparsevi_interpretability_experiments.main(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option, load_previous_ax_client=load_previous_ax_client, seed=seed)
    elif task == "hyper":
        import tools_hyper
        tools_hyper.hyper_wrapper(task, not_run_hyper, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option, load_previous_ax_client=load_previous_ax_client, obo_file=obo_file, filter_namespace=filter_namespace, seed=seed)
    elif task == "enrichment_analysis":
        import enrichment_analysis
        enrichment_analysis.main(task, chunk, not_run_hyper, hyper_name, tool_name, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, enrichment_method=enrichment_method)
    elif task == "single_hyper_variations":
        import single_hyper_variations
        single_hyper_variations.run_single_hyperparameter_variations(task, hyper_name, tool_name, covariates_config, inject_batch_covariates, inject_patient_covariates, inject_further_batch_covariates, inject_further_continuous_batch_covariates, incorporate_batch_scib, incorporate_patient_scib, optimization_metric, dataset_name, subsampled, paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata, train_indices, val_indices, n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key, patient_key, further_batch_keys, further_continuous_batch_keys, n_patient_covariates, protein_expression_obsm_key, use_mean_mixing, use_transformer, use_transformer_latents, use_product_of_experts, use_mixture_of_experts, use_rectified_flows, use_gating_network, calculate_model_interpretability, keep_activations, perform_perturbation_analysis, calculate_covariate_attention, perform_scarches, perform_scarches_hao_query, scarches_modality_mode, sparsevi_config_settings, default_option=default_option, obo_file=obo_file, filter_namespace=filter_namespace, seed=seed, ablation_group=ablation_group, ablation_value=ablation_value)
    elif task == "benchmarking_analysis":
        import benchmarking_analysis
        benchmarking_analysis.run_scaling_analysis(
            hyper_name, tool_name, covariates_config, inject_batch_covariates,
            inject_patient_covariates, inject_further_batch_covariates,
            inject_further_continuous_batch_covariates, incorporate_batch_scib,
            incorporate_patient_scib, optimization_metric, dataset_name, subsampled,
            paired_rate, utilize_highly_variable, utilize_highly_abundant, utilize_de,
            spike_in, mdata, mdata_counts, adata, train_indices, val_indices,
            n_genes, n_regions, n_snps, labels_key, further_labels_keys, batch_key,
            patient_key, further_batch_keys, further_continuous_batch_keys,
            n_patient_covariates, protein_expression_obsm_key, use_mean_mixing,
            use_transformer, use_transformer_latents, use_product_of_experts,
            use_mixture_of_experts, use_rectified_flows, use_gating_network,
            calculate_model_interpretability, keep_activations,
            perform_perturbation_analysis, calculate_covariate_attention,
            perform_scarches, perform_scarches_hao_query, scarches_modality_mode,
            sparsevi_config_settings, default_option, obo_file, filter_namespace, seed, skip_training=benchmarking_analysis_skip_training
        )

    elif task == "sparsevi_reload_interpretability":
        import sparsevi_interpretability_experiments
        if model_path is None:
            raise ValueError("--model_path must be provided for task 'sparsevi_reload_interpretability'.")
        sparsevi_interpretability_experiments.main(
            task, model_path, tool_name,
            covariates_config={}, inject_batch_covariates=None,
            inject_patient_covariates=None, inject_further_batch_covariates=None,
            inject_further_continuous_batch_covariates=None,
            incorporate_batch_scib=None, incorporate_patient_scib=None,
            optimization_metric="scib",
            dataset_name=dataset_name, subsampled=subsampled,
            paired_rate=paired_rate, utilize_highly_variable=utilize_highly_variable,
            utilize_highly_abundant=utilize_highly_abundant, utilize_de=utilize_de,
            spike_in=spike_in,
            mdata=mdata, mdata_counts=mdata_counts, adata=adata,
            train_indices=train_indices, val_indices=val_indices,
            n_genes=n_genes, n_regions=n_regions, n_snps=n_snps,
            labels_key=labels_key, further_labels_keys=further_labels_keys,
            batch_key=batch_key, patient_key=patient_key,
            further_batch_keys=further_batch_keys,
            further_continuous_batch_keys=further_continuous_batch_keys,
            n_patient_covariates=n_patient_covariates,
            protein_expression_obsm_key=protein_expression_obsm_key,
            use_mean_mixing=None, use_transformer=None,
            use_transformer_latents=None, use_product_of_experts=None,
            use_mixture_of_experts=None, use_rectified_flows=None,
            use_gating_network=None,
            calculate_model_interpretability=calculate_model_interpretability,
            keep_activations=keep_activations,
            perform_perturbation_analysis=perform_perturbation_analysis,
            calculate_covariate_attention=calculate_covariate_attention,
            perform_scarches=perform_scarches,
            perform_scarches_hao_query=perform_scarches_hao_query,
            scarches_modality_mode=scarches_modality_mode,
            sparsevi_config_settings=sparsevi_config_settings,
            default_option=default_option,
            load_previous_ax_client=False,
            seed=seed,
            model_path=model_path
        )
    elif task == "grn":
        import grn_pipeline
        grn_results = grn_pipeline.run_grn_pipeline(
            mdata_counts,
            outdir=hyper_name,
            skip_qc=True,
            n_hvg=4000,
            random_state=seed,
        )
    elif task == "grn_analysis":
        import grn_analysis
        grn_analysis_results = grn_analysis.run_analysis(
            adata=mdata.mod["rna"],
            cell_type_col=labels_key,
            outdir=hyper_name,
            seed=seed if seed is not None else 42,
            sif_path="./pyscenic_0.12.1.sif",
        )
    else:
        raise NotImplementedError("Argument task must be either 'go_gene_stability', 'run_tool_parameters_different_seeds' or 'hyper'.")

