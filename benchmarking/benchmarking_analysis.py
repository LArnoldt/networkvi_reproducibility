import os
import time
import psutil
import gc
import sys
import tools
import sparsevi_evaluation
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
        }
    return {"allocated_mb": 0, "reserved_mb": 0}

def reset_gpu_memory():
    """Reset GPU memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def run_benchmarking_experiment(
        task, hyper_name, tool_name, covariates_config, inject_batch_covariates,
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
        sparsevi_config_settings, parameters, obo_file, filter_namespace, seed,
        benchmark_interpretability=True, skip_training=False
):
    '''Run a single benchmarking experiment and collect metrics'''

    metrics = {
        'experiment': hyper_name,
        'dataset': dataset_name,
        'n_cells': len(train_indices) + len(val_indices),
        'n_genes': n_genes,
        'n_regions': n_regions,
        'n_go_terms': None,
        'n_layers_encoder': parameters.get('n_layers_encoder', None),
        'standard_go_size': parameters.get('standard_go_size', None),
        'standard_gene_size': parameters.get('standard_gene_size', None),
    }

    reset_gpu_memory()

    if "modality_penalty" in sparsevi_config_settings:
        del sparsevi_config_settings["modality_penalty"]

    if "map_ensembl_go" in parameters:
        if parameters["map_ensembl_go"] == "map_ensembl_go_standard_with_rna":
            parameters["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources",
                             "ebi_ensembl_go_mappings",
                             "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources",
                             "ebi_ensembl_go_mappings",
                             "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources",
                             "ebi_ensembl_go_mappings",
                             "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf")
            ]

    sparsevi_config = sparsevi_evaluation.get_sparsevi_config(
        **parameters, **sparsevi_config_settings
    )
    if "map_ensembl_go" in sparsevi_config:
        del sparsevi_config['map_ensembl_go']

    hyper_name
    os.makedirs(hyper_name, exist_ok=True)

    process = psutil.Process(os.getpid())

    if not skip_training:
        print(f"\n{'=' * 60}")
        print(f"Training Benchmark: {hyper_name}")
        print(f"{'=' * 60}")
        training_start_time = time.time()
        metric = tools.sparsevi(
            task, hyper_name, inject_batch_covariates, inject_patient_covariates,
            inject_further_batch_covariates, inject_further_continuous_batch_covariates,
            incorporate_batch_scib, incorporate_patient_scib, optimization_metric,
            dataset_name, subsampled, paired_rate, utilize_highly_variable,
            utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata,
            train_indices, val_indices, n_genes, n_regions, n_snps, labels_key,
            further_labels_keys, batch_key, patient_key, further_batch_keys,
            further_continuous_batch_keys, n_patient_covariates,
            protein_expression_obsm_key, use_mean_mixing, use_transformer,
            use_transformer_latents, use_product_of_experts, use_mixture_of_experts,
            use_rectified_flows, use_gating_network,
            default=True, best=False, hyper=False,
            calculate_model_interpretability=False,
            keep_activations=False,
            perform_perturbation_analysis=False,
            calculate_covariate_attention=False,
            perform_scarches=perform_scarches,
            perform_scarches_hao_query=perform_scarches_hao_query,
            scarches_modality_mode=scarches_modality_mode,
            **covariates_config, **parameters, **sparsevi_config,
            best_optimization_metric=0, obo_file=obo_file,
            filter_namespace=filter_namespace, seed=seed, perform_scib_calculation=False, perform_umap_plots=False, perform_model_downstream=False
        )

        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        gpu_memory_training = get_gpu_memory_usage()

        cpu_memory_training = process.memory_info().rss / 1024 ** 2

        metrics['training_time_seconds'] = training_duration
        metrics['training_time_hours'] = training_duration / 3600
        metrics['gpu_memory_training_allocated_mb'] = gpu_memory_training["allocated_mb"]
        metrics['gpu_memory_training_reserved_mb'] = gpu_memory_training["reserved_mb"]
        metrics['cpu_memory_training_mb'] = cpu_memory_training
        metrics['scib_score'] = metric

        print(f"\nTraining completed in {training_duration:.2f} seconds "
              f"({training_duration / 3600:.2f} hours)")
        print(f"GPU Memory (allocated): {gpu_memory_training['allocated_mb']:.2f} MB")
        print(f"GPU Memory (reserved): {gpu_memory_training['reserved_mb']:.2f} MB")
        print(f"CPU Memory: {cpu_memory_training:.2f} MB")

    if benchmark_interpretability:
        print(f"\n{'=' * 60}")
        print(f"Interpretability Benchmark: {hyper_name}")
        print(f"{'=' * 60}")

        sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/"
                        "development/arnoldtl/code/scvi-tools-original/src/")
        import scvi

        import scanpy as sc
        import anndata as ad

        adata_mvi = sc.concat([mdata_counts.mod[mod] for mod in mdata_counts.mod
                               if mod != "prot"], join="outer", merge="first", axis=1)
        adata_mvi.X = np.array(adata_mvi.X)
        if "prot" in mdata_counts.mod:
            adata_mvi.obsm["protein_expression"] = mdata_counts.mod["prot"].X
            adata_mvi.uns['protein_expression'] = {
                'var': mdata_counts.mod["prot"].var,
                'obs': mdata_counts.mod["prot"].obs
            }

        categorical_covariate_keys = (
            ([batch_key] if inject_batch_covariates else []) +
            ([patient_key] if inject_patient_covariates else []) +
            (further_batch_keys if inject_further_batch_covariates else [])
            if inject_batch_covariates or inject_patient_covariates or
               inject_further_batch_covariates else None
        )
        continuous_covariate_keys = (
            further_continuous_batch_keys if
            inject_further_continuous_batch_covariates else None
        )

        scvi.model.SPARSEVI.setup_anndata(
            adata_mvi,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            patient_key=patient_key,
            protein_expression_obsm_key=protein_expression_obsm_key
        )

        model = scvi.model.SPARSEVI.load(hyper_name, adata=adata_mvi)

        if hasattr(model.module, 'n_go_terms'):
            metrics['n_go_terms'] = model.module.n_go_terms

        BENCHMARK_PHENOTYPE = 'CD8-positive, alpha-beta T cell'
        BENCHMARK_COMPARISON = 'CD4-positive, alpha-beta T cell'
        from collections import Counter
        label_counts = Counter(adata_mvi.obs[labels_key])
        n_phenotype = label_counts[BENCHMARK_PHENOTYPE]
        n_comparison = label_counts[BENCHMARK_COMPARISON]

        print(f"\nComparison cell counts:")
        print(f"  '{BENCHMARK_PHENOTYPE}': {n_phenotype} cells")
        print(f"  '{BENCHMARK_COMPARISON}': {n_comparison} cells")

        reset_gpu_memory()
        interp_start_time = time.time()

        model.get_go_gene_importances(
            labels_column=labels_key,
            results_dir=f"{hyper_name}/benchmark_go_importance",
            shuffle_set_split=False,
            batch_size=512,
            train_idx=train_indices,
            validation_idx=val_indices,
            calc_go_terms=True,
            calc_gene_groups=False,
            labels_selection=BENCHMARK_PHENOTYPE,
            comparison=BENCHMARK_COMPARISON,
        )

        interp_end_time = time.time()
        interp_duration = interp_end_time - interp_start_time
        gpu_memory_interp = get_gpu_memory_usage()
        cpu_memory_interp = process.memory_info().rss / 1024 ** 2

        metrics['interpretability_go_importance_time_seconds'] = interp_duration
        metrics['interpretability_go_importance_time_hours'] = interp_duration / 3600
        metrics['gpu_memory_go_importance_allocated_mb'] = gpu_memory_interp["allocated_mb"]
        metrics['gpu_memory_go_importance_reserved_mb'] = gpu_memory_interp["reserved_mb"]
        metrics['cpu_memory_go_importance_mb'] = cpu_memory_interp

        print(f"\nGO Importance completed in {interp_duration:.2f}s ({interp_duration / 3600:.2f}h)")
        print(f"GPU Memory (allocated): {gpu_memory_interp['allocated_mb']:.2f} MB")
        print(f"GPU Memory (reserved): {gpu_memory_interp['reserved_mb']:.2f} MB")
        print(f"CPU Memory: {cpu_memory_interp:.2f} MB")

        pd.DataFrame([metrics]).to_csv(f"{hyper_name}/benchmark_metrics_partial.csv", index=False)

        try:
            reset_gpu_memory()
            attn_start_time = time.time()

            model.calculate_covariate_attention(
                labels_column=labels_key,
                results_dir=f"{hyper_name}/benchmark_go_importance",
                shuffle_set_split=False,
                batch_size=512,
                train_idx=train_indices,
                validation_idx=val_indices,
                modality_categorical_covariate_keys=(
                                                        ["modality"] if paired_rate not in (1, 1.0) else []
                                                    ) + categorical_covariate_keys,
                continuous_covariate_keys=[],
                labels_selection=BENCHMARK_PHENOTYPE,
            )

            attn_end_time = time.time()
            attn_duration = attn_end_time - attn_start_time
            gpu_memory_attn = get_gpu_memory_usage()
            cpu_memory_attn = process.memory_info().rss / 1024 ** 2

            metrics['interpretability_time_seconds'] = attn_duration
            metrics['interpretability_time_hours'] = attn_duration / 3600
            metrics['gpu_memory_covariate_attention_allocated_mb'] = gpu_memory_attn["allocated_mb"]
            metrics['gpu_memory_covariate_attention_reserved_mb'] = gpu_memory_attn["reserved_mb"]
            metrics['cpu_memory_covariate_attention_mb'] = cpu_memory_attn

            print(f"\nCovariate Attention completed in {attn_duration:.2f}s ({attn_duration / 3600:.2f}h)")
            print(f"GPU Memory (allocated): {gpu_memory_attn['allocated_mb']:.2f} MB")
            print(f"GPU Memory (reserved): {gpu_memory_attn['reserved_mb']:.2f} MB")
            print(f"CPU Memory: {cpu_memory_attn:.2f} MB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"WARNING: Covariate attention OOM — skipping. Error: {e}")
            metrics['interpretability_time_seconds'] = None
            metrics['gpu_memory_covariate_attention_allocated_mb'] = None

        import shutil
        if os.path.exists(f"{hyper_name}/benchmark_go_importance"):
            shutil.rmtree(f"{hyper_name}/benchmark_go_importance")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{hyper_name}/benchmark_metrics.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"Benchmark Summary")
    print(f"{'=' * 60}")
    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value}")

    return metrics


def run_scaling_analysis(
        base_hyper_name, tool_name, covariates_config, inject_batch_covariates,
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
        sparsevi_config_settings, default_option=None, obo_file=None,
        filter_namespace=True, seed=None, skip_training=False
):
    '''Run comprehensive scaling analysis varying:'''

    is_go_model = "go" in sparsevi_config_settings.get('model_type', '')
    is_gene_model = "gene" in sparsevi_config_settings.get('model_type', '')

    if is_go_model:
        baseline_parameters = {
            "lr": 0.0001,
            "modality_penalty": "MMD",
            "gene_likelihood": "zinb",
            "n_layers_encoder": 4,
            "n_layers_decoder": 2,
            "n_hidden": 384,
            "dropout_rate": 0.3,
            "activation_fn": "swish",
            "standard_go_size": 2,
            "map_ensembl_go": "map_ensembl_go_standard_with_rna"
        }
    else:
        baseline_parameters = {
            "lr": 0.0001,
            "modality_penalty": "Jeffreys",
            "gene_likelihood": "zinb",
            "n_layers_encoder": 2,
            "n_layers_decoder": 2,
            "n_hidden": -1,
            "dropout_rate": 0.1,
            "activation_fn": "relu",
            "standard_gene_size": 5,
            "map_ensembl_go": "map_ensembl_go_standard_with_rna"
        }

    parameter_variations = {
        'n_layers_encoder': [2, 3, 4, 5],
    }

    if is_go_model:
        parameter_variations['standard_go_size'] = [2, 3, 4, 5]

    if is_gene_model:
        parameter_variations['standard_gene_size'] = [2, 3, 4, 5]

    results = []

    print(f"\n{'#' * 80}")
    print(f"Running BASELINE experiment")
    print(f"{'#' * 80}\n")

    base_hyper_name = base_hyper_name + "_ml_benchmark"
    baseline_hyper_name = f"{base_hyper_name}/baseline"

    if not os.path.exists(f"{baseline_hyper_name}/benchmark_metrics.csv"):
        baseline_metrics = run_benchmarking_experiment(
            "benchmarking_analysis", baseline_hyper_name, tool_name, covariates_config,
            inject_batch_covariates, inject_patient_covariates,
            inject_further_batch_covariates, inject_further_continuous_batch_covariates,
            incorporate_batch_scib, incorporate_patient_scib, optimization_metric,
            dataset_name, subsampled, paired_rate, utilize_highly_variable,
            utilize_highly_abundant, utilize_de, spike_in, mdata, mdata_counts, adata,
            train_indices, val_indices, n_genes, n_regions, n_snps, labels_key,
            further_labels_keys, batch_key, patient_key, further_batch_keys,
            further_continuous_batch_keys, n_patient_covariates,
            protein_expression_obsm_key, use_mean_mixing, use_transformer,
            use_transformer_latents, use_product_of_experts, use_mixture_of_experts,
            use_rectified_flows, use_gating_network, calculate_model_interpretability,
            keep_activations, perform_perturbation_analysis, calculate_covariate_attention,
            perform_scarches, perform_scarches_hao_query, scarches_modality_mode,
            sparsevi_config_settings, baseline_parameters, obo_file, filter_namespace,
            seed, benchmark_interpretability=True, skip_training=skip_training
        )
        results.append(baseline_metrics)
    else:
        print(f"Baseline already exists, skipping...")
        baseline_metrics = pd.read_csv(
            f"{baseline_hyper_name}/benchmark_metrics.csv"
        ).iloc[0].to_dict()
        results.append(baseline_metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{base_hyper_name}/scaling_analysis_results.csv", index=False)

    print(f"\n{'=' * 80}")
    print(f"SCALING ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")
    print(results_df.to_string())


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    summary_stats = {k: (
        v if v is None else (int(v) if isinstance(v, np.integer) else (float(v) if isinstance(v, np.floating) else v)))
                     for k, v in baseline_metrics.items()}
    summary_stats['dataset'] = dataset_name
    summary_stats['total_cells'] = int(len(train_indices) + len(val_indices))

    with open(f"{base_hyper_name}/summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=4, cls=NumpyEncoder)

    return results_df