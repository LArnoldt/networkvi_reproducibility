import os
import time
import pandas as pd
import tools
import sparsevi_evaluation


def run_single_hyperparameter_variations(
        task, base_hyper_name, tool_name, covariates_config,
        inject_batch_covariates, inject_patient_covariates,
        inject_further_batch_covariates, inject_further_continuous_batch_covariates,
        incorporate_batch_scib, incorporate_patient_scib,
        optimization_metric, dataset_name, subsampled, paired_rate,
        utilize_highly_variable, utilize_highly_abundant, utilize_de,
        spike_in, mdata, mdata_counts, adata, train_indices, val_indices,
        n_genes, n_regions, n_snps, labels_key, further_labels_keys,
        batch_key, patient_key, further_batch_keys, further_continuous_batch_keys,
        n_patient_covariates, protein_expression_obsm_key,
        use_mean_mixing, use_transformer, use_transformer_latents,
        use_product_of_experts, use_mixture_of_experts, use_rectified_flows,
        use_gating_network, calculate_model_interpretability, keep_activations,
        perform_perturbation_analysis, calculate_covariate_attention,
        perform_scarches, perform_scarches_hao_query, scarches_modality_mode,
        sparsevi_config_settings, default_option=None,
        obo_file=None, filter_namespace=True, seed=None,
        ablation_group=None,
        ablation_value=None,
):
    '''When ablation_group and ablation_value are provided (the normal cluster use'''

    is_go_model = "go" in sparsevi_config_settings.get("model_type", "")

    if is_go_model:
        baseline_parameters = {
            "lr": 0.0001,
            "modality_penalty": "MMD",
            "gene_likelihood": "zinb",
            "n_layers_encoder": 5,
            "n_layers_decoder": 2,
            "n_hidden": 384,
            "dropout_rate": 0.3,
            "activation_fn": "swish",
            "standard_go_size": 2,
            "standard_gene_size": 5,
            "map_ensembl_go": "map_ensembl_go_standard_with_rna",
        }
    else:
        baseline_parameters = {
            "lr": 0.0001,
            "modality_penalty": "Jeffreys",
            "gene_likelihood": "zinb",
            "n_layers_encoder": 5,
            "n_layers_decoder": 2,
            "n_hidden": -1,
            "dropout_rate": 0.1,
            "activation_fn": "relu",
            "standard_gene_size": 5,
            "map_ensembl_go": "map_ensembl_go_standard_with_rna",
        }

    ablation_grid = {
        "depth":     ("n_layers_encoder",  [3, 4, 5, 6, 7]),
        "go_size":   ("standard_go_size",  [2, 3, 4, 5]),
        "gene_size": ("standard_gene_size",[2, 3, 4, 5]),
    }

    ablation_root = base_hyper_name + "_ablation"
    os.makedirs(ablation_root, exist_ok=True)
    results_path = os.path.join(ablation_root, "ablation_results.csv")

    if ablation_group is not None and ablation_value is not None:
        assert ablation_group in ablation_grid, (
            f"Unknown ablation_group '{ablation_group}'. "
            f"Choose from: {list(ablation_grid.keys())}"
        )
        runs = [(ablation_group, int(ablation_value))]
    else:
        runs = [
            (group, value)
            for group, (_, values) in ablation_grid.items()
            for value in values
            if group != "go_size" or is_go_model
        ]

    results = []

    for group, value in runs:
        param_name, _ = ablation_grid[group]

        hyper_name = os.path.join(ablation_root, f"ablation_{group}_{value}")
        os.makedirs(hyper_name, exist_ok=True)

        run_parameters = baseline_parameters.copy()
        run_parameters[param_name] = value

        print(f"\n[ablation] group={group}  {param_name}={value}  → {hyper_name}")

        start = time.time()
        try:
            scib = _run_single_experiment(
                task=task,
                hyper_name=hyper_name,
                tool_name=tool_name,
                covariates_config=covariates_config,
                inject_batch_covariates=inject_batch_covariates,
                inject_patient_covariates=inject_patient_covariates,
                inject_further_batch_covariates=inject_further_batch_covariates,
                inject_further_continuous_batch_covariates=inject_further_continuous_batch_covariates,
                incorporate_batch_scib=incorporate_batch_scib,
                incorporate_patient_scib=incorporate_patient_scib,
                optimization_metric=optimization_metric,
                dataset_name=dataset_name,
                subsampled=subsampled,
                paired_rate=paired_rate,
                utilize_highly_variable=utilize_highly_variable,
                utilize_highly_abundant=utilize_highly_abundant,
                utilize_de=utilize_de,
                spike_in=spike_in,
                mdata=mdata,
                mdata_counts=mdata_counts,
                adata=adata,
                train_indices=train_indices,
                val_indices=val_indices,
                n_genes=n_genes,
                n_regions=n_regions,
                n_snps=n_snps,
                labels_key=labels_key,
                further_labels_keys=further_labels_keys,
                batch_key=batch_key,
                patient_key=patient_key,
                further_batch_keys=further_batch_keys,
                further_continuous_batch_keys=further_continuous_batch_keys,
                n_patient_covariates=n_patient_covariates,
                protein_expression_obsm_key=protein_expression_obsm_key,
                use_mean_mixing=use_mean_mixing,
                use_transformer=use_transformer,
                use_transformer_latents=use_transformer_latents,
                use_product_of_experts=use_product_of_experts,
                use_mixture_of_experts=use_mixture_of_experts,
                use_rectified_flows=use_rectified_flows,
                use_gating_network=use_gating_network,
                sparsevi_config_settings=sparsevi_config_settings,
                parameters=run_parameters,
                obo_file=obo_file,
                filter_namespace=filter_namespace,
                seed=seed,
            )
            error = None
        except Exception as e:
            print(f"  [FAILED] {e}")
            scib  = None
            error = str(e)

        elapsed = time.time() - start

        row = {
            "group":      group,
            "param_name": param_name,
            "value":      value,
            "scib":       scib,
            "runtime_s":  elapsed,
            "hyper_name": hyper_name,
            "error":      error,
        }
        results.append(row)

        new_row_df = pd.DataFrame([row])
        if os.path.exists(results_path):
            existing = pd.read_csv(results_path)
            existing = existing[
                ~((existing["group"] == group) &
                  (existing["value"] == value))
            ]
            updated = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            updated = new_row_df
        updated.to_csv(results_path, index=False)

        print(f"  scib={scib}  ({elapsed:.0f}s)  → saved to {results_path}")

    return pd.DataFrame(results)


def _run_single_experiment(
        task, hyper_name, tool_name, covariates_config,
        inject_batch_covariates, inject_patient_covariates,
        inject_further_batch_covariates, inject_further_continuous_batch_covariates,
        incorporate_batch_scib, incorporate_patient_scib,
        optimization_metric, dataset_name, subsampled, paired_rate,
        utilize_highly_variable, utilize_highly_abundant, utilize_de,
        spike_in, mdata, mdata_counts, adata, train_indices, val_indices,
        n_genes, n_regions, n_snps, labels_key, further_labels_keys,
        batch_key, patient_key, further_batch_keys, further_continuous_batch_keys,
        n_patient_covariates, protein_expression_obsm_key,
        use_mean_mixing, use_transformer, use_transformer_latents,
        use_product_of_experts, use_mixture_of_experts, use_rectified_flows,
        use_gating_network,
        sparsevi_config_settings, parameters, obo_file, filter_namespace, seed):

    sparsevi_config_settings = dict(sparsevi_config_settings)
    parameters = dict(parameters)

    if "modality_penalty" in sparsevi_config_settings:
        del sparsevi_config_settings["modality_penalty"]

    gaf_base = os.path.join(
        os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings"
    )
    if parameters.get("map_ensembl_go") == "map_ensembl_go_standard_with_rna":
        parameters["map_ensembl_go"] = [
            os.path.join(gaf_base, "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
            os.path.join(gaf_base, "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
            os.path.join(gaf_base, "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
        ]
    elif parameters.get("map_ensembl_go") == "map_ensembl_go_standard_with_rna_with_npintertar":
        parameters["map_ensembl_go"] = [
            os.path.join(gaf_base, "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
            os.path.join(gaf_base, "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
            os.path.join(gaf_base, "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
            os.path.join(gaf_base, "goa_human_npinter_tar_v2.gaf"),
        ]

    sparsevi_config = sparsevi_evaluation.get_sparsevi_config(
        **parameters, **sparsevi_config_settings
    )
    if "map_ensembl_go" in sparsevi_config:
        del sparsevi_config["map_ensembl_go"]

    return tools.sparsevi(
        task, hyper_name,
        inject_batch_covariates, inject_patient_covariates,
        inject_further_batch_covariates, inject_further_continuous_batch_covariates,
        incorporate_batch_scib, incorporate_patient_scib,
        optimization_metric, dataset_name, subsampled, paired_rate,
        utilize_highly_variable, utilize_highly_abundant, utilize_de,
        spike_in, mdata, mdata_counts, adata,
        train_indices, val_indices,
        n_genes, n_regions, n_snps, labels_key, further_labels_keys,
        batch_key, patient_key, further_batch_keys, further_continuous_batch_keys,
        n_patient_covariates, protein_expression_obsm_key,
        use_mean_mixing, use_transformer, use_transformer_latents,
        use_product_of_experts, use_mixture_of_experts, use_rectified_flows,
        use_gating_network,
        default=True, best=False, hyper=False,
        calculate_model_interpretability=False,
        keep_activations=False,
        perform_perturbation_analysis=False,
        calculate_covariate_attention=False,
        perform_scarches=False,
        perform_scarches_hao_query=False,
        scarches_modality_mode=None,
        **covariates_config, **parameters, **sparsevi_config,
        best_optimization_metric=0,
        obo_file=obo_file,
        filter_namespace=filter_namespace,
        seed=seed,
        perform_scib_calculation=True,
        perform_umap_plots=True,
        perform_model_downstream=False,
    )