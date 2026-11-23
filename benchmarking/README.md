# NetworkVI benchmarking pipeline

This is the training/evaluation pipeline for all experiments in the manuscript. 

## Running it

```
python ./benchmarking/main.py --task=<task> --dataset_name=<dataset> --tool_name=<tool> [options...]
```

Main hyperparameters: `--task` (usually `hyper`, i.e. train + hyperparameter search),
`--tool_name` (`networkvi` is NetworkVI; also `wnn`, `mofa`, `totalvi`, `midas`, `multimil`),
`--dataset_name` (see `datasets.py`), and for NetworkVI, `--model_type` (`go_with_interaction_gene_layer`
is the published architecture; `dense` is the fully-connected baseline).

The hyperparameter search runs for NetworkVI, MultiVI (`--model_type=dense`), TotalVI and MultiMIL. For MOFA+, WNN and
MIDAS, `--task=hyper` trains only their default configuration (`tools_hyper.py`).

## NetworkVI configurations

Without further options, NetworkVI is trained with mean mixing of the modality-specific posteriors, the Jeffreys
penalty, 4 encoder and 2 decoder layers, 384 hidden units, dropout 0.3, hard swish, GO node size 2, gene node size 5,
learning rate 1e-4, weight decay 1e-3 and a ZINB likelihood.

- `--modality_penalty` (`Jeffreys`, `MMD`, `realMMD`) sets the penalty of all models of a run, including the
  hyperparameter trials and the single-parameter ablations; the penalty is not searched.
- `--modality_fusion` (`mean_mixing`, `mixture_of_experts`, `product_of_experts`) sets the fusion; it is not searched.
- `--not_run_hyper` trains only the default model; `--seed` sets the seed.
- `--task=run_tool_parameters_different_seeds --different_seeds <seeds>` trains a fixed configuration for several
  seeds, from the default configuration or from `--model_parameters_json=<file>` (keys as above, e.g. `modality_penalty`,
  `n_layers_encoder`, `n_hidden`, `dropout_rate`, `activation_fn`, `standard_go_size`, `standard_gene_size`).

Common options for the NeurIPS 2021 CITE-seq BMMC models:

```
python ./benchmarking/main.py --task=hyper --dataset_name=neurips2021_cite_bmmc --tool_name=networkvi --paired_rate=1.0 \
  --optimization_metric=scib --utilize_highly_variable --model_type=go_with_interaction_gene_layer \
  --gaf_source=standard_with_rna --gene_layer_interaction_source=ENCFF173VDJ \
  --networkvi_inject_covariates_mode=deeply_inject_covariates --inject_batch_covariates --incorporate_batch_scib \
  --not_run_hyper --seed=0
```

Penalty and fusion of the models used in the manuscript, with the options that select them:

| Analysis | Penalty and fusion | Additional options |
|---|---|---|
| GO and gene interpretability, label transfer, Multiome downstream tasks (Figures 2d–g, 3; Supplementary Figures 8–10, 14–18) | Jeffreys, mean mixing | none |
| TAD analyses (Figure 4, Supplementary Figures 19–23) | Jeffreys, mean mixing | `--seed=11`, `13`, `14`; without the TAD layer `--model_type=go_with_gene_layer` |
| Benchmark with hyperparameter search (Supplementary Figures 1–12) | Jeffreys, mean mixing | without `--not_run_hyper`, per dataset and `--paired_rate` |
| GO-shuffled controls (Supplementary Figure 13) | MMD, mean mixing | `--modality_penalty=MMD --gaf_source=standard_with_rna_gshuff`; for 13b also the two covariate options of the next row |
| Covariate attention (Figure 5b, Supplementary Figures 13b, 24) | MMD, mean mixing | `--modality_penalty=MMD --inject_further_batch_covariates --inject_further_continuous_batch_covariates` |
| Perturbation analysis (Figure 6, Supplementary Figures 25–26) | MMD, mean mixing | `--dataset_name=frangieh2021 --model_type=go_with_gene_layer --modality_penalty=MMD`, without batch covariates |
| Perturbation analysis (Figure 6a, c; Supplementary Figure 25a) | MMD, mixture of experts | as above, plus `--modality_fusion=mixture_of_experts` |
| Prior-knowledge sources (Figure 2b) | MMD, mean mixing | `--modality_penalty=MMD`, plus the source options (e.g. `--gene_layer_interaction_source`, `--gaf_source`) |
| Stability of the interpretability | MMD, mean mixing | `--modality_penalty=MMD`, different seeds |
| GO hierarchy ablations (Supplementary Table 7) | MMD, mean mixing | `--task=single_hyper_variations --ablation_group=<depth\|go_size\|gene_size> --ablation_value=<value>` |
| MultiVI | Jeffreys, mean mixing | `--model_type=dense` |

## Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point / dispatcher |
| `tools.py` | Per-tool training/evaluation implementations |
| `tools_hyper.py` | Ax hyperparameter search spaces per tool |
| `datasets.py` | Dataset loaders |
| `metrics.py` | scIB integration metrics |
| `networkvi_evaluation.py` | NetworkVI config builder, incl. Fig. 2b's alternative prior-knowledge sources |
| `networkvi_label_transfer.py` | scArches query-to-reference label transfer (Fig. S8-S10) |
| `networkvi_interpretability_experiments.py` | Sensitivity analysis (Fig. 2e, S17) |
| `enrichment_analysis.py` | GSEA/ORA baseline (Fig. 6, S25-S26) |
| `grn_pipeline.py` / `grn_analysis.py` | GRN/Leiden/SCENIC baselines |
| `single_hyper_variations.py` | Single-hyperparameter sweep |
| `run_tool_parameters_different_seeds.py` | Seed-reproducibility reruns |
| `benchmarking_analysis.py` / `benchmarking_analysis_plots.py` | Compute-scalability benchmark (HLCA) |
