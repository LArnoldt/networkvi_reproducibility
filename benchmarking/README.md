# NetworkVI benchmarking pipeline

This is the training/evaluation pipeline for all experiments in the manuscript. 

## Running it

```
python ./benchmarking/main.py --task=<task> --dataset_name=<dataset> --tool_name=<tool> [options...]
```

Main hyperparameters: `--task` (usually `hyper`, i.e. train + hyperparameter search),
`--tool_name` (`sparsevi` is NetworkVI; also `wnn`, `mofa`, `totalvi`, `midas`, `multimil`),
`--dataset_name` (see `datasets.py`), and for NetworkVI, `--model_type` (`go_with_interaction_gene_layer`
is the published architecture; `dense` is the fully-connected baseline).


## Files

| File | Purpose |
|---|---|
| `main.py` | CLI entry point / dispatcher |
| `tools.py` | Per-tool training/evaluation implementations |
| `tools_hyper.py` | Ax hyperparameter search spaces per tool |
| `datasets.py` | Dataset loaders |
| `metrics.py` | scIB integration metrics |
| `sparsevi_evaluation.py` | NetworkVI config builder, incl. Fig. 2b's alternative prior-knowledge sources |
| `sparsevi_label_transfer.py` | scArches query-to-reference label transfer (Fig. S8-S10) |
| `sparsevi_interpretability_experiments.py` | Sensitivity analysis + spike-in experiments (Fig. 2e, S17, S27) |
| `enrichment_analysis.py` | GSEA/ORA baseline (Fig. 6, S25-S26) |
| `go_gene_stability.py` | GO-importance stability across seeds/subsampling |
| `grn_pipeline.py` / `grn_analysis.py` | GRN/Leiden/SCENIC baselines |
| `single_hyper_variations.py` | Single-hyperparameter sweep |
| `run_tool_parameters_different_seeds.py` | Seed-reproducibility reruns |
| `benchmarking_analysis.py` / `benchmarking_analysis_plots.py` | Compute-scalability benchmark (HLCA) |
