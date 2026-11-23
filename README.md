# NetworkVI reproducibility

This repository contains the code to reproduce the analysis and figures of NetworkVI.

Please find the `NetworkVI` package repository [here](https://github.com/LArnoldt/networkvi).

SparseVI is an earlier name of NetworkVI and is a synonym for it. Result folders and files written by the original runs keep this name (for example `sparsevi_<dataset>_…/` and `X_sparsevi.npy`), so paths in the notebooks and the result folder prefix of new NetworkVI runs still contain `sparsevi`.

## Installation

TotalVI and MultiMIL need different `scvi-tools` versions than each other, so each runs in its own environment:

| Environment | Used for | Install |
|---|---|---|
| NetworkVI | NetworkVI models, interpretability, scIB, enrichment, GRN analyses | `pip install -r requirements.txt` |
| TotalVI | TotalVI benchmarks (`--tool_name=totalvi`) | `pip install -r envs/requirements-totalvi.txt` |
| MultiMIL | MultiMIL benchmarks (`--tool_name=multimil`) | `pip install -r envs/requirements-multimil.txt` |

All environments use `scib-metrics` 0.5.1 for the scIB metrics.