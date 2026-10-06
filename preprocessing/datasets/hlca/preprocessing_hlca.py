import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import muon as mu
from tqdm import tqdm
tqdm.pandas()

adata_core = sc.read_h5ad("./data/hlca_full.h5ad")
adata_core

adata = adata_core

random_state = np.random.RandomState()
indices = random_state.permutation(list(range(len(adata.obs))))
adata = adata[indices,:].copy()
rna = adata
rna
rna.var["gene_stable_id"] = rna.var.index


rna.layers["counts"] = rna.X.copy()

sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

sc.pp.highly_variable_genes(rna, n_top_genes=4000)


mdata = mu.MuData({"rna": rna})

mdata.write("./data/hlca_full.h5mu")
