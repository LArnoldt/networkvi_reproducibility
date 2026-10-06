import muon as mu

mdata = mu.read("./data/hlca_full.h5mu")
adata = mdata["rna"].copy()
del adata.uns
del adata.obsm
del adata.varm
del adata.obsp
del adata.varp
adata = adata[:, adata.var["highly_variable"]].copy()
adata.write_h5ad("./data/hlca_hvg.h5ad")


import mudata
import numpy as np
import scipy.sparse as sp
adata = mdata["rna"].copy()
adata.X = sp.csr_matrix(adata.X)
for k in adata.obsp:
    adata.obsp[k] = sp.csr_matrix(adata.obsp[k])
for k in adata.obsm:
    adata.obsm[k] = np.asarray(adata.obsm[k])
for k in list(adata.uns.keys()):
    if not isinstance(adata.uns[k], (dict, str, int, float, list)):
        del adata.uns[k]
del adata.uns
mdata_legacy = mudata.MuData({"rna": adata})
mdata_legacy.write("./data/hlca_full_legacy.h5mu")
