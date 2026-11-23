import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_neurips2021_multiome_BMMC():
    '''Function downloads M-seq datasets of GSE194122.'''

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122%5Fopenproblems%5Fneurips2021%5Fmultiome%5FBMMC%5Fprocessed.h5ad.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "neurips2021_multiome_BMMC", "data", "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "neurips2021_multiome_BMMC", "data", "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "neurips2021_multiome_BMMC", "data", "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "datasets", "neurips2021_multiome_BMMC", "data"), exist_ok=True)

    download_neurips2021_multiome_BMMC()
