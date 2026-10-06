import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_hlca():
    '''Function downloads RNA-seq dataset of Sikkema 2023.'''

    r = requests.get("https://datasets.cellxgene.cziscience.com/cd9ac225-682a-4f39-b0de-29caeb532bec.h5ad", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "hlca", "data", "hlca_full.h5ad"), 'wb').write(r.content)

    r = requests.get("https://datasets.cellxgene.cziscience.com/b351804c-293e-4aeb-9c4c-043db67f4540.h5ad", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "hlca", "data", "hlca_core.h5ad"), 'wb').write(r.content)


if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "datasets", "hlca", "data"), exist_ok=True)

    download_hlca()
