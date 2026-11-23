import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_Frangieh2021():
    '''Function downloads CITE-seq datasets of GSE194122.'''

    r = requests.get("https://zenodo.org/records/7041849/files/FrangiehIzar2021_protein.h5ad?download=1", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Frangieh2021", "data", "FrangiehIzar2021_protein.h5ad"), 'wb').write(r.content)

    r = requests.get("https://zenodo.org/records/7041849/files/FrangiehIzar2021_RNA.h5ad?download=1", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Frangieh2021", "data", "FrangiehIzar2021_RNA.h5ad"), 'wb').write(r.content)

if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "datasets", "Frangieh2021", "data"), exist_ok=True)

    download_Frangieh2021()
