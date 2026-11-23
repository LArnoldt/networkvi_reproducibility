import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_multimodal_pbmc_Hao2021():
    '''Function downloads CITE-seq datasets of Hao 2021.'''

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-barcodes.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "multimodal_pbmc_Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "multimodal_pbmc_Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "multimodal_pbmc_Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)