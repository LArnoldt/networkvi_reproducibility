import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_Hao2021():
    '''Function downloads CITE-seq datasets of Hao 2021.'''

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-barcodes.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-barcodes.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-features.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-features.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-features.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-features.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-matrix.mtx.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-matrix.mtx.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-matrix.mtx.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008737_RNA_3P-matrix.mtx"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-barcodes.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-barcodes.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-barcodes.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-barcodes.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-features.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-features.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-features.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-features.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-matrix.mtx.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-matrix.mtx.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-matrix.mtx.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008738_ADT_3P-matrix.mtx"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008739/suppl/GSM5008739_HTO_3P-barcodes.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-barcodes.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-barcodes.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-barcodes.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008739/suppl/GSM5008739_HTO_3P-features.tsv.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-features.tsv.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-features.tsv.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-features.tsv"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008739/suppl/GSM5008739_HTO_3P-matrix.mtx.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-matrix.mtx.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-matrix.mtx.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data", "GSM5008739_HTO_3P-matrix.mtx"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "datasets", "Hao2021", "data"), exist_ok=True)

    download_Hao2021()
