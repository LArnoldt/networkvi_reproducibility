import requests
import os
import zipfile
import glob
import subprocess
import gzip
import shutil

def download_Stephenson2021():
    '''
    Function downloads CITE-seq datasets of Stephenson 2021.
    Database Link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378/ / https://atlas.fredhutch.org/nygc/multimodal-pbmc/
    DOI: 10.1016/j.cell.2021.04.048
    '''

    r = requests.get("https://covid19.cog.sanger.ac.uk/submissions/release1/haniffa21.processed.h5ad", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "datasets", "Stephenson2021", "data", "haniffa21.processed.h5ad"), 'wb').write(r.content)

if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "datasets", "Stephenson2021", "data"), exist_ok=True)

    download_Stephenson2021()
