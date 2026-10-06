import requests
import os
import pandas as pd
import tarfile
import gzip
import shutil
import json
from pathlib import Path

def download_K562_encode_contact_domains():
    '''Function downloads contact domains (genome subcompartments, loops) from ENCODE database.'''

    r = requests.get("https://www.encodeproject.org/files/ENCFF173VDJ/@@download/ENCFF173VDJ.bedpe.gz", allow_redirects=True)
    open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    r = requests.get("https://www.encodeproject.org/files/ENCFF134HIZ/@@download/ENCFF134HIZ.bedpe.gz", allow_redirects=True)
    open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def _download_genetic_marker_annotation(accession_code, cell_name, genetic_marker, assembly, date):

    out_path = os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode", f"{accession_code}_{cell_name}_{genetic_marker}_{assembly}_{date}_fc.bigWig")
    if os.path.exists(out_path):
        print(f"  already downloaded, skipping: {out_path}")
        return
    r = requests.get(f"https://www.encodeproject.org/files/{accession_code}/@@download/{accession_code}.bigWig", allow_redirects=True)
    open(out_path, 'wb').write(r.content)

def download_encode_genetic_marker_annotations_from_list(file_name, cell_name):
    '''Function downloads genetic markers from ENCODE database.'''

    with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", file_name)) as f:
        selected_BigWig_files = f.readlines()

    for file in selected_BigWig_files:

        file = Path(file).stem

        url = f'https://www.encodeproject.org/{file}?frame=object'
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)
        try:
            biosample = response.json()
            if biosample['file_type'] == "bigWig" and biosample["biological_replicates"] == [1, 2]:
                print(file.strip("/files"))
                print(biosample["target"])
                _download_genetic_marker_annotation(file, cell_name, biosample["target"].strip("/targets/").strip("-human"), biosample["assembly"], biosample['date_created'].split("T")[0])
        except requests.exceptions.JSONDecodeError:
            print("JSONDecodeError")
            print(file.strip("/files"))
            continue

if __name__ == "__main__":

    os.makedirs(os.path.join("/sc-scratch/sc-scratch-dh-ukb-intergenics/encode"), exist_ok=True)
    download_K562_encode_contact_domains()

    download_encode_genetic_marker_annotations_from_list("selected_BigWig_files_K562_default.txt", "K562")

