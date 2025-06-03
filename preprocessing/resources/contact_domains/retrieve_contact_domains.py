import requests
import os
import pandas as pd
import tarfile
import gzip
import shutil
import json
from pathlib import Path

def download_GM12878_encode_contact_domains():
    '''
    Function downloads contact domains (genome subcompartments, loops) from ENCODE database.
    Database Link: https://www.encodeproject.org/ and https://www.encodeproject.org/search/?type=Experiment&control_type%21=%2A&status=released&perturbed=false&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=in+situ+Hi-C&biosample_ontology.classification=cell+line&biosample_ontology.term_name=GM12878&files.file_type=bed+bed9%2B&files.file_type=bedpe
    DOI: 10.1038/nature11247 and 10.1093/nar/gkz1062
    '''

    #https://www.encodeproject.org/experiments/ENCSR410MDC/

    r = requests.get("https://www.encodeproject.org/files/ENCFF203AKP/@@download/ENCFF203AKP.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    r = requests.get("https://www.encodeproject.org/files/ENCFF781ASD/@@download/ENCFF781ASD.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF781ASD_GM12878_GRCh38_loops_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF781ASD_GM12878_GRCh38_loops_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF781ASD_GM12878_GRCh38_loops_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    #https://www.encodeproject.org/experiments/ENCSR968KAY/

    r = requests.get("https://www.encodeproject.org/files/ENCFF531LSJ/@@download/ENCFF531LSJ.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    r = requests.get("https://www.encodeproject.org/files/ENCFF041XLP/@@download/ENCFF041XLP.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF041XLP_GM12878_GRCh38_loops_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF041XLP_GM12878_GRCh38_loops_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF041XLP_GM12878_GRCh38_loops_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_K562_encode_contact_domains():
    '''
    Function downloads contact domains (genome subcompartments, loops) from ENCODE database.
    Database Link: https://www.encodeproject.org/ and https://www.encodeproject.org/search/?type=Experiment&control_type!=*&status=released&perturbed=false&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=in+situ+Hi-C&biosample_ontology.classification=cell+line&files.file_type=bed+bed9%2B&files.file_type=bedpe&biosample_ontology.term_name=K562
    DOI: 10.1038/nature11247 and 10.1093/nar/gkz1062
    '''

    #https://www.encodeproject.org/experiments/ENCSR545YBD/

    r = requests.get("https://www.encodeproject.org/files/ENCFF173VDJ/@@download/ENCFF173VDJ.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    r = requests.get("https://www.encodeproject.org/files/ENCFF134HIZ/@@download/ENCFF134HIZ.bedpe.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", "ENCFF134HIZ_K562_GRCh38_loops_juicertools.bedpe"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def _download_genetic_marker_annotation(accession_code, cell_name, genetic_marker, assembly, date):

    r = requests.get(f"https://www.encodeproject.org/files/{accession_code}/@@download/{accession_code}.bigWig", allow_redirects=True)
    #open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", f"{accession_code}_{cell_name}_{genetic_marker}_{assembly}_{date}.bigWig"), 'wb').write(r.content)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode", f"{accession_code}_{cell_name}_{genetic_marker}_{assembly}_{date}_fc.bigWig"), 'wb').write(r.content) #TOOD ARNODLT MAKE THIS WORK WITH PVAL AS WELL

def download_encode_genetic_marker_annotations_from_list(file_name, cell_name):
    '''
    Function downloads genetic markers from ENCODE database.
    Database Link: https://www.encodeproject.org/
    DOI: 10.1038/nature11247 and 10.1093/nar/gkz1062
    '''

    with open(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", file_name)) as f:
        selected_BigWig_files = f.readlines()

    for file in selected_BigWig_files:

        file = Path(file).stem

        url = f'https://www.encodeproject.org/{file}?frame=object'
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers)
        try:
            biosample = response.json()
            if biosample['file_type'] == "bigWig" and biosample["biological_replicates"] == [1, 2]: # and biosample['output_type'] == "fold change over control": #'preferred_default' in biosample.keys() and biosample['preferred_default'] and biosample['output_type'] == "signal p-value":
                print(file.strip("/files"))
                print(biosample["target"])
                _download_genetic_marker_annotation(file, cell_name, biosample["target"].strip("/targets/").strip("-human"), biosample["assembly"], biosample['date_created'].split("T")[0])
        except requests.exceptions.JSONDecodeError:
            print("JSONDecodeError")
            print(file.strip("/files"))
            continue

if __name__ == "__main__":

    #download_tad_kb()

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "resources", "contact_domains", "encode"), exist_ok=True)
    #download_GM12878_encode_contact_domains()
    download_K562_encode_contact_domains()
    #download_encode_genetic_marker_annotations()

    #File from https://www.encodeproject.org/search/?type=Experiment&control_type%21=%2A&status=released&perturbed=false&biosample_ontology.term_name=GM12878&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&target.investigated_as=transcription+factor&target.investigated_as=chromatin+remodeler&target.investigated_as=RNA+binding+protein&target.investigated_as=histone&target.investigated_as=cohesin&target.investigated_as=DNA+repair&target.investigated_as=broad+histone+mark&target.investigated_as=narrow+histone+mark&target.investigated_as=cofactor&target.investigated_as=DNA+replication&target.investigated_as=other+context&target.investigated_as=RNA+polymerase+complex&assembly=GRCh38&files.file_type=bigWig
    #download_encode_genetic_marker_annotations_from_list("selected_BigWig_files_GM12878_default.txt", "GM12878")
    #download_encode_genetic_marker_annotations_from_list("selected_BigWig_files_GM12878_default_analysis.txt", "GM12878")

    #File from https://www.encodeproject.org/search/?type=Experiment&control_type%21=%2A&status=released&perturbed=false&biosample_ontology.term_name=GM12878&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&target.investigated_as=transcription+factor&target.investigated_as=chromatin+remodeler&target.investigated_as=RNA+binding+protein&target.investigated_as=histone&target.investigated_as=cohesin&target.investigated_as=DNA+repair&target.investigated_as=broad+histone+mark&target.investigated_as=narrow+histone+mark&target.investigated_as=cofactor&target.investigated_as=DNA+replication&target.investigated_as=other+context&target.investigated_as=RNA+polymerase+complex&assembly=GRCh38&files.file_type=bigWig
    download_encode_genetic_marker_annotations_from_list("selected_BigWig_files_K562_default.txt", "K562")

    #File from
    #download_encode_genetic_marker_annotations_from_list("selected_BigWig_files_Bladder.txt", "Bladder")
