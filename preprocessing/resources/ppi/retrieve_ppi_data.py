import requests
import os
import pandas as pd
from pybiomart import Server
import tarfile
import zipfile
import gzip
import shutil

def _get_ensembl_gene_id_ensembl_peptide_id_mapping():
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    ensembl_gene_id_ensembl_peptide_id_mapping = dataset.query(attributes=['ensembl_gene_id', 'ensembl_peptide_id'])
    ensembl_gene_id_ensembl_peptide_id_mapping = ensembl_gene_id_ensembl_peptide_id_mapping[~ensembl_gene_id_ensembl_peptide_id_mapping["Protein stable ID"].isnull()]
    ensembl_gene_id_ensembl_peptide_id_mapping = ensembl_gene_id_ensembl_peptide_id_mapping.rename(columns={"Protein stable ID": "Uniprot/Ensembl ID"})
    return ensembl_gene_id_ensembl_peptide_id_mapping

def _get_ensembl_gene_id_uniprotswissprot_mapping():
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    ensembl_gene_id_uniprotswissprot_mapping = dataset.query(attributes=['ensembl_gene_id', 'uniprotswissprot'])
    ensembl_gene_id_uniprotswissprot_mapping = ensembl_gene_id_uniprotswissprot_mapping[~ensembl_gene_id_uniprotswissprot_mapping["UniProtKB/Swiss-Prot ID"].isnull()]
    ensembl_gene_id_uniprotswissprot_mapping = ensembl_gene_id_uniprotswissprot_mapping.rename(columns={"UniProtKB/Swiss-Prot ID": "Uniprot/Ensembl ID"})
    return ensembl_gene_id_uniprotswissprot_mapping

def download_biogrid_database():
    '''
    Function downloads BIOGRID Database.
    Database Link: https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.235/
    DOI: 10.1002/pro.3978
    '''

    r = requests.get("https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.235/BIOGRID-ALL-4.4.235.tab3.zip", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "BIOGRID-ALL-4.4.235.tab3.zip"), 'wb').write(r.content)
    with zipfile.ZipFile(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "BIOGRID-ALL-4.4.235.tab3.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "BIOGRID-ALL-4.4.235.tab3"))

    biogrid = pd.read_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "BIOGRID-ALL-4.4.235.tab3", "BIOGRID-ALL-4.4.235.tab3.txt"), delimiter="\t")
    biogrid = biogrid[biogrid["Organism Name Interactor A"] == "Homo sapiens"]
    biogrid = biogrid[biogrid["Organism Name Interactor B"] == "Homo sapiens"]

    ensembl_gene_id_uniprotswissprot_mapping = _get_ensembl_gene_id_uniprotswissprot_mapping()
    biogrid = biogrid.join(ensembl_gene_id_uniprotswissprot_mapping.set_index("Uniprot/Ensembl ID"), on="SWISS-PROT Accessions Interactor A", rsuffix=" A")
    biogrid = biogrid.join(ensembl_gene_id_uniprotswissprot_mapping.set_index("Uniprot/Ensembl ID"), on="SWISS-PROT Accessions Interactor B", rsuffix=" B")
    biogrid = biogrid[~biogrid["Gene stable ID"].isnull()]
    biogrid = biogrid[~biogrid["Gene stable ID B"].isnull()]

    biogrid_ppi_layer = pd.DataFrame(data={
        "gene1": biogrid["Gene stable ID"],
        "gene2": biogrid["Gene stable ID B"],
        "combined_score": [1] * len(biogrid["Gene stable ID"]),
    })
    biogrid_ppi_layer = biogrid_ppi_layer.drop_duplicates(keep="first")

    biogrid_ppi_layer.to_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "biogrid.csv"))

def download_string_database():
    '''
    Function downloads STRING Database.
    Database Link: https://string-db.org/cgi/download?sessionId=bGXKHG0flP57
    DOI: 10.1093/nar/gkac1000
    '''

    r = requests.get("https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "9606.protein.links.full.v12.0.txt.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "9606.protein.links.full.v12.0.txt.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "9606.protein.links.full.v12.0.txt"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    string = pd.read_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "9606.protein.links.full.v12.0.txt"), delimiter=" ")
    string["protein1"] = string["protein1"].apply(lambda x: x.split(".")[1])
    string["protein2"] = string["protein2"].apply(lambda x: x.split(".")[1])

    ensembl_gene_id_ensembl_peptide_id_mapping = _get_ensembl_gene_id_ensembl_peptide_id_mapping()
    string = string.join(ensembl_gene_id_ensembl_peptide_id_mapping.set_index("Uniprot/Ensembl ID"), on="protein1", rsuffix="1")
    string = string.join(ensembl_gene_id_ensembl_peptide_id_mapping.set_index("Uniprot/Ensembl ID"), on="protein2", rsuffix="2")
    string = string[~string["Gene stable ID"].isnull()]
    string = string[~string["Gene stable ID2"].isnull()]

    string_ppi_layer = pd.DataFrame(data={
        "gene1": string["Gene stable ID"],
        "gene2": string["Gene stable ID2"],
        "combined_score": string["combined_score"],
    })
    string_ppi_layer = string_ppi_layer.drop_duplicates(keep="first")

    string_ppi_layer.to_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "string.csv"))

    string_ppi_layer_filt = string_ppi_layer[string_ppi_layer["combined_score"] > 250]

    string_ppi_layer_filt.to_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ppi", "string_filt.csv"))

if __name__ == "__main__":

    download_biogrid_database()

    download_string_database()
