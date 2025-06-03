import requests
import os
import tarfile
import gzip
import shutil
import subprocess
import argparse
import pandas as pd
from pybiomart import Server
from itertools import islice
from datetime import date

def _get_ensembl_gene_id_uniprotswissprot_mapping():
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    ensembl_gene_id_uniprotswissprot_mapping = dataset.query(attributes=['ensembl_gene_id', 'uniprotswissprot'])
    ensembl_gene_id_uniprotswissprot_mapping = ensembl_gene_id_uniprotswissprot_mapping[~ensembl_gene_id_uniprotswissprot_mapping["UniProtKB/Swiss-Prot ID"].isnull()]
    ensembl_gene_id_uniprotswissprot_mapping = ensembl_gene_id_uniprotswissprot_mapping.rename(columns={"UniProtKB/Swiss-Prot ID": "external_identifier"})
    return ensembl_gene_id_uniprotswissprot_mapping

def _get_ensembl_gene_id_rnacentral_mapping():
    server = Server(host='http://www.ensembl.org')
    dataset = (server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl'])
    ensembl_gene_id_rnacentral_mapping = dataset.query(attributes=['ensembl_gene_id', 'rnacentral'])
    ensembl_gene_id_rnacentral_mapping = ensembl_gene_id_rnacentral_mapping[~ensembl_gene_id_rnacentral_mapping["RNAcentral ID"].isnull()]
    ensembl_gene_id_rnacentral_mapping = ensembl_gene_id_rnacentral_mapping.rename(columns={"RNAcentral ID": "external_identifier"})
    ensembl_gene_id_rnacentral_mapping["external_identifier"] = ensembl_gene_id_rnacentral_mapping["external_identifier"].apply(lambda x: x+"_9606")
    return ensembl_gene_id_rnacentral_mapping

def download_go_gaf():
    '''
    Function downloads GO Ontology and GO-Gene mapping files.
    '''

    r = requests.get("http://purl.obolibrary.org/obo/go/go-basic.obo", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "go-basic.obo"), 'wb').write(r.content)

    r = requests.get("https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_{date.today()}.gaf.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_{date.today()}.gaf.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_{date.today()}.gaf"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_isoform.gaf.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_isoform_{date.today()}.gaf.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_isoform_{date.today()}.gaf.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_isoform_{date.today()}.gaf"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_rna.gaf.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_rna_{date.today()}.gaf.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_rna_{date.today()}.gaf.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_rna_{date.today()}.gaf"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    r = requests.get("https://current.geneontology.org/annotations/reactome.gaf.gz", allow_redirects=True)
    open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"reactome_{date.today()}.gaf.gz"), 'wb').write(r.content)
    with gzip.open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"reactome_{date.today()}.gaf.gz"), 'rb') as f_in:
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"reactome_{date.today()}.gaf"), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def process_gaf():

    def write_gaf(goa, goa_head, goa_name, mapping_file):
        goa = goa.join(mapping_file.set_index("external_identifier"), on="DB_Object_ID")
        goa["DB_Object_ID"] = goa["Gene stable ID"]
        del goa["Gene stable ID"]
        goa = goa[~goa["DB_Object_ID"].isnull()]
        with open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"{goa_name}_ensembl_gene_mapping_{date.today()}.gaf"), "w") as f:
            for item in goa_head:
                f.write(item)
        goa.to_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"{goa_name}_ensembl_gene_mapping_{date.today()}.gaf"), mode="a", sep="\t", header=False, index=False)

    ensembl_gene_id_uniprotswissprot_mapping = _get_ensembl_gene_id_uniprotswissprot_mapping()
    ensembl_gene_id_rnacentral_mapping = _get_ensembl_gene_id_rnacentral_mapping()

    columns = ["DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID", "DB_Reference", "Evidence_Code",
               "With_or_From", "Aspect", "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type", "Taxon", "Date",
               "Assigned_by", "Annotation_Extension", "Annotation_Properties"]
    goa_human = pd.read_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_{date.today()}.gaf"), delimiter="\t", header=None, skiprows=13)
    goa_human.columns = columns
    goa_human_head = list(islice(open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_{date.today()}.gaf"), "r"), 13))
    goa_human_isoform = pd.read_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_isoform_{date.today()}.gaf"), delimiter="\t", header=None, skiprows=13)
    goa_human_isoform.columns = columns
    goa_human_isoform_head = list(islice(open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_isoform_{date.today()}.gaf"), "r"), 13))
    goa_human_rna = pd.read_csv(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_rna_{date.today()}.gaf"), delimiter="\t", header=None, skiprows=13)
    goa_human_rna.columns = columns
    goa_human_rna_head = list(islice(open(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", f"goa_human_rna_{date.today()}.gaf"), "r"), 13))

    write_gaf(goa_human, goa_human_head, "goa_human", ensembl_gene_id_uniprotswissprot_mapping)
    write_gaf(goa_human_isoform, goa_human_isoform_head, "goa_human_isoform", ensembl_gene_id_uniprotswissprot_mapping)
    write_gaf(goa_human_rna, goa_human_rna_head, "goa_human_rna", ensembl_gene_id_rnacentral_mapping)

if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings"), exist_ok=True)

    download_go_gaf()
    process_gaf()