import os
from collections import defaultdict
from functools import lru_cache

from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import GODag

GO_RESOURCES = os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings")
OBO_FILE = os.path.join(GO_RESOURCES, "go-basic.obo")
GAF_FILES = (
    os.path.join(GO_RESOURCES, "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
    os.path.join(GO_RESOURCES, "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
    os.path.join(GO_RESOURCES, "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
)


@lru_cache(maxsize=None)
def networkvi_go_gene_sets(obo_file=OBO_FILE, gaf_files=GAF_FILES, namespace="biological_process"):
    godag = GODag(obo_file, prt=None)
    upper_terms = {}

    def _upper(term):
        if term.id not in upper_terms:
            terms = {term.id}
            for parent in term.parents:
                if parent.namespace == namespace:
                    terms |= _upper(parent)
            upper_terms[term.id] = terms
        return upper_terms[term.id]

    genes = defaultdict(set)
    for gaf_file in gaf_files:
        for gene, go_ids in GafReader(gaf_file, prt=None).get_id2gos_nss().items():
            for go_id in go_ids:
                if go_id in godag and godag[go_id].id == go_id and godag[go_id].namespace == namespace:
                    for term_id in _upper(godag[go_id]):
                        genes[term_id].add(gene)
    return {f"{godag[term_id].name} ({term_id})": sorted(term_genes) for term_id, term_genes in genes.items()}


def to_gene_symbols(gene_sets, gene_symbols, gene_ids):
    symbol = dict(zip(gene_ids, gene_symbols))
    translated = {term: sorted({symbol[gene] for gene in genes if gene in symbol}) for term, genes in gene_sets.items()}
    return {term: genes for term, genes in translated.items() if genes}


def write_gmt(gene_sets, path):
    with open(path, "w") as f:
        for term, genes in gene_sets.items():
            f.write("\t".join([term, "NetworkVI GO (go-basic.obo, GOA human 2024-08-08)", *genes]) + "\n")
