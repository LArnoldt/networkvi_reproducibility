import pandas as pd
import scanpy as sc
import anndata as ad
import os
import sys
from torchinfo import summary
import torch
import pickle
import numpy as np
import shutil

def get_sparsevi_config(model_type=None, gene_interaction_layer_dynamic=None, rigl_dynamic_tend_epoch_num=None, gene_layer_interaction_source=None, gene_interaction_layer_nlayers=None, group_loss_modality_penalty=None, group_loss_max_iter=None, gaf_source=None, n_layers_encoder=None, **kwargs):

    sparsevi_config = {}

    if model_type.lower() == "go":
        sparsevi_config["layers_encoder_type"] = "go"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = None
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "go_with_gene_layer":
        sparsevi_config["layers_encoder_type"] = "go"
        sparsevi_config["expression_gene_layer_type"] = "standard"
        sparsevi_config["accessibility_gene_layer_type"] = "standard"
        sparsevi_config["genotype_gene_layer_type"] = "standard"
        sparsevi_config["protein_gene_layer_type"] = "standard"
    elif model_type.lower() == "go_with_gene_layer_accessibility_only":
        sparsevi_config["layers_encoder_type"] = "go"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = "standard"
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "go_with_interaction_gene_layer":
        sparsevi_config["layers_encoder_type"] = "go"
        sparsevi_config["expression_gene_layer_type"] = "interaction"
        sparsevi_config["accessibility_gene_layer_type"] = "interaction"
        sparsevi_config["genotype_gene_layer_type"] = "interaction"
        sparsevi_config["protein_gene_layer_type"] = "interaction"
    elif model_type.lower() == "go_with_interaction_gene_layer_accessibility_only":
        sparsevi_config["layers_encoder_type"] = "go"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = "interaction"
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "dense":
        sparsevi_config["layers_encoder_type"] = "dense"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = None
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "dense_with_gene_layer":
        sparsevi_config["layers_encoder_type"] = "dense"
        sparsevi_config["expression_gene_layer_type"] = "standard"
        sparsevi_config["accessibility_gene_layer_type"] = "standard"
        sparsevi_config["genotype_gene_layer_type"] = "standard"
        sparsevi_config["protein_gene_layer_type"] = "standard"
    elif model_type.lower() == "dense_with_interaction_gene_layer":
        sparsevi_config["layers_encoder_type"] = "dense"
        sparsevi_config["expression_gene_layer_type"] = "interaction"
        sparsevi_config["accessibility_gene_layer_type"] = "interaction"
        sparsevi_config["genotype_gene_layer_type"] = "interaction"
        sparsevi_config["protein_gene_layer_type"] = "interaction"
    elif model_type.lower() == "rigl":
        sparsevi_config["layers_encoder_type"] = "rigl"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = None
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
        sparsevi_config["sparsities"] = [0.5] * n_layers_encoder
        sparsevi_config["dynamic"] = False
    elif model_type.lower() == "rigl_with_gene_layer":
        sparsevi_config["layers_encoder_type"] = "rigl"
        sparsevi_config["expression_gene_layer_type"] = "standard"
        sparsevi_config["accessibility_gene_layer_type"] = "standard"
        sparsevi_config["genotype_gene_layer_type"] = "standard"
        sparsevi_config["protein_gene_layer_type"] = "standard"
        sparsevi_config["sparsities"] = [0.5] * n_layers_encoder
        sparsevi_config["dynamic"] = False
    elif model_type.lower() == "rigl_dynamic":
        sparsevi_config["layers_encoder_type"] = "rigl"
        sparsevi_config["expression_gene_layer_type"] = "standard"
        sparsevi_config["accessibility_gene_layer_type"] = "standard"
        sparsevi_config["genotype_gene_layer_type"] = "standard"
        sparsevi_config["protein_gene_layer_type"] = "standard"
        sparsevi_config["sparsities"] = [0.5] * n_layers_encoder
        sparsevi_config["dynamic"] = True
        sparsevi_config["dynamic_tend_epoch_num"] = rigl_dynamic_tend_epoch_num
    elif model_type.lower() == "rigl_dynamic_without_gene_layer":
        sparsevi_config["layers_encoder_type"] = "rigl"
        sparsevi_config["expression_gene_layer_type"] = None
        sparsevi_config["accessibility_gene_layer_type"] = None
        sparsevi_config["genotype_gene_layer_type"] = None
        sparsevi_config["protein_gene_layer_type"] = None
        sparsevi_config["sparsities"] = [0.5] * n_layers_encoder
        sparsevi_config["dynamic"] = True
        sparsevi_config["dynamic_tend_epoch_num"] = rigl_dynamic_tend_epoch_num

    if gene_layer_interaction_source is not None:
        sparsevi_config["gene_interaction_layer_dynamic"] = gene_interaction_layer_dynamic
        sparsevi_config["gene_interaction_layer_nlayers"] = gene_interaction_layer_nlayers
        if gene_layer_interaction_source.lower() == "string":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/ppi/string_filt.csv"
        elif gene_layer_interaction_source.lower() == "biogrid":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/ppi/biogrid.csv"
        elif gene_layer_interaction_source == "ENCFF203AKP":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config["gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF781ASD":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_loops_juicertools.csv"
            sparsevi_config[ "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_loops_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF531LSJ":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF041XLP":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_loops_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_loops_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ_shuff":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_shuffled.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF134HIZ":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_loops_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_loops_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF203AKP_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            sparsevi_config["gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF781ASD_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_loops_juicertools_shifted_benchmark.csv"
            sparsevi_config[ "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_loops_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF531LSJ_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF041XLP_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_loops_juicertools_shifted_benchmark.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF134HIZ_shifted_benchmark":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_loops_juicertools_shifted_benchmark.csv"
            sparsevi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_loops_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "grand_k562_grn":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/grand/grand_k562_grn_filt_2.0.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "grand_k562_ppi":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/grand/grand_k562_ppi_all.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_all":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_all.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c2":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c2.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c3":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c3.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c5":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c5.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c8":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c8.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_h":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_h.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "htftarget_bone_marrow":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/transcription_factors/htftarget_bone_marrow.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "tflink":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/transcription_factors/tflink_all.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "MCF7":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF164AGX_MCF-7_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "T47D":
            sparsevi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF804SET_T47D_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF164AGX":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF164AGX_MCF-7_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF804SET":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF804SET_T47D_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF223RVL":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF223RVL_IMR-90_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF549OBE":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF549OBE_HepG2_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF958WAA":
            sparsevi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF958WAA_HCT116_GRCh38_contact_domains_juicertools.csv"
            sparsevi_config[
                "gene_groups"] = ""

    if group_loss_modality_penalty is not None:
        if group_loss_modality_penalty.lower() == "probs":
            sparsevi_config["group_loss_modality_penalty"] = group_loss_modality_penalty
            sparsevi_config["group_loss_max_iter"] = group_loss_max_iter
        elif group_loss_modality_penalty.lower() == "dynamics":
            sparsevi_config["group_loss_modality_penalty"] = group_loss_modality_penalty
            sparsevi_config["group_loss_max_iter"] = group_loss_max_iter
        else:
            sparsevi_config["group_loss_modality_penalty"] = None

    if gaf_source is not None:
        if gaf_source.lower() == "standard":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf")]
        elif gaf_source.lower() == "standard_with_rna":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf")]
        elif gaf_source.lower() == "standard_with_rna_lshuff":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf")]
        elif gaf_source.lower() == "standard_with_rna_gshuff":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npinternc":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npintertar":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npinter":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_npinternc":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf")]
        elif gaf_source.lower() == "standard_with_npintertar":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_npinter":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "go2022":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2022-06-15.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2022-06-15.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2022-06-15.gaf")]
        elif gaf_source.lower() == "go2019":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2019-06-09.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2019-06-09.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2019-06-09.gaf")]
        elif gaf_source.lower() == "go2014":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2014-06-01.gaf")]
        elif gaf_source.lower() == "reactome":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "reactome_human_ensembl_gene_mapping_2026-01-20.gaf")]
        elif gaf_source.lower() == "pcommons":
            sparsevi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "pathwaycommons_human_ensembl_gene_mapping.gaf")]
        else:
            sparsevi_config["map_ensembl_go"] = [gaf_source]
            sparsevi_config["filter_namespace"] = False

    return sparsevi_config

def sparsevi_model_introspection(model, adata_mvi, categorical_covariate_keys):

    sys.path.append("/sc-projects/sc-proj-dh-ukb-intergenics/analysis/development/arnoldtl/code/scvi-tools/src/")
    import scvi

    model.view_anndata_setup()

    tensors = {
        "X": torch.zeros((scvi.settings.batch_size, adata_mvi.shape[1])),
        "batch": torch.zeros((scvi.settings.batch_size, 1), dtype=torch.int64),
        "patient": torch.zeros((scvi.settings.batch_size, 1), dtype=torch.int64),
        "ind_x": torch.zeros((scvi.settings.batch_size, 1), dtype=torch.int64),
        "labels": torch.zeros((scvi.settings.batch_size, 1), dtype=torch.int64),
        "proteins": torch.zeros((scvi.settings.batch_size, adata_mvi.obsm['protein_expression'].shape[1])),
        "batch_index": torch.zeros((scvi.settings.batch_size, 1), dtype=torch.int64),
    }
    if categorical_covariate_keys is not None:
        tensors["extra_categorical_covs"] = torch.zeros((scvi.settings.batch_size, len(categorical_covariate_keys)), dtype=torch.int64)

    print(summary(model.module, input_data=[tensors]))


    try:
        print("Expression:")
        model.module.z_encoder_expression.encoder.gomodel.block_structure
        for layer1 in model.module.z_encoder_expression.encoder.gomodel.block_structure.keys():
            for layer2 in model.module.z_encoder_expression.encoder.gomodel.block_structure[layer1].keys():
                print(f"{layer1}_{layer2}: {len(model.module.z_encoder_expression.encoder.gomodel.block_structure[layer1][layer2])}")
    except AttributeError:
        pass

    try:
        print("Accessibility:")
        model.module.z_encoder_accessibility.encoder.gomodel.block_structure
        for layer1 in model.module.z_encoder_accessibility.encoder.gomodel.block_structure.keys():
            for layer2 in model.module.z_encoder_accessibility.encoder.gomodel.block_structure[layer1].keys():
                print(f"{layer1}_{layer2}: {len(model.module.z_encoder_accessibility.encoder.gomodel.block_structure[layer1][layer2])}")
    except AttributeError:
        pass

    try:
        print("Genotype:")
        model.module.z_encoder_genotype.encoder.gomodel.block_structure
        for layer1 in model.module.z_encoder_genotype.encoder.gomodel.block_structure.keys():
            for layer2 in model.module.z_encoder_genotype.encoder.gomodel.block_structure[layer1].keys():
                print(f"{layer1}_{layer2}: {len(model.module.z_encoder_genotype.encoder.gomodel.block_structure[layer1][layer2])}")
    except AttributeError:
        pass

    try:
        print("Protein:")
        model.module.z_encoder_protein.encoder.gomodel.block_structure
        for layer1 in model.module.z_encoder_protein.encoder.gomodel.block_structure.keys():
            for layer2 in model.module.z_encoder_protein.encoder.gomodel.block_structure[layer1].keys():
                print(f"{layer1}_{layer2}: {len(model.module.z_encoder_protein.encoder.gomodel.block_structure[layer1][layer2])}")
    except AttributeError:
        pass

def model_downstream(name, adata_mvi=None, mdata=None, model=None, labels_key=None, train_indices=None, val_indices=None):
    
    if type(adata_mvi) == str:
        adata_mvi = sc.read_h5ad(os.path.join(f"{name}", "adata_mvi.h5ad"))
    else:
        if type(adata_mvi) != ad._core.anndata.AnnData:
            raise TypeError("adata_mvi can't be load and is not AnnData.")

    if model is None:
        model = scvi.model.SPARSEVI.load(name, adata=adata_mvi)

    library_size_factors = model.get_library_size_factors()
    with open(os.path.join(f"{name}/", "library_size_factors.pickle"), 'wb') as handle:
        pickle.dump(library_size_factors, handle, protocol=pickle.DEFAULT_PROTOCOL)

    imputed_expression = model.get_normalized_expression()
    np.save(os.path.join(f"{name}/", "imputed_expression.npy"), imputed_expression)

    differential_expression_df = model.differential_expression(groupby=labels_key)
    differential_expression_df.to_pickle(os.path.join(f"{name}/", f"differential_expression_df_{labels_key}.pickle"))

    if "atac" in mdata.mod:
        region_factors = model.get_region_factors()
        np.save(os.path.join(f"{name}/", "region_factors.npy"), region_factors)

        accessibility_estimates_df = model.get_accessibility_estimates()
        accessibility_estimates_df.to_pickle(os.path.join(f"{name}/", "accessibility_estimates_df.pickle"))

        differential_accessibility_df = model.differential_accessibility(groupby=labels_key)
        differential_accessibility_df.to_pickle(os.path.join(f"{name}/", f"differential_accessibility_df_{labels_key}.pickle"))


def model_interpretability(name, adata_mvi=None, mdata=None, model=None, labels_key=None, train_indices=None, val_indices=None, calc_go_terms=False, calc_gene_groups=False, perform_perturbation_analysis=False, keep_activations=False):

    if type(adata_mvi) == str:
        adata_mvi = sc.read_h5ad(os.path.join(f"{name}", "adata_mvi.h5ad"))
    else:
        if type(adata_mvi) != ad._core.anndata.AnnData:
            raise TypeError("adata_mvi can't be load and is not AnnData.")

    if model is None:
        model = scvi.model.SPARSEVI.load(name, adata=adata_mvi)

    available_gene_stable_ids = \
    list(adata_mvi.var[adata_mvi.var["modality"] == "expression"]["gene_stable_id"].to_numpy()) + \
    list(adata_mvi.var[adata_mvi.var["modality"] == "Peaks"]["gene_stable_id"].to_numpy()) + \
    list(adata_mvi.var[adata_mvi.var["modality"] == "SNP"]["gene_stable_id"].to_numpy()) + \
    list(mdata.mod["prot"].var["gene_stable_id"].to_numpy()) if "prot" in mdata.mod else []


    perturbation_gene_ids = ['IL7R', 'TRAC', 'GATA3', 'ITGB1', 'SLC4A10', 'FOXP3', 'IL2RA', 'CD8A', 'CD8B', 'CD248',
                             'CCL5', 'GNLY', 'NKG7', 'KLRC2', 'CD79A', 'MS4A1', 'IGHM', 'IGHD', 'TNFRSF13C', 'IL4R',
                             'TCL1A', 'KLF4', 'LYZ', 'S100A8', 'ITGAM', 'CD14', 'FCGR3A', 'MS4A7', 'CST3', 'IRF8',
                             'TCF4']
    perturbation_gene_stable_ids = ['ENSG00000168685', 'ENSG00000277734', 'ENSG00000107485', 'ENSG00000150093',
                                    'ENSG00000144290', 'ENSG00000049768', 'ENSG00000134460', 'ENSG00000153563',
                                    'ENSG00000172116', 'ENSG00000174807', 'ENSG00000271503', 'ENSG00000115523',
                                    'ENSG00000105374', 'ENSG00000205809', 'ENSG00000105369', 'ENSG00000156738',
                                    'ENSG00000211899', 'ENSG00000211898', 'ENSG00000159958', 'ENSG00000077238',
                                    'ENSG00000100721', 'ENSG00000136826', 'ENSG00000090382', 'ENSG00000143546',
                                    'ENSG00000169896', 'ENSG00000170458', 'ENSG00000203747', 'ENSG00000110079',
                                    'ENSG00000101439', 'ENSG00000140968', 'ENSG00000196628']
    perturbation_gene_stable_ids = ["ENSG00000203747", "ENSG00000153563"]
    perturbation_gene_stable_ids = set(available_gene_stable_ids) & set(perturbation_gene_stable_ids)

    name = f"/sc-scratch/sc-scratch-dh-ukb-intergenics/sparsevi/{name}"
    os.makedirs(f"{name}", exist_ok=True)

    os.makedirs(f"{name}/calc_go_terms_{labels_key}", exist_ok=True)
    if calc_go_terms:
        model.get_go_gene_importances(labels_column=labels_key,
                                                    results_dir=f"{name}/calc_go_terms_{labels_key}",
                                                    shuffle_set_split=False,
                                                    batch_size=512, train_idx=train_indices,
                                                    validation_idx=val_indices, calc_go_terms=True,
                                                    calc_gene_groups=False, keep_activations=keep_activations)
    os.makedirs(f"{name}/calc_gene_groups_{labels_key}", exist_ok=True)
    if calc_gene_groups is not None and calc_gene_groups != False:
        model.get_go_gene_importances(labels_column=labels_key,
                                                    results_dir=f"{name}/calc_gene_groups_{labels_key}",
                                                    shuffle_set_split=False,
                                                    batch_size=512, train_idx=train_indices,
                                                    validation_idx=val_indices, calc_go_terms=False,
                                                    calc_gene_groups=calc_gene_groups, save_fit=False,
                                                    overwrite_save_fit=False, keep_activations=keep_activations)

    os.makedirs(f"{name}/calc_go_terms_perturbation_analysis_{labels_key}", exist_ok=True)
    if calc_go_terms and perform_perturbation_analysis:
        _ = model.get_perturbation_go_gene_importances(labels_column=labels_key,
                                                       modality_key="modality",
                                                       gene_stable_id_key="gene_stable_id",
                                                       perturbation_gene_stable_ids=perturbation_gene_stable_ids,
                                                       batch_size=512,
                                                       train_idx=train_indices,
                                                       validation_idx=val_indices,
                                                       shuffle_set_split=False,
                                                       calc_go_terms=True,
                                                       results_dir=f"{name}/calc_go_terms_{labels_key}",
                                                       keep_activations=keep_activations)
