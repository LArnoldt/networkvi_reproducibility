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
import networkvi
from networkvi.model import NETWORKVI

def get_networkvi_config(model_type=None, gene_interaction_layer_dynamic=None, rigl_dynamic_tend_epoch_num=None, gene_layer_interaction_source=None, gene_interaction_layer_nlayers=None, gaf_source=None, n_layers_encoder=None, **kwargs):

    networkvi_config = {}

    if model_type.lower() == "go":
        networkvi_config["layers_encoder_type"] = "go"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = None
        networkvi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "go_with_gene_layer":
        networkvi_config["layers_encoder_type"] = "go"
        networkvi_config["expression_gene_layer_type"] = "standard"
        networkvi_config["accessibility_gene_layer_type"] = "standard"
        networkvi_config["protein_gene_layer_type"] = "standard"
    elif model_type.lower() == "go_with_gene_layer_accessibility_only":
        networkvi_config["layers_encoder_type"] = "go"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = "standard"
        networkvi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "go_with_interaction_gene_layer":
        networkvi_config["layers_encoder_type"] = "go"
        networkvi_config["expression_gene_layer_type"] = "interaction"
        networkvi_config["accessibility_gene_layer_type"] = "interaction"
        networkvi_config["protein_gene_layer_type"] = "interaction"
    elif model_type.lower() == "go_with_interaction_gene_layer_accessibility_only":
        networkvi_config["layers_encoder_type"] = "go"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = "interaction"
        networkvi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "dense":
        networkvi_config["layers_encoder_type"] = "dense"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = None
        networkvi_config["protein_gene_layer_type"] = None
    elif model_type.lower() == "dense_with_gene_layer":
        networkvi_config["layers_encoder_type"] = "dense"
        networkvi_config["expression_gene_layer_type"] = "standard"
        networkvi_config["accessibility_gene_layer_type"] = "standard"
        networkvi_config["protein_gene_layer_type"] = "standard"
    elif model_type.lower() == "dense_with_interaction_gene_layer":
        networkvi_config["layers_encoder_type"] = "dense"
        networkvi_config["expression_gene_layer_type"] = "interaction"
        networkvi_config["accessibility_gene_layer_type"] = "interaction"
        networkvi_config["protein_gene_layer_type"] = "interaction"
    elif model_type.lower() == "rigl":
        networkvi_config["layers_encoder_type"] = "rigl"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = None
        networkvi_config["protein_gene_layer_type"] = None
        networkvi_config["sparsities"] = [0.5] * n_layers_encoder
        networkvi_config["dynamic"] = False
    elif model_type.lower() == "rigl_with_gene_layer":
        networkvi_config["layers_encoder_type"] = "rigl"
        networkvi_config["expression_gene_layer_type"] = "standard"
        networkvi_config["accessibility_gene_layer_type"] = "standard"
        networkvi_config["protein_gene_layer_type"] = "standard"
        networkvi_config["sparsities"] = [0.5] * n_layers_encoder
        networkvi_config["dynamic"] = False
    elif model_type.lower() == "rigl_dynamic":
        networkvi_config["layers_encoder_type"] = "rigl"
        networkvi_config["expression_gene_layer_type"] = "standard"
        networkvi_config["accessibility_gene_layer_type"] = "standard"
        networkvi_config["protein_gene_layer_type"] = "standard"
        networkvi_config["sparsities"] = [0.5] * n_layers_encoder
        networkvi_config["dynamic"] = True
        networkvi_config["dynamic_tend_epoch_num"] = rigl_dynamic_tend_epoch_num
    elif model_type.lower() == "rigl_dynamic_without_gene_layer":
        networkvi_config["layers_encoder_type"] = "rigl"
        networkvi_config["expression_gene_layer_type"] = None
        networkvi_config["accessibility_gene_layer_type"] = None
        networkvi_config["protein_gene_layer_type"] = None
        networkvi_config["sparsities"] = [0.5] * n_layers_encoder
        networkvi_config["dynamic"] = True
        networkvi_config["dynamic_tend_epoch_num"] = rigl_dynamic_tend_epoch_num

    if gene_layer_interaction_source is not None:
        networkvi_config["gene_interaction_layer_dynamic"] = gene_interaction_layer_dynamic
        networkvi_config["gene_interaction_layer_nlayers"] = gene_interaction_layer_nlayers
        if gene_layer_interaction_source.lower() == "string":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/ppi/string_filt.csv"
        elif gene_layer_interaction_source.lower() == "biogrid":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/ppi/biogrid.csv"
        elif gene_layer_interaction_source == "ENCFF203AKP":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools.csv"
            networkvi_config["gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF781ASD":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[ "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF531LSJ":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF041XLP":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_contact_domains_uicertools.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ_shuff":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_shuffled.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF134HIZ":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_contact_domains_uicertools.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_contact_domains_juicertools_gene_groups.npy"
        elif gene_layer_interaction_source == "ENCFF203AKP_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config["gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF203AKP_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF781ASD_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config[ "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF781ASD_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF531LSJ_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF531LSJ_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF041XLP_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF041XLP_GM12878_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF173VDJ_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF173VDJ_K562_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "ENCFF134HIZ_shifted_benchmark":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_contact_domains_juicertools_shifted_benchmark.csv"
            networkvi_config[
                "gene_groups"] = "./preprocessing/resources/contact_domains/encode/ENCFF134HIZ_K562_GRCh38_contact_domains_juicertools_gene_groups_shifted_benchmark.npy"
        elif gene_layer_interaction_source == "grand_k562_grn":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/grand/grand_k562_grn_filt_2.0.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "grand_k562_ppi":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/grand/grand_k562_ppi_all.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_all":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_all.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c2":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c2.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c3":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c3.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c5":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c5.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_c8":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_c8.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "msigdb_h":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/msigdb/msigdb_h.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "htftarget_bone_marrow":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/transcription_factors/htftarget_bone_marrow.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "tflink":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/transcription_factors/tflink_all.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "MCF7":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF164AGX_MCF-7_GRCh38_contact_domains_juicertools.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "T47D":
            networkvi_config["gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF804SET_T47D_GRCh38_contact_domains_juicertools.csv"
            networkvi_config["gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF164AGX":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF164AGX_MCF-7_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF804SET":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF804SET_T47D_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF223RVL":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF223RVL_IMR-90_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF549OBE":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF549OBE_HepG2_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = ""
        elif gene_layer_interaction_source == "ENCFF958WAA":
            networkvi_config[
                "gene_layer_interaction_source"] = "./preprocessing/resources/contact_domains/encode/ENCFF958WAA_HCT116_GRCh38_contact_domains_juicertools.csv"
            networkvi_config[
                "gene_groups"] = ""


    if gaf_source is not None:
        if gaf_source.lower() == "standard":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf")]
        elif gaf_source.lower() == "standard_with_rna":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf")]
        elif gaf_source.lower() == "standard_with_rna_lshuff":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08_level_shuffled.gaf")]
        elif gaf_source.lower() == "standard_with_rna_gshuff":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08_global_shuffled.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npinternc":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npintertar":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_rna_with_npinter":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_npinternc":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf")]
        elif gaf_source.lower() == "standard_with_npintertar":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "standard_with_npinter":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2024-08-08.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_nc.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_npinter_tar_v2.gaf")]
        elif gaf_source.lower() == "go2022":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2022-06-15.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2022-06-15.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2022-06-15.gaf")]
        elif gaf_source.lower() == "go2019":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2019-06-09.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_isoform_ensembl_gene_mapping_2019-06-09.gaf"),
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_rna_ensembl_gene_mapping_2019-06-09.gaf")]
        elif gaf_source.lower() == "go2014":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "goa_human_ensembl_gene_mapping_2014-06-01.gaf")]
        elif gaf_source.lower() == "reactome":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "reactome_human_ensembl_gene_mapping_2026-01-20.gaf")]
        elif gaf_source.lower() == "pcommons":
            networkvi_config["map_ensembl_go"] = [
                os.path.join(os.getcwd(), "preprocessing", "resources", "ebi_ensembl_go_mappings", "pathwaycommons_human_ensembl_gene_mapping.gaf")]
        else:
            networkvi_config["map_ensembl_go"] = [gaf_source]
            networkvi_config["filter_namespace"] = False

    return networkvi_config

def networkvi_model_introspection(model, adata_mvi, categorical_covariate_keys):

    model.view_anndata_setup()

    tensors = {
        "X": torch.zeros((networkvi.settings.batch_size, adata_mvi.shape[1])),
        "batch": torch.zeros((networkvi.settings.batch_size, 1), dtype=torch.int64),
        "patient": torch.zeros((networkvi.settings.batch_size, 1), dtype=torch.int64),
        "ind_x": torch.zeros((networkvi.settings.batch_size, 1), dtype=torch.int64),
        "labels": torch.zeros((networkvi.settings.batch_size, 1), dtype=torch.int64),
        "proteins": torch.zeros((networkvi.settings.batch_size, adata_mvi.obsm['protein_expression'].shape[1])),
        "batch_index": torch.zeros((networkvi.settings.batch_size, 1), dtype=torch.int64),
    }
    if categorical_covariate_keys is not None:
        tensors["extra_categorical_covs"] = torch.zeros((networkvi.settings.batch_size, len(categorical_covariate_keys)), dtype=torch.int64)

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
        model = NETWORKVI.load(name, adata=adata_mvi)

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
        model = NETWORKVI.load(name, adata=adata_mvi)

    available_gene_stable_ids = \
    list(adata_mvi.var[adata_mvi.var["modality"] == "expression"]["gene_stable_id"].to_numpy()) + \
    list(adata_mvi.var[adata_mvi.var["modality"] == "Peaks"]["gene_stable_id"].to_numpy()) + \
    list(adata_mvi.var[adata_mvi.var["modality"] == "SNP"]["gene_stable_id"].to_numpy()) + \
    list(mdata.mod["prot"].var["gene_stable_id"].to_numpy()) if "prot" in mdata.mod else []


    perturbation_gene_ids = []
    perturbation_gene_stable_ids = []
    perturbation_gene_stable_ids = set(available_gene_stable_ids) & set(perturbation_gene_stable_ids)

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
