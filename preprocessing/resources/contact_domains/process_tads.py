import os
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from pathlib import Path
from tad_utils import download_file, EnsemblMapper, ensure_directory

CHROMOSOMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
               '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'MT', 'X', 'Y']


def download_encode_tad_file(accession: str, cell_line: str, data_type: str,
                             output_dir: str = "data/tads/encode") -> str:
    """
    Download ENCODE TAD file.

    Args:
        accession: ENCODE accession ID (e.g., 'ENCFF203AKP')
        cell_line: Cell line name (e.g., 'GM12878', 'K562', 'MCF-7')
        data_type: Type of data ('contact_domains' or 'loops')
        output_dir: Output directory

    Returns:
        Path to downloaded file
    """
    ensure_directory(output_dir)

    url = f"https://www.encodeproject.org/files/{accession}/@@download/{accession}.bedpe.gz"
    filename = f"{accession}_{cell_line}_GRCh38_{data_type}_juicertools.bedpe"
    gz_path = os.path.join(output_dir, filename + ".gz")

    txt_path = download_file(url, gz_path, extract=True)
    return txt_path


def load_tad_file(tad_file_path: str) -> pd.DataFrame:
    """Load TAD file in BEDPE format."""
    tad_list = pd.read_csv(tad_file_path, delimiter="\t", header=None, skiprows=2)
    tad_list = tad_list[[0, 1, 2]]
    tad_list[0] = tad_list[0].apply(lambda x: x.split("chr")[-1])
    return tad_list


def generate_shifted_tad_regions(tad_list: pd.DataFrame) -> pd.DataFrame:
    """Generate shifted TAD regions for benchmark/control."""
    length_tads = []
    for _, row in tad_list.iterrows():
        if row[0] in CHROMOSOMES:
            length_tads.append(row[2] - row[1])

    mean_length = int(np.mean(length_tads))

    tad_list_shifted = tad_list.copy()
    tad_list_shifted[1] = tad_list_shifted[1] - mean_length
    tad_list_shifted[2] = tad_list_shifted[2] - mean_length

    return tad_list_shifted


def generate_tad_gene_dict(tad_list: pd.DataFrame,
                           gene_positions: pd.DataFrame) -> dict:
    """
    Map TADs to genes based on genomic positions.

    Args:
        tad_list: DataFrame with TAD regions
        gene_positions: DataFrame with gene positions from Ensembl

    Returns:
        Dictionary mapping chromosomes to lists of gene sets in TADs
    """
    tad_gene_dict = {chromosome: [] for chromosome in CHROMOSOMES}

    for _, row in tqdm(tad_list.iterrows(), total=len(tad_list), desc="Processing TADs"):
        chromosome_genes = gene_positions[
            gene_positions["Chromosome/scaffold name"] == row[0]
            ]

        overlapping_genes = chromosome_genes[
            (chromosome_genes["Gene end (bp)"].between(row[1], row[2])) |
            (chromosome_genes["Gene start (bp)"].between(row[1], row[2]))
            ]["Gene stable ID"].tolist()

        if len(overlapping_genes) > 1:
            tad_gene_dict[row[0]].append(overlapping_genes)

    return tad_gene_dict


def generate_tad_gene_groups(tad_gene_dict: dict) -> np.ndarray:
    """Convert TAD gene dictionary to array of gene sets."""
    gene_groups = [
        set(gene_group)
        for chromosome in tad_gene_dict.keys()
        for gene_group in tad_gene_dict[chromosome]
    ]
    return np.array(gene_groups, dtype=object)


def generate_tad_interactions(tad_gene_dict: dict) -> pd.DataFrame:
    """
    Generate gene-gene interactions from TAD regions.

    Creates bidirectional interactions between all gene pairs within each TAD.
    """
    gene1_list = []
    gene2_list = []

    for chromosome in tad_gene_dict.keys():
        for tad in tad_gene_dict[chromosome]:
            for gene_a, gene_b in combinations(tad, 2):
                gene1_list.extend([gene_a, gene_b])
                gene2_list.extend([gene_b, gene_a])

    df = pd.DataFrame({
        'gene1': gene1_list,
        'gene2': gene2_list,
        'combined_score': [1] * len(gene1_list),
    })

    df = df.drop_duplicates(keep="first")
    df = df.reset_index(drop=True)

    return df

def shuffle_gene_interactions(
    interactions: pd.DataFrame,
    shuffle="gene2",
    seed: int = 42
) -> pd.DataFrame:
    """
    Randomize gene-gene pairs reproducibly.

    Args:
        interactions:
            DataFrame with columns:
            ['gene1', 'gene2', 'combined_score']

        shuffle:
            Which columns to shuffle:
            - "gene1"
            - "gene2"
            - "both"

        seed:
            Random seed

    Returns:
        Shuffled interaction DataFrame
    """
    shuffled = interactions.copy()

    rng = np.random.default_rng(seed)

    if shuffle in ["gene1", "both"]:
        shuffled["gene1"] = rng.permutation(shuffled["gene1"].values)

    if shuffle in ["gene2", "both"]:
        shuffled["gene2"] = rng.permutation(shuffled["gene2"].values)

    return shuffled.reset_index(drop=True)

def process_tad_cell_line(tad_files: list, cell_line: str,
                          output_dir: str = "data/tads/encode") -> None:
    """
    Process all TAD files for a specific cell line.

    Args:
        tad_files: List of tuples (accession, data_type)
        cell_line: Cell line name
        output_dir: Output directory
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {cell_line} TAD data")
    print(f"{'=' * 60}")

    ensure_directory(output_dir)

    tad_file_paths = []
    for accession, data_type in tad_files:
        path = download_encode_tad_file(accession, cell_line, data_type, output_dir)
        tad_file_paths.append(path)

    tad_lists = [load_tad_file(path) for path in tad_file_paths]

    tad_lists_shifted = [generate_shifted_tad_regions(tad) for tad in tad_lists]

    mapper = EnsemblMapper()
    gene_positions = mapper.get_ensembl_gene_positions()

    for i, (tad_file_path, tad_list, tad_list_shifted) in enumerate(
            zip(tad_file_paths, tad_lists, tad_lists_shifted)
    ):
        file_stem = Path(tad_file_path).stem
        print(f"\nProcessing {file_stem}...")

        tad_gene_dict = generate_tad_gene_dict(tad_list, gene_positions)
        tad_gene_dict_shifted = generate_tad_gene_dict(tad_list_shifted, gene_positions)

        tad_gene_groups = generate_tad_gene_groups(tad_gene_dict)
        tad_gene_groups_shifted = generate_tad_gene_groups(tad_gene_dict_shifted)

        tad_interactions = generate_tad_interactions(tad_gene_dict)
        tad_interactions_shifted = generate_tad_interactions(tad_gene_dict_shifted)
        tad_interactions_shuffled = shuffle_gene_interactions(
            tad_interactions,
            shuffle="gene2",
            seed=42
        )

        tad_interactions.to_csv(os.path.join(output_dir, f"{file_stem}.csv"))
        tad_interactions_shifted.to_csv(os.path.join(output_dir, f"{file_stem}_shifted_benchmark.csv"))
        tad_interactions_shuffled.to_csv(os.path.join(output_dir, f"{file_stem}_shuffled.csv"))
        np.save(os.path.join(output_dir, f"{file_stem}_gene_groups.npy"), tad_gene_groups)
        np.save(os.path.join(output_dir, f"{file_stem}_gene_groups_shifted_benchmark.npy"), tad_gene_groups_shifted)

        print(f"  Saved {len(tad_interactions)} interactions")

    print(f"\nGenerating consensus for {cell_line}...")
    all_interactions = [generate_tad_interactions(
        generate_tad_gene_dict(tad, gene_positions)
    ) for tad in tad_lists]
    all_interactions_shifted = [generate_tad_interactions(
        generate_tad_gene_dict(tad, gene_positions)
    ) for tad in tad_lists_shifted]

    consensus = pd.concat(all_interactions, ignore_index=True)
    consensus = consensus.drop_duplicates(keep="first").reset_index(drop=True)

    consensus_shifted = pd.concat(all_interactions_shifted, ignore_index=True)
    consensus_shifted = consensus_shifted.drop_duplicates(keep="first").reset_index(drop=True)

    consensus_name = "_".join([Path(p).stem for p in tad_file_paths])
    consensus.to_csv(os.path.join(output_dir, f"{consensus_name}_consensus.csv"))
    consensus_shifted.to_csv(
        os.path.join(output_dir, f"{consensus_name}_consensus_shifted_benchmark.csv")
    )

    print(f"  Saved consensus with {len(consensus)} interactions")


if __name__ == "__main__":

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode")
    process_tad_cell_line([("ENCFF173VDJ", "contact_domains")], "K562", output_dir=output_dir)

