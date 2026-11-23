import os
import requests
import gzip
import shutil
import zipfile
import tarfile
import pandas as pd
from pybiomart import Server
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_file(url: str, output_path: str, extract: bool = False) -> str:
    """
    Download a file from URL to output_path.

    Args:
        url: URL to download from
        output_path: Path to save the file
        extract: If True, extract compressed files

    Returns:
        Path to the final file (extracted if applicable)
    """
    print(f"Downloading {url}...")
    ensure_directory(os.path.dirname(output_path))

    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()

    with open(output_path, 'wb') as f:
        f.write(r.content)

    if not extract:
        return output_path

    if output_path.endswith('.gz') and not output_path.endswith('.tar.gz'):
        extracted_path = output_path[:-3]
        with gzip.open(output_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return extracted_path

    elif output_path.endswith('.zip'):
        extract_dir = output_path[:-4]
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir

    elif output_path.endswith('.tar.gz') or output_path.endswith('.tgz'):
        extract_dir = output_path.replace('.tar.gz', '').replace('.tgz', '')
        with tarfile.open(output_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        return extract_dir

    return output_path


class EnsemblMapper:
    """Handle Ensembl gene ID mappings using BioMart."""

    def __init__(self):
        self.server = Server(host='http://www.ensembl.org')
        self.dataset = self.server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
        self._cache = {}

    def get_gene_id_to_ensembl_mapping(self, id_type: str) -> pd.DataFrame:
        """
        Get mapping from various ID types to Ensembl gene IDs.

        Args:
            id_type: Type of ID to map from. Options:
                - 'entrezgene_id': NCBI Gene ID
                - 'uniprotswissprot': UniProt/Swiss-Prot ID
                - 'ensembl_peptide_id': Ensembl Peptide ID
                - 'hgnc_symbol': HGNC gene symbol

        Returns:
            DataFrame with columns for the ID type and 'ensembl_gene_id'
        """
        cache_key = id_type
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        print(f"Fetching {id_type} to Ensembl mapping from BioMart...")

        mapping = self.dataset.query(attributes=['ensembl_gene_id', id_type])
        mapping = mapping[~mapping[mapping.columns[1]].isnull()]

        self._cache[cache_key] = mapping
        return mapping.copy()

    def get_ensembl_gene_positions(self) -> pd.DataFrame:
        """Get Ensembl gene IDs with chromosome positions."""
        cache_key = 'positions'
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        print("Fetching gene positions from BioMart...")

        positions = self.dataset.query(
            attributes=['ensembl_gene_id', 'chromosome_name', 'start_position', 'end_position']
        )

        self._cache[cache_key] = positions
        return positions.copy()

    def map_ids_to_ensembl(self, ids: List[str], id_type: str) -> Dict[str, str]:
        """
        Map a list of IDs to Ensembl gene IDs.

        Args:
            ids: List of IDs to map
            id_type: Type of the input IDs

        Returns:
            Dictionary mapping input IDs to Ensembl gene IDs
        """
        mapping_df = self.get_gene_id_to_ensembl_mapping(id_type)

        lookup = {}
        for _, row in mapping_df.iterrows():
            source_id = str(row[mapping_df.columns[1]])
            ensembl_id = row['Gene stable ID']
            lookup[source_id] = ensembl_id

        return lookup


def create_gene_interaction_df(gene_pairs: List[Tuple[str, str]],
                               scores: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Create a standardized gene interaction DataFrame.

    Args:
        gene_pairs: List of (gene1, gene2) tuples
        scores: Optional list of interaction scores

    Returns:
        DataFrame with columns: gene1, gene2, combined_score
    """
    if scores is None:
        scores = [1] * len(gene_pairs)

    df = pd.DataFrame({
        'gene1': [pair[0] for pair in gene_pairs],
        'gene2': [pair[1] for pair in gene_pairs],
        'combined_score': scores
    })

    df = df.drop_duplicates(subset=['gene1', 'gene2'], keep='first')
    df = df.reset_index(drop=True)

    return df


def create_bidirectional_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bidirectional interactions from a unidirectional DataFrame.

    Args:
        df: DataFrame with gene1, gene2, combined_score

    Returns:
        DataFrame with both directions of each interaction
    """
    df_reverse = df.copy()
    df_reverse['gene1'], df_reverse['gene2'] = df['gene2'].copy(), df['gene1'].copy()

    df_bidirectional = pd.concat([df, df_reverse], ignore_index=True)
    df_bidirectional = df_bidirectional.drop_duplicates(subset=['gene1', 'gene2'], keep='first')
    df_bidirectional = df_bidirectional.reset_index(drop=True)

    return df_bidirectional


def filter_human_only(df: pd.DataFrame, organism_column: str) -> pd.DataFrame:
    """
    Filter DataFrame to keep only Homo sapiens entries.

    Args:
        df: Input DataFrame
        organism_column: Name of the column containing organism information

    Returns:
        Filtered DataFrame
    """
    return df[df[organism_column] == "Homo sapiens"].copy()


def save_interaction_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save interaction DataFrame to CSV.

    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    ensure_directory(os.path.dirname(output_path))
    df.to_csv(output_path, index=True)
    print(f"Saved {len(df)} interactions to {output_path}")