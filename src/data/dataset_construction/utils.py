#!/usr/bin/env python3
"""
Shared utilities for CD-HIT cluster processing and FASTA file operations.
"""

import pandas as pd
import subprocess
import logging
import re
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_blast_installation():
    """Check if BLAST+ is installed and accessible."""
    try:
        result = subprocess.run(
            ["blastp", "-version"], capture_output=True, text=True, check=True
        )
        logger.info(f"BLAST+ version detected: {result.stdout.split()[1]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("BLAST+ not found. Please install BLAST+ toolkit.")
        logger.error("Installation: conda install -c bioconda blast")
        return False


def sequences_to_fasta(sequences_dict, output_path):
    """Convert sequence dictionary to FASTA file.

    Args:
        sequences_dict (dict): Dictionary of {sequence_id: sequence}
        output_path (str): Path to output FASTA file
    """
    records = []
    for seq_id, sequence in sequences_dict.items():
        record = SeqRecord(Seq(sequence), id=seq_id, description="")
        records.append(record)

    with open(output_path, "w") as f:
        SeqIO.write(records, f, "fasta")

    logger.info(f"Wrote {len(records)} sequences to {output_path}")


def load_fasta_sequences(fasta_file):
    """Load sequences from FASTA file into a dictionary.

    Args:
        fasta_file (str): Path to FASTA file

    Returns:
        dict: Dictionary mapping sequence_id -> sequence
    """
    sequences = {}

    logger.info(f"Loading sequences from FASTA file: {fasta_file}")

    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)

        logger.info(f"Loaded {len(sequences)} sequences from FASTA file")

    except Exception as e:
        logger.error(f"Error reading FASTA file: {e}")
        raise

    return sequences


def parse_cdhit_clusters(cluster_file, mode="representatives_only"):
    """Parse CD-HIT cluster file to extract sequence information.

    Args:
        cluster_file (str): Path to CD-HIT .clstr file
        mode (str): Parsing mode - "representatives_only" or "full_clusters"

    Returns:
        dict: For "representatives_only" mode: {cluster_id: representative_sequence_id}
              For "full_clusters" mode: {cluster_id: {
                  'representative': sequence_id,
                  'all_sequences': [sequence_id1, sequence_id2, ...]
              }}
    """
    if mode == "representatives_only":
        return _parse_cdhit_representatives(cluster_file)
    elif mode == "full_clusters":
        return _parse_cdhit_clusters_full(cluster_file)
    else:
        raise ValueError(f"Unknown parsing mode: {mode}")


def _parse_cdhit_representatives(cluster_file):
    """Parse CD-HIT cluster file to extract only representative sequences.

    Args:
        cluster_file (str): Path to CD-HIT .clstr file

    Returns:
        dict: Dictionary mapping cluster_id -> representative_sequence_id
    """
    representatives = {}
    current_cluster = None

    logger.info(f"Parsing CD-HIT cluster file (representatives only): {cluster_file}")

    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                # Extract cluster number
                current_cluster = line.split()[1]

            elif line and current_cluster is not None:
                # Parse sequence line
                if line.endswith("*"):
                    # This is the representative sequence
                    # Extract sequence ID from the line
                    # Format: "0	50aa, >jgi|Sphst1|249440|gm1.11330_g... *"

                    # Find the sequence ID between ">" and "..."
                    match = re.search(r">([^\.]+)", line)
                    if match:
                        seq_id = match.group(1)
                        representatives[current_cluster] = seq_id
                        logger.debug(
                            f"Cluster {current_cluster}: representative = {seq_id}"
                        )
                    else:
                        logger.warning(
                            f"Could not extract sequence ID from line: {line}"
                        )

    logger.info(f"Found {len(representatives)} representative sequences from clusters")
    return representatives


def _parse_cdhit_clusters_full(cluster_file):
    """Parse CD-HIT cluster file to extract all sequences per cluster.

    Args:
        cluster_file (str): Path to CD-HIT .clstr file

    Returns:
        dict: Dictionary mapping cluster_id -> {
            'representative': sequence_id,
            'all_sequences': [sequence_id1, sequence_id2, ...]
        }
    """
    clusters = {}
    current_cluster = None

    logger.info(f"Parsing CD-HIT cluster file (full clusters): {cluster_file}")

    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                # Extract cluster number
                current_cluster = line.split()[1]
                clusters[current_cluster] = {
                    "representative": None,
                    "all_sequences": [],
                }

            elif line and current_cluster is not None:
                # Parse sequence line
                # Format: "0	50aa, >jgi|Sphst1|249440|gm1.11330_g... *"
                # or:     "1	45aa, >jgi|Sphst1|249441|gm1.11331_g..."

                # Extract sequence ID
                match = re.search(r">([^\.]+)", line)
                if match:
                    seq_id = match.group(1)
                    clusters[current_cluster]["all_sequences"].append(seq_id)

                    # Check if this is the representative (marked with *)
                    if line.endswith("*"):
                        clusters[current_cluster]["representative"] = seq_id
                        logger.debug(
                            f"Cluster {current_cluster}: representative = {seq_id}"
                        )
                else:
                    logger.warning(f"Could not extract sequence ID from line: {line}")

    # Validate clusters
    valid_clusters = {}
    for cluster_id, cluster_info in clusters.items():
        if cluster_info["representative"] and cluster_info["all_sequences"]:
            valid_clusters[cluster_id] = cluster_info
        else:
            logger.warning(
                f"Invalid cluster {cluster_id}: missing representative or sequences"
            )

    logger.info(f"Found {len(valid_clusters)} valid clusters")
    total_sequences = sum(
        len(cluster["all_sequences"]) for cluster in valid_clusters.values()
    )
    logger.info(f"Total sequences across all clusters: {total_sequences}")

    return valid_clusters


def match_sequences_to_fasta(sequence_ids, fasta_sequences, mode="exact"):
    """Match sequence IDs to actual sequences from FASTA.

    Args:
        sequence_ids (list or dict): Sequence IDs to match
        fasta_sequences (dict): Sequences from FASTA file
        mode (str): Matching mode - "exact", "partial", or "representatives"

    Returns:
        dict: Dictionary mapping sequence_id -> sequence for matched sequences
    """
    matched_sequences = {}
    unmatched_ids = []

    logger.info(f"Matching sequences to FASTA sequences (mode: {mode})...")

    # Convert to list if needed
    if isinstance(sequence_ids, dict):
        if mode == "representatives":
            # Extract representative IDs from cluster dict
            seq_id_list = [
                cluster_info["representative"] for cluster_info in sequence_ids.values()
            ]
        else:
            seq_id_list = list(sequence_ids.values())
    else:
        seq_id_list = sequence_ids

    for seq_id in seq_id_list:
        matched_seq = None

        # Try exact match first
        if seq_id in fasta_sequences:
            matched_seq = fasta_sequences[seq_id]
        elif mode in ["partial", "representatives"]:
            # Try partial matching
            for fasta_id in fasta_sequences:
                if fasta_id.startswith(seq_id) or seq_id in fasta_id:
                    matched_seq = fasta_sequences[fasta_id]
                    logger.debug(f"Partial match: {seq_id} -> {fasta_id}")
                    break

        if matched_seq:
            matched_sequences[seq_id] = matched_seq
        else:
            unmatched_ids.append(seq_id)

    logger.info(f"Successfully matched {len(matched_sequences)} sequences")

    if unmatched_ids:
        logger.warning(f"Could not match {len(unmatched_ids)} sequences:")
        for seq_id in unmatched_ids[:10]:  # Show first 10
            logger.warning(f"  - {seq_id}")
        if len(unmatched_ids) > 10:
            logger.warning(f"  ... and {len(unmatched_ids) - 10} more")

    return matched_sequences


def create_csv_for_filtering(
    matched_sequences, output_csv, dataset_name="train", label=0
):
    """Create CSV file compatible with filter_negatives_by_similarity.py.

    Args:
        matched_sequences (dict): Dictionary of sequence_id -> sequence
        output_csv (str): Path to output CSV file
        dataset_name (str): Dataset name (default: "train")
        label (int): Label for sequences (default: 0 for negatives)
    """
    logger.info(f"Creating CSV file: {output_csv}")

    # Create list of records
    csv_records = []
    for seq_id, sequence in matched_sequences.items():
        csv_records.append(
            {
                "sequence_id": seq_id,
                "sequence": sequence,
                "label": label,
                "dataset": dataset_name,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(csv_records)

    # Sort by sequence_id for consistency
    df = df.sort_values("sequence_id").reset_index(drop=True)

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created CSV with {len(df)} sequences")
    logger.info(f"CSV saved to: {output_csv}")

    # Log statistics
    logger.info("\nDataset statistics:")
    logger.info(f"  Total sequences: {len(df)}")
    logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    logger.info(f"  Dataset distribution: {df['dataset'].value_counts().to_dict()}")

    # Show sample sequences
    logger.info("\nSample sequences:")
    for i, row in df.head(3).iterrows():
        seq_preview = (
            row["sequence"][:50] + "..."
            if len(row["sequence"]) > 50
            else row["sequence"]
        )
        logger.info(f"  {row['sequence_id']}: {seq_preview}")

    return df
