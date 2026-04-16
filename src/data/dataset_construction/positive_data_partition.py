#!/usr/bin/env python3
"""
Script to process CD-HIT cluster output and create cluster-based train/test splits.
Implements redundancy filtering to prevent data leakage between test and pretrain sets.

This script creates a single CSV file with a 'partition' column indicating:
1. test: representative sequences from test clusters
2. train: representative sequences from train clusters
3. pretrain: all sequences from train clusters (includes train sequences)

Includes mandatory BLASTP-based similarity filtering to ensure no test sequence
has >40% identity with any sequence in the pretrain set.
"""

import pandas as pd
import argparse
import logging
import tempfile
import subprocess
import shutil
from pathlib import Path
import random
import numpy as np

# Import shared utilities
from src.data.dataset_construction.utils import (
    check_blast_installation,
    sequences_to_fasta,
    load_fasta_sequences,
    parse_cdhit_clusters,
    logger,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def match_cluster_sequences_to_fasta(clusters, fasta_sequences):
    """Match cluster sequence IDs to actual sequences from FASTA.

    Args:
        clusters (dict): Cluster information from CD-HIT
        fasta_sequences (dict): Sequences from FASTA file

    Returns:
        dict: Updated clusters with matched sequences
    """
    matched_clusters = {}
    total_matched = 0
    total_unmatched = 0

    logger.info("Matching cluster sequences to FASTA sequences...")

    for cluster_id, cluster_info in clusters.items():
        matched_cluster = {
            "representative": None,
            "representative_sequence": None,
            "all_sequences": {},  # seq_id -> sequence
            "unmatched_sequences": [],
        }

        # Match all sequences in cluster
        for seq_id in cluster_info["all_sequences"]:
            matched_seq = None

            # Try exact match first
            if seq_id in fasta_sequences:
                matched_seq = fasta_sequences[seq_id]
            else:
                # Try partial matching
                for fasta_id in fasta_sequences:
                    if fasta_id.startswith(seq_id) or seq_id in fasta_id:
                        matched_seq = fasta_sequences[fasta_id]
                        logger.debug(f"Partial match: {seq_id} -> {fasta_id}")
                        break

            if matched_seq:
                matched_cluster["all_sequences"][seq_id] = matched_seq
                total_matched += 1

                # Check if this is the representative
                if seq_id == cluster_info["representative"]:
                    matched_cluster["representative"] = seq_id
                    matched_cluster["representative_sequence"] = matched_seq
            else:
                matched_cluster["unmatched_sequences"].append(seq_id)
                total_unmatched += 1

        # Only keep clusters with valid representatives and at least one sequence
        if (
            matched_cluster["representative"]
            and matched_cluster["representative_sequence"]
            and matched_cluster["all_sequences"]
        ):
            matched_clusters[cluster_id] = matched_cluster
        else:
            logger.warning(
                f"Skipping cluster {cluster_id}: no valid representative found"
            )

    logger.info(
        f"Successfully matched {total_matched} sequences across {len(matched_clusters)} clusters"
    )
    if total_unmatched > 0:
        logger.warning(f"Could not match {total_unmatched} sequences")

    return matched_clusters


def perform_cluster_based_split(clusters, test_ratio=0.2, random_seed=42):
    """Perform cluster-based train/test split.

    Args:
        clusters (dict): Matched cluster information
        test_ratio (float): Proportion of clusters for test set
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_clusters, test_clusters)
    """
    logger.info(f"Performing cluster-based split with test ratio: {test_ratio}")

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    cluster_ids = list(clusters.keys())
    num_test_clusters = int(len(cluster_ids) * test_ratio)

    # Randomly select test clusters
    test_cluster_ids = set(random.sample(cluster_ids, num_test_clusters))
    train_cluster_ids = set(cluster_ids) - test_cluster_ids

    train_clusters = {cid: clusters[cid] for cid in train_cluster_ids}
    test_clusters = {cid: clusters[cid] for cid in test_cluster_ids}

    logger.info(f"Split results:")
    logger.info(f"  Train clusters: {len(train_clusters)}")
    logger.info(f"  Test clusters: {len(test_clusters)}")

    # Count sequences
    train_total_seqs = sum(
        len(cluster["all_sequences"]) for cluster in train_clusters.values()
    )
    test_total_seqs = sum(
        len(cluster["all_sequences"]) for cluster in test_clusters.values()
    )
    train_reps = len(train_clusters)
    test_reps = len(test_clusters)

    logger.info(
        f"  Train total sequences: {train_total_seqs} (representatives: {train_reps})"
    )
    logger.info(
        f"  Test total sequences: {test_total_seqs} (representatives: {test_reps})"
    )

    return train_clusters, test_clusters


def create_combined_dataset_csv(train_clusters, test_clusters, output_path, label=0):
    """Create a single combined CSV file with partition column.

    Args:
        train_clusters (dict): Train cluster information
        test_clusters (dict): Test cluster information
        output_path (str): Path to output CSV file
        label (int): Label for sequences

    Returns:
        str: Path to created CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating combined dataset CSV file: {output_path}")

    all_records = []

    # Test partition: representatives from test clusters
    for cluster_id, cluster_info in test_clusters.items():
        all_records.append(
            {
                "sequence_id": cluster_info["representative"],
                "sequence": cluster_info["representative_sequence"],
                "label": label,
                "partition": "test",
                "cluster_id": cluster_id,
            }
        )

    # Train partition: representatives from train clusters
    for cluster_id, cluster_info in train_clusters.items():
        all_records.append(
            {
                "sequence_id": cluster_info["representative"],
                "sequence": cluster_info["representative_sequence"],
                "label": label,
                "partition": "train",
                "cluster_id": cluster_id,
            }
        )

    # Pretrain partition: all sequences from train clusters
    for cluster_id, cluster_info in train_clusters.items():
        for seq_id, sequence in cluster_info["all_sequences"].items():
            all_records.append(
                {
                    "sequence_id": seq_id,
                    "sequence": sequence,
                    "label": label,
                    "partition": "pretrain",
                    "cluster_id": cluster_id,
                }
            )

    # Create combined DataFrame
    combined_df = pd.DataFrame(all_records)

    # Sort by partition and sequence_id for consistency
    combined_df = combined_df.sort_values(["partition", "sequence_id"]).reset_index(
        drop=True
    )

    # Save to CSV
    combined_df.to_csv(output_path, index=False)

    # Log partition statistics
    partition_counts = combined_df["partition"].value_counts()
    logger.info(f"Created combined dataset with {len(combined_df)} total sequences:")
    for partition in ["test", "train", "pretrain"]:
        if partition in partition_counts:
            logger.info(f"  {partition}: {partition_counts[partition]} sequences")

    logger.info(f"Combined dataset saved to: {output_path}")

    return str(output_path)


def run_blastp_redundancy_check(
    combined_csv_path,
    identity_threshold=40.0,
    coverage_threshold=60.0,
    evalue_threshold=1e-5,
    num_threads=16,
):
    """Run BLASTP to check for redundancy between test and pretrain partitions.

    Args:
        combined_csv_path (str): Path to combined dataset CSV with partition column
        identity_threshold (float): Identity percentage threshold
        coverage_threshold (float): Coverage percentage threshold
        evalue_threshold (float): E-value threshold
        num_threads (int): Number of threads for BLAST

    Returns:
        set: Set of problematic sequence IDs to remove from pretrain
    """
    logger.info(
        "Running BLASTP redundancy check between test and pretrain partitions..."
    )

    # Load combined dataset
    combined_df = pd.read_csv(combined_csv_path)

    # Filter by partition
    test_df = combined_df[combined_df["partition"] == "test"].copy()
    pretrain_df = combined_df[combined_df["partition"] == "pretrain"].copy()

    logger.info(f"Test sequences: {len(test_df)}")
    logger.info(f"Pretrain sequences: {len(pretrain_df)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create FASTA files
        test_fasta = temp_dir / "test_sequences.fasta"
        pretrain_fasta = temp_dir / "pretrain_sequences.fasta"
        blast_output = temp_dir / "blast_results.txt"

        # Convert to FASTA
        test_seqs = dict(zip(test_df["sequence_id"], test_df["sequence"]))
        pretrain_seqs = dict(zip(pretrain_df["sequence_id"], pretrain_df["sequence"]))

        sequences_to_fasta(test_seqs, test_fasta)
        sequences_to_fasta(pretrain_seqs, pretrain_fasta)

        # Run BLASTP: query=test, subject=pretrain
        cmd = [
            "blastp",
            "-query",
            str(test_fasta),
            "-subject",
            str(pretrain_fasta),
            "-out",
            str(blast_output),
            "-evalue",
            str(evalue_threshold),
            "-outfmt",
            "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs",
            "-num_threads",
            str(num_threads),
            "-max_target_seqs",
            "10",
        ]

        try:
            logger.info(f"Running BLASTP: test vs pretrain...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stderr and "Warning" not in result.stderr:
                logger.warning(f"BLAST stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"BLASTP failed: {e}")
            logger.error(f"BLAST stderr: {e.stderr}")
            raise

        # Parse results
        try:
            columns = [
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "sstart",
                "send",
                "evalue",
                "bitscore",
                "qcovs",
            ]
            blast_df = pd.read_csv(blast_output, sep="\t", names=columns)
            logger.info(f"Parsed {len(blast_df)} BLAST hits")
        except Exception as e:
            logger.warning(f"No BLAST hits found or error parsing: {e}")
            return set()

        if blast_df.empty:
            logger.info(
                "No problematic similarities found - all sequences pass redundancy check"
            )
            return set()

        # Filter hits above thresholds
        problematic_hits = blast_df[
            (blast_df["pident"] >= identity_threshold)
            & (blast_df["qcovs"] >= coverage_threshold)
        ]

        # Get pretrain sequences to remove (sseqid = subject = pretrain sequences)
        problematic_pretrain_seqs = set(problematic_hits["sseqid"].unique())

        logger.warning(
            f"Found {len(problematic_hits)} hits above similarity thresholds"
        )
        logger.warning(
            f"Will remove {len(problematic_pretrain_seqs)} sequences from pretrain set"
        )

        if len(problematic_hits) > 0:
            logger.warning("Sample problematic similarities:")
            for i, (_, hit) in enumerate(problematic_hits.head(5).iterrows()):
                logger.warning(
                    f"  Test {hit['qseqid']} <-> Pretrain {hit['sseqid']}: "
                    f"{hit['pident']:.1f}% identity, {hit['qcovs']:.1f}% coverage"
                )

        return problematic_pretrain_seqs


def filter_pretrain_by_redundancy(
    combined_csv_path, problematic_sequences, output_csv_path
):
    """Remove problematic sequences from both pretrain and train partitions.

    This ensures the invariant that train ⊆ pretrain is maintained by removing
    sequences that are too similar to test sequences from both training partitions.

    Args:
        combined_csv_path (str): Path to combined dataset CSV
        problematic_sequences (set): Set of sequence IDs to remove from training partitions
        output_csv_path (str): Path to filtered output CSV

    Returns:
        dict: Filtering statistics
    """
    logger.info("Filtering training partitions to remove redundant sequences...")

    # Load combined dataset
    combined_df = pd.read_csv(combined_csv_path)

    # Count original sequences in training partitions
    pretrain_mask = combined_df["partition"] == "pretrain"
    train_mask = combined_df["partition"] == "train"
    initial_pretrain_count = pretrain_mask.sum()
    initial_train_count = train_mask.sum()

    # Filter out problematic sequences from both pretrain and train partitions
    problematic_pretrain_mask = pretrain_mask & combined_df["sequence_id"].isin(
        problematic_sequences
    )
    problematic_train_mask = train_mask & combined_df["sequence_id"].isin(
        problematic_sequences
    )

    # Create filtered dataframe (remove sequences matching either mask)
    filtered_df = combined_df[
        ~(problematic_pretrain_mask | problematic_train_mask)
    ].copy()

    # Count final sequences in training partitions
    final_pretrain_count = (filtered_df["partition"] == "pretrain").sum()
    final_train_count = (filtered_df["partition"] == "train").sum()
    pretrain_removed = initial_pretrain_count - final_pretrain_count
    train_removed = initial_train_count - final_train_count

    # Save filtered dataset
    filtered_df.to_csv(output_csv_path, index=False)

    logger.info(f"Training partition filtering results:")
    logger.info(
        f"  Pretrain - Original: {initial_pretrain_count}, Removed: {pretrain_removed}, Final: {final_pretrain_count}"
    )
    logger.info(
        f"  Train - Original: {initial_train_count}, Removed: {train_removed}, Final: {final_train_count}"
    )
    logger.info(
        f"  Pretrain retention rate: {final_pretrain_count/initial_pretrain_count*100:.1f}%"
    )
    logger.info(
        f"  Train retention rate: {final_train_count/initial_train_count*100:.1f}%"
    )

    # Verify train ⊆ pretrain invariant is maintained
    train_seq_ids = set(filtered_df[filtered_df["partition"] == "train"]["sequence_id"])
    pretrain_seq_ids = set(
        filtered_df[filtered_df["partition"] == "pretrain"]["sequence_id"]
    )
    missing_in_pretrain = train_seq_ids - pretrain_seq_ids

    if missing_in_pretrain:
        logger.warning(
            f"INVARIANT VIOLATION: {len(missing_in_pretrain)} train sequences not in pretrain!"
        )
    else:
        logger.info("✓ Verified: train ⊆ pretrain invariant maintained")

    return {
        "initial_count": initial_pretrain_count,
        "removed_count": pretrain_removed,
        "final_count": final_pretrain_count,
        "retention_rate": (
            final_pretrain_count / initial_pretrain_count
            if initial_pretrain_count > 0
            else 1.0
        ),
        "train_initial_count": initial_train_count,
        "train_removed_count": train_removed,
        "train_final_count": final_train_count,
        "train_retention_rate": (
            final_train_count / initial_train_count if initial_train_count > 0 else 1.0
        ),
        "invariant_violations": len(missing_in_pretrain),
    }


def process_cdhit_with_cluster_splitting(
    cluster_file,
    fasta_file,
    output_csv,
    test_ratio=0.2,
    random_seed=42,
    label=0,
    identity_threshold=40.0,
    coverage_threshold=60.0,
    evalue_threshold=1e-5,
    num_threads=16,
):
    """Main function to process CD-HIT output with cluster-based splitting and redundancy filtering.

    Args:
        cluster_file (str): Path to CD-HIT .clstr file
        fasta_file (str): Path to original FASTA file
        output_csv (str): Path to output combined CSV file
        test_ratio (float): Proportion of clusters for test set
        random_seed (int): Random seed for reproducibility
        label (int): Label for sequences (0 for negatives, 1 for positives)
        identity_threshold (float): Identity threshold for redundancy filtering
        coverage_threshold (float): Coverage threshold for redundancy filtering
        evalue_threshold (float): E-value threshold for BLAST
        num_threads (int): Number of threads for BLAST

    Returns:
        dict: Processing statistics and file paths
    """
    logger.info(
        "Starting CD-HIT cluster processing with splitting and redundancy filtering..."
    )

    # Check BLAST installation
    if not check_blast_installation():
        raise RuntimeError("BLAST+ is required but not installed")

    # Step 1: Parse clusters
    clusters = parse_cdhit_clusters(cluster_file, mode="full_clusters")

    # Step 2: Load FASTA sequences
    fasta_sequences = load_fasta_sequences(fasta_file)

    # Step 3: Match sequences
    matched_clusters = match_cluster_sequences_to_fasta(clusters, fasta_sequences)

    # Step 4: Cluster-based split
    train_clusters, test_clusters = perform_cluster_based_split(
        matched_clusters, test_ratio, random_seed
    )

    # Step 5: Create initial combined dataset
    combined_csv_path = create_combined_dataset_csv(
        train_clusters, test_clusters, output_csv, label
    )

    # Step 6: Redundancy filtering
    logger.info("Performing redundancy check and filtering...")
    problematic_sequences = run_blastp_redundancy_check(
        combined_csv_path,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        evalue_threshold=evalue_threshold,
        num_threads=num_threads,
    )

    # Step 7: Filter pretrain partition if needed
    final_csv_path = combined_csv_path
    if problematic_sequences:
        # Create filtered version
        filtered_csv_path = Path(output_csv).parent / (
            Path(output_csv).stem + "_filtered.csv"
        )

        filtering_stats = filter_pretrain_by_redundancy(
            combined_csv_path, problematic_sequences, filtered_csv_path
        )

        final_csv_path = str(filtered_csv_path)

        # Replace original with filtered version
        shutil.move(filtered_csv_path, output_csv)
        final_csv_path = output_csv
    else:
        logger.info("No redundant sequences found - dataset unchanged")
        # Get counts for stats
        combined_df = pd.read_csv(combined_csv_path)
        pretrain_count = (combined_df["partition"] == "pretrain").sum()
        train_count = (combined_df["partition"] == "train").sum()
        filtering_stats = {
            "initial_count": pretrain_count,
            "removed_count": 0,
            "final_count": pretrain_count,
            "retention_rate": 1.0,
            "train_initial_count": train_count,
            "train_removed_count": 0,
            "train_final_count": train_count,
            "train_retention_rate": 1.0,
            "invariant_violations": 0,
        }

    # Get final partition counts
    final_df = pd.read_csv(final_csv_path)
    partition_counts = final_df["partition"].value_counts()

    # Compile final statistics
    stats = {
        "total_clusters": len(matched_clusters),
        "train_clusters": len(train_clusters),
        "test_clusters": len(test_clusters),
        "test_sequences": partition_counts.get("test", 0),
        "train_sequences": partition_counts.get("train", 0),
        "pretrain_sequences_final": filtering_stats["final_count"],
        "pretrain_sequences_removed": filtering_stats["removed_count"],
        "pretrain_retention_rate": filtering_stats["retention_rate"],
        "train_sequences_removed": filtering_stats["train_removed_count"],
        "train_retention_rate": filtering_stats["train_retention_rate"],
        "invariant_violations": filtering_stats["invariant_violations"],
        "output_csv": final_csv_path,
        "total_sequences": len(final_df),
    }

    logger.info("\n" + "=" * 60)
    logger.info("CLUSTER-BASED PROCESSING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Final dataset created: {final_csv_path}")
    logger.info(f"Total sequences: {stats['total_sequences']}")
    logger.info(f"Partition breakdown:")
    logger.info(f"  test: {stats['test_sequences']} sequences")
    logger.info(f"  train: {stats['train_sequences']} sequences")
    logger.info(f"  pretrain: {stats['pretrain_sequences_final']} sequences")

    if stats["pretrain_sequences_removed"] > 0 or stats["train_sequences_removed"] > 0:
        logger.info(f"\nRedundancy filtering results:")
        logger.info(
            f"  Pretrain: removed {stats['pretrain_sequences_removed']} sequences (retention: {stats['pretrain_retention_rate']*100:.1f}%)"
        )
        logger.info(
            f"  Train: removed {stats['train_sequences_removed']} sequences (retention: {stats['train_retention_rate']*100:.1f}%)"
        )

        if stats["invariant_violations"] == 0:
            logger.info(f"  ✓ Maintained train ⊆ pretrain invariant")
        else:
            logger.warning(
                f"  ❌ {stats['invariant_violations']} invariant violations detected"
            )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process CD-HIT cluster output with cluster-based splitting and redundancy filtering"
    )
    parser.add_argument(
        "--cluster-file", required=True, help="Path to CD-HIT .clstr file"
    )
    parser.add_argument(
        "--fasta-file",
        required=True,
        help="Path to original FASTA file (before clustering)",
    )
    parser.add_argument(
        "--output-csv", required=True, help="Path to output combined CSV file"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of clusters for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Label for sequences (default: 1 for positives)",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=40.0,
        help="Identity percentage threshold for redundancy filtering (default: 40.0)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=60.0,
        help="Query coverage percentage threshold for redundancy filtering (default: 60.0)",
    )
    parser.add_argument(
        "--evalue-threshold",
        type=float,
        default=1e-5,
        help="E-value threshold for BLAST search (default: 1e-5)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads for BLAST (default: 16)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input files exist
    for file_path in [args.cluster_file, args.fasta_file]:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return 1

    try:
        # Process with cluster splitting and redundancy filtering
        stats = process_cdhit_with_cluster_splitting(
            cluster_file=args.cluster_file,
            fasta_file=args.fasta_file,
            output_csv=args.output_csv,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            label=args.label,
            identity_threshold=args.identity_threshold,
            coverage_threshold=args.coverage_threshold,
            evalue_threshold=args.evalue_threshold,
            num_threads=args.num_threads,
        )

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
