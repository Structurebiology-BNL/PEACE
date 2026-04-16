#!/usr/bin/env python3
"""
Script to filter negative samples by checking similarity against positive samples.
Uses BLASTP to identify and remove negatives that are too similar to positives,
preventing label noise in the training dataset.

Supports two modes:
1. existing: Filter new negatives against positives from an existing mixed dataset
2. separate: Filter negatives against positives from separate CSV files
"""

import pandas as pd
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path
import shutil

# Import shared utilities
from .utils import check_blast_installation, sequences_to_fasta, logger

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_blastp_search(
    query_fasta, subject_fasta, output_file, evalue=1e-5, num_threads=4
):
    """Run BLASTP search of query sequences against subject sequences.

    Args:
        query_fasta (str): Path to query FASTA file (new negatives)
        subject_fasta (str): Path to subject FASTA file (existing positives)
        output_file (str): Path to output file
        evalue (float): E-value threshold for BLAST search
        num_threads (int): Number of threads to use

    Returns:
        bool: True if BLAST search successful
    """
    # Convert paths to strings to handle both str and Path objects
    cmd = [
        "blastp",
        "-query",
        str(query_fasta),
        "-subject",
        str(subject_fasta),
        "-out",
        str(output_file),
        "-evalue",
        str(evalue),
        "-outfmt",
        "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs",
        "-num_threads",
        str(num_threads),
        "-max_target_seqs",
        "10",  # Keep top 10 hits per query
    ]

    try:
        logger.info(f"Running BLASTP search...")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stderr and "Warning" not in result.stderr:
            logger.warning(f"BLAST stderr: {result.stderr}")

        logger.info(f"BLASTP search completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"BLASTP search failed: {e}")
        logger.error(f"BLAST stderr: {e.stderr}")
        return False


def parse_blast_results(
    blast_output_file, identity_threshold=30.0, coverage_threshold=50.0
):
    """Parse BLAST results and identify problematic sequences.

    Args:
        blast_output_file (str): Path to BLAST output file
        identity_threshold (float): Minimum identity percentage to flag as problematic
        coverage_threshold (float): Minimum query coverage percentage to flag as problematic

    Returns:
        tuple: (problematic_sequences, blast_df)
            - problematic_sequences (set): Set of query sequence IDs to remove
            - blast_df (pd.DataFrame): Full BLAST results dataframe
    """
    # BLAST output format 6 columns
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

    try:
        blast_df = pd.read_csv(blast_output_file, sep="\t", names=columns)
        logger.info(f"Parsed {len(blast_df)} BLAST hits")
    except Exception as e:
        logger.error(f"Error parsing BLAST results: {e}")
        return set(), pd.DataFrame()

    if blast_df.empty:
        logger.info("No BLAST hits found - all sequences pass similarity filter")
        return set(), blast_df

    # Filter based on identity and coverage thresholds
    problematic_hits = blast_df[
        (blast_df["pident"] >= identity_threshold)
        & (blast_df["qcovs"] >= coverage_threshold)
    ]

    problematic_sequences = set(problematic_hits["qseqid"].unique())

    logger.info(f"Found {len(problematic_hits)} hits above thresholds")
    logger.info(
        f"Identified {len(problematic_sequences)} problematic sequences to remove"
    )

    if len(problematic_sequences) > 0:
        logger.info("Sample of problematic sequences:")
        for i, (_, hit) in enumerate(problematic_hits.head(5).iterrows()):
            logger.info(
                f"  {hit['qseqid']} -> {hit['sseqid']}: "
                f"{hit['pident']:.1f}% identity, {hit['qcovs']:.1f}% coverage, "
                f"E-value: {hit['evalue']:.2e}"
            )

    return problematic_sequences, blast_df


def filter_sequences_by_similarity(
    existing_csv_path,
    new_negatives_csv_path,
    output_csv_path,
    evalue_threshold=1e-5,
    identity_threshold=30.0,
    coverage_threshold=50.0,
    num_threads=4,
    keep_blast_files=False,
    blast_output_dir=None,
):
    """Main function to filter new negative sequences by similarity to existing positives.

    Args:
        existing_csv_path (str): Path to existing dataset CSV
        new_negatives_csv_path (str): Path to new negative samples CSV
        output_csv_path (str): Path to output filtered CSV
        evalue_threshold (float): E-value threshold for BLAST search
        identity_threshold (float): Identity percentage threshold for filtering
        coverage_threshold (float): Query coverage percentage threshold for filtering
        num_threads (int): Number of threads for BLAST
        keep_blast_files (bool): Whether to keep temporary BLAST files
        blast_output_dir (str): Directory to save BLAST output files (if keep_blast_files=True)

    Returns:
        dict: Summary statistics of the filtering process
    """
    # Check BLAST installation
    if not check_blast_installation():
        raise RuntimeError("BLAST+ is required but not installed")

    # Load existing dataset
    logger.info(f"Loading existing dataset from {existing_csv_path}")
    try:
        existing_df = pd.read_csv(existing_csv_path)
        logger.info(f"Loaded {len(existing_df)} sequences from existing dataset")
    except Exception as e:
        logger.error(f"Error loading existing dataset: {e}")
        raise

    # Load new negatives
    logger.info(f"Loading new negatives from {new_negatives_csv_path}")
    try:
        new_negatives_df = pd.read_csv(new_negatives_csv_path)
        logger.info(f"Loaded {len(new_negatives_df)} new negative sequences")
    except Exception as e:
        logger.error(f"Error loading new negatives: {e}")
        raise

    # Extract existing positives (label = 1)
    existing_positives = existing_df[existing_df["label"] == 1].copy()
    logger.info(f"Found {len(existing_positives)} existing positive sequences")

    if len(existing_positives) == 0:
        logger.warning(
            "No existing positive sequences found - skipping similarity check"
        )
        # Just add new negatives to existing dataset
        combined_df = pd.concat([existing_df, new_negatives_df], ignore_index=True)
        combined_df.to_csv(output_csv_path, index=False)
        return {
            "initial_negatives": len(new_negatives_df),
            "filtered_negatives": 0,
            "final_negatives": len(new_negatives_df),
            "existing_positives": 0,
            "problematic_sequences": [],
        }

    # Verify new negatives are all labeled as negative
    if "label" in new_negatives_df.columns:
        non_negative_count = len(new_negatives_df[new_negatives_df["label"] != 0])
        if non_negative_count > 0:
            logger.warning(
                f"Found {non_negative_count} non-negative labels in new negatives"
            )
    else:
        logger.info("Adding label=0 to new negatives")
        new_negatives_df["label"] = 0

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Convert sequences to FASTA format
        positives_fasta = temp_dir / "existing_positives.fasta"
        negatives_fasta = temp_dir / "new_negatives.fasta"
        blast_output = temp_dir / "blast_results.txt"

        # Create FASTA files
        positives_seqs = dict(
            zip(existing_positives["sequence_id"], existing_positives["sequence"])
        )
        negatives_seqs = dict(
            zip(new_negatives_df["sequence_id"], new_negatives_df["sequence"])
        )

        sequences_to_fasta(positives_seqs, positives_fasta)
        sequences_to_fasta(negatives_seqs, negatives_fasta)

        # Run BLASTP search
        logger.info(f"Running BLASTP with E-value threshold: {evalue_threshold}")
        blast_success = run_blastp_search(
            query_fasta=negatives_fasta,
            subject_fasta=positives_fasta,
            output_file=blast_output,
            evalue=evalue_threshold,
            num_threads=num_threads,
        )

        if not blast_success:
            raise RuntimeError("BLASTP search failed")

        # Parse results and identify problematic sequences
        logger.info(
            f"Filtering with identity ≥ {identity_threshold}% and coverage ≥ {coverage_threshold}%"
        )
        problematic_sequences, blast_df = parse_blast_results(
            blast_output, identity_threshold, coverage_threshold
        )

        # Save BLAST files if requested
        if keep_blast_files and blast_output_dir:
            blast_output_dir = Path(blast_output_dir)
            blast_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to permanent location
            shutil.copy2(blast_output, blast_output_dir / "blast_results.txt")
            shutil.copy2(positives_fasta, blast_output_dir / "existing_positives.fasta")
            shutil.copy2(negatives_fasta, blast_output_dir / "new_negatives.fasta")

            # Save blast results as CSV for easier analysis
            if not blast_df.empty:
                blast_df.to_csv(blast_output_dir / "blast_results.csv", index=False)

            logger.info(f"BLAST files saved to {blast_output_dir}")

    # Filter new negatives
    initial_count = len(new_negatives_df)
    filtered_negatives = new_negatives_df[
        ~new_negatives_df["sequence_id"].isin(problematic_sequences)
    ].copy()
    filtered_count = len(problematic_sequences)
    final_count = len(filtered_negatives)

    logger.info(f"Filtering summary:")
    logger.info(f"  Initial new negatives: {initial_count}")
    logger.info(f"  Problematic sequences removed: {filtered_count}")
    logger.info(f"  Final new negatives: {final_count}")
    logger.info(f"  Retention rate: {final_count/initial_count*100:.1f}%")

    # Combine with existing dataset
    # Ensure consistent column order
    required_columns = ["sequence_id", "sequence", "label"]
    if "dataset" in existing_df.columns:
        required_columns.append("dataset")
        if "dataset" not in filtered_negatives.columns:
            filtered_negatives["dataset"] = "train"  # Default to train

    # Reorder columns to match existing dataset
    existing_columns = existing_df.columns.tolist()
    filtered_negatives = filtered_negatives.reindex(
        columns=existing_columns, fill_value=None
    )

    combined_df = pd.concat([existing_df, filtered_negatives], ignore_index=True)

    # Save filtered dataset
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Filtered dataset saved to {output_csv_path}")
    logger.info(f"Final dataset size: {len(combined_df)} sequences")

    # Log final distribution
    logger.info("\nFinal dataset distribution:")
    print(combined_df["label"].value_counts().sort_index())
    if "dataset" in combined_df.columns:
        logger.info("\nDataset split distribution:")
        print(combined_df.groupby(["dataset", "label"]).size().unstack(fill_value=0))

    return {
        "initial_negatives": initial_count,
        "filtered_negatives": filtered_count,
        "final_negatives": final_count,
        "retention_rate": final_count / initial_count,
        "existing_positives": len(existing_positives),
        "problematic_sequences": list(problematic_sequences),
        "final_dataset_size": len(combined_df),
        "blast_hits_total": len(blast_df) if not blast_df.empty else 0,
    }


def filter_negatives_simple(
    positives_csv_path,
    negatives_csv_path,
    output_csv_path,
    evalue_threshold=1e-5,
    identity_threshold=40.0,
    coverage_threshold=60.0,
    num_threads=4,
    keep_blast_files=False,
    blast_output_dir=None,
):
    """Filter negative sequences by similarity to positive sequences from separate CSV files.

    Args:
        positives_csv_path (str): Path to positive sequences CSV file
        negatives_csv_path (str): Path to negative sequences CSV file
        output_csv_path (str): Path to output filtered negatives CSV
        evalue_threshold (float): E-value threshold for BLAST search
        identity_threshold (float): Identity percentage threshold for filtering
        coverage_threshold (float): Query coverage percentage threshold for filtering
        num_threads (int): Number of threads for BLAST
        keep_blast_files (bool): Whether to keep temporary BLAST files
        blast_output_dir (str): Directory to save BLAST output files (if keep_blast_files=True)

    Returns:
        dict: Summary statistics of the filtering process
    """
    # Check BLAST installation
    if not check_blast_installation():
        raise RuntimeError("BLAST+ is required but not installed")

    # Load positive sequences
    logger.info(f"Loading positive sequences from {positives_csv_path}")
    try:
        positives_df = pd.read_csv(positives_csv_path)
        logger.info(f"Loaded {len(positives_df)} positive sequences")
    except Exception as e:
        logger.error(f"Error loading positive sequences: {e}")
        raise

    # Load negative sequences
    logger.info(f"Loading negative sequences from {negatives_csv_path}")
    try:
        negatives_df = pd.read_csv(negatives_csv_path)
        logger.info(f"Loaded {len(negatives_df)} negative sequences")
    except Exception as e:
        logger.error(f"Error loading negative sequences: {e}")
        raise

    # Validate required columns
    required_columns = ["sequence_id", "sequence"]
    for df_name, df in [("positives", positives_df), ("negatives", negatives_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"{df_name} CSV missing required columns: {missing_columns}"
            )

    if len(positives_df) == 0:
        logger.warning(
            "No positive sequences found - returning all negatives unchanged"
        )
        negatives_df.to_csv(output_csv_path, index=False)
        return {
            "initial_negatives": len(negatives_df),
            "filtered_negatives": 0,
            "final_negatives": len(negatives_df),
            "existing_positives": 0,
            "problematic_sequences": [],
        }

    # Ensure negatives have label column (set to 0 if missing)
    if "label" not in negatives_df.columns:
        logger.info("Adding label=0 to negative sequences")
        negatives_df["label"] = 0

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Convert sequences to FASTA format
        positives_fasta = temp_dir / "positive_sequences.fasta"
        negatives_fasta = temp_dir / "negative_sequences.fasta"
        blast_output = temp_dir / "blast_results.txt"

        # Create FASTA files
        positives_seqs = dict(
            zip(positives_df["sequence_id"], positives_df["sequence"])
        )
        negatives_seqs = dict(
            zip(negatives_df["sequence_id"], negatives_df["sequence"])
        )

        sequences_to_fasta(positives_seqs, positives_fasta)
        sequences_to_fasta(negatives_seqs, negatives_fasta)

        # Run BLASTP search
        logger.info(f"Running BLASTP with E-value threshold: {evalue_threshold}")
        blast_success = run_blastp_search(
            query_fasta=negatives_fasta,
            subject_fasta=positives_fasta,
            output_file=blast_output,
            evalue=evalue_threshold,
            num_threads=num_threads,
        )

        if not blast_success:
            raise RuntimeError("BLASTP search failed")

        # Parse results and identify problematic sequences
        logger.info(
            f"Filtering with identity ≥ {identity_threshold}% and coverage ≥ {coverage_threshold}%"
        )
        problematic_sequences, blast_df = parse_blast_results(
            blast_output, identity_threshold, coverage_threshold
        )

        # Save BLAST files if requested
        if keep_blast_files and blast_output_dir:
            blast_output_dir = Path(blast_output_dir)
            blast_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to permanent location
            shutil.copy2(blast_output, blast_output_dir / "blast_results.txt")
            shutil.copy2(positives_fasta, blast_output_dir / "positive_sequences.fasta")
            shutil.copy2(negatives_fasta, blast_output_dir / "negative_sequences.fasta")

            # Save blast results as CSV for easier analysis
            if not blast_df.empty:
                blast_df.to_csv(blast_output_dir / "blast_results.csv", index=False)

            logger.info(f"BLAST files saved to {blast_output_dir}")

    # Filter negative sequences
    initial_count = len(negatives_df)
    filtered_negatives = negatives_df[
        ~negatives_df["sequence_id"].isin(problematic_sequences)
    ].copy()
    filtered_count = len(problematic_sequences)
    final_count = len(filtered_negatives)

    logger.info(f"Filtering summary:")
    logger.info(f"  Initial negative sequences: {initial_count}")
    logger.info(f"  Problematic sequences removed: {filtered_count}")
    logger.info(f"  Final negative sequences: {final_count}")
    logger.info(f"  Retention rate: {final_count/initial_count*100:.1f}%")

    # Save filtered negatives
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_negatives.to_csv(output_path, index=False)

    logger.info(f"Filtered negatives saved to {output_csv_path}")

    return {
        "initial_negatives": initial_count,
        "filtered_negatives": filtered_count,
        "final_negatives": final_count,
        "retention_rate": final_count / initial_count,
        "existing_positives": len(positives_df),
        "problematic_sequences": list(problematic_sequences),
        "final_dataset_size": final_count,
        "blast_hits_total": len(blast_df) if not blast_df.empty else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter negative samples by checking similarity against positive samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode descriptions:
  existing: Filter new negatives against positives from an existing mixed dataset
  separate: Filter negatives against positives from separate CSV files

Examples:
  # Mode 1: Existing dataset + new negatives (original functionality)
  python filter_negatives_by_similarity.py --mode existing \\
    --existing-csv dataset.csv --new-negatives-csv new_negatives.csv \\
    --output-csv filtered_dataset.csv

  # Mode 2: Separate positive and negative files (new functionality)
  python filter_negatives_by_similarity.py --mode separate \\
    --positives-csv positives.csv --negatives-csv negatives.csv \\
    --output-csv filtered_negatives.csv
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["existing", "separate"],
        default="existing",
        help="Filtering mode: 'existing' (original) or 'separate' (new two-file mode)",
    )

    # Arguments for existing mode
    parser.add_argument(
        "--existing-csv",
        help="Path to existing dataset CSV file (required for 'existing' mode)",
    )
    parser.add_argument(
        "--new-negatives-csv",
        help="Path to new negative samples CSV file (required for 'existing' mode)",
    )

    # Arguments for separate mode
    parser.add_argument(
        "--positives-csv",
        help="Path to positive sequences CSV file (required for 'separate' mode)",
    )
    parser.add_argument(
        "--negatives-csv",
        help="Path to negative sequences CSV file (required for 'separate' mode)",
    )

    # Common arguments
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--evalue-threshold",
        type=float,
        default=1e-5,
        help="E-value threshold for BLAST search (default: 1e-5)",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=40.0,
        help="Identity percentage threshold for filtering (default: 40.0)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=60.0,
        help="Query coverage percentage threshold for filtering (default: 60.0)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=16,
        help="Number of threads for BLAST (default: 16)",
    )
    parser.add_argument(
        "--keep-blast-files",
        action="store_true",
        help="Keep BLAST output files for inspection",
    )
    parser.add_argument(
        "--blast-output-dir",
        help="Directory to save BLAST output files (requires --keep-blast-files)",
    )

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "existing":
        if not args.existing_csv or not args.new_negatives_csv:
            parser.error(
                "'existing' mode requires --existing-csv and --new-negatives-csv"
            )
        input_files = [args.existing_csv, args.new_negatives_csv]
    elif args.mode == "separate":
        if not args.positives_csv or not args.negatives_csv:
            parser.error("'separate' mode requires --positives-csv and --negatives-csv")
        input_files = [args.positives_csv, args.negatives_csv]

    # Validate other arguments
    if args.keep_blast_files and not args.blast_output_dir:
        parser.error(
            "--blast-output-dir is required when --keep-blast-files is specified"
        )

    # Validate input files exist
    for file_path in input_files:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return 1

    try:
        # Run appropriate filtering function based on mode
        if args.mode == "existing":
            logger.info(
                "Running in 'existing' mode - filtering new negatives against existing dataset"
            )
            results = filter_sequences_by_similarity(
                existing_csv_path=args.existing_csv,
                new_negatives_csv_path=args.new_negatives_csv,
                output_csv_path=args.output_csv,
                evalue_threshold=args.evalue_threshold,
                identity_threshold=args.identity_threshold,
                coverage_threshold=args.coverage_threshold,
                num_threads=args.num_threads,
                keep_blast_files=args.keep_blast_files,
                blast_output_dir=args.blast_output_dir,
            )
            success_message = "FILTERING COMPLETED SUCCESSFULLY (EXISTING MODE)"

        elif args.mode == "separate":
            logger.info(
                "Running in 'separate' mode - filtering negatives against positives from separate files"
            )
            results = filter_negatives_simple(
                positives_csv_path=args.positives_csv,
                negatives_csv_path=args.negatives_csv,
                output_csv_path=args.output_csv,
                evalue_threshold=args.evalue_threshold,
                identity_threshold=args.identity_threshold,
                coverage_threshold=args.coverage_threshold,
                num_threads=args.num_threads,
                keep_blast_files=args.keep_blast_files,
                blast_output_dir=args.blast_output_dir,
            )
            success_message = "FILTERING COMPLETED SUCCESSFULLY (SEPARATE MODE)"

        logger.info("\n" + "=" * 60)
        logger.info(success_message)
        logger.info("=" * 60)
        logger.info(f"Summary:")
        logger.info(f"  Retention rate: {results['retention_rate']*100:.1f}%")
        logger.info(f"  Sequences removed: {results['filtered_negatives']}")
        logger.info(f"  Final dataset size: {results['final_dataset_size']}")

        if results["filtered_negatives"] > 0:
            logger.info(
                f"\nRecommendation: Review the {results['filtered_negatives']} removed sequences"
            )
            logger.info(f"to ensure they were correctly filtered.")

        return 0

    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
