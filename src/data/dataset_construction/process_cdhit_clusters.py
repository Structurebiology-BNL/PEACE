#!/usr/bin/env python3
"""
Script to process CD-HIT cluster output and extract representative sequences.
Creates a CSV file compatible with filter_negatives_by_similarity.py script.
"""
import argparse
import logging
from pathlib import Path

# Import shared utilities
from .utils import (
    load_fasta_sequences,
    parse_cdhit_clusters,
    match_sequences_to_fasta,
    create_csv_for_filtering,
    logger,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_cdhit_output(
    cluster_file, fasta_file, output_csv, dataset_name="train", label=0
):
    """Main function to process CD-HIT output and create filtered CSV.

    Args:
        cluster_file (str): Path to CD-HIT .clstr file
        fasta_file (str): Path to original FASTA file
        output_csv (str): Path to output CSV file
        dataset_name (str): Dataset name for sequences
        label (int): Label for sequences (0 for negatives, 1 for positives)

    Returns:
        pd.DataFrame: The created DataFrame
    """
    logger.info("Starting CD-HIT cluster processing...")
    logger.info(f"Cluster file: {cluster_file}")
    logger.info(f"FASTA file: {fasta_file}")
    logger.info(f"Output CSV: {output_csv}")

    # Step 1: Parse cluster file to get representatives
    representatives = parse_cdhit_clusters(cluster_file, mode="representatives_only")

    # Step 2: Load sequences from FASTA file
    fasta_sequences = load_fasta_sequences(fasta_file)

    # Step 3: Match representatives to actual sequences
    matched_sequences = match_sequences_to_fasta(
        representatives, fasta_sequences, mode="partial"
    )

    # Step 4: Create CSV file
    df = create_csv_for_filtering(matched_sequences, output_csv, dataset_name, label)

    logger.info("CD-HIT processing completed successfully!")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process CD-HIT cluster output to extract representative sequences for similarity filtering"
    )
    parser.add_argument(
        "--cluster-file", required=True, help="Path to CD-HIT .clstr file"
    )
    parser.add_argument(
        "--fasta-file",
        required=True,
        help="Path to original FASTA file (before clustering)",
    )
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--dataset-name",
        default="train",
        help="Dataset name for sequences (default: train)",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Label for sequences (default: 0 for negatives)",
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
        # Process CD-HIT output
        df = process_cdhit_output(
            cluster_file=args.cluster_file,
            fasta_file=args.fasta_file,
            output_csv=args.output_csv,
            dataset_name=args.dataset_name,
            label=args.label,
        )

        logger.info("\n" + "=" * 60)
        logger.info("CD-HIT PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Representative sequences extracted: {len(df)}")
        logger.info(f"Output CSV file: {args.output_csv}")
        logger.info(f"\nNext step: Run similarity filtering with:")
        logger.info(f"python src/data/fungtion/filter_negatives_by_similarity.py \\")
        logger.info(f"    --existing-csv src/data/fungtion/fungtion_dataset.csv \\")
        logger.info(f"    --new-negatives-csv {args.output_csv} \\")
        logger.info(
            f"    --output-csv src/data/fungtion/fungtion_dataset_updated.csv \\"
        )
        logger.info(f"    --keep-blast-files \\")
        logger.info(f"    --blast-output-dir blast_similarity_check")

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
