import pandas as pd
import numpy as np
import argparse
import os


def split_and_combine_datasets(
    positive_csv_path: str,
    negative_csv_path: str,
    pretrain_output_path: str,
    finetune_output_path: str,
    negative_ratio: int = 50,
    random_seed: int = 42,
) -> tuple:
    """
    Split negative sequences into partitions and combine with positive sequences.
    Output two files: pretraining and finetuning CSVs as described in the requirements.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Load the datasets
    print(f"Loading positive sequences from: {positive_csv_path}")
    positive_df = pd.read_csv(positive_csv_path)

    print(f"Loading negative sequences from: {negative_csv_path}")
    negative_df = pd.read_csv(negative_csv_path)

    # Validate positive dataset structure
    required_cols = ["sequence_id", "sequence", "label", "partition"]
    if not all(col in positive_df.columns for col in required_cols):
        raise ValueError(f"Positive dataset must have columns: {required_cols}")

    # Validate negative dataset structure
    required_neg_cols = ["sequence_id", "sequence", "label"]
    if not all(col in negative_df.columns for col in required_neg_cols):
        raise ValueError(f"Negative dataset must have columns: {required_neg_cols}")

    # Infer positive train and test counts from the CSV
    if not set(["train", "test"]).issubset(set(positive_df["partition"].unique())):
        raise ValueError(
            "Positive CSV must contain both 'train' and 'test' partitions in the 'partition' column."
        )
    positive_train_count = len(positive_df[positive_df["partition"] == "train"])
    positive_test_count = len(positive_df[positive_df["partition"] == "test"])

    # Calculate negative sequence counts
    negative_test_count = positive_test_count * negative_ratio
    negative_train_count = positive_train_count * negative_ratio

    total_negative_needed = negative_test_count + negative_train_count
    total_negatives_available = len(negative_df)

    print(f"\nDataset Statistics:")
    print(f"Positive sequences: {len(positive_df)}")
    print(f"  Train: {positive_train_count}")
    print(f"  Test: {positive_test_count}")
    print(f"Negative sequences available: {total_negatives_available}")
    print(f"Negative sequences needed: {total_negative_needed}")

    if total_negatives_available < total_negative_needed:
        raise ValueError(
            f"Not enough negative sequences! Need {total_negative_needed}, "
            f"but only have {total_negatives_available}"
        )

    # Calculate partition counts for negatives
    remaining_negatives = total_negatives_available - total_negative_needed
    negative_pretrain_only_count = remaining_negatives  # These go to pretrain only

    print(f"\nNegative Sequence Allocation:")
    print(f"Test partition: {negative_test_count}")
    print(f"Train partition: {negative_train_count}")
    print(f"Pretrain-only partition: {negative_pretrain_only_count}")
    print(
        f"Total pretrain (train + pretrain-only): {negative_train_count + negative_pretrain_only_count}"
    )

    # Shuffle negative sequences
    negative_df_shuffled = negative_df.sample(
        frac=1, random_state=random_seed
    ).reset_index(drop=True)

    # Split negative sequences into partitions
    partitions = []
    current_idx = 0

    # Test partition
    test_negatives = negative_df_shuffled.iloc[
        current_idx : current_idx + negative_test_count
    ].copy()
    test_negatives["partition"] = "test"
    partitions.append(test_negatives)
    current_idx += negative_test_count

    # Train partition (subset of pretrain)
    train_negatives = negative_df_shuffled.iloc[
        current_idx : current_idx + negative_train_count
    ].copy()
    train_negatives["partition"] = "train"
    partitions.append(train_negatives)
    current_idx += negative_train_count

    # Pretrain-only partition
    if negative_pretrain_only_count > 0:
        pretrain_only_negatives = negative_df_shuffled.iloc[
            current_idx : current_idx + negative_pretrain_only_count
        ].copy()
        pretrain_only_negatives["partition"] = "pretrain"
        partitions.append(pretrain_only_negatives)

    # Combine all negative partitions
    negative_df_with_partitions = pd.concat(partitions, ignore_index=True)

    # Combine positive and negative datasets
    combined_df = pd.concat(
        [positive_df, negative_df_with_partitions], ignore_index=True
    )

    # --- NEW LOGIC: Create Pretraining and Finetuning DataFrames ---
    # Pretraining CSV: all 'pretrain' and 'train' partitions, relabel all as 'train', exclude 'test'
    pretrain_df = combined_df[
        combined_df["partition"].isin(["pretrain", "train"])
    ].copy()
    pretrain_df["partition"] = "train"  # relabel all as 'train'
    pretrain_df.to_csv(pretrain_output_path, index=False)
    print(f"\nPretraining CSV saved to: {pretrain_output_path}")
    print(f"  Total sequences: {len(pretrain_df)}")
    print(f"  Label distribution: {pretrain_df['label'].value_counts().to_dict()}")

    # Finetuning CSV: positive 'train', all negatives (from train/test only), all 'test'
    # Identify positives and negatives
    is_positive = combined_df["label"] == 1
    is_negative = combined_df["label"] == 0
    is_train = combined_df["partition"] == "train"
    is_test = combined_df["partition"] == "test"

    # Positive 'train'
    finetune_pos_train = combined_df[is_positive & is_train]
    # All negatives from train and test partitions only
    finetune_neg = combined_df[
        is_negative & (combined_df["partition"].isin(["train", "test"]))
    ]
    # All 'test' (both positive and negative)
    finetune_test = combined_df[is_test]
    # Concatenate and drop duplicates (in case of overlap)
    finetune_df = pd.concat(
        [finetune_pos_train, finetune_neg, finetune_test], ignore_index=True
    )
    finetune_df = finetune_df.drop_duplicates(subset=["sequence_id"])
    # Warn if any 'pretrain' entries remain (should not happen)
    if "pretrain" in finetune_df["partition"].unique():
        print(
            "WARNING: 'pretrain' entries found in finetuning CSV. These will be removed."
        )
        finetune_df = finetune_df[finetune_df["partition"].isin(["train", "test"])]
    finetune_df.to_csv(finetune_output_path, index=False)
    print(f"\nFinetuning CSV saved to: {finetune_output_path}")
    print(f"  Total sequences: {len(finetune_df)}")
    print(f"  Label distribution: {finetune_df['label'].value_counts().to_dict()}")
    print(
        f"  Partition distribution: {finetune_df['partition'].value_counts().to_dict()}"
    )

    return pretrain_df, finetune_df


def verify_partition_relationships(df: pd.DataFrame) -> None:
    """
    Verify that train is indeed a subset of pretrain in terms of available sequences.
    """
    print("\nVerifying partition relationships:")

    # Count sequences available for each stage
    pretrain_sequences = len(df[df["partition"].isin(["train", "pretrain"])])
    train_sequences = len(df[df["partition"] == "train"])
    test_sequences = len(df[df["partition"] == "test"])

    print(f"Sequences available for pretraining: {pretrain_sequences}")
    print(f"Sequences available for finetuning: {train_sequences}")
    print(f"Sequences available for testing: {test_sequences}")

    # Verify train is subset of pretrain
    assert train_sequences <= pretrain_sequences, "Train should be a subset of pretrain"
    print("✓ Partition relationships are correct")


def main():
    parser = argparse.ArgumentParser(
        description="Combine positive and negative protein sequence CSVs into partitioned pretraining and finetuning datasets."
    )
    parser.add_argument(
        "--positive-csv", required=True, help="Path to positive CSV file"
    )
    parser.add_argument(
        "--negative-csv", required=True, help="Path to negative CSV file"
    )
    parser.add_argument(
        "--pretrain-csv", required=True, help="Path to output pretraining CSV file"
    )
    parser.add_argument(
        "--finetune-csv", required=True, help="Path to output finetuning CSV file"
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=50,
        help="Negative:positive ratio (default: 50)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Validate file existence
    if not os.path.isfile(args.positive_csv):
        print(f"❌ Error: Positive CSV file not found: {args.positive_csv}")
        exit(1)
    if not os.path.isfile(args.negative_csv):
        print(f"❌ Error: Negative CSV file not found: {args.negative_csv}")
        exit(1)

    print(
        "🧬 Protein Sequence Dataset Splitting and Combining (Pretraining/Finetuning Mode)"
    )
    print("=" * 60)

    try:
        pretrain_df, finetune_df = split_and_combine_datasets(
            positive_csv_path=args.positive_csv,
            negative_csv_path=args.negative_csv,
            pretrain_output_path=args.pretrain_csv,
            finetune_output_path=args.finetune_csv,
            negative_ratio=args.negative_ratio,
            random_seed=args.random_seed,
        )

        print("\n" + "=" * 60)
        print("✅ Dataset processing completed successfully!")
        verify_partition_relationships(pretrain_df)
        # Additional analysis for finetune_df
        print("\n📊 Finetuning CSV Analysis:")
        test_pos = len(
            finetune_df[
                (finetune_df["partition"] == "test") & (finetune_df["label"] == 1)
            ]
        )
        test_neg = len(
            finetune_df[
                (finetune_df["partition"] == "test") & (finetune_df["label"] == 0)
            ]
        )
        train_pos = len(
            finetune_df[
                (finetune_df["partition"] == "train") & (finetune_df["label"] == 1)
            ]
        )
        train_neg = len(
            finetune_df[
                (finetune_df["partition"] == "train") & (finetune_df["label"] == 0)
            ]
        )
        print(
            f"Test set - Positive: {test_pos}, Negative: {test_neg}, Ratio: 1:{test_neg//test_pos if test_pos > 0 else 'N/A'}"
        )
        print(
            f"Train set - Positive: {train_pos}, Negative: {train_neg}, Ratio: 1:{train_neg//train_pos if train_pos > 0 else 'N/A'}"
        )
        summary_stats = {
            "total_sequences": len(finetune_df),
            "positive_sequences": len(finetune_df[finetune_df["label"] == 1]),
            "negative_sequences": len(finetune_df[finetune_df["label"] == 0]),
            "test_sequences": len(finetune_df[finetune_df["partition"] == "test"]),
            "train_sequences": len(finetune_df[finetune_df["partition"] == "train"]),
        }
        print(f"\n📈 Finetuning Summary Statistics:")
        for key, value in summary_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
