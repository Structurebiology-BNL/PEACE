#!/usr/bin/env python3
# ruff: noqa
"""
EffectorP Prediction Analysis Script

This script parses EffectorP 3.0 output and evaluates predictions against ground truth data.
It calculates comprehensive evaluation metrics including accuracy, precision, recall, F1-score, and MCC.
Now includes threshold analysis to find optimal thresholds on training data and apply them to test data.

Usage:
    # For threshold analysis (recommended):
    python analyze_effectorP_predictions.py --train_predictions train_predictions.txt --test_predictions test_predictions.txt --ground_truth ground_truth.csv --threshold_analysis --output results_analysis.txt --threshold_output threshold_analysis.txt

    # For single file analysis (backward compatibility):
    python analyze_effectorP_predictions.py --predictions temp_results.txt --ground_truth ground_truth.csv --output results_analysis.txt

    # For threshold analysis with custom plot directory:
    python analyze_effectorP_predictions.py --train_predictions train_predictions.txt --test_predictions test_predictions.txt --ground_truth ground_truth.csv --threshold_analysis --plot_dir ./plots

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import argparse
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional plotting dependency
    sns = None

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import threshold analysis utilities
try:
    from effector_bincls.metrics import find_optimal_threshold, high_recall_auprc
    from effector_bincls.plotting import plot_threshold_analysis

    THRESHOLD_ANALYSIS_AVAILABLE = True
except ImportError:
    print(
        "Warning: Threshold analysis utilities not available. Install required dependencies for full functionality."
    )
    THRESHOLD_ANALYSIS_AVAILABLE = False


class EffectorPAnalyzer:
    """
    Class to analyze EffectorP predictions against ground truth data.
    """

    def __init__(self):
        self.predictions_df = None
        self.ground_truth_df = None
        self.merged_df = None
        self.evaluation_results = {}

    def parse_effectorP_output(self, filepath):
        """
        Parse EffectorP output file to extract predictions.

        Args:
            filepath (str): Path to EffectorP output file

        Returns:
            pd.DataFrame: DataFrame with parsed predictions
        """
        predictions = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Find the header line
        header_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("# Identifier"):
                header_found = True
                header_idx = i
                break

        if not header_found:
            raise ValueError("Could not find header line starting with '# Identifier'")

        # Parse prediction lines
        for line in lines[header_idx + 1 :]:
            line = line.strip()

            # Skip empty lines and summary lines
            if not line or line.startswith("-") or "proteins were provided" in line:
                continue

            # Split by tabs
            parts = line.split("\t")
            if len(parts) < 5:
                continue

            identifier = parts[0].strip()
            cyto_pred = parts[1].strip()
            apo_pred = parts[2].strip()
            noneff_pred = parts[3].strip()
            final_pred = parts[4].strip()

            # Extract probabilities
            cyto_prob = self._extract_probability(cyto_pred)
            apo_prob = self._extract_probability(apo_pred)
            noneff_prob = self._extract_probability(noneff_pred)

            # Determine binary prediction (effector vs non-effector)
            binary_pred = 1 if final_pred != "Non-effector" else 0

            # Determine effector type
            effector_type = self._classify_effector_type(final_pred)

            predictions.append(
                {
                    "sequence_id": identifier,
                    "cytoplasmic_prob": cyto_prob,
                    "apoplastic_prob": apo_prob,
                    "noneffector_prob": noneff_prob,
                    "final_prediction": final_pred,
                    "binary_prediction": binary_pred,
                    "effector_type": effector_type,
                    "max_effector_prob": max(
                        cyto_prob if cyto_prob is not None else 0,
                        apo_prob if apo_prob is not None else 0,
                    ),
                }
            )

        return pd.DataFrame(predictions)

    def parse_multiple_effectorP_outputs(self, train_filepath, test_filepath):
        """
        Parse separate train and test EffectorP output files and merge them with partition information.

        Args:
            train_filepath (str): Path to training EffectorP output file
            test_filepath (str): Path to test EffectorP output file

        Returns:
            pd.DataFrame: DataFrame with parsed predictions and partition information
        """
        print(f"Parsing training predictions from: {train_filepath}")
        train_predictions = self.parse_effectorP_output(train_filepath)
        train_predictions["partition"] = "train"

        print(f"Parsing test predictions from: {test_filepath}")
        test_predictions = self.parse_effectorP_output(test_filepath)
        test_predictions["partition"] = "test"

        # Combine predictions
        self.predictions_df = pd.concat(
            [train_predictions, test_predictions], ignore_index=True
        )

        print(
            f"Combined predictions: {len(train_predictions)} training + {len(test_predictions)} test = {len(self.predictions_df)} total"
        )

        return self.predictions_df

    def _extract_probability(self, pred_string):
        """
        Extract probability from prediction string like 'Y (0.951)' or return None for '-'.

        Args:
            pred_string (str): Prediction string

        Returns:
            float or None: Extracted probability
        """
        if pred_string == "-":
            return None

        match = re.search(r"Y \(([\d.]+)\)", pred_string)
        if match:
            return float(match.group(1))
        return None

    def _classify_effector_type(self, final_pred):
        """
        Classify the effector type based on final prediction.

        Args:
            final_pred (str): Final prediction string

        Returns:
            str: Effector type classification
        """
        if final_pred == "Non-effector":
            return "non-effector"
        elif final_pred == "Cytoplasmic effector":
            return "cytoplasmic"
        elif final_pred == "Apoplastic effector":
            return "apoplastic"
        elif "Cytoplasmic/apoplastic" in final_pred:
            return "dual_cyto_primary"
        elif "Apoplastic/cytoplasmic" in final_pred:
            return "dual_apo_primary"
        else:
            return "unknown"

    def load_ground_truth(self, filepath):
        """
        Load ground truth CSV file.

        Args:
            filepath (str): Path to ground truth CSV file

        Returns:
            pd.DataFrame: Ground truth data
        """
        self.ground_truth_df = pd.read_csv(filepath)

        # Show available columns for debugging
        print(
            f"Available columns in ground truth file: {list(self.ground_truth_df.columns)}"
        )

        # Check for common typos and auto-correct
        column_mapping = {}
        if (
            "parition" in self.ground_truth_df.columns
            and "partition" not in self.ground_truth_df.columns
        ):
            print("Found 'parition' column, renaming to 'partition'")
            column_mapping["parition"] = "partition"

        # Apply column mapping if needed
        if column_mapping:
            self.ground_truth_df = self.ground_truth_df.rename(columns=column_mapping)
            print(f"Applied column mapping: {column_mapping}")

        # Validate required columns
        required_cols = ["sequence_id", "sequence", "label", "partition"]
        missing_cols = [
            col for col in required_cols if col not in self.ground_truth_df.columns
        ]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {list(self.ground_truth_df.columns)}")
            raise ValueError(
                f"Missing required columns in ground truth file: {missing_cols}"
            )

        # Show partition values for debugging
        if "partition" in self.ground_truth_df.columns:
            partition_counts = self.ground_truth_df["partition"].value_counts()
            print(f"Partition distribution: {dict(partition_counts)}")

        return self.ground_truth_df

    def merge_data(self):
        """
        Merge predictions with ground truth data.

        Returns:
            pd.DataFrame: Merged data
        """
        if self.predictions_df is None:
            raise ValueError(
                "Predictions not loaded. Call parse_effectorP_output() first."
            )

        if self.ground_truth_df is None:
            raise ValueError("Ground truth not loaded. Call load_ground_truth() first.")

        # Remove the partition column from predictions since we want to use the ground truth partition
        predictions_for_merge = self.predictions_df.drop(
            columns=["partition"], errors="ignore"
        )
        print(
            f"Predictions columns after dropping partition: {list(predictions_for_merge.columns)}"
        )

        self.merged_df = pd.merge(
            self.ground_truth_df, predictions_for_merge, on="sequence_id", how="inner"
        )

        print(f"Merged {len(self.merged_df)} sequences successfully.")
        print(f"Ground truth had {len(self.ground_truth_df)} sequences.")
        print(f"Predictions had {len(self.predictions_df)} sequences.")

        # Check for missing sequences
        missing_in_pred = set(self.ground_truth_df["sequence_id"]) - set(
            self.predictions_df["sequence_id"]
        )
        missing_in_gt = set(self.predictions_df["sequence_id"]) - set(
            self.ground_truth_df["sequence_id"]
        )

        if missing_in_pred:
            print(
                f"Warning: {len(missing_in_pred)} sequences from ground truth not found in predictions."
            )
            print(f"First few missing: {list(missing_in_pred)[:5]}")
        if missing_in_gt:
            print(
                f"Warning: {len(missing_in_gt)} sequences from predictions not found in ground truth."
            )
            print(f"First few missing: {list(missing_in_gt)[:5]}")

        # Show merged data info
        print(f"Merged data columns: {list(self.merged_df.columns)}")
        print(f"Merged data shape: {self.merged_df.shape}")

        return self.merged_df

    def calculate_metrics(self, dataset_filter=None):
        """
        Calculate evaluation metrics.

        Args:
            dataset_filter (str, optional): Filter by dataset ('train' or 'test')

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Filter by dataset if specified
        if dataset_filter:
            df = self.merged_df[self.merged_df["partition"] == dataset_filter].copy()
            dataset_name = dataset_filter
        else:
            df = self.merged_df.copy()
            dataset_name = "all"

        if len(df) == 0:
            raise ValueError(f"No data available for dataset: {dataset_filter}")

        y_true = df["label"].values
        y_pred = df["binary_prediction"].values

        # Get probability scores for ROC-AUC and AUPRC
        y_prob = df["max_effector_prob"].values

        # Filter out None values for probability-based metrics
        valid_prob_mask = ~np.isnan(y_prob) & (y_prob is not None)
        y_true_valid = y_true[valid_prob_mask]
        y_prob_valid = y_prob[valid_prob_mask]

        # Calculate metrics
        metrics = {
            "partition": dataset_name,
            "n_samples": len(df),
            "n_positive": np.sum(y_true),
            "n_negative": len(y_true) - np.sum(y_true),
            "n_pred_positive": np.sum(y_pred),
            "n_pred_negative": len(y_pred) - np.sum(y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": self._calculate_specificity(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

        # Calculate ROC-AUC and AUPRC if we have valid probability scores
        if len(y_true_valid) > 0 and len(np.unique(y_true_valid)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_valid, y_prob_valid)
            except ValueError:
                metrics["roc_auc"] = None
                print(
                    f"Warning: Could not calculate ROC-AUC for {dataset_name} dataset"
                )

            try:
                metrics["auprc"] = average_precision_score(y_true_valid, y_prob_valid)
            except ValueError:
                metrics["auprc"] = None
                print(f"Warning: Could not calculate AUPRC for {dataset_name} dataset")

            try:
                metrics["high_recall_auprc_0.7"] = high_recall_auprc(
                    y_true_valid, y_prob_valid, recall_threshold=0.7
                )
            except Exception as e:
                metrics["high_recall_auprc_0.7"] = None
                print(
                    f"Warning: Could not calculate high-recall AUPRC (0.7) for {dataset_name} dataset: {e}"
                )

            try:
                metrics["high_recall_auprc_0.8"] = high_recall_auprc(
                    y_true_valid, y_prob_valid, recall_threshold=0.8
                )
            except Exception as e:
                metrics["high_recall_auprc_0.8"] = None
                print(
                    f"Warning: Could not calculate high-recall AUPRC (0.8) for {dataset_name} dataset: {e}"
                )
        else:
            metrics["roc_auc"] = None
            metrics["auprc"] = None
            metrics["high_recall_auprc_0.7"] = None
            metrics["high_recall_auprc_0.8"] = None
            print(
                f"Warning: Insufficient data for ROC-AUC and AUPRC calculation for {dataset_name} dataset"
            )

        # Store results
        self.evaluation_results[dataset_name] = metrics

        return metrics

    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def find_optimal_thresholds(self, threshold_methods=None, target_recalls=None):
        """
        Find optimal thresholds on training data using different methods.

        Args:
            threshold_methods (list, optional): List of threshold methods to use
            target_recalls (list, optional): List of target recall values for recall_constrained method

        Returns:
            dict: Dictionary containing optimal thresholds and corresponding metrics
        """
        if not THRESHOLD_ANALYSIS_AVAILABLE:
            raise ImportError(
                "Threshold analysis utilities not available. Install required dependencies."
            )

        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Filter to training data only
        train_df = self.merged_df[self.merged_df["partition"] == "train"].copy()
        if len(train_df) == 0:
            raise ValueError("No training data found. Check partition column values.")

        # Get training predictions and labels
        train_probs = train_df["max_effector_prob"].values
        train_labels = train_df["label"].values

        # Filter out None/NaN values
        valid_mask = ~np.isnan(train_probs) & (train_probs is not None)
        train_probs = train_probs[valid_mask]
        train_labels = train_labels[valid_mask]

        if len(train_probs) == 0:
            raise ValueError("No valid training predictions found.")

        # Default parameters
        if threshold_methods is None:
            threshold_methods = ["youden"]
        if target_recalls is None:
            target_recalls = [0.7, 0.8, 0.9, 0.95]

        threshold_results = {}

        # Add default threshold (0.5) as baseline comparison
        default_threshold = 0.5
        y_pred_default = (train_probs >= default_threshold).astype(int)
        default_metrics = self._calculate_metrics_at_threshold(
            train_labels, y_pred_default, train_probs, default_threshold
        )

        threshold_results["default_0.5"] = {
            "threshold": default_threshold,
            "method": "default",
            "train_metrics": default_metrics,
        }

        print(f"Added default threshold (0.5) baseline for comparison")

        for method in threshold_methods:
            try:
                if method == "recall_constrained":
                    # Handle recall_constrained method with multiple target recalls
                    for target_recall in target_recalls:
                        try:
                            optimal_thresh = find_optimal_threshold(
                                train_probs,
                                train_labels,
                                method=method,
                                target_recall=target_recall,
                            )

                            # Calculate metrics at this threshold
                            y_pred = (train_probs >= optimal_thresh).astype(int)
                            metrics = self._calculate_metrics_at_threshold(
                                train_labels, y_pred, train_probs, optimal_thresh
                            )

                            key = f"{method}_R{target_recall}"
                            threshold_results[key] = {
                                "threshold": optimal_thresh,
                                "method": method,
                                "target_recall": target_recall,
                                "train_metrics": metrics,
                            }

                        except Exception as e:
                            print(
                                f"Warning: Could not calculate {method} threshold for target_recall={target_recall}: {e}"
                            )
                else:
                    # Handle other methods (including youden)
                    optimal_thresh = find_optimal_threshold(
                        train_probs, train_labels, method=method
                    )

                    # Calculate metrics at this threshold
                    y_pred = (train_probs >= optimal_thresh).astype(int)
                    metrics = self._calculate_metrics_at_threshold(
                        train_labels, y_pred, train_probs, optimal_thresh
                    )

                    threshold_results[method] = {
                        "threshold": optimal_thresh,
                        "method": method,
                        "train_metrics": metrics,
                    }

            except Exception as e:
                print(f"Warning: Could not calculate {method} threshold: {e}")

        return threshold_results

    def _calculate_metrics_at_threshold(self, y_true, y_pred, y_prob, threshold):
        """Calculate all metrics at a specific threshold."""
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "specificity": self._calculate_specificity(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def evaluate_with_thresholds(self, threshold_results, dataset_filter="test"):
        """
        Evaluate test data using the optimal thresholds found on training data.

        Args:
            threshold_results (dict): Results from find_optimal_thresholds()
            dataset_filter (str): Dataset to evaluate ('test' or 'train')

        Returns:
            dict: Evaluation results for each threshold method
        """
        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Filter to specified dataset
        eval_df = self.merged_df[self.merged_df["partition"] == dataset_filter].copy()
        if len(eval_df) == 0:
            raise ValueError(
                f"No {dataset_filter} data found. Check partition column values."
            )

        # Get evaluation predictions and labels
        eval_probs = eval_df["max_effector_prob"].values
        eval_labels = eval_df["label"].values

        # Filter out None/NaN values
        valid_mask = ~np.isnan(eval_probs) & (eval_probs is not None)
        eval_probs = eval_probs[valid_mask]
        eval_labels = eval_labels[valid_mask]

        if len(eval_probs) == 0:
            raise ValueError(f"No valid {dataset_filter} predictions found.")

        evaluation_results = {}

        for method_name, threshold_info in threshold_results.items():
            threshold = threshold_info["threshold"]

            # Apply threshold to get predictions
            y_pred = (eval_probs >= threshold).astype(int)

            # Calculate metrics
            metrics = self._calculate_metrics_at_threshold(
                eval_labels, y_pred, eval_probs, threshold
            )

            evaluation_results[method_name] = {
                "threshold": threshold,
                "method": threshold_info["method"],
                "target_recall": threshold_info.get("target_recall"),
                "train_metrics": threshold_info["train_metrics"],
                "test_metrics": metrics,
            }

        return evaluation_results

    def generate_threshold_analysis_plots(
        self, threshold_results, save_dir=None, threshold_method="youden"
    ):
        """
        Generate threshold analysis plots using the existing visualization function.

        Args:
            threshold_results (dict): Results from find_optimal_thresholds()
            save_dir (str, optional): Directory to save plots
            threshold_method (str): The threshold method to use for plotting (default: "youden")
        """
        if not THRESHOLD_ANALYSIS_AVAILABLE:
            print("Warning: Threshold analysis plotting not available.")
            return

        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Use training data for threshold analysis plots
        train_df = self.merged_df[self.merged_df["partition"] == "train"].copy()
        if len(train_df) == 0:
            print("Warning: No training data found for threshold analysis plots.")
            return

        # Get training predictions and labels
        train_probs = train_df["max_effector_prob"].values
        train_labels = train_df["label"].values

        # Filter out None/NaN values
        valid_mask = ~np.isnan(train_probs) & (train_probs is not None)
        train_probs = train_probs[valid_mask]
        train_labels = train_labels[valid_mask]

        if len(train_probs) == 0:
            print(
                "Warning: No valid training predictions found for threshold analysis plots."
            )
            return

        # Find the threshold for plotting using the specified method
        best_threshold = None
        threshold_method_used = threshold_method

        # Use the specified threshold method from threshold_results
        if threshold_method in threshold_results:
            best_threshold = threshold_results[threshold_method]["threshold"]
            threshold_method_used = threshold_method
        elif len(threshold_results) > 0:
            # Use the first available threshold method if specified method is not available
            first_method = list(threshold_results.keys())[0]
            best_threshold = threshold_results[first_method]["threshold"]
            threshold_method_used = first_method

        # Generate plots
        try:
            plot_threshold_analysis(
                outputs=train_probs,
                labels=train_labels,
                save_dir=Path(save_dir) if save_dir else Path("."),
                fold_number="effectorP_training",
                optimal_threshold=best_threshold,
                threshold_method_used=threshold_method_used,
                logger=None,
            )
            print(f"Training threshold analysis plots saved to: {save_dir or '.'}")
        except Exception as e:
            print(f"Warning: Could not generate training threshold analysis plots: {e}")

    def generate_test_threshold_analysis_plots(
        self,
        threshold_results,
        evaluation_results,
        save_dir=None,
        threshold_method="youden",
    ):
        """
        Generate threshold analysis plots for test data evaluation.

        Args:
            threshold_results (dict): Results from find_optimal_thresholds()
            evaluation_results (dict): Results from evaluate_with_thresholds()
            save_dir (str, optional): Directory to save plots
            threshold_method (str): The threshold method to use for plotting (default: "youden")
        """
        if not THRESHOLD_ANALYSIS_AVAILABLE:
            print("Warning: Threshold analysis plotting not available.")
            return

        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Use test data for test evaluation plots
        test_df = self.merged_df[self.merged_df["partition"] == "test"].copy()
        if len(test_df) == 0:
            print("Warning: No test data found for test threshold analysis plots.")
            return

        # Get test predictions and labels
        test_probs = test_df["max_effector_prob"].values
        test_labels = test_df["label"].values

        # Filter out None/NaN values
        valid_mask = ~np.isnan(test_probs) & (test_probs is not None)
        test_probs = test_probs[valid_mask]
        test_labels = test_labels[valid_mask]

        if len(test_probs) == 0:
            print(
                "Warning: No valid test predictions found for test threshold analysis plots."
            )
            return

        # Find the threshold for plotting using the specified method
        best_threshold = None
        threshold_method_used = threshold_method

        # Use the specified threshold method from threshold_results
        if threshold_method in threshold_results:
            best_threshold = threshold_results[threshold_method]["threshold"]
            threshold_method_used = threshold_method
        elif len(threshold_results) > 0:
            # Use the first available threshold method if specified method is not available
            first_method = list(threshold_results.keys())[0]
            best_threshold = threshold_results[first_method]["threshold"]
            threshold_method_used = first_method

        # Generate test data plots
        try:
            plot_threshold_analysis(
                outputs=test_probs,
                labels=test_labels,
                save_dir=Path(save_dir) if save_dir else Path("."),
                fold_number="effectorP_test",
                optimal_threshold=best_threshold,
                threshold_method_used=threshold_method_used,
                logger=None,
            )
            print(f"Test threshold analysis plots saved to: {save_dir or '.'}")
        except Exception as e:
            print(f"Warning: Could not generate test threshold analysis plots: {e}")

        # Also generate a comparison plot showing all thresholds on test data
        try:
            self._generate_threshold_comparison_plot(
                test_probs, test_labels, threshold_results, evaluation_results, save_dir
            )
        except Exception as e:
            print(f"Warning: Could not generate threshold comparison plot: {e}")

    def _generate_threshold_comparison_plot(
        self, test_probs, test_labels, threshold_results, evaluation_results, save_dir
    ):
        """
        Generate a comparison plot showing performance of all thresholds on test data.

        Args:
            test_probs: Test prediction probabilities
            test_labels: Test ground truth labels
            threshold_results: Results from find_optimal_thresholds()
            evaluation_results: Results from evaluate_with_thresholds()
            save_dir: Directory to save plots
        """
        if not evaluation_results:
            return

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: F1 score vs threshold
        thresholds = []
        f1_scores = []
        mcc_scores = []
        method_names = []

        for method_name, eval_info in evaluation_results.items():
            threshold = eval_info["threshold"]
            f1 = eval_info["test_metrics"]["f1_score"]
            mcc = eval_info["test_metrics"]["mcc"]

            thresholds.append(threshold)
            f1_scores.append(f1)
            mcc_scores.append(mcc)
            method_names.append(method_name)

        # Sort by threshold for better visualization
        sorted_indices = np.argsort(thresholds)
        thresholds = [thresholds[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        mcc_scores = [mcc_scores[i] for i in sorted_indices]
        method_names = [method_names[i] for i in sorted_indices]

        # F1 score plot
        ax1.plot(thresholds, f1_scores, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("F1 Score")
        ax1.set_title("F1 Score vs Threshold on Test Data")
        ax1.grid(True, alpha=0.3)

        # Add method labels
        for i, (threshold, f1, method) in enumerate(
            zip(thresholds, f1_scores, method_names)
        ):
            ax1.annotate(
                method,
                (threshold, f1),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        # MCC plot
        ax2.plot(
            thresholds, mcc_scores, "s-", linewidth=2, markersize=8, color="orange"
        )
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("MCC Score")
        ax2.set_title("MCC Score vs Threshold on Test Data")
        ax2.grid(True, alpha=0.3)

        # Add method labels
        for i, (threshold, mcc, method) in enumerate(
            zip(thresholds, mcc_scores, method_names)
        ):
            ax2.annotate(
                method,
                (threshold, mcc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        plt.tight_layout()

        # Save plot
        save_path = Path(save_dir) if save_dir else Path(".")
        plot_filename = "threshold_comparison_test_data.png"
        plt.savefig(save_path / plot_filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Threshold comparison plot saved to: {save_path / plot_filename}")

        # Save comparison data to CSV
        import pandas as pd

        comparison_data = []
        for method_name, eval_info in evaluation_results.items():
            comparison_data.append(
                {
                    "method": method_name,
                    "threshold": eval_info["threshold"],
                    "f1_score": eval_info["test_metrics"]["f1_score"],
                    "mcc_score": eval_info["test_metrics"]["mcc"],
                    "accuracy": eval_info["test_metrics"]["accuracy"],
                    "precision": eval_info["test_metrics"]["precision"],
                    "recall": eval_info["test_metrics"]["recall"],
                }
            )

        df = pd.DataFrame(comparison_data)
        csv_filename = "threshold_comparison_test_data.csv"
        df.to_csv(save_path / csv_filename, index=False)
        print(f"Threshold comparison data saved to: {save_path / csv_filename}")

    def generate_threshold_report(
        self,
        threshold_results,
        evaluation_results,
        output_file=None,
        threshold_method="youden",
    ):
        """
        Generate a comprehensive report comparing different threshold methods.

        Args:
            threshold_results (dict): Results from find_optimal_thresholds()
            evaluation_results (dict): Results from evaluate_with_thresholds()
            output_file (str, optional): Output file path for the report
            threshold_method (str): The threshold method to compare with default (default: "youden")

        Returns:
            str: Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EffectorP Threshold Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary of threshold methods
        report_lines.append("Threshold Methods Analyzed:")
        report_lines.append("-" * 40)

        # Show default threshold first for emphasis
        if "default_0.5" in threshold_results:
            default_info = threshold_results["default_0.5"]
            report_lines.append(
                f"  default_0.5: DEFAULT (EffectorP paper) -> T={default_info['threshold']:.4f}"
            )

        # Show other methods
        for method_name, threshold_info in threshold_results.items():
            if method_name == "default_0.5":
                continue  # Already shown above

            threshold = threshold_info["threshold"]
            method = threshold_info["method"]
            target_recall = threshold_info.get("target_recall")

            if target_recall:
                report_lines.append(
                    f"  {method_name}: {method} (R={target_recall}) -> T={threshold:.4f}"
                )
            else:
                report_lines.append(f"  {method_name}: {method} -> T={threshold:.4f}")
        report_lines.append("")

        # Training performance comparison
        report_lines.append("Training Performance Comparison:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"{'Method':<25} {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}"
        )
        report_lines.append("-" * 80)

        # Show default threshold first
        if "default_0.5" in threshold_results:
            metrics = threshold_results["default_0.5"]["train_metrics"]
            report_lines.append(
                f"{'default_0.5':<25} {metrics['threshold']:<12.4f} "
                f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                f"{metrics['mcc']:<10.4f} {'(EffectorP baseline)':<15}"
            )

        # Show other methods
        for method_name, threshold_info in threshold_results.items():
            if method_name == "default_0.5":
                continue  # Already shown above

            metrics = threshold_info["train_metrics"]
            report_lines.append(
                f"{method_name:<25} {metrics['threshold']:<12.4f} "
                f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                f"{metrics['mcc']:<10.4f}"
            )
        report_lines.append("")

        # Test performance comparison
        if evaluation_results:
            report_lines.append("Test Performance Comparison:")
            report_lines.append("-" * 40)
            report_lines.append(
                f"{'Method':<25} {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}"
            )
            report_lines.append("-" * 80)

            # Show default threshold first
            if "default_0.5" in evaluation_results:
                metrics = evaluation_results["default_0.5"]["test_metrics"]
                report_lines.append(
                    f"{'default_0.5':<25} {metrics['threshold']:<12.4f} "
                    f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                    f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                    f"{metrics['mcc']:<10.4f} {'(EffectorP baseline)':<15}"
                )

            # Show other methods
            for method_name, eval_info in evaluation_results.items():
                if method_name == "default_0.5":
                    continue  # Already shown above

                metrics = eval_info["test_metrics"]
                report_lines.append(
                    f"{method_name:<25} {metrics['threshold']:<12.4f} "
                    f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                    f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                    f"{metrics['mcc']:<10.4f}"
                )
            report_lines.append("")

        # Recommendations
        report_lines.append("Recommendations:")
        report_lines.append("-" * 20)

        # Show youden method results
        if evaluation_results:
            # Find youden method results
            youden_results = None
            for method_name, eval_info in evaluation_results.items():
                if method_name == "youden":
                    youden_results = eval_info
                    break

            if youden_results:
                youden_metrics = youden_results["test_metrics"]
                report_lines.append(
                    f"  Youden method: F1={youden_metrics['f1_score']:.4f}, MCC={youden_metrics['mcc']:.4f}"
                )

                # Compare with default threshold
                if "default_0.5" in evaluation_results:
                    default_metrics = evaluation_results["default_0.5"]["test_metrics"]
                    report_lines.append("")
                    report_lines.append(
                        "Comparison with EffectorP Default Threshold (0.5):"
                    )
                    report_lines.append(
                        f"  Default (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                    )
                    report_lines.append(
                        f"  Youden method: F1={youden_metrics['f1_score']:.4f}, MCC={youden_metrics['mcc']:.4f}"
                    )
                    report_lines.append(
                        f"  F1 improvement: {((youden_metrics['f1_score'] / default_metrics['f1_score']) - 1) * 100:+.1f}%"
                    )
                    report_lines.append(
                        f"  MCC improvement: {((youden_metrics['mcc'] / default_metrics['mcc']) - 1) * 100:+.1f}%"
                    )

        # Show chosen method and default threshold results
        if evaluation_results:
            # Find chosen method and default threshold results
            chosen_results = None
            default_results = None
            for method_name, eval_info in evaluation_results.items():
                if method_name == threshold_method:
                    chosen_results = eval_info
                elif method_name == "default_0.5":
                    default_results = eval_info

            if chosen_results:
                chosen_metrics = chosen_results["test_metrics"]
                report_lines.append(
                    f"  {threshold_method} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                )

            if default_results:
                default_metrics = default_results["test_metrics"]
                report_lines.append(
                    f"  Default threshold (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                )

            # Compare with default threshold
            if chosen_results and default_results:
                report_lines.append("")
                report_lines.append(
                    f"Comparison: {threshold_method} vs EffectorP Default Threshold (0.5):"
                )
                report_lines.append(
                    f"  {threshold_method} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                )
                report_lines.append(
                    f"  Default threshold (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                )
                report_lines.append(
                    f"  F1 improvement: {((chosen_metrics['f1_score'] / default_metrics['f1_score']) - 1) * 100:+.1f}%"
                )
                report_lines.append(
                    f"  MCC improvement: {((chosen_metrics['mcc'] / default_metrics['mcc']) - 1) * 100:+.1f}%"
                )
            elif chosen_results and "default_0.5" in evaluation_results:
                # Fallback comparison
                default_metrics = evaluation_results["default_0.5"]["test_metrics"]
                report_lines.append("")
                report_lines.append(
                    "Comparison with EffectorP Default Threshold (0.5):"
                )
                report_lines.append(
                    f"  Default (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                )
                report_lines.append(
                    f"  {threshold_method.capitalize()} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                )
                report_lines.append(
                    f"  F1 improvement: {((chosen_metrics['f1_score'] / default_metrics['f1_score']) - 1) * 100:+.1f}%"
                )
                report_lines.append(
                    f"  MCC improvement: {((chosen_metrics['mcc'] / default_metrics['mcc']) - 1) * 100:+.1f}%"
                )

        report_text = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"Threshold analysis report saved to: {output_file}")

        return report_text

    def analyze_by_effector_type(self, dataset_filter=None):
        """
        Analyze predictions by effector type.

        Args:
            dataset_filter (str, optional): Filter by dataset ('train' or 'test')

        Returns:
            pd.DataFrame: Analysis by effector type
        """
        if self.merged_df is None:
            raise ValueError("Data not merged. Call merge_data() first.")

        # Filter by dataset if specified
        if dataset_filter:
            df = self.merged_df[self.merged_df["partition"] == dataset_filter].copy()
        else:
            df = self.merged_df.copy()

        # Analyze by effector type
        type_analysis = (
            df.groupby("effector_type")
            .agg(
                {
                    "label": ["count", "sum", "mean"],
                    "binary_prediction": ["sum", "mean"],
                    "max_effector_prob": "mean",
                }
            )
            .round(3)
        )

        type_analysis.columns = [
            "count",
            "true_effectors",
            "true_effector_rate",
            "pred_effectors",
            "pred_effector_rate",
            "avg_max_prob",
        ]

        # Calculate accuracy for each type
        type_accuracy = (
            df.groupby("effector_type")
            .apply(lambda x: accuracy_score(x["label"], x["binary_prediction"]))
            .round(3)
        )

        type_analysis["accuracy"] = type_accuracy

        return type_analysis

    def generate_report(self, output_file=None):
        """
        Generate a comprehensive analysis report.

        Args:
            output_file (str, optional): Output file path for the report

        Returns:
            str: Report text
        """
        if not self.evaluation_results:
            # Calculate metrics for all data if not already done
            self.calculate_metrics()

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EffectorP 3.0 Prediction Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Dataset overview
        if self.merged_df is not None:
            report_lines.append("Dataset Overview:")
            report_lines.append(f"Total sequences analyzed: {len(self.merged_df)}")

            # Dataset split
            dataset_counts = self.merged_df["partition"].value_counts()
            for dataset, count in dataset_counts.items():
                report_lines.append(f"  {dataset.capitalize()} set: {count} sequences")

            # Label distribution
            label_counts = self.merged_df["label"].value_counts()
            report_lines.append(f"Ground truth effectors: {label_counts.get(1, 0)}")
            report_lines.append(f"Ground truth non-effectors: {label_counts.get(0, 0)}")
            report_lines.append("")

        # Metrics for each dataset
        for dataset_name, metrics in self.evaluation_results.items():
            report_lines.append(
                f"Evaluation Metrics - {dataset_name.capitalize()} Dataset:"
            )
            report_lines.append("-" * 50)
            report_lines.append(f"Samples: {metrics['n_samples']}")
            report_lines.append(f"True positives (effectors): {metrics['n_positive']}")
            report_lines.append(
                f"True negatives (non-effectors): {metrics['n_negative']}"
            )
            report_lines.append(f"Predicted positives: {metrics['n_pred_positive']}")
            report_lines.append(f"Predicted negatives: {metrics['n_pred_negative']}")
            report_lines.append("")
            report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
            report_lines.append(f"Precision: {metrics['precision']:.4f}")
            report_lines.append(f"Recall (Sensitivity): {metrics['recall']:.4f}")
            report_lines.append(f"Specificity: {metrics['specificity']:.4f}")
            report_lines.append(f"F1-score: {metrics['f1_score']:.4f}")
            report_lines.append(
                f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}"
            )

            # Add ROC-AUC and AUPRC if available
            if metrics.get("roc_auc") is not None:
                report_lines.append(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            else:
                report_lines.append("ROC-AUC: Not available")

            if metrics.get("auprc") is not None:
                report_lines.append(f"AUPRC: {metrics['auprc']:.4f}")
            else:
                report_lines.append("AUPRC: Not available")

            if metrics.get("high_recall_auprc_0.7") is not None:
                report_lines.append(
                    f"High-recall AUPRC (0.7): {metrics['high_recall_auprc_0.7']:.4f}"
                )
            else:
                report_lines.append("High-recall AUPRC (0.7): Not available")

            if metrics.get("high_recall_auprc_0.8") is not None:
                report_lines.append(
                    f"High-recall AUPRC (0.8): {metrics['high_recall_auprc_0.8']:.4f}"
                )
            else:
                report_lines.append("High-recall AUPRC (0.8): Not available")
            report_lines.append("")

            # Confusion matrix
            cm = metrics["confusion_matrix"]
            report_lines.append("Confusion Matrix:")
            report_lines.append("                 Predicted")
            report_lines.append("                 Non-eff  Effector")
            report_lines.append(f"Actual Non-eff     {cm[0, 0]:>4}     {cm[0, 1]:>4}")
            report_lines.append(f"       Effector    {cm[1, 0]:>4}     {cm[1, 1]:>4}")
            report_lines.append("")

        # Effector type analysis
        if self.merged_df is not None:
            report_lines.append("Analysis by Effector Type:")
            report_lines.append("-" * 50)

            type_analysis = self.analyze_by_effector_type()
            report_lines.append(str(type_analysis))
            report_lines.append("")

        # Performance summary
        report_lines.append("Performance Summary:")
        report_lines.append("-" * 30)

        all_metrics = self.evaluation_results.get("all", {})
        if all_metrics:
            report_lines.append(f"Overall Accuracy: {all_metrics['accuracy']:.4f}")
            report_lines.append(f"Overall F1-score: {all_metrics['f1_score']:.4f}")
            report_lines.append(f"Overall MCC: {all_metrics['mcc']:.4f}")
            if all_metrics.get("roc_auc") is not None:
                report_lines.append(f"Overall ROC-AUC: {all_metrics['roc_auc']:.4f}")
            if all_metrics.get("auprc") is not None:
                report_lines.append(f"Overall AUPRC: {all_metrics['auprc']:.4f}")
            if all_metrics.get("high_recall_auprc_0.7") is not None:
                report_lines.append(
                    f"Overall High-recall AUPRC (0.7): {all_metrics['high_recall_auprc_0.7']:.4f}"
                )
            if all_metrics.get("high_recall_auprc_0.8") is not None:
                report_lines.append(
                    f"Overall High-recall AUPRC (0.8): {all_metrics['high_recall_auprc_0.8']:.4f}"
                )

        report_text = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")

        return report_text

    def plot_confusion_matrix(self, dataset_filter=None, save_path=None):
        """
        Plot confusion matrix.

        Args:
            dataset_filter (str, optional): Filter by dataset ('train' or 'test')
            save_path (str, optional): Path to save the plot
        """
        if not self.evaluation_results:
            self.calculate_metrics(dataset_filter)

        dataset_name = dataset_filter if dataset_filter else "all"
        metrics = self.evaluation_results.get(dataset_name, {})

        if "confusion_matrix" not in metrics:
            raise ValueError(f"No confusion matrix found for dataset: {dataset_name}")

        cm = metrics["confusion_matrix"]

        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non-effector", "Effector"],
                yticklabels=["Non-effector", "Effector"],
            )
        else:
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            tick_labels = ["Non-effector", "Effector"]
            plt.xticks(range(len(tick_labels)), tick_labels)
            plt.yticks(range(len(tick_labels)), tick_labels)
            for row_index in range(cm.shape[0]):
                for column_index in range(cm.shape[1]):
                    plt.text(
                        column_index,
                        row_index,
                        f"{cm[row_index, column_index]:d}",
                        ha="center",
                        va="center",
                        color="black",
                    )
        plt.title(f"Confusion Matrix - {dataset_name.capitalize()} Dataset")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix plot saved to: {save_path}")

        plt.show()


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze EffectorP predictions against ground truth with optional threshold analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Threshold analysis (recommended for optimal performance):
  %(prog)s --train_predictions train_preds.txt --test_predictions test_preds.txt --ground_truth data.csv --threshold_analysis
  
  # Single file analysis (backward compatibility):
  %(prog)s --predictions all_preds.txt --ground_truth data.csv
  
  # Threshold analysis with custom output:
  %(prog)s --train_predictions train_preds.txt --test_predictions test_preds.txt --ground_truth data.csv --threshold_analysis --output results.txt --threshold_output thresholds.txt --plot_dir ./plots
        """,
    )
    parser.add_argument(
        "--train_predictions",
        help="Path to EffectorP output file for training data (required for threshold analysis)",
    )
    parser.add_argument(
        "--test_predictions",
        help="Path to EffectorP output file for test data (required for threshold analysis)",
    )
    parser.add_argument(
        "--predictions",
        help="Path to single EffectorP output file (for backward compatibility, not used with threshold analysis)",
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Path to ground truth CSV file with sequence_id, sequence, label, and partition columns",
    )
    parser.add_argument(
        "--output",
        default="effectorP_analysis_report.txt",
        help="Output file for standard analysis report",
    )
    parser.add_argument(
        "--partition",
        choices=["train", "test"],
        default="test",
        help="Dataset to analyze when using single predictions file (default: test)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate confusion matrix plot"
    )
    parser.add_argument(
        "--threshold_analysis",
        action="store_true",
        help="Perform threshold analysis using training data to find optimal thresholds and apply to test data",
    )
    parser.add_argument(
        "--threshold_output",
        default="effectorP_threshold_analysis.txt",
        help="Output file for threshold analysis report",
    )
    parser.add_argument(
        "--plot_dir",
        default=".",
        help="Directory to save threshold analysis plots (default: current directory)",
    )

    args = parser.parse_args()

    # Validate input arguments
    if args.threshold_analysis:
        if not args.train_predictions or not args.test_predictions:
            print(
                "Error: Threshold analysis requires both --train_predictions and --test_predictions files."
            )
            print(
                "Use --train_predictions for training data and --test_predictions for test data."
            )
            return 1
    elif not args.predictions and not (
        args.train_predictions and args.test_predictions
    ):
        print(
            "Error: Must provide either --predictions (single file) or both --train_predictions and --test_predictions."
        )
        return 1

    # Initialize analyzer
    analyzer = EffectorPAnalyzer()

    try:
        # Load and parse data
        print("Loading EffectorP predictions...")
        if args.predictions:
            analyzer.parse_effectorP_output(args.predictions)
        else:
            analyzer.parse_multiple_effectorP_outputs(
                args.train_predictions, args.test_predictions
            )

        print("Loading ground truth data...")
        analyzer.load_ground_truth(args.ground_truth)

        print("Merging data...")
        analyzer.merge_data()

        # Check if we have both train and test partitions for threshold analysis
        available_partitions = analyzer.merged_df["partition"].unique()
        print(f"Available partitions: {available_partitions}")

        # Validate partition data for threshold analysis
        if args.threshold_analysis:
            if (
                "train" not in available_partitions
                or "test" not in available_partitions
            ):
                print(
                    "Error: Threshold analysis requires both 'train' and 'test' partitions in ground truth data."
                )
                print("Available partitions:", available_partitions)
                print(
                    "Please ensure your ground truth CSV has a 'partition' column with 'train' and 'test' values."
                )
                return 1

            # Check data distribution
            train_count = len(
                analyzer.merged_df[analyzer.merged_df["partition"] == "train"]
            )
            test_count = len(
                analyzer.merged_df[analyzer.merged_df["partition"] == "test"]
            )
            print(
                f"Data distribution: {train_count} training samples, {test_count} test samples"
            )

            if train_count == 0 or test_count == 0:
                print("Error: No data found for one or both partitions.")
                return 1

        # Calculate metrics
        print("Calculating evaluation metrics...")
        if args.partition == "all":
            analyzer.calculate_metrics()
            # Also calculate for train and test separately if available
            if "train" in analyzer.merged_df["partition"].values:
                analyzer.calculate_metrics("train")
            if "test" in analyzer.merged_df["partition"].values:
                analyzer.calculate_metrics("test")
        else:
            analyzer.calculate_metrics(args.partition)

        # Generate standard report
        print("Generating analysis report...")
        report = analyzer.generate_report(args.output)

        # Perform threshold analysis if requested and we have both train/test data
        if args.threshold_analysis:
            if (
                "train" not in available_partitions
                or "test" not in available_partitions
            ):
                print(
                    "Warning: Threshold analysis requires both 'train' and 'test' partitions."
                )
                print("Available partitions:", available_partitions)
                print("Skipping threshold analysis.")
            else:
                print("\n" + "=" * 60)
                print("PERFORMING THRESHOLD ANALYSIS")
                print("=" * 60)

                try:
                    # Step 1: Find optimal thresholds on training data using specified method
                    threshold_method = "youden"  # Default method
                    print(
                        f"Finding optimal thresholds on training data using {threshold_method} method..."
                    )
                    threshold_results = analyzer.find_optimal_thresholds(
                        threshold_methods=[threshold_method]
                    )

                    if threshold_results:
                        print(
                            f"Found optimal thresholds for {len(threshold_results)} methods:"
                        )
                        for method_name, threshold_info in threshold_results.items():
                            threshold = threshold_info["threshold"]
                            method = threshold_info["method"]
                            target_recall = threshold_info.get("target_recall")

                            if target_recall:
                                print(
                                    f"  {method_name}: {method} (R={target_recall}) -> T={threshold:.4f}"
                                )
                            else:
                                print(f"  {method_name}: {method} -> T={threshold:.4f}")

                        # Step 2: Evaluate test data using these thresholds
                        print("\nEvaluating test data with optimal thresholds...")
                        evaluation_results = analyzer.evaluate_with_thresholds(
                            threshold_results, "test"
                        )

                        # Step 3: Generate threshold analysis plots
                        print("Generating threshold analysis plots...")
                        analyzer.generate_threshold_analysis_plots(
                            threshold_results,
                            args.plot_dir,
                            threshold_method=threshold_method,
                        )
                        analyzer.generate_test_threshold_analysis_plots(
                            threshold_results,
                            evaluation_results,
                            args.plot_dir,
                            threshold_method=threshold_method,
                        )

                        # Step 4: Generate threshold analysis report
                        print("Generating threshold analysis report...")
                        threshold_report = analyzer.generate_threshold_report(
                            threshold_results,
                            evaluation_results,
                            args.threshold_output,
                            threshold_method=threshold_method,
                        )

                        # Print summary of threshold analysis
                        print("\n" + "=" * 60)
                        print("THRESHOLD ANALYSIS SUMMARY")
                        print("=" * 60)

                        # Show chosen method and default threshold results
                        if evaluation_results:
                            # Find chosen method and default threshold results
                            chosen_results = None
                            default_results = None
                            for method_name, eval_info in evaluation_results.items():
                                if method_name == threshold_method:
                                    chosen_results = eval_info
                                elif method_name == "default_0.5":
                                    default_results = eval_info

                            if chosen_results:
                                chosen_metrics = chosen_results["test_metrics"]
                                print(
                                    f"{threshold_method.capitalize()} method results:"
                                )
                                print(f"  Threshold: {chosen_results['threshold']:.4f}")
                                print(f"  F1-score: {chosen_metrics['f1_score']:.4f}")
                                print(f"  MCC: {chosen_metrics['mcc']:.4f}")
                                print(f"  Accuracy: {chosen_metrics['accuracy']:.4f}")
                                print(f"  Precision: {chosen_metrics['precision']:.4f}")
                                print(f"  Recall: {chosen_metrics['recall']:.4f}")

                            if default_results:
                                default_metrics = default_results["test_metrics"]
                                print(f"\nEffectorP Default Threshold (0.5) results:")
                                print(
                                    f"  Threshold: {default_results['threshold']:.4f}"
                                )
                                print(f"  F1-score: {default_metrics['f1_score']:.4f}")
                                print(f"  MCC: {default_metrics['mcc']:.4f}")
                                print(f"  Accuracy: {default_metrics['accuracy']:.4f}")
                                print(
                                    f"  Precision: {default_metrics['precision']:.4f}"
                                )
                                print(f"  Recall: {default_metrics['recall']:.4f}")

                            # Show comparison between methods
                            if chosen_results and default_results:
                                print(
                                    f"\nComparison: {threshold_method.capitalize()} vs EffectorP Default Threshold (0.5):"
                                )
                                print(
                                    f"  {threshold_method.capitalize()} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                                )
                                print(
                                    f"  Default threshold (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                                )
                                print(
                                    f"  F1 improvement: {((chosen_metrics['f1_score'] / default_metrics['f1_score']) - 1) * 100:+.1f}%"
                                )
                                print(
                                    f"  MCC improvement: {((chosen_metrics['mcc'] / default_metrics['mcc']) - 1) * 100:+.1f}%"
                                )
                            elif chosen_results and "default_0.5" in evaluation_results:
                                # Fallback to old comparison if default not available
                                default_metrics = evaluation_results["default_0.5"][
                                    "test_metrics"
                                ]
                                print(
                                    f"\nComparison with EffectorP Default Threshold (0.5):"
                                )
                                print(
                                    f"  Default threshold (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                                )
                                print(
                                    f"  {threshold_method.capitalize()} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                                )
                                print(
                                    f"  F1 improvement: {((chosen_metrics['f1_score'] / default_metrics['f1_score']) - 1) * 100:+.1f}%"
                                )
                                print(
                                    f"  MCC improvement: {((chosen_metrics['mcc'] / default_metrics['mcc']) - 1) * 100:+.1f}%"
                                )
                            elif chosen_results:
                                # Fallback to old comparison if default not available
                                print(f"\nComparison with default threshold (0.5):")
                                default_metrics = analyzer.calculate_metrics("test")
                                print(
                                    f"  Default threshold (0.5): F1={default_metrics['f1_score']:.4f}, MCC={default_metrics['mcc']:.4f}"
                                )
                                print(
                                    f"  {threshold_method.capitalize()} method: F1={chosen_metrics['f1_score']:.4f}, MCC={chosen_metrics['mcc']:.4f}"
                                )

                        print(
                            f"\nDetailed threshold analysis saved to: {args.threshold_output}"
                        )
                        print(f"Threshold analysis plots saved to: {args.plot_dir}")

                    else:
                        print(
                            "Warning: No optimal thresholds found. Check training data quality."
                        )

                except Exception as e:
                    print(f"Error during threshold analysis: {str(e)}")
                    print("Continuing with standard analysis...")

        # Print summary to console
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)

        # Print key metrics
        all_metrics = analyzer.evaluation_results.get("all", {})
        if all_metrics:
            print(f"Overall Accuracy: {all_metrics['accuracy']:.4f}")
            print(f"Overall Precision: {all_metrics['precision']:.4f}")
            print(f"Overall Recall: {all_metrics['recall']:.4f}")
            print(f"Overall F1-score: {all_metrics['f1_score']:.4f}")
            print(f"Overall MCC: {all_metrics['mcc']:.4f}")
            if all_metrics.get("roc_auc") is not None:
                print(f"Overall ROC-AUC: {all_metrics['roc_auc']:.4f}")
            if all_metrics.get("auprc") is not None:
                print(f"Overall AUPRC: {all_metrics['auprc']:.4f}")
            if all_metrics.get("high_recall_auprc_0.7") is not None:
                print(
                    f"Overall High-recall AUPRC (0.7): {all_metrics['high_recall_auprc_0.7']:.4f}"
                )
            if all_metrics.get("high_recall_auprc_0.8") is not None:
                print(
                    f"Overall High-recall AUPRC (0.8): {all_metrics['high_recall_auprc_0.8']:.4f}"
                )

        print(f"\nDetailed report saved to: {args.output}")

        # Generate plot if requested
        if args.plot:
            plot_path = Path(args.output).stem + "_confusion_matrix.png"
            analyzer.plot_confusion_matrix(
                dataset_filter=args.partition if args.partition != "all" else None,
                save_path=plot_path,
            )

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
