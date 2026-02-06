"""Evaluation metrics for comparing simulation results with transaction data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def load_transaction_data(transaction_csv_path: str) -> pd.DataFrame:
    """Load transaction data CSV file.

    Expected columns:
    - household_id: Household identifier
    - purchased: 1 if purchased, 0 if not (or boolean)
    - purchase_date: Purchase date (optional, for time-based analysis)
    - product_name: Product name (optional)
    - launch_date: Launch date (optional, for new product validation)
    """
    df = pd.read_csv(transaction_csv_path)

    # Validate that this is a new product (no purchases before launch date)
    if 'launch_date' in df.columns and 'purchase_date' in df.columns:
        launch_date = pd.to_datetime(df['launch_date'].iloc[0])
        purchase_dates = pd.to_datetime(df[df['purchased'] == 1]['purchase_date'], errors='coerce')

        # Check if any purchases occurred before launch
        if purchase_dates.notna().any():
            before_launch = purchase_dates < launch_date
            if before_launch.any():
                print(f"Warning: Found {before_launch.sum()} purchases before launch date {launch_date.date()}")
                print("This should be a new product with no prior purchases.")

    return df


def prepare_evaluation_data(
    simulation_results: Dict,
    transaction_data: pd.DataFrame,
    household_id_col: str = "household_id",
    purchased_col: str = "purchased",
) -> tuple:
    """Prepare data for evaluation by aligning simulation and transaction data.

    Args:
        simulation_results: Results dict from run_simulation (should contain household_predictions)
        transaction_data: DataFrame with transaction records
        household_id_col: Column name for household ID in transaction data
        purchased_col: Column name for purchase indicator in transaction data

    Returns:
        Tuple of (y_true, y_pred) arrays for evaluation
    """
    # Extract simulation predictions from household_predictions if available
    household_predictions = simulation_results.get("household_predictions", {})

    if household_predictions and "household_ids" in household_predictions and "purchased" in household_predictions:
        # Use per-household predictions from simulation
        sim_hh_ids = household_predictions["household_ids"]
        sim_purchased = household_predictions["purchased"]

        # Create mapping from household ID to prediction
        sim_dict = dict(zip(sim_hh_ids, sim_purchased))
    else:
        # Fallback: create predictions from aggregate count
        n_households = simulation_results.get("n_households", 0)
        purchasing_count = int(simulation_results.get("purchasing_count", 0))
        sim_hh_ids = list(range(1, n_households + 1))
        sim_purchased = [1] * purchasing_count + [0] * (n_households - purchasing_count)
        sim_dict = dict(zip(sim_hh_ids, sim_purchased))

    # Get all unique household IDs from both sources
    all_hh_ids = set(sim_hh_ids)
    if household_id_col in transaction_data.columns:
        all_hh_ids.update(transaction_data[household_id_col].unique())
    else:
        # Assume transaction data household IDs match simulation
        all_hh_ids.update(range(1, len(transaction_data) + 1))

    all_hh_ids = sorted(list(all_hh_ids))
    n_total = len(all_hh_ids)

    # Create prediction array
    y_pred = np.zeros(n_total)
    for i, hh_id in enumerate(all_hh_ids):
        y_pred[i] = sim_dict.get(hh_id, 0)

    # Create true labels from transaction data
    y_true = np.zeros(n_total)

    if purchased_col in transaction_data.columns:
        # Create mapping from transaction data
        trans_dict = {}
        for idx, row in transaction_data.iterrows():
            hh_id = row.get(household_id_col, idx + 1)
            purchased = row.get(purchased_col, 0)

            # Convert to binary
            if isinstance(purchased, bool):
                purchased_val = 1 if purchased else 0
            elif isinstance(purchased, (int, float, np.integer, np.floating)):
                purchased_val = 1 if float(purchased) > 0 else 0
            else:
                purchased_val = 1 if str(purchased).lower() in ['true', '1', 'yes', 'y'] else 0

            trans_dict[hh_id] = purchased_val

        # Map to evaluation array
        for i, hh_id in enumerate(all_hh_ids):
            y_true[i] = trans_dict.get(hh_id, 0)
    else:
        # No purchase column - assume all transactions are purchases
        if household_id_col in transaction_data.columns:
            trans_hh_ids = set(transaction_data[household_id_col].unique())
            for i, hh_id in enumerate(all_hh_ids):
                if hh_id in trans_hh_ids:
                    y_true[i] = 1

    return y_true, y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate classification metrics.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary with precision, recall, F1-score, accuracy, and confusion matrix
    """
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract confusion matrix components
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "confusion_matrix_array": cm.tolist(),
    }

    return metrics


def evaluate_simulation(
    simulation_results_path: str,
    transaction_data_path: str,
    output_path: Optional[str] = None,
) -> Dict:
    """Evaluate simulation results against transaction data.

    Args:
        simulation_results_path: Path to simulation results JSON file
        transaction_data_path: Path to transaction data CSV file
        output_path: Optional path to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    # Load simulation results
    with open(simulation_results_path, "r") as f:
        simulation_results = json.load(f)

    # Load transaction data
    transaction_data = load_transaction_data(transaction_data_path)

    # Prepare evaluation data
    y_true, y_pred = prepare_evaluation_data(simulation_results, transaction_data)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Add summary
    evaluation_results = {
        "simulation_results_path": simulation_results_path,
        "transaction_data_path": transaction_data_path,
        "n_households": len(y_true),
        "metrics": metrics,
        "summary": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
        },
    }

    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "w") as f:
            json.dump(evaluation_results, f, indent=2)

    return evaluation_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate simulation results against transaction data")
    parser.add_argument("--simulation-results", type=str, required=True, help="Path to simulation results JSON")
    parser.add_argument("--transaction-data", type=str, required=True, help="Path to transaction data CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results")

    args = parser.parse_args()

    results = evaluate_simulation(
        args.simulation_results,
        args.transaction_data,
        args.output,
    )

    print("\nEvaluation Results:")
    print(f"  Precision: {results['metrics']['precision']:.4f}")
    print(f"  Recall: {results['metrics']['recall']:.4f}")
    print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positive: {results['metrics']['confusion_matrix']['true_positive']}")
    print(f"  False Positive: {results['metrics']['confusion_matrix']['false_positive']}")
    print(f"  False Negative: {results['metrics']['confusion_matrix']['false_negative']}")
    print(f"  True Negative: {results['metrics']['confusion_matrix']['true_negative']}")
