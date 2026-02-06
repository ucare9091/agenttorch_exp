"""Generate simulated transaction data for 2 months evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd


def generate_transaction_data(
    household_csv_path: str,
    output_path: str,
    product_name: str = "new_ice_cream_flavor",
    start_date: str = "2024-06-01",
    months: int = 2,
    purchase_probability_base: float = 0.56,
    seed: int = 42,
) -> None:
    """
    Generate simulated transaction data for 2 months.

    This simulates a new ice cream flavor launch that was NOT available before.
    Only purchases after the launch date are recorded.

    Args:
        household_csv_path: Path to household CSV file
        output_path: Path to save transaction data CSV
        product_name: Name of the new product
        start_date: Launch date (YYYY-MM-DD)
        months: Number of months to simulate
        purchase_probability_base: Base probability of purchase (0.0-1.0)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load household data
    df_households = pd.read_csv(household_csv_path)
    n_households = len(df_households)

    # Parse start date
    launch_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = launch_date + timedelta(days=months * 30)

    # Generate transaction records
    transactions = []

    # For each household, simulate purchase behavior over 2 months
    for idx, household in df_households.iterrows():
        household_id = household.get('household_id', idx + 1)

        # Adjust purchase probability based on household characteristics
        base_prob = purchase_probability_base

        # Income effect: higher income = more likely to try new premium product
        income = household.get('income', 50000)
        income_factor = min(income / 100000.0, 1.0)  # Normalize to 0-1
        income_boost = income_factor * 0.15  # Up to 15% increase

        # Price sensitivity: lower sensitivity = more likely to purchase
        price_sensitivity = household.get('price_sensitivity', 0.5)
        sensitivity_penalty = price_sensitivity * 0.2  # Up to 20% decrease

        # Emotional state: happy households more likely to try new treats
        emotional_state = str(household.get('emotional_state', 'neutral')).lower()
        emotion_boost = 0.0
        if 'happy' in emotional_state or 'very_happy' in emotional_state:
            emotion_boost = 0.1  # 10% boost
        elif 'stressed' in emotional_state or 'very_stressed' in emotional_state:
            emotion_boost = -0.05  # 5% penalty

        # Calculate adjusted probability
        adjusted_prob = base_prob + income_boost - sensitivity_penalty + emotion_boost
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))  # Clamp to 0-1

        # Simulate purchase decision (1 = purchased, 0 = not purchased)
        # This represents whether the household purchased the new flavor at least once in 2 months
        purchased = 1 if random.random() < adjusted_prob else 0

        # If purchased, simulate purchase date (within 2 months after launch)
        purchase_date = None
        if purchased:
            # Purchase date is random within the 2-month window
            days_after_launch = random.randint(0, months * 30)
            purchase_date = launch_date + timedelta(days=days_after_launch)

        transactions.append({
            'household_id': household_id,
            'product_name': product_name,
            'purchased': purchased,
            'purchase_date': purchase_date.strftime('%Y-%m-%d') if purchase_date else None,
            'launch_date': launch_date.strftime('%Y-%m-%d'),
            'evaluation_period_start': launch_date.strftime('%Y-%m-%d'),
            'evaluation_period_end': end_date.strftime('%Y-%m-%d'),
        })

    # Create DataFrame and save
    df_transactions = pd.DataFrame(transactions)
    df_transactions.to_csv(output_path, index=False)

    # Print summary
    total_purchases = df_transactions['purchased'].sum()
    purchase_rate = total_purchases / n_households

    print(f"\nGenerated transaction data:")
    print(f"  Total households: {n_households}")
    print(f"  Purchasing households: {total_purchases}")
    print(f"  Purchase rate: {purchase_rate:.2%}")
    print(f"  Product: {product_name}")
    print(f"  Launch date: {launch_date.strftime('%Y-%m-%d')}")
    print(f"  Evaluation period: {launch_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate simulated transaction data for evaluation")
    parser.add_argument("--household-csv", type=str, required=True, help="Path to household CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save transaction data CSV")
    parser.add_argument("--product-name", type=str, default="new_ice_cream_flavor", help="Product name")
    parser.add_argument("--start-date", type=str, default="2024-06-01", help="Launch date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=2, help="Number of months to simulate")
    parser.add_argument("--purchase-probability", type=float, default=0.56, help="Base purchase probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_transaction_data(
        household_csv_path=args.household_csv,
        output_path=args.output,
        product_name=args.product_name,
        start_date=args.start_date,
        months=args.months,
        purchase_probability_base=args.purchase_probability,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
