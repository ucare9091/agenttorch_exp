#!/usr/bin/env python3
"""Run complete simulation and evaluation pipeline."""

import argparse
import json
import sys
from pathlib import Path

from generate_transaction_data import generate_transaction_data
from runner_dbx import run_simulation
from evaluation_metrics import evaluate_simulation


def main():
    parser = argparse.ArgumentParser(description="Run complete simulation and evaluation pipeline")
    parser.add_argument("--csv", type=str, required=True, help="Path to household CSV file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--product-category", type=str, default="new_ice_cream_flavor", help="Product category name")
    parser.add_argument("--new-price", type=float, default=4.99, help="New product price")
    parser.add_argument("--baseline-price", type=float, default=3.99, help="Baseline competitor price")
    parser.add_argument("--launch-date", type=str, default="2024-06-01", help="Launch date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=2, help="Number of months to simulate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")

    # LLM configuration
    parser.add_argument("--llm-enabled", action="store_true", help="Enable LLM integration")
    parser.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "anthropic"], help="LLM provider")
    parser.add_argument("--llm-model", type=str, default="gpt-3.5-turbo", help="LLM model name (e.g., 'gpt-3.5-turbo' or 'gpt-5.1' for proxy)")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key (or set MODEL_PROXY_API_KEY/OPENAI_API_KEY env var)")
    parser.add_argument("--llm-api-base-url", type=str, default=None, help="LLM API base URL for proxy support (or set MODEL_PROXY_API_BASE/OPENAI_API_BASE env var)")
    parser.add_argument("--llm-temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--llm-sample-size", type=int, default=0, help="Number of households to use LLM for")
    parser.add_argument("--llm-use-for-all", action="store_true", help="Use LLM for all households")
    parser.add_argument("--llm-weight", type=float, default=0.3, help="LLM weight vs deterministic logic")

    # Transaction data options
    parser.add_argument("--transaction-data", type=str, default=None, help="Path to existing transaction data CSV file (if provided, skips generation)")
    parser.add_argument("--purchase-probability", type=float, default=0.56, help="Base purchase probability for transaction data generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run simulation
    print("STEP 1: Running Simulation")

    scenario = {
        "product_category": args.product_category,
        "new_price": args.new_price,
        "baseline_price": args.baseline_price,
    }

    llm_config = None
    if args.llm_enabled:
        # Safely get llm_api_base_url attribute (argparse converts --llm-api-base-url to llm_api_base_url)
        api_base_url = getattr(args, 'llm_api_base_url', None)
        api_key = getattr(args, 'llm_api_key', None)

        llm_config = {
            "enabled": True,
            "provider": args.llm_provider,
            "model": args.llm_model,
            "api_key": api_key,
            "api_base_url": api_base_url,
            "temperature": args.llm_temperature,
            "max_tokens": 150,
            "sample_size": args.llm_sample_size,
            "use_for_all": args.llm_use_for_all,
            "llm_weight": args.llm_weight,
        }
        print(f"LLM enabled: {args.llm_provider}/{args.llm_model}")
        if api_base_url:
            print(f"  Using custom API base URL: {api_base_url}")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        simulation_results = run_simulation(
            csv_path=str(csv_path),
            scenario=scenario,
            device=args.device,
            llm_config=llm_config,
        )

        simulation_output = output_dir / "simulation_results.json"
        with open(simulation_output, "w") as f:
            json.dump(simulation_results, f, indent=2)

        print(f"\nSimulation completed!")
        print(f"  Total households: {simulation_results['n_households']}")
        print(f"  Purchasing households: {simulation_results['purchasing_count']:.0f}")
        print(f"  Purchase rate: {simulation_results['purchase_rate']:.2%}")
        print(f"  Total revenue: ${simulation_results['total_revenue']:.2f}")
        print(f"  Total units sold: {simulation_results['total_units']:.2f}")
        print(f"  Results saved to: {simulation_output}")

    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Get transaction data (generate or use existing)
    if args.transaction_data:
        # Use existing transaction data file
        print("\nSTEP 2: Using Existing Transaction Data")
        transaction_path = Path(args.transaction_data)
        if not transaction_path.exists():
            print(f"Error: Transaction data file not found: {transaction_path}", file=sys.stderr)
            sys.exit(1)

        # Validate transaction data format
        try:
            import pandas as pd
            df_trans = pd.read_csv(transaction_path)
            required_cols = ["household_id", "purchased"]
            missing_cols = [col for col in required_cols if col not in df_trans.columns]
            if missing_cols:
                print(f"Error: Transaction data missing required columns: {missing_cols}", file=sys.stderr)
                print(f"Expected columns: {required_cols}", file=sys.stderr)
                print(f"Found columns: {list(df_trans.columns)}", file=sys.stderr)
                sys.exit(1)
            print(f"  Using transaction data from: {transaction_path}")
            print(f"  Records: {len(df_trans)}")
            print(f"  Purchases: {df_trans['purchased'].sum()}")
            transaction_output = transaction_path
        except Exception as e:
            print(f"Error reading transaction data: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Generate transaction data
        print("\nSTEP 2: Generating Transaction Data")
        transaction_output = output_dir / "transaction_data.csv"

        try:
            generate_transaction_data(
                household_csv_path=str(csv_path),
                output_path=str(transaction_output),
                product_name=args.product_category,
                start_date=args.launch_date,
                months=args.months,
                purchase_probability_base=args.purchase_probability,
                seed=args.seed,
            )
            print(f"  Generated transaction data: {transaction_output}")
        except Exception as e:
            print(f"Error generating transaction data: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Step 3: Evaluate simulation against transaction data
    print("\nSTEP 3: Evaluating Simulation Against Transaction Data")

    evaluation_output = output_dir / "evaluation_results.json"

    try:
        eval_results = evaluate_simulation(
            str(simulation_output),
            str(transaction_output),
            str(evaluation_output),
        )

        print("\nEvaluation Metrics:")
        print(f"  Precision: {eval_results['metrics']['precision']:.4f}")
        print(f"  Recall: {eval_results['metrics']['recall']:.4f}")
        print(f"  F1-Score: {eval_results['metrics']['f1_score']:.4f}")
        print(f"  Accuracy: {eval_results['metrics']['accuracy']:.4f}")

        print("\nConfusion Matrix:")
        cm = eval_results['metrics']['confusion_matrix']
        print(f"  True Positive:  {cm['true_positive']:4d}  (Correctly predicted purchases)")
        print(f"  False Positive: {cm['false_positive']:4d}  (Predicted purchase, but didn't purchase)")
        print(f"  False Negative: {cm['false_negative']:4d}  (Didn't predict purchase, but purchased)")
        print(f"  True Negative:  {cm['true_negative']:4d}  (Correctly predicted non-purchases)")

        print(f"\nEvaluation results saved to: {evaluation_output}")

        # Print summary
        print("\nSUMMARY")
        print(f"Product: {args.product_category}")
        print(f"Launch Date: {args.launch_date}")
        print(f"Evaluation Period: {args.months} months")
        print(f"Simulated Purchase Rate: {simulation_results['purchase_rate']:.2%}")
        print(f"Model Performance:")
        print(f"  - Precision: {eval_results['metrics']['precision']:.2%} (of predicted purchases, {eval_results['metrics']['precision']:.2%} were correct)")
        print(f"  - Recall: {eval_results['metrics']['recall']:.2%} (of actual purchases, {eval_results['metrics']['recall']:.2%} were predicted)")
        print(f"  - F1-Score: {eval_results['metrics']['f1_score']:.4f} (balanced metric)")
        print(f"  - Accuracy: {eval_results['metrics']['accuracy']:.2%} (overall correctness)")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()
