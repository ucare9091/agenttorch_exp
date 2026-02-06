#!/usr/bin/env python3
"""Main entry point for running the price impact simulation in Docker."""

import argparse
import json
import sys
from pathlib import Path

from runner_dbx import run_simulation
from evaluation_metrics import evaluate_simulation


def main():
    parser = argparse.ArgumentParser(description="Run price impact simulation with household CSV data")
    parser.add_argument("--csv", type=str, default="data/households.csv", help="Path to household CSV file")
    parser.add_argument("--output", type=str, default="output/results.json", help="Path to output JSON file")
    parser.add_argument("--product-category", type=str, default="new_ice_cream_flavor", help="Product category name")
    parser.add_argument("--new-price", type=float, default=4.99, help="New product price")
    parser.add_argument("--baseline-price", type=float, default=3.99, help="Baseline competitor price")
    parser.add_argument("--transaction-data", type=str, default=None, help="Path to transaction data CSV for evaluation")
    parser.add_argument("--simulation-months", type=int, default=2, help="Number of months to simulate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")

    # LLM configuration arguments
    parser.add_argument("--llm-enabled", action="store_true", help="Enable LLM integration")
    parser.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "anthropic"], help="LLM provider")
    parser.add_argument("--llm-model", type=str, default="gpt-3.5-turbo", help="LLM model name (e.g., 'gpt-3.5-turbo' or 'gpt-5.1' for proxy)")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key (or set MODEL_PROXY_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY env var)")
    parser.add_argument("--llm-api-base-url", type=str, default=None, help="LLM API base URL for proxy support (or set MODEL_PROXY_API_BASE/OPENAI_API_BASE env var)")
    parser.add_argument("--llm-temperature", type=float, default=0.7, help="LLM temperature (0.0-2.0)")
    parser.add_argument("--llm-max-tokens", type=int, default=150, help="LLM max tokens")
    parser.add_argument("--llm-sample-size", type=int, default=0, help="Number of households to use LLM for (0 = all if --llm-use-for-all)")
    parser.add_argument("--llm-use-for-all", action="store_true", help="Use LLM for all households (not just sample)")
    parser.add_argument("--llm-weight", type=float, default=0.3, help="LLM weight vs deterministic logic (0.0-1.0)")

    args = parser.parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    scenario = {
        "product_category": args.product_category,
        "new_price": args.new_price,
        "baseline_price": args.baseline_price,
    }

    # Build LLM config if enabled
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
            "max_tokens": args.llm_max_tokens,
            "sample_size": args.llm_sample_size,
            "use_for_all": args.llm_use_for_all,
            "llm_weight": args.llm_weight,
        }
        print(f"LLM enabled: {args.llm_provider}/{args.llm_model}")
        if api_base_url:
            print(f"  Using custom API base URL: {api_base_url}")
        if args.llm_sample_size > 0:
            print(f"  Using LLM for {args.llm_sample_size} sample households")
        elif args.llm_use_for_all:
            print(f"  Using LLM for all households")
    else:
        print("LLM disabled (use --llm-enabled to enable)")

    print(f"Loading households from: {csv_path}")
    print(f"Running simulation with scenario: {scenario}")
    print(f"Using device: {args.device}")

    try:
        results = run_simulation(csv_path=str(csv_path), scenario=scenario, device=args.device, llm_config=llm_config)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("\nSimulation completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"  Total households: {results['n_households']}")
        print(f"  Purchasing households: {results['purchasing_count']:.0f}")
        print(f"  Purchase rate: {results['purchase_rate']:.2%}")
        print(f"  Total revenue: ${results['total_revenue']:.2f}")
        print(f"  Total units sold: {results['total_units']:.2f}")

        # Evaluate against transaction data if provided
        if args.transaction_data:
            transaction_path = Path(args.transaction_data)
            if transaction_path.exists():
                print("\nEvaluating against transaction data...")
                eval_output = output_path.parent / f"{output_path.stem}_evaluation.json"
                eval_results = evaluate_simulation(
                    str(output_path),
                    str(transaction_path),
                    str(eval_output),
                )
                print("\nEvaluation Metrics:")
                print(f"  Precision: {eval_results['metrics']['precision']:.4f}")
                print(f"  Recall: {eval_results['metrics']['recall']:.4f}")
                print(f"  F1-Score: {eval_results['metrics']['f1_score']:.4f}")
                print(f"  Accuracy: {eval_results['metrics']['accuracy']:.4f}")
                print("\nConfusion Matrix:")
                cm = eval_results['metrics']['confusion_matrix']
                print(f"  True Positive: {cm['true_positive']}")
                print(f"  False Positive: {cm['false_positive']}")
                print(f"  False Negative: {cm['false_negative']}")
                print(f"  True Negative: {cm['true_negative']}")
                print(f"\nEvaluation results saved to: {eval_output}")
            else:
                print(f"\nWarning: Transaction data file not found: {transaction_path}")
    except Exception as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
