# New Ice Cream Flavor Price Impact Simulator with LLM Integration

This simulator models purchase behavior for a new ice cream flavor launch, simulating 3000 households over a 2-month period. The simulation uses LLM-enhanced decision-making to predict purchase behavior and evaluates results against transaction data using precision, recall, F1-score, accuracy, and confusion matrix metrics.

## Overview

This simulator extends the base price impact simulator with:
- Ice cream product scenario (new flavor launch)
- LLM integration for enhanced behavioral modeling
- 2-month simulation period
- Evaluation metrics against transaction data

## Scenario

**Product**: New ice cream flavor
**New Price**: $4.99
**Baseline Competitor Price**: $3.99
**Simulation Period**: 2 months
**Households**: 3000

## Contents

- `run_simulation.py` - Main entry point with evaluation integration
- `runner_dbx.py` - Simulation runner
- `evaluation_metrics.py` - Evaluation metrics calculation
- `agenttorch_model/` - AgentTorch model package with LLM integration
- `requirements.txt` - Python dependencies

## Usage

### Basic Simulation

```bash
python run_simulation.py \
    --csv data/households.csv \
    --output output/results.json \
    --llm-enabled \
    --llm-provider openai \
    --llm-model gpt-3.5-turbo
```

### Simulation with Evaluation

```bash
python run_simulation.py \
    --csv data/households.csv \
    --output output/results.json \
    --transaction-data data/transaction_records.csv \
    --llm-enabled
```

### Standalone Evaluation

```bash
python evaluation_metrics.py \
    --simulation-results output/results.json \
    --transaction-data data/transaction_records.csv \
    --output output/evaluation.json
```

### Full Evaluation Pipeline (with option to use existing transaction data)

```bash
# Generate transaction data automatically
python run_full_evaluation.py \
    --csv data/households.csv \
    --output-dir output \
    --llm-enabled

# Or use existing transaction data CSV
python run_full_evaluation.py \
    --csv data/households.csv \
    --output-dir output \
    --transaction-data data/existing_transactions.csv \
    --llm-enabled
```

## Transaction Data Format

The transaction data CSV should have the following columns:

- `household_id`: Household identifier (integer)
- `purchased`: Purchase indicator (1 = purchased, 0 = not purchased, or boolean)

Example:
```csv
household_id,purchased
1,1
2,0
3,1
...
```

## Evaluation Metrics

The simulator calculates:
- **Precision**: Proportion of predicted purchases that were actual purchases
- **Recall**: Proportion of actual purchasers that were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall prediction correctness
- **Confusion Matrix**: True Positive, False Positive, False Negative, True Negative counts

## LLM Configuration

The simulator supports LLM integration for enhanced decision-making:

- **Provider**: OpenAI (default) or Anthropic
- **Model**: gpt-3.5-turbo (default) or other models (e.g., gpt-5.1 for proxy)
- **API Base URL**: Optional custom API base URL for proxy support
- **Temperature**: 0.7 (default) - controls randomness (0.0 = deterministic)
- **LLM Weight**: 0.3 (default) - balances LLM vs deterministic logic

### Using Custom API Proxy

To use a custom OpenAI proxy (like the example code), you can set environment variables or pass command-line arguments:

**Environment Variables:**
```bash
export MODEL_PROXY_API_KEY="your_proxy_api_key"
export MODEL_PROXY_API_BASE="https://your-proxy-url.com"
```

**Command-Line Arguments:**
```bash
python run_simulation.py \
    --csv data/households.csv \
    --output output/results.json \
    --llm-enabled \
    --llm-provider openai \
    --llm-model gpt-5.1 \
    --llm-api-key "your_proxy_api_key" \
    --llm-api-base-url "https://your-proxy-url.com"
```

**Important:** Always quote the URL value when using `--llm-api-base-url` to prevent shell interpretation issues:
- Correct: `--llm-api-base-url "https://example.com"`
- Correct: `--llm-api-base-url='https://example.com'`
- Incorrect: `--llm-api-base-url https://example.com` (may cause "command not found" errors)

The code automatically uses `MODEL_PROXY_API_KEY` and `MODEL_PROXY_API_BASE` environment variables if available, falling back to `OPENAI_API_KEY` and `OPENAI_API_BASE` if not set.

## Installation

```bash
pip install -r requirements.txt
```

## Dataloader Fix

This project includes a patch script to fix a TypeError in the AgentTorch dataloader when handling string/object data types. The error occurs when `LoadPopulation` tries to convert numpy.object arrays to PyTorch tensors.

**Apply the fix:**

```bash
python apply_dataloader_fix.py
```

This script will:
1. Locate the `agent_torch/core/dataloader.py` file in your Python environment
2. Add the `_fix_df` function that converts object-type columns to numeric types
3. Update the tensor creation to use the fixed DataFrame

The fix handles:
- Object-type columns: Converts to numeric if >50% can be converted, otherwise uses categorical codes
- Boolean columns: Converts to int8
- Ensures all data types are compatible with PyTorch tensors

**Note:** This fix must be applied after installing AgentTorch but before running simulations. The script is idempotent and can be run multiple times safely.

## Output

The simulation generates:
1. `results.json` - Simulation results with per-household predictions
2. `results_evaluation.json` - Evaluation metrics (if transaction data provided)

## Example Results

```json
{
  "scenario": {
    "product_category": "new_ice_cream_flavor",
    "new_price": 4.99,
    "baseline_price": 3.99
  },
  "purchasing_count": 1680.0,
  "purchase_rate": 0.56,
  "total_revenue": 8383.20,
  "total_units": 1680.0,
  "n_households": 3000,
  "household_predictions": {
    "household_ids": [1, 2, 3, ...],
    "purchased": [1, 0, 1, ...]
  }
}
```

## Evaluation Results

```json
{
  "metrics": {
    "precision": 0.82,
    "recall": 0.75,
    "f1_score": 0.78,
    "accuracy": 0.81,
    "confusion_matrix": {
      "true_positive": 1260,
      "false_positive": 420,
      "false_negative": 420,
      "true_negative": 900
    }
  }
}
```
