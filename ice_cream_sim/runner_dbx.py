from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch

import yaml
from agent_torch.core.dataloader import LoadPopulation
from agent_torch.core.executor import Executor


def find_column(df: pd.DataFrame, possible_names: list, default_value=None):
    """
    Find column by multiple possible names (case-insensitive, partial match).

    Handles any column names by trying:
    1. Exact match (case-insensitive)
    2. Partial match (substring in either direction)
    3. Word-based match (matches key words like 'income', 'size', etc.)

    Returns the matching column name or None if not found.
    """
    if df is None or len(df.columns) == 0:
        return None

    df_cols_lower = {col.lower(): col for col in df.columns}

    # Try exact match first (case-insensitive)
    for name in possible_names:
        name_lower = name.lower()
        if name_lower in df_cols_lower:
            return df_cols_lower[name_lower]

    # Try partial match (substring in either direction)
    for name in possible_names:
        name_lower = name.lower()
        for col_lower, col_orig in df_cols_lower.items():
            if name_lower in col_lower or col_lower in name_lower:
                return col_orig

    # Try word-based matching (extract key words from possible names)
    for name in possible_names:
        name_lower = name.lower()
        # Extract key words (remove common prefixes/suffixes)
        key_words = [w for w in name_lower.split('_') if len(w) > 2]
        for col_lower, col_orig in df_cols_lower.items():
            col_words = [w for w in col_lower.split('_') if len(w) > 2]
            # Check if any key word matches any column word
            if any(kw in cw or cw in kw for kw in key_words for cw in col_words):
                return col_orig

    return None


def load_households(csv_path: str) -> pd.DataFrame:
    """Load household CSV with flexible column name matching."""
    df = pd.read_csv(csv_path)
    return df


def build_agent_states(df: pd.DataFrame) -> Dict:
    """Build agent states from DataFrame with flexible column matching.

    Handles any column names and any number of rows efficiently.
    Missing columns use sensible defaults.

    This function is designed to work with ANY CSV structure:
    - Any number of rows (1 to millions)
    - Any column names (will try to match common patterns)
    - Missing columns are filled with sensible defaults
    - Extra columns are ignored
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty or None")

    n_rows = len(df)

    # Flexible column matching - tries many common variations
    id_col = find_column(df, ["id", "household_id", "householdid", "hh_id", "hhid", "household", "hh"])
    income_col = find_column(df, ["income", "household_income", "hh_income", "annual_income", "salary", "earnings", "revenue"])
    size_col = find_column(df, ["household_size", "hh_size", "householdsize", "people_in_hh", "num_people", "size", "people", "members", "persons"])
    children_col = find_column(df, ["num_children", "children", "num_children_in_hh", "kids", "child", "dependents"])
    sensitivity_col = find_column(df, ["price_sensitivity", "price_sens", "sensitivity", "price_elasticity", "elasticity", "sens"])
    location_col = find_column(df, ["location", "region", "geo", "geography", "area", "zip", "zipcode", "state", "city"])
    emotional_state_col = find_column(df, ["emotional_state", "emotion", "mood", "feeling", "stress_level", "happiness", "anxiety", "sentiment", "emotional", "mood_state"])

    # Extract or default values for each required field
    # Handle any data type issues gracefully
    try:
        if id_col:
            household_ids = pd.to_numeric(df[id_col], errors='coerce').fillna(0).astype(int).tolist()
        else:
            household_ids = list(range(1, n_rows + 1))
    except Exception:
        household_ids = list(range(1, n_rows + 1))

    try:
        if income_col:
            incomes = pd.to_numeric(df[income_col], errors='coerce').fillna(50000.0).astype(float).tolist()
        else:
            incomes = [50000.0] * n_rows
    except Exception:
        incomes = [50000.0] * n_rows

    try:
        if size_col:
            household_sizes = pd.to_numeric(df[size_col], errors='coerce').fillna(2.0).astype(float).tolist()
        else:
            household_sizes = [2.0] * n_rows
    except Exception:
        household_sizes = [2.0] * n_rows

    try:
        if children_col:
            num_children = pd.to_numeric(df[children_col], errors='coerce').fillna(0.0).astype(float).tolist()
        else:
            num_children = [0.0] * n_rows
    except Exception:
        num_children = [0.0] * n_rows

    try:
        if sensitivity_col:
            price_sensitivities = pd.to_numeric(df[sensitivity_col], errors='coerce').fillna(0.5).astype(float).tolist()
        else:
            price_sensitivities = [0.5] * n_rows
    except Exception:
        price_sensitivities = [0.5] * n_rows

    try:
        if location_col:
            locations = df[location_col].fillna("unknown").astype(str).tolist()
        else:
            locations = ["unknown"] * n_rows
    except Exception:
        locations = ["unknown"] * n_rows

    try:
        if emotional_state_col:
            # Handle both numeric (0-1 scale) and string (descriptive) emotional states
            emotional_states_raw = df[emotional_state_col].fillna("neutral").astype(str).tolist()
            # Normalize to common emotional states if numeric
            emotional_states = []
            for emo in emotional_states_raw:
                try:
                    emo_val = float(emo)
                    # Map numeric to emotional states: 0.0-0.2=very_stressed, 0.2-0.4=stressed, 0.4-0.6=neutral, 0.6-0.8=happy, 0.8-1.0=very_happy
                    if emo_val < 0.2:
                        emotional_states.append("very_stressed")
                    elif emo_val < 0.4:
                        emotional_states.append("stressed")
                    elif emo_val < 0.6:
                        emotional_states.append("neutral")
                    elif emo_val < 0.8:
                        emotional_states.append("happy")
                    else:
                        emotional_states.append("very_happy")
                except (ValueError, TypeError):
                    # Already a string, normalize common variations
                    emo_lower = str(emo).lower().strip()
                    if emo_lower in ["very_stressed", "very stressed", "anxious", "worried", "distressed"]:
                        emotional_states.append("very_stressed")
                    elif emo_lower in ["stressed", "stress", "concerned", "tense"]:
                        emotional_states.append("stressed")
                    elif emo_lower in ["neutral", "normal", "calm", "balanced", ""]:
                        emotional_states.append("neutral")
                    elif emo_lower in ["happy", "content", "satisfied", "pleased"]:
                        emotional_states.append("happy")
                    elif emo_lower in ["very_happy", "very happy", "elated", "joyful", "excited", "euphoric"]:
                        emotional_states.append("very_happy")
                    else:
                        emotional_states.append("neutral")  # Default fallback
        else:
            emotional_states = ["neutral"] * n_rows
    except Exception:
        emotional_states = ["neutral"] * n_rows

    households = [
        {
            "household_id": int(hid),
            "income": float(inc),
            "household_size": float(size),
            "num_children": float(children),
            "price_sensitivity": float(sens),
            "location": str(loc),
            "emotional_state": str(emo),
        }
        for hid, inc, size, children, sens, loc, emo in zip(
            household_ids, incomes, household_sizes, num_children, price_sensitivities, locations, emotional_states
        )
    ]

    return {"households": households, "n_agents": len(households)}


def write_population_pickles(base_dir: Path, agent_states: Dict) -> Path:
    """Create agenttorch_population directory with pickle files and __init__.py."""
    population_dir = base_dir / "agenttorch_population"
    population_dir.mkdir(parents=True, exist_ok=True)
    init_path = population_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text('"""AgentTorch population package."""\n')

    households = agent_states["households"]

    # Numeric data for LoadPopulation (will be converted to torch tensors)
    numeric_data = {
        "household_id": [h.get("household_id", 0) for h in households],
        "income": [h.get("income", 0.0) for h in households],
        "household_size": [h.get("household_size", 0.0) for h in households],
        "price_sensitivity": [h.get("price_sensitivity", 0.0) for h in households],
    }

    # String data for load_population_attribute (saved separately, not loaded by LoadPopulation)
    string_data = {
        "location": [h.get("location", "unknown") for h in households],
        "emotional_state": [h.get("emotional_state", "neutral") for h in households],
    }

    # Save numeric data as pickle (will be loaded by LoadPopulation)
    for key, values in numeric_data.items():
        series = pd.Series(values)
        series.to_pickle(population_dir / f"{key}.pickle")

    # Save string data as pickle (loaded by load_population_attribute, not LoadPopulation)
    # LoadPopulation will try to load these but fail, so we need to handle this
    # We'll save them but LoadPopulation should skip non-numeric types
    for key, values in string_data.items():
        series = pd.Series(values)
        series.to_pickle(population_dir / f"{key}.pickle")

    return population_dir


def build_model_config(
    base_dir: Path,
    agent_states: Dict,
    device: str = "cpu",
    product_category: str = "new_ice_cream_flavor",
    new_price: float = 4.99,
    baseline_price: float = 3.99,
    llm_config: Optional[Dict] = None,
) -> Path:
    """Build AgentTorch config YAML directly without using config builders."""
    model_dir = base_dir / "agenttorch_model"
    yamls_dir = model_dir / "yamls"
    yamls_dir.mkdir(parents=True, exist_ok=True)

    num_agents = len(agent_states["households"])
    population_dir = base_dir / "agenttorch_population"

    def _pop_prop_dict(name: str, dtype: str):
        """Create property dict for population-loaded attributes."""
        return {
            "dtype": dtype,
            "initialization_function": {
                "generator": "load_population_attribute",
                "arguments": {
                    "file_path": {
                        "value": str(population_dir / f"{name}.pickle"),
                        "shape": None,
                        "learnable": False,
                        "initialization_function": None,
                    }
                },
            },
            "learnable": False,
            "name": name,
            "shape": [num_agents, 1],
            "value": None,
        }

    config_dict = {
        "simulation_metadata": {
            "calibration": False,
            "device": device,
            "num_agents": num_agents,
            "num_episodes": 1,
            "num_steps_per_episode": 1,
            "num_substeps_per_step": 1,
            "population_dir": str(population_dir),
        },
        "state": {
            "agents": {
                "households": {
                    "number": num_agents,
                    "properties": {
                        "household_id": _pop_prop_dict("household_id", "int"),
                        "income": _pop_prop_dict("income", "float"),
                        "household_size": _pop_prop_dict("household_size", "float"),
                        "price_sensitivity": _pop_prop_dict("price_sensitivity", "float"),
                        "location": {
                            "dtype": "str",
                            "initialization_function": {
                                "generator": "load_population_attribute",
                                "arguments": {
                                    "file_path": {
                                        "value": str(population_dir / "location.pickle"),
                                        "shape": None,
                                        "learnable": False,
                                        "initialization_function": None,
                                    }
                                },
                            },
                            "learnable": False,
                            "name": "location",
                            "shape": [num_agents, 1],
                            "value": None,
                        },
                        "emotional_state": {
                            "dtype": "str",
                            "initialization_function": {
                                "generator": "load_population_attribute",
                                "arguments": {
                                    "file_path": {
                                        "value": str(population_dir / "emotional_state.pickle"),
                                        "shape": None,
                                        "learnable": False,
                                        "initialization_function": None,
                                    }
                                },
                            },
                            "learnable": False,
                            "name": "emotional_state",
                            "shape": [num_agents, 1],
                            "value": None,
                        },
                        "will_purchase": {
                            "dtype": "float",
                            "initialization_function": None,
                            "learnable": False,
                            "name": "will_purchase",
                            "shape": [num_agents, 1],
                            "value": 0.0,
                        },
                        "quantity": {
                            "dtype": "float",
                            "initialization_function": None,
                            "learnable": False,
                            "name": "quantity",
                            "shape": [num_agents, 1],
                            "value": 0.0,
                        },
                        "total_spend": {
                            "dtype": "float",
                            "initialization_function": None,
                            "learnable": False,
                            "name": "total_spend",
                            "shape": [num_agents, 1],
                            "value": 0.0,
                        },
                    },
                },
            },
            "environment": {
                "new_price": {
                    "dtype": "float",
                    "initialization_function": None,
                    "learnable": False,
                    "name": "new_price",
                    "shape": [1],
                    "value": new_price,
                },
                "baseline_price": {
                    "dtype": "float",
                    "initialization_function": None,
                    "learnable": False,
                    "name": "baseline_price",
                    "shape": [1],
                    "value": baseline_price,
                },
                "product_category": {
                    "dtype": "str",
                    "initialization_function": None,
                    "learnable": False,
                    "name": "product_category",
                    "shape": None,
                    "value": product_category,
                },
                "llm_config": {
                    "dtype": "dict",
                    "initialization_function": None,
                    "learnable": False,
                    "name": "llm_config",
                    "shape": None,
                    "value": llm_config if llm_config else {},
                },
            },
            "network": {},
            "objects": None,
        },
        "substeps": {
            "0": {
                "active_agents": ["households"],
                "description": "Simulate household purchase decisions based on price",
                "name": "purchase_decision",
                "observation": {"households": None},
                "policy": {
                    "households": {
                        "purchase_decision": {
                            "generator": "purchase_decision",
                            "input_variables": {
                                "income": "agents/households/income",
                                "household_size": "agents/households/household_size",
                                "price_sensitivity": "agents/households/price_sensitivity",
                                "location": "agents/households/location",
                                "emotional_state": "agents/households/emotional_state",
                                "new_price": "environment/new_price",
                                "baseline_price": "environment/baseline_price",
                            },
                            "output_variables": ["will_purchase", "quantity", "total_spend"],
                            "arguments": {
                                "llm": llm_config if llm_config else {},
                            },
                        },
                    },
                },
                "reward": None,
                "transition": {
                    "apply_purchase_decision": {
                        "generator": "apply_purchase_decision",
                        "input_variables": {
                            "will_purchase": "agents/households/will_purchase",
                            "quantity": "agents/households/quantity",
                            "total_spend": "agents/households/total_spend",
                        },
                        "output_variables": ["will_purchase", "quantity", "total_spend"],
                        "arguments": {},
                    },
                },
            },
        },
    }

    config_path = yamls_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return config_path


def run_simulation(
    csv_path: str,
    scenario: Optional[Dict] = None,
    device: str = "cpu",
    llm_config: Optional[Dict] = None,
) -> Dict:
    """
    Run AgentTorch simulation using the original framework structure.

    This follows the AgentTorch large population model framework:
    - Uses Executor(model_module, pop_loader=population) pattern
    - Model module follows AgentTorch GitHub structure
    - Population loader uses LoadPopulation from agent_torch.core.dataloader

    Args:
        csv_path: Path to household CSV file
        scenario: Simulation scenario dict (product_category, new_price, baseline_price)
        device: Device to use ("cpu" or "cuda")
        llm_config: Optional LLM configuration dict with keys:
            - enabled: bool (whether to use LLM)
            - provider: str ("openai" or "anthropic")
            - model: str (model name)
            - api_key: str (optional, uses env var if not provided)
            - temperature: float (0.0-2.0)
            - max_tokens: int
            - sample_size: int (number of households to use LLM for, 0 = all)
            - use_for_all: bool (use LLM for all households)
            - llm_weight: float (0.0-1.0, how much to weight LLM vs deterministic)
    """
    scenario = scenario or {
        "product_category": "new_ice_cream_flavor",
        "new_price": 4.99,
        "baseline_price": 3.99,
    }

    # Use relative path based on script location (like price_sim and market_sim)
    base_dir = Path(__file__).resolve().parent

    # Load household data and build agent states
    df = load_households(csv_path)
    agent_states = build_agent_states(df)

    # Create AgentTorch population package (pickle files)
    write_population_pickles(base_dir, agent_states)

    # Build AgentTorch model config (YAML)
    build_model_config(
        base_dir,
        agent_states,
        device=device,
        product_category=scenario["product_category"],
        new_price=scenario["new_price"],
        baseline_price=scenario.get("baseline_price", scenario["new_price"]),
        llm_config=llm_config,
    )

    # Import model and population modules following AgentTorch structure
    # Import importlib and sys here (same as price_sim/market_sim pattern)
    import importlib
    import sys

    # Verify population directory exists before importing (for debugging)
    population_dir = base_dir / "agenttorch_population"
    if not population_dir.exists():
        raise RuntimeError(
            f"Population directory does not exist: {population_dir}\n"
            f"Base dir: {base_dir}\n"
            f"Base dir exists: {base_dir.exists()}\n"
            f"Current working directory: {Path.cwd()}"
        )

    init_file = population_dir / "__init__.py"
    if not init_file.exists():
        raise RuntimeError(f"Population __init__.py not found: {init_file}")

    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    # Load model module (agenttorch_model package)
    model_module = importlib.import_module("agenttorch_model")

    # Load population module (agenttorch_population package)
    # LoadPopulation only needs module.__path__[0], so we can create a simple mock module
    # This avoids import issues on Databricks
    try:
        # Try standard import first (works in most environments)
        population_module = importlib.import_module("agenttorch_population")
    except (ModuleNotFoundError, ImportError):
        # Fallback: Create a mock module object with just __path__ attribute
        # LoadPopulation only accesses region.__path__[0], so this is sufficient
        import types
        population_module = types.ModuleType("agenttorch_population")
        population_module.__path__ = [str(population_dir)]
        # Register in sys.modules so it can be found if needed elsewhere
        sys.modules["agenttorch_population"] = population_module

    # Create population loader using AgentTorch framework
    # LoadPopulation will try to convert all pickle files to torch tensors
    # We need to temporarily move string pickles (location, emotional_state) out of the way
    # since they can't be converted to tensors
    string_pickle_files = ["location.pickle", "emotional_state.pickle"]
    temp_files = {}

    # Temporarily move string pickles OUTSIDE the population directory
    # (just renaming with _ prefix still matches *.pickle glob pattern)
    temp_dir = base_dir / "_temp_population_strings"
    temp_dir.mkdir(exist_ok=True)

    for filename in string_pickle_files:
        src = population_dir / filename
        if src.exists():
            # Move to temp directory outside population_dir
            temp_path = temp_dir / filename
            src.rename(temp_path)
            temp_files[filename] = temp_path

    try:
        # LoadPopulation will only load numeric pickles now
        # (string pickles are in temp_dir, not in population_dir)
        population = LoadPopulation(population_module)
    finally:
        # Restore string pickles (needed for load_population_attribute)
        for filename, temp_path in temp_files.items():
            temp_path.rename(population_dir / filename)
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist

    # Initialize Executor using AgentTorch framework pattern
    # Executor(model_module, pop_loader=population) is the standard AgentTorch pattern
    executor = Executor(model_module, pop_loader=population)

    # Initialize executor
    executor.init()
    state = executor.runner.state

    # Set scenario parameters in environment after initialization
    if "environment" in state:
        # Set string value (not in config to avoid tensor conversion issue)
        state["environment"]["product_category"] = scenario["product_category"]

        # Set numeric values as tensors
        if "new_price" in state["environment"]:
            if torch.is_tensor(state["environment"]["new_price"]):
                state["environment"]["new_price"] = torch.tensor([scenario["new_price"]], device=state["environment"]["new_price"].device)
            else:
                state["environment"]["new_price"] = torch.tensor([scenario["new_price"]])

        if "baseline_price" in state["environment"]:
            baseline_val = scenario.get("baseline_price", scenario["new_price"])
            if torch.is_tensor(state["environment"]["baseline_price"]):
                state["environment"]["baseline_price"] = torch.tensor([baseline_val], device=state["environment"]["baseline_price"].device)
            else:
                state["environment"]["baseline_price"] = torch.tensor([baseline_val])

    # Execute simulation using AgentTorch framework
    executor.execute()

    # Extract results from AgentTorch state
    state = executor.runner.state
    households = state.get("agents", {}).get("households", {})

    # Get results with proper defaults
    will_purchase = households.get("will_purchase")
    if will_purchase is None:
        will_purchase = torch.zeros((agent_states["n_agents"], 1))

    quantity = households.get("quantity")
    if quantity is None:
        quantity = torch.zeros_like(will_purchase)
    else:
        quantity = torch.zeros_like(will_purchase) if quantity is None else quantity

    total_spend = households.get("total_spend")
    if total_spend is None:
        total_spend = torch.zeros_like(quantity)

    purchasing = (will_purchase > 0).float()
    purchasing_count = float(purchasing.sum().item())
    n_households = float(purchasing.numel())

    # Store per-household predictions for evaluation
    purchasing_list = purchasing.cpu().numpy().flatten().tolist() if torch.is_tensor(purchasing) else purchasing

    # Get household IDs from agent states
    household_ids = agent_states.get("households", [])
    hh_ids = [h.get("household_id", i+1) for i, h in enumerate(household_ids)] if household_ids else list(range(1, int(n_households) + 1))

    return {
        "scenario": scenario,
        "purchasing_count": purchasing_count,
        "purchase_rate": purchasing_count / max(n_households, 1.0),
        "total_revenue": float(total_spend.sum().item()),
        "total_units": float(quantity.sum().item()),
        "n_households": int(n_households),
        "household_predictions": {
            "household_ids": hh_ids,
            "purchased": purchasing_list,
        },
    }
