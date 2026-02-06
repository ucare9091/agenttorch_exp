from __future__ import annotations

import pandas as pd
import torch

from agent_torch.core.registry import Registry


@Registry.register_substep("load_population_attribute", "initialization")
def load_population_attribute(shape, params):
    """Load a single attribute from a pickle file into a tensor or list (for strings)."""
    file_path = params["file_path"]
    series = pd.read_pickle(file_path)
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            series = series.iloc[:, 0]

    # Handle string types - return as list instead of tensor
    if series.dtype == 'object' or isinstance(series.iloc[0] if len(series) > 0 else None, str):
        # Return as list for string attributes
        return series.tolist()

    # Handle numeric types - convert to tensor
    values = series.to_numpy()
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    # Handle numpy 2.x compatibility: ensure array is writable for PyTorch
    # numpy 2.x may return read-only arrays from pandas, which causes warnings
    if not values.flags.writeable:
        values = values.copy()
    tensor = torch.from_numpy(values)
    if shape is not None:
        assert tuple(tensor.shape) == tuple(shape), (
            f"Expected shape {shape}, got {tensor.shape} for {file_path}"
        )
    return tensor
