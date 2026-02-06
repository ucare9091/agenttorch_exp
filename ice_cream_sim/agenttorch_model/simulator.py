"""AgentTorch simulator module - provides registry for Executor."""

from agent_torch.core import Registry
from agenttorch_model import registry

def get_registry():
    """Return the AgentTorch registry with registered substeps."""
    return registry
