"""AgentTorch model package for the price simulator."""

from agent_torch.core import Registry
from agent_torch.core.helpers import *  # noqa: F401,F403

from .substeps import *  # noqa: F401,F403

# Create and populate registry as a module-level variable
# This follows AgentTorch framework convention
registry = Registry()
