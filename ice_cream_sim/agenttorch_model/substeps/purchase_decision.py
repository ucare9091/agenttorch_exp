from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import re

from agent_torch.core.helpers.general import get_by_path
from agent_torch.core.registry import Registry
from agent_torch.core.substep import SubstepAction, SubstepTransition

from .llm_helper import get_llm_helper_from_config, llm_enhanced_decision


def get_var(state: Dict[str, Any], var: str):
    """
    Retrieves a value from the current state of the model.
    Follows AgentTorch framework pattern.
    """
    return get_by_path(state, re.split("/", var))


@Registry.register_substep("purchase_decision", "policy")
class PurchaseDecisionPolicyAT(SubstepAction):
    def forward(self, state: Dict[str, Any], observation: Dict[str, Any]):
        # Extract input variables using AgentTorch framework pattern
        income = get_var(state, self.input_variables["income"])
        household_size = get_var(state, self.input_variables["household_size"])
        price_sensitivity = get_var(state, self.input_variables["price_sensitivity"])
        new_price = get_var(state, self.input_variables["new_price"])
        baseline_price = get_var(state, self.input_variables["baseline_price"])

        # Extract location if available (for LLM context)
        location = None
        if "location" in self.input_variables:
            try:
                location = get_var(state, self.input_variables["location"])
                # Handle location as list (string attribute) or tensor
                if isinstance(location, list):
                    pass  # Already a list
                elif torch.is_tensor(location):
                    # Convert tensor to list of strings (if it contains string indices)
                    # For now, treat as list of indices that map to location names
                    location = location.tolist()
                else:
                    location = None
            except (KeyError, TypeError, AttributeError):
                location = None

        # Extract emotional state if available (for LLM context)
        emotional_state = None
        if "emotional_state" in self.input_variables:
            try:
                emotional_state = get_var(state, self.input_variables["emotional_state"])
                # Handle emotional_state as list (string attribute) or tensor
                if isinstance(emotional_state, list):
                    pass  # Already a list
                elif torch.is_tensor(emotional_state):
                    emotional_state = emotional_state.tolist()
                else:
                    emotional_state = None
            except (KeyError, TypeError, AttributeError):
                emotional_state = None

        # Normalize to tensors for computation
        if not torch.is_tensor(new_price):
            new_price = torch.tensor(new_price)
        if not torch.is_tensor(baseline_price):
            baseline_price = torch.tensor(baseline_price)

        # Ensure compatible shapes and dtypes for broadcasting
        if torch.is_tensor(income):
            if new_price.dim() == 0:
                new_price = new_price.unsqueeze(0)
            if baseline_price.dim() == 0:
                baseline_price = baseline_price.unsqueeze(0)
            new_price = new_price.to(dtype=income.dtype, device=income.device)
            baseline_price = baseline_price.to(dtype=income.dtype, device=income.device)

        # Get LLM helper if configured (following AgentTorch pattern)
        llm_helper = None
        llm_config = self.arguments.get("llm", {}) if hasattr(self, "arguments") else {}
        if not llm_config:
            # Try to get from state environment
            try:
                env_llm = state.get("environment", {}).get("llm_config", {})
                if env_llm:
                    llm_config = env_llm
            except (KeyError, TypeError):
                pass

        if llm_config and llm_config.get("enabled", False):
            from .llm_helper import LLMHelper
            llm_helper = LLMHelper(
                provider=llm_config.get("provider", "openai"),
                model=llm_config.get("model", "gpt-3.5-turbo"),
                api_key=llm_config.get("api_key"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 150),
            )

        # Get product category from environment
        product_category = state.get("environment", {}).get("product_category", "product")

        # Compute deterministic purchase probability (base logic)
        price_change_pct = torch.where(
            baseline_price > 0,
            (new_price - baseline_price) / baseline_price,
            torch.zeros_like(new_price),
        )

        income_normalized = torch.clamp(income / 150000.0, max=1.0)
        price_sensitivity_factor = 1.0 - (income_normalized * 0.5)

        base_probability = torch.full_like(price_sensitivity_factor, 0.7)

        sensitivity_impact = torch.where(
            price_change_pct > 0,
            price_sensitivity * price_change_pct * 2.0,
            price_sensitivity * torch.abs(price_change_pct) * 0.5,
        )
        purchase_probability = torch.where(
            price_change_pct > 0,
            base_probability * (1.0 - torch.clamp(sensitivity_impact, max=0.8)),
            base_probability * (1.0 + torch.clamp(sensitivity_impact, max=0.5)),
        )

        income_factor = torch.clamp(income / 100000.0, max=1.0)
        purchase_probability = purchase_probability * (0.7 + 0.3 * income_factor)
        purchase_probability = torch.clamp(purchase_probability, min=0.0, max=1.0)

        # LLM-enhanced decision (if enabled and for sample households)
        # For efficiency, use LLM only for a sample or when explicitly enabled for all
        use_llm_sample = llm_helper and llm_config.get("sample_size", 0) > 0
        use_llm_all = llm_helper and llm_config.get("use_for_all", False)

        if use_llm_all or use_llm_sample:
            # Convert tensors to Python values for LLM
            n_agents = income.shape[0] if torch.is_tensor(income) else len(income) if isinstance(income, list) else 1

            # Determine which households to use LLM for
            if use_llm_all:
                llm_indices = list(range(n_agents))
            else:
                sample_size = min(llm_config.get("sample_size", 10), n_agents)
                import random
                llm_indices = random.sample(range(n_agents), sample_size)

            # Create LLM-enhanced probabilities
            llm_adjustments = torch.ones_like(purchase_probability)

            for idx in llm_indices:
                # Extract household context
                hh_income = float(income[idx].item() if torch.is_tensor(income) else income[idx])
                hh_size = float(household_size[idx].item() if torch.is_tensor(household_size) else household_size[idx])
                hh_sens = float(price_sensitivity[idx].item() if torch.is_tensor(price_sensitivity) else price_sensitivity[idx])
                if location is not None:
                    if isinstance(location, list) and idx < len(location):
                        hh_loc = str(location[idx])
                    elif torch.is_tensor(location) and idx < location.shape[0]:
                        hh_loc = str(location[idx].item())
                    else:
                        hh_loc = "unknown"
                else:
                    hh_loc = "unknown"

                if emotional_state is not None:
                    if isinstance(emotional_state, list) and idx < len(emotional_state):
                        hh_emo = str(emotional_state[idx])
                    elif torch.is_tensor(emotional_state) and idx < emotional_state.shape[0]:
                        hh_emo = str(emotional_state[idx].item())
                    else:
                        hh_emo = "neutral"
                else:
                    hh_emo = "neutral"

                household_context = {
                    "income": hh_income,
                    "household_size": hh_size,
                    "price_sensitivity": hh_sens,
                    "location": hh_loc,
                    "emotional_state": hh_emo,
                }

                market_context = {
                    "product_category": product_category,
                    "new_price": float(new_price[0].item() if torch.is_tensor(new_price) else new_price),
                    "baseline_price": float(baseline_price[0].item() if torch.is_tensor(baseline_price) else baseline_price),
                }

                # Get LLM decision
                llm_result = llm_enhanced_decision(llm_helper, household_context, market_context)

                # Adjust probability based on LLM decision
                llm_weight = llm_config.get("llm_weight", 0.3)  # How much to weight LLM vs deterministic
                base_prob = float(purchase_probability[idx].item() if torch.is_tensor(purchase_probability) else purchase_probability[idx])

                if llm_result["decision"]:
                    # LLM says yes - increase probability
                    llm_prob = llm_result["confidence"]
                    adjusted_prob = base_prob * (1 - llm_weight) + llm_prob * llm_weight
                else:
                    # LLM says no - decrease probability
                    llm_prob = 1.0 - llm_result["confidence"]
                    adjusted_prob = base_prob * (1 - llm_weight) + llm_prob * llm_weight

                llm_adjustments[idx] = adjusted_prob / max(base_prob, 0.01)  # Avoid division by zero

            # Apply LLM adjustments
            purchase_probability = purchase_probability * llm_adjustments
            purchase_probability = torch.clamp(purchase_probability, min=0.0, max=1.0)

        # Final decision
        rand_val = torch.rand_like(purchase_probability)
        will_purchase = (rand_val < purchase_probability).float()

        base_quantity = torch.clamp(household_size * 0.5, min=1.0)
        quantity = torch.where(
            will_purchase > 0,
            torch.where(
                price_change_pct < 0,
                base_quantity * (1.0 - price_change_pct * 0.3),
                base_quantity * (1.0 - price_change_pct * 0.5),
            ),
            torch.zeros_like(base_quantity),
        )
        quantity = torch.clamp(quantity, min=0.0)
        total_spend = new_price * quantity

        return {
            "will_purchase": will_purchase,
            "quantity": quantity,
            "total_spend": total_spend,
        }


@Registry.register_substep("apply_purchase_decision", "transition")
class ApplyPurchaseDecision(SubstepTransition):
    def forward(self, state: Dict[str, Any], action: Dict[str, Any]):
        # AgentTorch framework: action contains policy outputs nested under agent name
        # Structure: action['households'] = {'will_purchase': ..., 'quantity': ..., 'total_spend': ...}
        if action and isinstance(action, dict) and "households" in action:
            household_action = action["households"]
            if isinstance(household_action, dict):
                return {
                    "will_purchase": household_action.get("will_purchase"),
                    "quantity": household_action.get("quantity"),
                    "total_spend": household_action.get("total_spend"),
                }

        # Fallback for direct keys (should not happen in standard AgentTorch flow)
        return {
            "will_purchase": action.get("will_purchase") if action else None,
            "quantity": action.get("quantity") if action else None,
            "total_spend": action.get("total_spend") if action else None,
        }
