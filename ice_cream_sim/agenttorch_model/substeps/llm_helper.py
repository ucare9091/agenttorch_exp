"""LLM integration helper for AgentTorch substeps.

This module provides LLM integration following AgentTorch patterns,
allowing policies to use language models for decision-making.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, List
import torch

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMHelper:
    """Helper class for LLM API calls in AgentTorch substeps."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ):
        """
        Initialize LLM helper.

        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (e.g., "gpt-3.5-turbo", "claude-3-haiku-20240307", "gpt-5.1" for proxy)
            api_key: API key (if None, reads from environment)
            api_base_url: Custom API base URL for proxy support (if None, uses default)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        elif self.provider == "openai":
            # Try MODEL_PROXY_API_KEY first (for proxy), then OPENAI_API_KEY
            self.api_key = os.getenv("MODEL_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = None

        # Get API base URL from parameter or environment (for proxy support)
        if api_base_url:
            self.api_base_url = api_base_url
        elif self.provider == "openai":
            # Try MODEL_PROXY_API_BASE first (for proxy), then OPENAI_API_BASE
            self.api_base_url = os.getenv("MODEL_PROXY_API_BASE") or os.getenv("OPENAI_API_BASE")
        else:
            self.api_base_url = None

        # Initialize client if available
        if self.provider == "openai" and OPENAI_AVAILABLE:
            if self.api_key:
                # Initialize OpenAI client with optional base_url for proxy support
                client_kwargs = {"api_key": self.api_key}
                if self.api_base_url:
                    client_kwargs["base_url"] = self.api_base_url
                self.client = openai.OpenAI(**client_kwargs)
            else:
                self.client = None
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        else:
            self.client = None

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_json: bool = False,
    ) -> str:
        """
        Call LLM API with a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            parse_json: If True, try to parse response as JSON

        Returns:
            LLM response text (or parsed JSON dict if parse_json=True)
        """
        if not self.client:
            # Fallback: return default response if LLM not available
            return self._fallback_response(prompt, parse_json)

        try:
            if self.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = response.choices[0].message.content

            elif self.provider == "anthropic":
                messages = []
                if system_prompt:
                    # Anthropic uses system parameter
                    response = self.client.messages.create(
                        model=self.model,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                result = response.content[0].text

            else:
                return self._fallback_response(prompt, parse_json)

            if parse_json:
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    if "```json" in result:
                        json_start = result.find("```json") + 7
                        json_end = result.find("```", json_start)
                        if json_end > json_start:
                            result = result[json_start:json_end].strip()
                    elif "```" in result:
                        json_start = result.find("```") + 3
                        json_end = result.find("```", json_start)
                        if json_end > json_start:
                            result = result[json_start:json_end].strip()
                    return json.loads(result)

            return result

        except Exception as e:
            # Fallback on error
            print(f"LLM API error: {e}")
            return self._fallback_response(prompt, parse_json)

    def _fallback_response(self, prompt: str, parse_json: bool) -> Any:
        """Fallback response when LLM is not available."""
        if parse_json:
            return {"decision": "no", "reasoning": "LLM not available"}
        return "LLM not available - using default logic"

    def batch_call(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        parse_json: bool = False,
    ) -> List[str]:
        """
        Call LLM API for multiple prompts (sequential for now).

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            parse_json: If True, parse each response as JSON

        Returns:
            List of LLM responses
        """
        results = []
        for prompt in prompts:
            result = self.call(prompt, system_prompt, parse_json)
            results.append(result)
        return results


def get_llm_helper_from_config(config: Dict[str, Any]) -> Optional[LLMHelper]:
    """
    Create LLMHelper from AgentTorch config.

    Args:
        config: AgentTorch config dict (from state or arguments)

    Returns:
        LLMHelper instance or None if not configured
    """
    llm_config = config.get("llm", {})
    if not llm_config or not llm_config.get("enabled", False):
        return None

    return LLMHelper(
        provider=llm_config.get("provider", "openai"),
        model=llm_config.get("model", "gpt-3.5-turbo"),
        api_key=llm_config.get("api_key"),
        api_base_url=llm_config.get("api_base_url"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 150),
    )


def llm_enhanced_decision(
    llm_helper: Optional[LLMHelper],
    household_context: Dict[str, Any],
    market_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM to enhance purchase decision with natural language reasoning.

    Args:
        llm_helper: LLMHelper instance (None to skip LLM)
        household_context: Household attributes (income, size, etc.)
        market_context: Market conditions (price, baseline, etc.)

    Returns:
        Dict with 'decision' (bool), 'reasoning' (str), 'confidence' (float)
    """
    if not llm_helper:
        # Fallback to default decision
        return {
            "decision": False,
            "reasoning": "LLM not available",
            "confidence": 0.5,
        }

    # Get emotional state and create emotional context
    emotional_state = household_context.get('emotional_state', 'neutral')
    emotional_context = ""
    if emotional_state:
        emotional_state_lower = str(emotional_state).lower()
        if emotional_state_lower in ['very_stressed', 'very stressed', 'anxious', 'worried', 'distressed']:
            emotional_context = "\nEmotional State: VERY STRESSED - This household is experiencing high stress/anxiety. They may make impulsive purchases for comfort, or avoid purchases due to financial worry. Consider emotional spending patterns."
        elif emotional_state_lower in ['stressed', 'stress', 'concerned', 'tense']:
            emotional_context = "\nEmotional State: STRESSED - This household is under moderate stress. They may seek comfort purchases or be more price-conscious due to financial concerns."
        elif emotional_state_lower in ['happy', 'content', 'satisfied', 'pleased']:
            emotional_context = "\nEmotional State: HAPPY - This household is in a positive emotional state. They may be more open to purchases, especially for treats or items that enhance their happiness."
        elif emotional_state_lower in ['very_happy', 'very happy', 'elated', 'joyful', 'excited', 'euphoric']:
            emotional_context = "\nEmotional State: VERY HAPPY - This household is in an elevated positive emotional state. They may be more likely to make impulse purchases, treat themselves, or buy items that maintain their positive mood."
        else:
            emotional_context = "\nEmotional State: NEUTRAL - This household is in a balanced emotional state. Decisions will be primarily based on rational factors."

    # Build prompt for LLM with emotional state consideration
    prompt = f"""You are analyzing a household's purchase decision for a product, considering both rational and emotional factors.

Household Profile:
- Income: ${household_context.get('income', 0):,.0f}
- Household Size: {household_context.get('household_size', 0):.1f}
- Price Sensitivity: {household_context.get('price_sensitivity', 0.5):.2f}
- Location: {household_context.get('location', 'unknown')}{emotional_context}

Market Conditions:
- Product: {market_context.get('product_category', 'product')}
- New Price: ${market_context.get('new_price', 0):.2f}
- Baseline Price: ${market_context.get('baseline_price', 0):.2f}
- Price Change: {((market_context.get('new_price', 0) / max(market_context.get('baseline_price', 1), 0.01)) - 1) * 100:.1f}%

IMPORTANT: Consider how the emotional state affects purchase behavior:
- Stressed/anxious households may make comfort purchases OR avoid spending
- Happy households may be more open to purchases and treats
- Emotional state can override rational price sensitivity

Based on BOTH rational factors (income, price, sensitivity) AND emotional state, would this household purchase the product?
Respond with a JSON object:
{{
    "decision": true/false,
    "reasoning": "brief explanation that explicitly mentions emotional factors",
    "confidence": 0.0-1.0,
    "quantity": estimated quantity (0-10),
    "emotional_influence": "how emotional state influenced the decision"
}}
"""

    system_prompt = "You are an expert behavioral economist and market analyst. You understand how emotions influence purchasing decisions. Consider both rational economic factors AND emotional/psychological factors when making purchase predictions. Emotional states can significantly impact consumer behavior - stressed people may seek comfort purchases or avoid spending, while happy people may be more open to treats and impulse buys."

    try:
        response = llm_helper.call(prompt, system_prompt, parse_json=True)
        if isinstance(response, dict):
            return {
                "decision": response.get("decision", False),
                "reasoning": response.get("reasoning", ""),
                "confidence": float(response.get("confidence", 0.5)),
                "quantity": float(response.get("quantity", 0.0)),
                "emotional_influence": response.get("emotional_influence", ""),
            }
    except Exception as e:
        print(f"LLM decision error: {e}")

    # Fallback
    return {
        "decision": False,
        "reasoning": "LLM call failed",
        "confidence": 0.5,
        "quantity": 0.0,
        "emotional_influence": "",
    }
