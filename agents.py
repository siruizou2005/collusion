"""AgentState and PricingAgent classes for the collusion simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from prompts import (
    PROMPT_C,
    PROMPT_PREFIXES,
    STRUCTURED_OUTPUT_INSTRUCTIONS,
)


class AgentStructuredResponse(BaseModel):
    """Strict JSON schema for Gemini-native structured agent output."""

    model_config = ConfigDict(extra="forbid")

    observations_and_thoughts: str = Field(
        description="Concise reasoning about the current market state and pricing choice.",
    )
    plans: str = Field(
        description="Short note to store in PLANS.txt for the next period.",
    )
    insights: str = Field(
        description="Short note to store in INSIGHTS.txt about what was learned.",
    )
    chosen_price: float = Field(
        description="The price to charge this period as a JSON number.",
    )


def _strip_unsupported_schema_fields(value):
    """Remove JSON schema fields that Gemini's native API rejects."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "additionalProperties":
                continue
            cleaned[key] = _strip_unsupported_schema_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_unsupported_schema_fields(item) for item in value]
    return value


@dataclass
class AgentState:
    """Holds the dynamic state of an agent across periods."""
    plans: str = ""
    insights: str = ""
    price_history: List[float] = field(default_factory=list)
    quantity_history: List[float] = field(default_factory=list)
    profit_history: List[float] = field(default_factory=list)
    raw_prompts: List[str] = field(default_factory=list)
    raw_responses: List[str] = field(default_factory=list)

    def update_history(self, price: float, quantity: float, profit: float) -> None:
        self.price_history.append(price)
        self.quantity_history.append(quantity)
        self.profit_history.append(profit)


class PricingAgent:
    """Represents one of the LLM pricing agents (Firm 1 or Firm 2)."""

    def __init__(self, firm_id: str, prompt_family: str, cost: float, price_ceiling: float,
                 alpha: float, model: str = "gpt-3.5-turbo", memory_window: int = 100,
                 noise_sigma: float = 0.0, include_stochasticity_notice: bool = False):
        self.firm_id = firm_id
        if prompt_family not in PROMPT_PREFIXES:
            raise ValueError(f"Unknown prompt family: {prompt_family}")
        self.prompt_family = prompt_family
        self.cost = cost
        self.price_ceiling = price_ceiling
        self.alpha = alpha
        self.model = model
        self.memory_window = memory_window
        self.state = AgentState()
        self.noise_sigma = noise_sigma
        self.include_stochasticity_notice = include_stochasticity_notice

    def _prompt_prefix(self) -> str:
        prefix = PROMPT_PREFIXES[self.prompt_family]
        if self.include_stochasticity_notice:
            prefix += PROMPT_C
        return prefix

    def _product_info(self) -> str:
        return (
            f"\nProduct information:\n"
            f"- The cost to produce each unit is {self.cost * self.alpha:.2f}.\n"
            f"- No customer would pay more than {self.price_ceiling:.2f} for this product."
        )

    def _plans_insights_text(self) -> str:
        plans = self.state.plans if self.state.plans else "<empty>"
        insights = self.state.insights if self.state.insights else "<empty>"
        return (
            "\nYour previous PLANS.txt:\n" + plans +
            "\n\nYour previous INSIGHTS.txt:\n" + insights
        )

    def build_prompt(self, market_history: List[Tuple[int, float, float, float, float]]) -> str:
        """Construct the full prompt for this agent in the current period.

        Parameters
        ----------
        market_history : list of tuples
            Each tuple is (round_num, price_i, price_j, quantity_i, profit_i).
            Only the last `memory_window` entries are used.
        """
        prefix = self._prompt_prefix()
        product_info = self._product_info()

        if not market_history:
            history_text = "\nMarket history: (no previous rounds)"
        else:
            slice_history = market_history[-self.memory_window:]
            history_text = "\nMarket history (last {0} rounds):\n".format(len(slice_history))
            for (round_num, p_i, p_j, q_i, prof_i) in slice_history:
                history_text += (
                    f"Round {round_num}: your price={p_i:.2f}, competitor price={p_j:.2f}, "
                    f"your quantity={q_i:.2f}, your profit={prof_i:.2f}\n"
                )

        plans_insights = self._plans_insights_text()
        instructions = (
            "\nOutput requirements:\n"
            f"- {STRUCTURED_OUTPUT_INSTRUCTIONS}\n"
            "- Keep 'observations_and_thoughts' to at most 120 words.\n"
            "- Keep 'plans' to at most 60 words.\n"
            "- Keep 'insights' to at most 60 words.\n"
            f"- 'chosen_price' must be a JSON number between {self.cost * self.alpha:.2f} "
            f"and {self.price_ceiling:.2f}.\n"
            "- Focus on the decision itself; do not restate the schema."
        )

        return prefix + product_info + history_text + plans_insights + instructions

    @staticmethod
    def response_json_schema() -> dict:
        """Return a Gemini-compatible JSON schema for structured output."""
        return _strip_unsupported_schema_fields(AgentStructuredResponse.model_json_schema())

    def parse_response(self, response: str, price_floor: float,
                       price_ceiling: float) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Parse the raw JSON LLM response into (plans, insights, price).

        Parameters
        ----------
        response : str
            Raw JSON string returned by the LLM.
        price_floor : float
            Minimum allowable price (will be clamped to this if below).
        price_ceiling : float
            Maximum allowable price (will be clamped to this if above).

        Returns
        -------
        Tuple[Optional[str], Optional[str], Optional[float]]
            (plans, insights, price), any of which may be None if parsing fails.
        """
        if response is None:
            return None, None, None
        try:
            parsed = AgentStructuredResponse.model_validate_json(response)
            price_value = max(price_floor, min(float(parsed.chosen_price), price_ceiling))
            return parsed.plans.strip(), parsed.insights.strip(), price_value
        except ValidationError as exc:
            print(f"[WARN] Failed to validate structured response: {exc}\nResponse was:\n{response}\n")
            return None, None, None
        except Exception as exc:
            print(f"[WARN] Failed to parse response: {exc}\nResponse was:\n{response}\n")
            return None, None, None
