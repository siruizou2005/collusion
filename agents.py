"""AgentState and PricingAgent classes for the collusion simulation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from prompts import (
    OUTPUT_TEMPLATE,
    PROMPT_P0C,
)


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
                 alpha: float, model: str = "gpt-3.5-turbo", memory_window: int = 100):
        self.firm_id = firm_id
        assert prompt_family == "P0C"
        self.prompt_family = prompt_family
        self.cost = cost
        self.price_ceiling = price_ceiling
        self.alpha = alpha
        self.model = model
        self.memory_window = memory_window
        self.state = AgentState()

    def _prompt_prefix(self) -> str:
        return PROMPT_P0C

    def _product_info(self) -> str:
        return (
            f"\nProduct information:\n"
            f"- The cost to produce each unit is {self.cost:.2f}.\n"
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
            "\n\nPlease follow this format exactly:\n\n" + OUTPUT_TEMPLATE + "\n\n"
            "Keep the full response concise. Limit 'My observations and thoughts' to at most 120 words, "
            "'New content for PLANS.txt' to at most 60 words, and 'New content for INSIGHTS.txt' to at most 60 words.\n"
            "Remember to end with 'My chosen price:' followed by your numeric price "
            "without any additional words."
        )

        return prefix + product_info + history_text + plans_insights + instructions

    def parse_response(self, response: str, price_floor: float,
                       price_ceiling: float) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Parse the raw LLM response into (plans, insights, price).

        Parameters
        ----------
        response : str
            Raw text returned by the LLM.
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
            sections = response.split("My observations and thoughts:")
            if len(sections) < 2:
                raise ValueError("Missing observations section")
            rest = sections[1]
            idx_plans = rest.find("New content for PLANS.txt:")
            idx_insights = rest.find("New content for INSIGHTS.txt:")
            idx_price = rest.rfind("My chosen price")
            if idx_plans == -1 or idx_insights == -1 or idx_price == -1:
                raise ValueError("Missing required headings")
            plans_text = rest[idx_plans + len("New content for PLANS.txt:"):idx_insights].strip()
            insights_text = rest[idx_insights + len("New content for INSIGHTS.txt:"):idx_price].strip()
            price_text = rest[idx_price:].split(":", 1)[-1].strip()
            price_value = None
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", price_text)
            if match:
                price_value = float(match.group(0))
                price_value = max(price_floor, min(price_value, price_ceiling))
            else:
                raise ValueError("Failed to extract price value")
            return plans_text, insights_text, price_value
        except Exception as exc:
            print(f"[WARN] Failed to parse response: {exc}\nResponse was:\n{response}\n")
            return None, None, None
