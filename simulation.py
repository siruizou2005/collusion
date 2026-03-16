"""CollusionSimulation orchestration class."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents import AgentState, PricingAgent
from economics import (
    compute_expected_quantity,
    compute_profit,
    compute_realised_quantity,
    find_static_optima,
)
from llm_client import call_llm, dummy_price_strategy


class CollusionSimulation:
    """Orchestrates the repeated pricing game between two agents."""

    def __init__(self,
                 prompt_family: str,
                 alpha: float,
                 cycle_amplitude: float = 0.0,
                 cycle_period: int = 150,
                 cycle_baseline: float = 1.0,
                 cost: float = 1.0,
                 mu: float = 0.25,
                 beta: float = 100.0,
                 a_i: float = 2.0,
                 a0: float = 0.0,
                 n_periods: int = 300,
                 history_window: int = 100,
                 model: str = "gpt-3.5-turbo",
                 benchmark_price_max: Optional[float] = None,
                 temperature: float = 1.0,
                 price_floor: Optional[float] = None,
                 price_ceiling_multiplier: float = 2.34):
        self.prompt_family = prompt_family
        self.cycle_amplitude = cycle_amplitude
        self.cycle_period = cycle_period
        self.cycle_baseline = cycle_baseline
        self.alpha = alpha
        self.cost = cost
        self.mu = mu
        self.beta = beta
        self.a_i = a_i
        self.a0 = a0
        self.n_periods = n_periods
        self.history_window = history_window
        self.model = model
        self.benchmark_price_max = benchmark_price_max
        self.temperature = temperature
        self.price_floor = cost * alpha if price_floor is None else price_floor
        self.price_ceiling_multiplier = price_ceiling_multiplier

        p_nash, profit_nash, p_monop, profit_monop = find_static_optima(
            alpha, mu, beta, a_i, a0, cost,
            price_min=self.price_floor,
            price_max=benchmark_price_max,
        )
        self.p_nash = p_nash
        self.profit_nash = profit_nash
        self.p_monopoly = p_monop
        self.profit_monopoly = profit_monop
        self.price_ceiling = max(self.price_floor, self.price_ceiling_multiplier * self.p_monopoly)

        agent_family = prompt_family
        self.agents = {
            "A": PricingAgent("A", agent_family, cost, self.price_ceiling, alpha, model,
                              memory_window=history_window),
            "B": PricingAgent("B", agent_family, cost, self.price_ceiling, alpha, model,
                              memory_window=history_window),
        }
        self.market_history: List[Tuple[int, float, float, float, float, float, float, float]] = []

    def _market_factor(self, t: int) -> float:
        return self.cycle_baseline + self.cycle_amplitude * math.cos(
            2.0 * math.pi * (t - 1) / self.cycle_period
        )

    def _phase(self, t: int) -> str:
        return "boom" if (t - 1) % self.cycle_period < self.cycle_period // 2 else "bust"

    def _dummy_or_call_llm(self, prompt: str) -> str:
        """Call the LLM if credentials are available, otherwise use fallback."""
        response = call_llm(prompt, model=self.model, temperature=self.temperature)
        if response is not None:
            return response
        return ""

    def run(self) -> Dict[str, Any]:
        """Execute a single run of the simulation and return summary metrics."""
        for agent in self.agents.values():
            agent.state = AgentState()
        self.market_history = []

        for period in range(1, self.n_periods + 1):
            decisions = {}
            for firm in ["A", "B"]:
                agent = self.agents[firm]
                history_for_agent = []
                for entry in self.market_history[-agent.memory_window:]:
                    round_num, price_A, price_B, q_A, prof_A, q_B, prof_B, _mf = entry
                    if firm == "A":
                        history_for_agent.append((round_num, price_A, price_B, q_A, prof_A))
                    else:
                        history_for_agent.append((round_num, price_B, price_A, q_B, prof_B))

                prompt = agent.build_prompt(history_for_agent)
                agent.state.raw_prompts.append(prompt)

                price_value = None
                new_plans = None
                new_insights = None
                for attempt in range(10):
                    raw_response = self._dummy_or_call_llm(prompt)
                    agent.state.raw_responses.append(raw_response)
                    plans, insights, price_candidate = agent.parse_response(
                        raw_response, self.price_floor, self.price_ceiling
                    )
                    if price_candidate is not None:
                        price_value = price_candidate
                        new_plans = plans
                        new_insights = insights
                        break

                if price_value is None:
                    price_value = dummy_price_strategy(self.price_floor, self.price_ceiling)
                    new_plans = ""
                    new_insights = ""

                agent.state.plans = new_plans if new_plans is not None else agent.state.plans
                agent.state.insights = new_insights if new_insights is not None else agent.state.insights
                decisions[firm] = price_value

            price_A = decisions["A"]
            price_B = decisions["B"]
            mf = self._market_factor(period)
            q_A_exp, q_B_exp = compute_expected_quantity(
                price_A, price_B, self.alpha,
                self.beta, self.a_i, self.a_i, self.a0, self.mu,
                market_factor=mf,
            )
            q_A_real = compute_realised_quantity(q_A_exp, noise_sigma=0.0)
            q_B_real = compute_realised_quantity(q_B_exp, noise_sigma=0.0)
            profit_A = compute_profit(price_A, q_A_real, self.alpha, self.cost)
            profit_B = compute_profit(price_B, q_B_real, self.alpha, self.cost)

            self.market_history.append((period, price_A, price_B, q_A_real, profit_A, q_B_real, profit_B, mf))
            self.agents["A"].state.update_history(price_A, q_A_real, profit_A)
            self.agents["B"].state.update_history(price_B, q_B_real, profit_B)

        final_history = self.market_history[-50:]
        avg_price_A = np.mean([entry[1] for entry in final_history])
        avg_price_B = np.mean([entry[2] for entry in final_history])
        avg_total_profit = np.mean([entry[4] + entry[6] for entry in final_history])
        avg_price_norm_A = avg_price_A / self.alpha
        avg_price_norm_B = avg_price_B / self.alpha
        avg_total_profit_norm = avg_total_profit / self.alpha

        def collusion_index(val: float, nash: float, monop: float) -> float:
            if monop == nash:
                return 0.0
            return (val - nash) / (monop - nash)

        price_collusion_index = collusion_index(
            (avg_price_A + avg_price_B) / 2.0, self.p_nash, self.p_monopoly
        )
        profit_collusion_index = collusion_index(
            avg_total_profit, 2 * self.profit_nash, 2 * self.profit_monopoly
        )

        # Boom/bust phase metrics over full history
        boom_entries = [e for e in self.market_history if self._phase(e[0]) == "boom"]
        bust_entries = [e for e in self.market_history if self._phase(e[0]) == "bust"]

        boom_avg_price = (
            float(np.mean([(e[1] + e[2]) / 2.0 for e in boom_entries]))
            if boom_entries else float("nan")
        )
        bust_avg_price = (
            float(np.mean([(e[1] + e[2]) / 2.0 for e in bust_entries]))
            if bust_entries else float("nan")
        )
        boom_price_collusion_index = collusion_index(boom_avg_price, self.p_nash, self.p_monopoly)
        bust_price_collusion_index = collusion_index(bust_avg_price, self.p_nash, self.p_monopoly)

        return {
            "prompt_family": self.prompt_family,
            "cycle_amplitude": self.cycle_amplitude,
            "alpha": self.alpha,
            "avg_price_A": avg_price_A,
            "avg_price_B": avg_price_B,
            "avg_price_norm_A": avg_price_norm_A,
            "avg_price_norm_B": avg_price_norm_B,
            "avg_total_profit": avg_total_profit,
            "avg_total_profit_norm": avg_total_profit_norm,
            "price_collusion_index": price_collusion_index,
            "profit_collusion_index": profit_collusion_index,
            "boom_avg_price": boom_avg_price,
            "bust_avg_price": bust_avg_price,
            "boom_price_collusion_index": boom_price_collusion_index,
            "bust_price_collusion_index": bust_price_collusion_index,
            "p_nash": self.p_nash,
            "p_monopoly": self.p_monopoly,
            "profit_nash": self.profit_nash,
            "profit_monopoly": self.profit_monopoly,
            "run_history": self.market_history,
        }
