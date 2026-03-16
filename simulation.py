"""CollusionSimulation orchestration class."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents import AgentState, PricingAgent
from economics import (
    compute_expected_quantity,
    compute_profit,
    compute_realised_quantity,
    find_static_optima,
)
from llm_client import call_llm


class CollusionSimulation:
    """Orchestrates the repeated pricing game between two agents."""

    def __init__(self,
                 prompt_family: str,
                 noise_sigma: float,
                 alpha: float,
                 cost: float = 1.0,
                 mu: float = 0.25,
                 beta: float = 100.0,
                 a_i: float = 2.0,
                 a0: float = 0.0,
                 n_periods: int = 300,
                 history_window: int = 100,
                 model: str = "gemini-3-flash-preview",
                 benchmark_price_max: Optional[float] = None,
                 temperature: float = 1.0,
                 price_floor: Optional[float] = None,
                 price_ceiling_multiplier: float = 2.34,
                 cycle_effect_share: float = 0.0,
                 cycle_period: int = 150,
                 cycle_baseline: float = 1.0,
                 checkpoint_path: Optional[str] = None,
                 event_log_path: Optional[str] = None,
                 resume: bool = False):
        self.prompt_family = prompt_family
        self.noise_sigma = noise_sigma
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
        if cycle_effect_share < 0:
            raise ValueError("cycle_effect_share must be non-negative")
        if cycle_period <= 0:
            raise ValueError("cycle_period must be positive")
        if cycle_baseline <= 0:
            raise ValueError("cycle_baseline must be strictly positive")
        self.cycle_effect_share = cycle_effect_share
        self.cycle_period = cycle_period
        self.cycle_baseline = cycle_baseline
        self.cycle_amplitude = 0.5 * self.cycle_effect_share * self.cycle_baseline
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.resume = resume

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
                              memory_window=history_window, noise_sigma=noise_sigma,
                              include_stochasticity_notice=True),
            "B": PricingAgent("B", agent_family, cost, self.price_ceiling, alpha, model,
                              memory_window=history_window, noise_sigma=noise_sigma,
                              include_stochasticity_notice=True),
        }
        self.market_history: List[Tuple[int, float, float, float, float, float, float, float]] = []

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "prompt_family": self.prompt_family,
            "noise_sigma": self.noise_sigma,
            "alpha": self.alpha,
            "cost": self.cost,
            "mu": self.mu,
            "beta": self.beta,
            "a_i": self.a_i,
            "a0": self.a0,
            "n_periods": self.n_periods,
            "history_window": self.history_window,
            "model": self.model,
            "temperature": self.temperature,
            "price_floor": self.price_floor,
            "price_ceiling": self.price_ceiling,
            "cycle_effect_share": self.cycle_effect_share,
            "cycle_period": self.cycle_period,
            "cycle_baseline": self.cycle_baseline,
        }

    @staticmethod
    def _unpack_history_entry(
        entry: Tuple[Any, ...]
    ) -> Tuple[int, float, float, float, float, float, float, float]:
        if len(entry) == 7:
            period, price_a, price_b, quantity_a, profit_a, quantity_b, profit_b = entry
            return (
                int(period),
                1.0,
                float(price_a),
                float(price_b),
                float(quantity_a),
                float(profit_a),
                float(quantity_b),
                float(profit_b),
            )
        if len(entry) == 8:
            period, market_factor, price_a, price_b, quantity_a, profit_a, quantity_b, profit_b = entry
            return (
                int(period),
                float(market_factor),
                float(price_a),
                float(price_b),
                float(quantity_a),
                float(profit_a),
                float(quantity_b),
                float(profit_b),
            )
        raise ValueError(f"Unexpected market history entry length: {len(entry)}")

    def _market_factor(self, period: int) -> float:
        angle = 2.0 * math.pi * ((period - 1) % self.cycle_period) / self.cycle_period
        return self.cycle_baseline + self.cycle_amplitude * math.cos(angle)

    def _cycle_phase(self, market_factor: float) -> str:
        if self.cycle_effect_share <= 0:
            return "neutral"
        return "high" if market_factor >= self.cycle_baseline else "low"

    @staticmethod
    def _serialize_agent_state(agent: PricingAgent) -> Dict[str, Any]:
        state = agent.state
        return {
            "plans": state.plans,
            "insights": state.insights,
            "price_history": state.price_history,
            "quantity_history": state.quantity_history,
            "profit_history": state.profit_history,
            "raw_prompts": state.raw_prompts,
            "raw_responses": state.raw_responses,
        }

    @staticmethod
    def _restore_agent_state(agent: PricingAgent, payload: Dict[str, Any]) -> None:
        agent.state = AgentState(
            plans=payload.get("plans", ""),
            insights=payload.get("insights", ""),
            price_history=list(payload.get("price_history", [])),
            quantity_history=list(payload.get("quantity_history", [])),
            profit_history=list(payload.get("profit_history", [])),
            raw_prompts=list(payload.get("raw_prompts", [])),
            raw_responses=list(payload.get("raw_responses", [])),
        )

    def _append_event(self, payload: Dict[str, Any]) -> None:
        if self.event_log_path is None:
            return
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _save_checkpoint(self, last_completed_period: int, completed: bool = False) -> None:
        if self.checkpoint_path is None:
            return
        payload = {
            "config": self._checkpoint_config(),
            "last_completed_period": last_completed_period,
            "completed": completed,
            "market_history": self.market_history,
            "agents": {
                firm: self._serialize_agent_state(agent)
                for firm, agent in self.agents.items()
            },
        }
        self._write_json_atomic(self.checkpoint_path, payload)

    def _load_checkpoint(self) -> int:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            for agent in self.agents.values():
                agent.state = AgentState()
            self.market_history = []
            return 1

        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        saved_config = payload.get("config", {})
        current_config = self._checkpoint_config()
        default_config = {
            "cycle_effect_share": 0.0,
            "cycle_period": 150,
            "cycle_baseline": 1.0,
        }
        for key in [
            "prompt_family",
            "noise_sigma",
            "alpha",
            "n_periods",
            "model",
            "temperature",
            "cycle_effect_share",
            "cycle_period",
            "cycle_baseline",
        ]:
            saved_value = saved_config.get(key, default_config.get(key))
            if saved_value != current_config.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch on '{key}': "
                    f"{saved_value} != {current_config.get(key)}"
                )

        self.market_history = [tuple(entry) for entry in payload.get("market_history", [])]
        for firm, agent in self.agents.items():
            self._restore_agent_state(agent, payload.get("agents", {}).get(firm, {}))

        last_completed_period = int(payload.get("last_completed_period", 0))
        self._append_event({
            "event": "resume",
            "last_completed_period": last_completed_period,
        })
        return last_completed_period + 1

    def _reset_state(self) -> None:
        for agent in self.agents.values():
            agent.state = AgentState()
        self.market_history = []

    def _dummy_or_call_llm(self,
                           prompt: str,
                           response_schema: Optional[Any] = None,
                           response_json_schema: Optional[dict] = None) -> str:
        """Call the LLM if credentials are available, otherwise use fallback."""
        response = call_llm(
            prompt,
            model=self.model,
            temperature=self.temperature,
            response_schema=response_schema,
            response_json_schema=response_json_schema,
        )
        if response is not None:
            return response
        return ""

    def run(self) -> Dict[str, Any]:
        """Execute a single run of the simulation and return summary metrics."""
        if self.resume:
            start_period = self._load_checkpoint()
        else:
            self._reset_state()
            start_period = 1

        if start_period > self.n_periods:
            return self._build_summary()

        for period in range(start_period, self.n_periods + 1):
            decisions = {}
            for firm in ["A", "B"]:
                agent = self.agents[firm]
                history_for_agent = []
                for entry in self.market_history[-agent.memory_window:]:
                    round_num, _market_factor, price_A, price_B, q_A, prof_A, q_B, prof_B = (
                        self._unpack_history_entry(entry)
                    )
                    if firm == "A":
                        history_for_agent.append((round_num, price_A, price_B, q_A, prof_A))
                    else:
                        history_for_agent.append((round_num, price_B, price_A, q_B, prof_B))

                prompt = agent.build_prompt(history_for_agent)
                agent.state.raw_prompts.append(prompt)
                self._append_event({
                    "event": "prompt_built",
                    "period": period,
                    "firm": firm,
                    "prompt": prompt,
                })

                price_value = None
                new_plans = None
                new_insights = None
                for attempt in range(10):
                    raw_response = self._dummy_or_call_llm(
                        prompt,
                        response_json_schema=agent.response_json_schema(),
                    )
                    agent.state.raw_responses.append(raw_response)
                    self._append_event({
                        "event": "llm_response",
                        "period": period,
                        "firm": firm,
                        "attempt": attempt + 1,
                        "response": raw_response,
                    })
                    if raw_response:
                        plans, insights, price_candidate = agent.parse_response(
                            raw_response, self.price_floor, self.price_ceiling
                        )
                    else:
                        plans, insights, price_candidate = None, None, None
                    if price_candidate is not None:
                        price_value = price_candidate
                        new_plans = plans
                        new_insights = insights
                        break

                if price_value is None:
                    self._append_event({
                        "event": "run_aborted",
                        "period": period,
                        "firm": firm,
                        "reason": "llm_failed_to_return_valid_price_after_10_attempts",
                    })
                    self._save_checkpoint(max(period - 1, 0), completed=False)
                    raise RuntimeError(
                        f"Aborting run: firm {firm} failed to return a valid price "
                        f"after 10 attempts in period {period}."
                    )

                agent.state.plans = new_plans if new_plans is not None else agent.state.plans
                agent.state.insights = new_insights if new_insights is not None else agent.state.insights
                decisions[firm] = price_value
                self._append_event({
                    "event": "decision",
                    "period": period,
                    "firm": firm,
                    "price": price_value,
                    "plans": agent.state.plans,
                    "insights": agent.state.insights,
                })

            price_A = decisions["A"]
            price_B = decisions["B"]
            market_factor = self._market_factor(period)
            q_A_exp, q_B_exp = compute_expected_quantity(
                price_A, price_B, self.alpha,
                self.beta, self.a_i, self.a_i, self.a0, self.mu,
                market_factor=market_factor,
            )
            q_A_real = compute_realised_quantity(q_A_exp, noise_sigma=self.noise_sigma)
            q_B_real = compute_realised_quantity(q_B_exp, noise_sigma=self.noise_sigma)
            profit_A = compute_profit(price_A, q_A_real, self.alpha, self.cost)
            profit_B = compute_profit(price_B, q_B_real, self.alpha, self.cost)

            self.market_history.append(
                (period, market_factor, price_A, price_B, q_A_real, profit_A, q_B_real, profit_B)
            )
            self.agents["A"].state.update_history(price_A, q_A_real, profit_A)
            self.agents["B"].state.update_history(price_B, q_B_real, profit_B)
            self._append_event({
                "event": "period_complete",
                "period": period,
                "market_factor": market_factor,
                "cycle_phase": self._cycle_phase(market_factor),
                "price_A": price_A,
                "price_B": price_B,
                "expected_quantity_A": q_A_exp,
                "expected_quantity_B": q_B_exp,
                "quantity_A": q_A_real,
                "quantity_B": q_B_real,
                "profit_A": profit_A,
                "profit_B": profit_B,
            })
            self._save_checkpoint(period)

        summary = self._build_summary()
        self._append_event({
            "event": "run_complete",
            "summary": summary,
        })
        self._save_checkpoint(self.n_periods, completed=True)
        return summary

    def _build_summary(self) -> Dict[str, Any]:
        final_history = self.market_history[-50:]
        unpacked_final_history = [self._unpack_history_entry(entry) for entry in final_history]
        avg_price_A = np.mean([entry[2] for entry in unpacked_final_history])
        avg_price_B = np.mean([entry[3] for entry in unpacked_final_history])
        avg_total_profit = np.mean([entry[5] + entry[7] for entry in unpacked_final_history])
        avg_price_norm_A = avg_price_A / self.alpha
        avg_price_norm_B = avg_price_B / self.alpha
        avg_total_profit_norm = avg_total_profit / self.alpha

        unpacked_run_history = [self._unpack_history_entry(entry) for entry in self.market_history]
        market_factors = [entry[1] for entry in unpacked_run_history]
        high_phase_entries = [
            entry for entry in unpacked_run_history
            if self._cycle_phase(entry[1]) == "high"
        ]
        low_phase_entries = [
            entry for entry in unpacked_run_history
            if self._cycle_phase(entry[1]) == "low"
        ]

        def collusion_index(val: float, nash: float, monop: float) -> float:
            if monop == nash:
                return 0.0
            return (val - nash) / (monop - nash)

        def average_or_none(values: List[float]) -> Optional[float]:
            if not values:
                return None
            return float(np.mean(values))

        price_collusion_index = collusion_index(
            (avg_price_A + avg_price_B) / 2.0, self.p_nash, self.p_monopoly
        )
        profit_collusion_index = collusion_index(
            avg_total_profit, 2 * self.profit_nash, 2 * self.profit_monopoly
        )

        return {
            "prompt_family": self.prompt_family,
            "noise_sigma": self.noise_sigma,
            "alpha": self.alpha,
            "cycle_effect_share": self.cycle_effect_share,
            "cycle_period": self.cycle_period,
            "cycle_baseline": self.cycle_baseline,
            "avg_price_A": avg_price_A,
            "avg_price_B": avg_price_B,
            "avg_price_norm_A": avg_price_norm_A,
            "avg_price_norm_B": avg_price_norm_B,
            "avg_total_profit": avg_total_profit,
            "avg_total_profit_norm": avg_total_profit_norm,
            "price_collusion_index": price_collusion_index,
            "profit_collusion_index": profit_collusion_index,
            "p_nash": self.p_nash,
            "p_monopoly": self.p_monopoly,
            "profit_nash": self.profit_nash,
            "profit_monopoly": self.profit_monopoly,
            "avg_market_factor": average_or_none(market_factors),
            "min_market_factor": min(market_factors) if market_factors else None,
            "max_market_factor": max(market_factors) if market_factors else None,
            "high_phase_avg_price": average_or_none(
                [(entry[2] + entry[3]) / 2.0 for entry in high_phase_entries]
            ),
            "low_phase_avg_price": average_or_none(
                [(entry[2] + entry[3]) / 2.0 for entry in low_phase_entries]
            ),
            "high_phase_avg_total_profit": average_or_none(
                [entry[5] + entry[7] for entry in high_phase_entries]
            ),
            "low_phase_avg_total_profit": average_or_none(
                [entry[5] + entry[7] for entry in low_phase_entries]
            ),
            "run_history": self.market_history,
        }
