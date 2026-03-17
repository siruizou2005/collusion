"""Two-stage quality-choice experiment mode."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_client import call_llm
from prompts import PROMPT_PREFIXES

Quality = Literal["L", "H"]
QualityPair = Literal["LL", "LH", "HL", "HH"]
QUALITY_OPTIONS: Tuple[Quality, Quality] = ("L", "H")
QUALITY_PAIRS: Tuple[QualityPair, ...] = ("LL", "LH", "HL", "HH")


class QualityChoiceResponse(BaseModel):
    """Structured response for quality-choice rounds."""

    model_config = ConfigDict(extra="forbid")

    observations_and_thoughts: str = Field(
        description="Concise reasoning about the block-level quality choice.",
    )
    plans: str = Field(
        description="Short note to store for the next quality-choice round.",
    )
    insights: str = Field(
        description="Short note about what was learned from prior blocks.",
    )
    chosen_quality: Quality = Field(
        description="Chosen quality for the next block: 'L' or 'H'.",
    )


class QualityPriceResponse(BaseModel):
    """Structured response for pricing rounds in a locked quality block."""

    model_config = ConfigDict(extra="forbid")

    observations_and_thoughts: str = Field(
        description="Concise reasoning about the current pricing decision.",
    )
    plans: str = Field(
        description="Short note to store for the next pricing round.",
    )
    insights: str = Field(
        description="Short note about what was learned from this pricing round.",
    )
    chosen_price: float = Field(
        description="Chosen price for the current round.",
    )


def _strip_unsupported_schema_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "additionalProperties":
                continue
            cleaned[key] = _strip_unsupported_schema_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_unsupported_schema_fields(item) for item in value]
    return value


@dataclass(frozen=True)
class CostSpec:
    variable_cost: float
    fixed_cost: float


@dataclass(frozen=True)
class DemandSpec:
    intercept_a: float
    cross_a: float
    intercept_b: float
    cross_b: float


@dataclass(frozen=True)
class SegmentSpec:
    weight: float
    price_sensitivity: float
    outside_utility: float
    utilities: Dict[str, float]


@dataclass(frozen=True)
class QualityPreset:
    name: str
    price_cap: float
    cost_specs: Dict[str, Dict[Quality, CostSpec]]
    demand_mode: Literal["linear_pair", "two_segment_logit"] = "linear_pair"
    demand_specs: Optional[Dict[QualityPair, DemandSpec]] = None
    market_size: float = 0.0
    segment_specs: Optional[Dict[str, SegmentSpec]] = None
    expected_quality_nash_pair: QualityPair = "LL"
    expected_joint_optimum_pair: QualityPair = "LH"
    expected_joint_profit_order: Optional[Tuple[QualityPair, ...]] = None
    min_joint_vs_nash_avg_price_gap: Optional[Dict[QualityPair, float]] = None
    min_joint_price_ratio_by_pair: Optional[Dict[QualityPair, float]] = None
    quality_prompt_context: str = ""
    price_prompt_context: str = ""


@dataclass(frozen=True)
class StageGameOutcome:
    quality_pair: QualityPair
    nash_price_a: float
    nash_price_b: float
    nash_quantity_a: float
    nash_quantity_b: float
    nash_profit_a: float
    nash_profit_b: float
    joint_price_a: float
    joint_price_b: float
    joint_quantity_a: float
    joint_quantity_b: float
    joint_profit_a: float
    joint_profit_b: float

    @property
    def nash_total_profit(self) -> float:
        return self.nash_profit_a + self.nash_profit_b

    @property
    def joint_total_profit(self) -> float:
        return self.joint_profit_a + self.joint_profit_b


@dataclass(frozen=True)
class StaticQualityBenchmarks:
    preset_name: str
    price_cap: float
    stage_game_outcomes: Dict[QualityPair, StageGameOutcome]
    quality_nash_pair: QualityPair
    joint_optimum_pair: QualityPair


@dataclass
class QualityBlockAgentState:
    quality_plans: str = ""
    quality_insights: str = ""
    price_plans: str = ""
    price_insights: str = ""
    quality_choice_history: List[str] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)
    quantity_history: List[float] = field(default_factory=list)
    profit_history: List[float] = field(default_factory=list)
    raw_prompts: List[Dict[str, Any]] = field(default_factory=list)
    raw_responses: List[Dict[str, Any]] = field(default_factory=list)

    def update_round_history(self, price: float, quantity: float, profit: float) -> None:
        self.price_history.append(price)
        self.quantity_history.append(quantity)
        self.profit_history.append(profit)


def get_quality_preset(name: str) -> QualityPreset:
    if name == "segmentation_v1":
        return QualityPreset(
            name=name,
            price_cap=12.0,
            cost_specs={
                "A": {
                    "L": CostSpec(1.0, 0.0),
                    "H": CostSpec(5.0, 10.0),
                },
                "B": {
                    "L": CostSpec(2.5, 0.0),
                    "H": CostSpec(1.0, 35.0),
                },
            },
            demand_mode="linear_pair",
            demand_specs={
                "LL": DemandSpec(10.0, 0.1, 10.0, 0.1),
                "LH": DemandSpec(10.0, 0.3, 11.0, 0.6),
                "HL": DemandSpec(11.0, 0.6, 10.0, 0.3),
                "HH": DemandSpec(11.0, 0.1, 11.0, 0.1),
            },
            expected_quality_nash_pair="LL",
            expected_joint_optimum_pair="LH",
        )
    if name == "segmentation_v2":
        return QualityPreset(
            name=name,
            price_cap=12.0,
            cost_specs={
                "A": {
                    "L": CostSpec(0.91, 0.0),
                    "H": CostSpec(1.415, 7.2),
                },
                "B": {
                    "L": CostSpec(0.774, 0.0),
                    "H": CostSpec(1.625, 8.602),
                },
            },
            demand_mode="two_segment_logit",
            market_size=43.428,
            segment_specs={
                "premium": SegmentSpec(
                    weight=0.367,
                    price_sensitivity=0.602,
                    outside_utility=2.569,
                    utilities={
                        "AL": 3.593,
                        "AH": 5.958,
                        "BL": 3.772,
                        "BH": 4.524,
                    },
                ),
                "budget": SegmentSpec(
                    weight=1.0 - 0.367,
                    price_sensitivity=0.708,
                    outside_utility=2.804,
                    utilities={
                        "AL": 5.83,
                        "AH": 4.76,
                        "BL": 6.195,
                        "BH": 6.048,
                    },
                ),
            },
            expected_quality_nash_pair="LL",
            expected_joint_optimum_pair="HL",
            expected_joint_profit_order=("HL", "LL", "LH", "HH"),
            min_joint_vs_nash_avg_price_gap={
                "LL": 0.10,
                "HL": 0.10,
            },
            min_joint_price_ratio_by_pair={
                "HL": 1.15,
            },
            quality_prompt_context=(
                "\n- The market contains both premium and budget customers who value quality and price differently.\n"
                "- Firm A's high quality is positioned to serve the premium segment, while Firm B's low quality can remain attractive to budget buyers.\n"
                "- High quality mainly changes market positioning and requires a higher fixed investment; do not treat it as only a higher unit cost.\n"
                "- A quality choice can be worthwhile if it creates a profitable premium niche or reduces direct head-to-head price competition over the next block."
            ),
            price_prompt_context=(
                "\n- The locked quality pair can segment demand between premium and budget customers, not just shift one representative demand curve.\n"
                "- A high-quality offer may support a premium price without winning every customer, while a lower-quality offer can still earn strong profits if it remains attractive to budget buyers.\n"
                "- When qualities differ, Firm A may be better placed to earn from premium buyers and Firm B may still earn strongly from budget buyers; price for your role in the market, not only for volume."
            ),
        )
    raise ValueError(f"Unknown quality preset: {name}")


def quality_pair_label(quality_a: Quality, quality_b: Quality) -> QualityPair:
    return f"{quality_a}{quality_b}"  # type: ignore[return-value]


def compute_quality_quantities(
    preset: QualityPreset,
    quality_pair: QualityPair,
    price_a: float,
    price_b: float,
) -> Tuple[float, float]:
    if preset.demand_mode == "linear_pair":
        if preset.demand_specs is None:
            raise ValueError(f"Preset {preset.name} is missing linear demand specs")
        spec = preset.demand_specs[quality_pair]
        quantity_a = max(0.0, spec.intercept_a - price_a + spec.cross_a * price_b)
        quantity_b = max(0.0, spec.intercept_b - price_b + spec.cross_b * price_a)
        return quantity_a, quantity_b
    if preset.demand_mode == "two_segment_logit":
        if preset.segment_specs is None:
            raise ValueError(f"Preset {preset.name} is missing segmented demand specs")
        label_a = f"A{quality_pair[0]}"
        label_b = f"B{quality_pair[1]}"
        quantity_a = 0.0
        quantity_b = 0.0
        for segment in preset.segment_specs.values():
            utility_a = segment.utilities[label_a] - segment.price_sensitivity * price_a
            utility_b = segment.utilities[label_b] - segment.price_sensitivity * price_b
            exp_outside = math.exp(segment.outside_utility)
            exp_a = math.exp(utility_a)
            exp_b = math.exp(utility_b)
            denom = exp_outside + exp_a + exp_b
            segment_mass = preset.market_size * segment.weight
            quantity_a += segment_mass * exp_a / denom
            quantity_b += segment_mass * exp_b / denom
        return quantity_a, quantity_b
    raise ValueError(f"Unknown demand mode for preset {preset.name}: {preset.demand_mode}")


def compute_quality_profits(
    preset: QualityPreset,
    quality_a: Quality,
    quality_b: Quality,
    price_a: float,
    price_b: float,
) -> Tuple[float, float, float, float]:
    quality_pair = quality_pair_label(quality_a, quality_b)
    quantity_a, quantity_b = compute_quality_quantities(
        preset,
        quality_pair,
        price_a,
        price_b,
    )
    cost_a = preset.cost_specs["A"][quality_a]
    cost_b = preset.cost_specs["B"][quality_b]
    profit_a = (price_a - cost_a.variable_cost) * quantity_a - cost_a.fixed_cost
    profit_b = (price_b - cost_b.variable_cost) * quantity_b - cost_b.fixed_cost
    return quantity_a, quantity_b, profit_a, profit_b


def _quality_profit_matrices(
    preset: QualityPreset,
    quality_pair: QualityPair,
    price_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prices = np.linspace(0.0, preset.price_cap, price_steps)
    price_a = prices[:, None]
    price_b = prices[None, :]
    quality_a = quality_pair[0]
    quality_b = quality_pair[1]
    if preset.demand_mode == "linear_pair":
        if preset.demand_specs is None:
            raise ValueError(f"Preset {preset.name} is missing linear demand specs")
        demand_spec = preset.demand_specs[quality_pair]
        quantity_a = np.maximum(0.0, demand_spec.intercept_a - price_a + demand_spec.cross_a * price_b)
        quantity_b = np.maximum(0.0, demand_spec.intercept_b - price_b + demand_spec.cross_b * price_a)
    elif preset.demand_mode == "two_segment_logit":
        if preset.segment_specs is None:
            raise ValueError(f"Preset {preset.name} is missing segmented demand specs")
        label_a = f"A{quality_a}"
        label_b = f"B{quality_b}"
        quantity_a = np.zeros((price_a.shape[0], price_b.shape[1]))
        quantity_b = np.zeros_like(quantity_a)
        for segment in preset.segment_specs.values():
            utility_a = segment.utilities[label_a] - segment.price_sensitivity * price_a
            utility_b = segment.utilities[label_b] - segment.price_sensitivity * price_b
            exp_outside = math.exp(segment.outside_utility)
            exp_a = np.exp(utility_a)
            exp_b = np.exp(utility_b)
            denom = exp_outside + exp_a + exp_b
            segment_mass = preset.market_size * segment.weight
            quantity_a += segment_mass * exp_a / denom
            quantity_b += segment_mass * exp_b / denom
    else:
        raise ValueError(f"Unknown demand mode for preset {preset.name}: {preset.demand_mode}")
    cost_a = preset.cost_specs["A"][quality_a]  # type: ignore[index]
    cost_b = preset.cost_specs["B"][quality_b]  # type: ignore[index]
    profit_a = (price_a - cost_a.variable_cost) * quantity_a - cost_a.fixed_cost
    profit_b = (price_b - cost_b.variable_cost) * quantity_b - cost_b.fixed_cost
    return prices, quantity_a, quantity_b, profit_a, profit_b


def _find_stage_game_nash(
    prices: np.ndarray,
    quantity_a: np.ndarray,
    quantity_b: np.ndarray,
    profit_a: np.ndarray,
    profit_b: np.ndarray,
) -> Tuple[int, int]:
    br_a_indices = profit_a.argmax(axis=0)
    br_b_indices = profit_b.argmax(axis=1)
    intersections = [col for col, row in enumerate(br_a_indices) if br_b_indices[row] == col]
    if intersections:
        best_col = max(
            intersections,
            key=lambda col: float(profit_a[br_a_indices[col], col] + profit_b[br_a_indices[col], col]),
        )
        return int(br_a_indices[best_col]), int(best_col)
    composed = br_b_indices[br_a_indices]
    best_col = int(np.argmin(np.abs(composed - np.arange(prices.shape[0]))))
    return int(br_a_indices[best_col]), int(best_col)


def _find_stage_game_joint_optimum(
    profit_a: np.ndarray,
    profit_b: np.ndarray,
) -> Tuple[int, int]:
    total_profit = profit_a + profit_b
    optimum_index = np.unravel_index(int(np.argmax(total_profit)), total_profit.shape)
    return int(optimum_index[0]), int(optimum_index[1])


def _build_stage_game_outcome(
    preset: QualityPreset,
    quality_pair: QualityPair,
    price_steps: int,
) -> StageGameOutcome:
    prices, quantity_a, quantity_b, profit_a, profit_b = _quality_profit_matrices(
        preset,
        quality_pair,
        price_steps,
    )
    nash_row, nash_col = _find_stage_game_nash(prices, quantity_a, quantity_b, profit_a, profit_b)
    joint_row, joint_col = _find_stage_game_joint_optimum(profit_a, profit_b)
    return StageGameOutcome(
        quality_pair=quality_pair,
        nash_price_a=float(prices[nash_row]),
        nash_price_b=float(prices[nash_col]),
        nash_quantity_a=float(quantity_a[nash_row, nash_col]),
        nash_quantity_b=float(quantity_b[nash_row, nash_col]),
        nash_profit_a=float(profit_a[nash_row, nash_col]),
        nash_profit_b=float(profit_b[nash_row, nash_col]),
        joint_price_a=float(prices[joint_row]),
        joint_price_b=float(prices[joint_col]),
        joint_quantity_a=float(quantity_a[joint_row, joint_col]),
        joint_quantity_b=float(quantity_b[joint_row, joint_col]),
        joint_profit_a=float(profit_a[joint_row, joint_col]),
        joint_profit_b=float(profit_b[joint_row, joint_col]),
    )


@lru_cache(maxsize=32)
def get_static_quality_benchmarks(
    preset_name: str,
    price_steps: int = 1201,
) -> StaticQualityBenchmarks:
    preset = get_quality_preset(preset_name)
    outcomes = {
        quality_pair: _build_stage_game_outcome(preset, quality_pair, price_steps)
        for quality_pair in QUALITY_PAIRS
    }

    payoff_a = np.array([
        [outcomes["LL"].nash_profit_a, outcomes["LH"].nash_profit_a],
        [outcomes["HL"].nash_profit_a, outcomes["HH"].nash_profit_a],
    ])
    payoff_b = np.array([
        [outcomes["LL"].nash_profit_b, outcomes["LH"].nash_profit_b],
        [outcomes["HL"].nash_profit_b, outcomes["HH"].nash_profit_b],
    ])
    best_rows = payoff_a.argmax(axis=0)
    best_cols = payoff_b.argmax(axis=1)
    equilibria = [
        (row, col)
        for row in range(2)
        for col in range(2)
        if best_rows[col] == row and best_cols[row] == col
    ]
    if not equilibria:
        raise RuntimeError("No stage-1 quality Nash equilibrium found for the preset")
    nash_row, nash_col = equilibria[0]
    quality_nash_pair = ("LL", "LH", "HL", "HH")[2 * nash_row + nash_col]

    joint_optimum_pair = max(
        QUALITY_PAIRS,
        key=lambda pair: outcomes[pair].joint_total_profit,
    )

    benchmarks = StaticQualityBenchmarks(
        preset_name=preset_name,
        price_cap=preset.price_cap,
        stage_game_outcomes=outcomes,
        quality_nash_pair=quality_nash_pair,  # type: ignore[arg-type]
        joint_optimum_pair=joint_optimum_pair,
    )
    validate_quality_benchmarks(benchmarks)
    return benchmarks


def validate_quality_benchmarks(benchmarks: StaticQualityBenchmarks) -> None:
    preset = get_quality_preset(benchmarks.preset_name)
    if benchmarks.quality_nash_pair != preset.expected_quality_nash_pair:
        raise ValueError(
            f"Expected quality Nash to be {preset.expected_quality_nash_pair}, "
            f"got {benchmarks.quality_nash_pair}"
        )
    if benchmarks.joint_optimum_pair != preset.expected_joint_optimum_pair:
        raise ValueError(
            f"Expected bounded joint optimum to be {preset.expected_joint_optimum_pair}, "
            f"got {benchmarks.joint_optimum_pair}"
        )
    quality_nash = benchmarks.stage_game_outcomes[preset.expected_quality_nash_pair]
    joint_optimum = benchmarks.stage_game_outcomes[preset.expected_joint_optimum_pair]
    if (
        joint_optimum.joint_profit_a <= quality_nash.nash_profit_a
        or joint_optimum.joint_profit_b <= quality_nash.nash_profit_b
    ):
        raise ValueError(
            f"Expected {preset.expected_joint_optimum_pair} joint optimum to improve both firms over "
            f"{preset.expected_quality_nash_pair} Nash"
        )
    if preset.expected_joint_profit_order is not None:
        ordered_pairs = sorted(
            QUALITY_PAIRS,
            key=lambda pair: benchmarks.stage_game_outcomes[pair].joint_total_profit,
            reverse=True,
        )
        if tuple(ordered_pairs) != preset.expected_joint_profit_order:
            raise ValueError(
                f"Expected joint-profit ranking {preset.expected_joint_profit_order}, got {tuple(ordered_pairs)}"
            )
    if preset.min_joint_vs_nash_avg_price_gap is not None:
        for pair, minimum_gap in preset.min_joint_vs_nash_avg_price_gap.items():
            outcome = benchmarks.stage_game_outcomes[pair]
            avg_nash_price = (outcome.nash_price_a + outcome.nash_price_b) / 2.0
            avg_joint_price = (outcome.joint_price_a + outcome.joint_price_b) / 2.0
            if avg_nash_price <= 0.0:
                raise ValueError(f"Non-positive Nash average price for {pair}")
            realized_gap = (avg_joint_price - avg_nash_price) / avg_nash_price
            if realized_gap < minimum_gap:
                raise ValueError(
                    f"Expected {pair} average price gap at least {minimum_gap:.3f}, got {realized_gap:.3f}"
                )
    if preset.min_joint_price_ratio_by_pair is not None:
        for pair, minimum_ratio in preset.min_joint_price_ratio_by_pair.items():
            outcome = benchmarks.stage_game_outcomes[pair]
            if outcome.joint_price_b <= 0.0:
                raise ValueError(f"Non-positive joint price for firm B in {pair}")
            realized_ratio = outcome.joint_price_a / outcome.joint_price_b
            if realized_ratio < minimum_ratio:
                raise ValueError(
                    f"Expected {pair} joint A/B price ratio at least {minimum_ratio:.3f}, got {realized_ratio:.3f}"
                )


class QualityBlockAgent:
    """LLM agent for the blocked quality-choice experiment."""

    def __init__(
        self,
        firm_id: str,
        prompt_family: str,
        preset: QualityPreset,
        model: str,
        memory_window: int = 100,
    ):
        if prompt_family not in PROMPT_PREFIXES:
            raise ValueError(f"Unknown prompt family: {prompt_family}")
        self.firm_id = firm_id
        self.prompt_family = prompt_family
        self.preset = preset
        self.model = model
        self.memory_window = memory_window
        self.state = QualityBlockAgentState()

    def _prompt_prefix(self) -> str:
        return PROMPT_PREFIXES[self.prompt_family]

    def _quality_cost_menu(self) -> str:
        cost_specs = self.preset.cost_specs[self.firm_id]
        low = cost_specs["L"]
        high = cost_specs["H"]
        return (
            f"\nYou are Firm {self.firm_id}."
            "\nYour quality options:\n"
            f"- L: unit cost={low.variable_cost:.2f}, fixed cost per round={low.fixed_cost:.2f}\n"
            f"- H: unit cost={high.variable_cost:.2f}, fixed cost per round={high.fixed_cost:.2f}"
        )

    @staticmethod
    def _history_text(history: List[Dict[str, Any]], memory_window: int) -> str:
        if not history:
            return "\nMarket history: (no previous rounds)"
        slice_history = history[-memory_window:]
        lines = [f"\nMarket history (last {len(slice_history)} rounds):"]
        for entry in slice_history:
            lines.append(
                "Round {round}: your quality={your_quality}, competitor quality={competitor_quality}, "
                "your price={your_price:.2f}, competitor price={competitor_price:.2f}, "
                "your quantity={your_quantity:.2f}, your profit={your_profit:.2f}".format(**entry)
            )
        return "\n".join(lines)

    @staticmethod
    def _quality_block_summary_text(
        history: List[Dict[str, Any]],
        quality_block_length: int,
        max_blocks: int = 6,
    ) -> str:
        if not history:
            return "\nRecent block summaries: (no previous blocks)"

        complete_rounds = len(history) - (len(history) % quality_block_length)
        if complete_rounds == 0:
            return "\nRecent block summaries: (no completed blocks yet)"

        completed_history = history[:complete_rounds]
        blocks: List[List[Dict[str, Any]]] = [
            completed_history[idx: idx + quality_block_length]
            for idx in range(0, len(completed_history), quality_block_length)
        ]
        recent_blocks = blocks[-max_blocks:]
        lines = [f"\nRecent block summaries (last {len(recent_blocks)} completed blocks):"]
        for block in recent_blocks:
            first_entry = block[0]
            start_round = first_entry["round"]
            end_round = block[-1]["round"]
            avg_price = float(np.mean([entry["your_price"] for entry in block]))
            avg_quantity = float(np.mean([entry["your_quantity"] for entry in block]))
            total_profit = float(np.sum([entry["your_profit"] for entry in block]))
            avg_profit = float(np.mean([entry["your_profit"] for entry in block]))
            lines.append(
                "Rounds {start_round}-{end_round}: your quality={your_quality}, competitor quality={competitor_quality}, "
                "avg price={avg_price:.2f}, avg quantity={avg_quantity:.2f}, avg profit={avg_profit:.2f}, "
                "total profit={total_profit:.2f}".format(
                    start_round=start_round,
                    end_round=end_round,
                    your_quality=first_entry["your_quality"],
                    competitor_quality=first_entry["competitor_quality"],
                    avg_price=avg_price,
                    avg_quantity=avg_quantity,
                    avg_profit=avg_profit,
                    total_profit=total_profit,
                )
            )
        return "\n".join(lines)

    def build_quality_prompt(
        self,
        history: List[Dict[str, Any]],
        current_round: int,
        block_end_round: int,
        quality_block_length: int,
    ) -> str:
        instructions = (
            "\nThis is a quality-choice round.\n"
            f"- You must choose exactly one quality from {{'L','H'}} for rounds {current_round}-{block_end_round}.\n"
            f"- Once chosen, your quality will remain locked for the next {quality_block_length} rounds in this block.\n"
            "- You will incur the chosen quality's fixed cost in each round of the block.\n"
            "- After both firms choose quality, you and your competitor will set prices simultaneously each round.\n"
            "- Quality affects both your cost structure and market demand / competitive positioning.\n"
            "- A higher quality may support higher willingness to pay or a different competitive position, but it may also be more expensive.\n"
            "- Do not treat H as only a cost increase or L as automatically safer; infer the quality trade-off from prior block outcomes.\n"
            "- Compare full-block profitability, not just one-period margins or fixed costs in isolation.\n"
            "- Choose the quality that maximizes long-run profit over the full block and the repeated game.\n"
            "- You do not observe the competitor's current quality choice yet."
        )
        if self.preset.quality_prompt_context:
            instructions += self.preset.quality_prompt_context
        notes = (
            "\nYour previous QUALITY_PLANS.txt:\n"
            + (self.state.quality_plans or "<empty>")
            + "\n\nYour previous QUALITY_INSIGHTS.txt:\n"
            + (self.state.quality_insights or "<empty>")
        )
        output_rules = (
            "\nOutput requirements:\n"
            "- Return only a JSON object with exactly these keys: "
            "'observations_and_thoughts', 'plans', 'insights', 'chosen_quality'.\n"
            "- Keep 'observations_and_thoughts' to at most 120 words.\n"
            "- Keep 'plans' to at most 60 words.\n"
            "- Keep 'insights' to at most 60 words.\n"
            "- 'chosen_quality' must be either 'L' or 'H'."
        )
        return (
            self._prompt_prefix()
            + self._quality_cost_menu()
            + instructions
            + self._history_text(history, self.memory_window)
            + self._quality_block_summary_text(history, quality_block_length)
            + notes
            + output_rules
        )

    def build_price_prompt(
        self,
        history: List[Dict[str, Any]],
        current_round: int,
        current_quality: Quality,
        competitor_quality: Quality,
        block_end_round: int,
        rounds_remaining_after_current: int,
    ) -> str:
        cost_spec = self.preset.cost_specs[self.firm_id][current_quality]
        instructions = (
            "\nThis is a pricing round inside a locked quality block.\n"
            f"- Your current quality is {current_quality}; the competitor's current quality is {competitor_quality}.\n"
            f"- These qualities remain fixed through round {block_end_round}.\n"
            f"- After this round, {rounds_remaining_after_current} pricing rounds remain before the next quality-choice opportunity.\n"
            f"- Your current unit cost is {cost_spec.variable_cost:.2f}, and your fixed cost this round for this quality is {cost_spec.fixed_cost:.2f}.\n"
            f"- Your chosen price must stay between 0.00 and {self.preset.price_cap:.2f}.\n"
            "- Quality affects demand and competitive positioning as well as cost, so use this block to learn how the current quality pair changes pricing power, volume, and profit.\n"
            "- Maximize long-run profit while accounting for the locked quality pair."
        )
        if self.preset.price_prompt_context:
            instructions += self.preset.price_prompt_context
        notes = (
            "\nYour previous PRICE_PLANS.txt:\n"
            + (self.state.price_plans or "<empty>")
            + "\n\nYour previous PRICE_INSIGHTS.txt:\n"
            + (self.state.price_insights or "<empty>")
        )
        output_rules = (
            "\nOutput requirements:\n"
            "- Return only a JSON object with exactly these keys: "
            "'observations_and_thoughts', 'plans', 'insights', 'chosen_price'.\n"
            "- Keep 'observations_and_thoughts' to at most 120 words.\n"
            "- Keep 'plans' to at most 60 words.\n"
            "- Keep 'insights' to at most 60 words.\n"
            f"- 'chosen_price' must be a JSON number between 0.00 and {self.preset.price_cap:.2f}."
        )
        return (
            self._prompt_prefix()
            + instructions
            + self._history_text(history, self.memory_window)
            + notes
            + output_rules
        )

    @staticmethod
    def quality_response_json_schema() -> Dict[str, Any]:
        return _strip_unsupported_schema_fields(QualityChoiceResponse.model_json_schema())

    @staticmethod
    def price_response_json_schema() -> Dict[str, Any]:
        return _strip_unsupported_schema_fields(QualityPriceResponse.model_json_schema())

    def parse_quality_response(
        self,
        response: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[Quality]]:
        if response is None:
            return None, None, None
        try:
            parsed = QualityChoiceResponse.model_validate_json(response)
            return parsed.plans.strip(), parsed.insights.strip(), parsed.chosen_quality
        except ValidationError as exc:
            print(f"[WARN] Failed to validate quality response: {exc}\nResponse was:\n{response}\n")
            return None, None, None
        except Exception as exc:
            print(f"[WARN] Failed to parse quality response: {exc}\nResponse was:\n{response}\n")
            return None, None, None

    def parse_price_response(
        self,
        response: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        if response is None:
            return None, None, None
        try:
            parsed = QualityPriceResponse.model_validate_json(response)
            price = max(0.0, min(float(parsed.chosen_price), self.preset.price_cap))
            return parsed.plans.strip(), parsed.insights.strip(), price
        except ValidationError as exc:
            print(f"[WARN] Failed to validate price response: {exc}\nResponse was:\n{response}\n")
            return None, None, None
        except Exception as exc:
            print(f"[WARN] Failed to parse price response: {exc}\nResponse was:\n{response}\n")
            return None, None, None


class QualityTwoStageSimulation:
    """Deterministic blocked quality-choice simulation."""

    def __init__(
        self,
        prompt_family: str,
        quality_preset: str = "segmentation_v1",
        n_periods: int = 500,
        quality_block_length: int = 10,
        history_window: int = 100,
        model: str = "gemini-3-flash-preview",
        temperature: float = 1.0,
        checkpoint_path: Optional[str] = None,
        event_log_path: Optional[str] = None,
        resume: bool = False,
    ):
        if n_periods <= 0:
            raise ValueError("n_periods must be positive")
        if quality_block_length <= 0:
            raise ValueError("quality_block_length must be positive")
        if n_periods % quality_block_length != 0:
            raise ValueError("n_periods must be divisible by quality_block_length")
        self.experiment_mode = "quality_two_stage"
        self.prompt_family = prompt_family
        self.quality_preset = quality_preset
        self.n_periods = n_periods
        self.quality_block_length = quality_block_length
        self.history_window = history_window
        self.model = model
        self.temperature = temperature
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.resume = resume
        self.preset = get_quality_preset(quality_preset)
        self.static_benchmarks = get_static_quality_benchmarks(quality_preset)
        self.price_floor = 0.0
        self.price_ceiling = self.preset.price_cap
        self.agents = {
            "A": QualityBlockAgent("A", prompt_family, self.preset, model, history_window),
            "B": QualityBlockAgent("B", prompt_family, self.preset, model, history_window),
        }
        self.market_history: List[Dict[str, Any]] = []
        self.block_history: List[Dict[str, Any]] = []
        self.current_block_qualities: Optional[Tuple[Quality, Quality]] = None

    def _checkpoint_config(self) -> Dict[str, Any]:
        return {
            "experiment_mode": self.experiment_mode,
            "prompt_family": self.prompt_family,
            "quality_preset": self.quality_preset,
            "n_periods": self.n_periods,
            "quality_block_length": self.quality_block_length,
            "history_window": self.history_window,
            "model": self.model,
            "temperature": self.temperature,
            "price_floor": self.price_floor,
            "price_ceiling": self.price_ceiling,
        }

    @staticmethod
    def _serialize_agent_state(agent: QualityBlockAgent) -> Dict[str, Any]:
        return asdict(agent.state)

    @staticmethod
    def _restore_agent_state(agent: QualityBlockAgent, payload: Dict[str, Any]) -> None:
        agent.state = QualityBlockAgentState(
            quality_plans=payload.get("quality_plans", ""),
            quality_insights=payload.get("quality_insights", ""),
            price_plans=payload.get("price_plans", ""),
            price_insights=payload.get("price_insights", ""),
            quality_choice_history=list(payload.get("quality_choice_history", [])),
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
            "current_block_qualities": list(self.current_block_qualities) if self.current_block_qualities else None,
            "market_history": self.market_history,
            "block_history": self.block_history,
            "agents": {
                firm: self._serialize_agent_state(agent)
                for firm, agent in self.agents.items()
            },
        }
        self._write_json_atomic(self.checkpoint_path, payload)

    def _load_checkpoint(self) -> int:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            self._reset_state()
            return 1

        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        saved_config = payload.get("config", {})
        current_config = self._checkpoint_config()
        saved_n_periods = int(saved_config.get("n_periods", 0))
        if saved_n_periods <= 0:
            raise ValueError("Checkpoint is missing a valid 'n_periods' value")
        if saved_n_periods > self.n_periods:
            raise ValueError(
                f"Checkpoint config mismatch on 'n_periods': "
                f"{saved_n_periods} > {self.n_periods}"
            )
        for key in [
            "experiment_mode",
            "prompt_family",
            "quality_preset",
            "quality_block_length",
            "model",
            "temperature",
        ]:
            if saved_config.get(key) != current_config.get(key):
                raise ValueError(
                    f"Checkpoint config mismatch on '{key}': "
                    f"{saved_config.get(key)} != {current_config.get(key)}"
                )
        self.market_history = list(payload.get("market_history", []))
        self.block_history = list(payload.get("block_history", []))
        raw_qualities = payload.get("current_block_qualities")
        if raw_qualities is None:
            self.current_block_qualities = None
        else:
            self.current_block_qualities = (raw_qualities[0], raw_qualities[1])
        for firm, agent in self.agents.items():
            self._restore_agent_state(agent, payload.get("agents", {}).get(firm, {}))
        last_completed_period = int(payload.get("last_completed_period", 0))
        self._append_event({
            "event": "resume",
            "last_completed_period": last_completed_period,
            "saved_n_periods": saved_n_periods,
            "target_n_periods": self.n_periods,
        })
        return last_completed_period + 1

    def _reset_state(self) -> None:
        for agent in self.agents.values():
            agent.state = QualityBlockAgentState()
        self.market_history = []
        self.block_history = []
        self.current_block_qualities = None

    def _dummy_or_call_llm(self, prompt: str, response_json_schema: Dict[str, Any]) -> str:
        response = call_llm(
            prompt,
            model=self.model,
            temperature=self.temperature,
            response_json_schema=response_json_schema,
        )
        if response is not None:
            return response
        return ""

    def _is_quality_choice_round(self, period: int) -> bool:
        return (period - 1) % self.quality_block_length == 0

    def _block_bounds(self, period: int) -> Tuple[int, int, int]:
        block_index = (period - 1) // self.quality_block_length + 1
        block_start = (block_index - 1) * self.quality_block_length + 1
        block_end = min(self.n_periods, block_start + self.quality_block_length - 1)
        return block_index, block_start, block_end

    def _history_for_agent(self, firm: str) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for entry in self.market_history[-self.history_window:]:
            if firm == "A":
                history.append({
                    "round": entry["round"],
                    "your_quality": entry["quality_A"],
                    "competitor_quality": entry["quality_B"],
                    "your_price": entry["price_A"],
                    "competitor_price": entry["price_B"],
                    "your_quantity": entry["quantity_A"],
                    "your_profit": entry["profit_A"],
                })
            else:
                history.append({
                    "round": entry["round"],
                    "your_quality": entry["quality_B"],
                    "competitor_quality": entry["quality_A"],
                    "your_price": entry["price_B"],
                    "competitor_price": entry["price_A"],
                    "your_quantity": entry["quantity_B"],
                    "your_profit": entry["profit_B"],
                })
        return history

    def _call_quality_stage(self, firm: str, period: int, block_end: int) -> Quality:
        agent = self.agents[firm]
        prompt = agent.build_quality_prompt(
            self._history_for_agent(firm),
            current_round=period,
            block_end_round=block_end,
            quality_block_length=self.quality_block_length,
        )
        agent.state.raw_prompts.append({"stage": "quality", "period": period, "prompt": prompt})
        self._append_event({
            "event": "quality_prompt_built",
            "period": period,
            "firm": firm,
            "prompt": prompt,
        })
        for attempt in range(10):
            raw_response = self._dummy_or_call_llm(
                prompt,
                response_json_schema=agent.quality_response_json_schema(),
            )
            agent.state.raw_responses.append({
                "stage": "quality",
                "period": period,
                "attempt": attempt + 1,
                "response": raw_response,
            })
            self._append_event({
                "event": "quality_llm_response",
                "period": period,
                "firm": firm,
                "attempt": attempt + 1,
                "response": raw_response,
            })
            if raw_response:
                plans, insights, quality = agent.parse_quality_response(raw_response)
            else:
                plans, insights, quality = None, None, None
            if quality is not None:
                agent.state.quality_plans = plans if plans is not None else agent.state.quality_plans
                agent.state.quality_insights = (
                    insights if insights is not None else agent.state.quality_insights
                )
                agent.state.quality_choice_history.append(quality)
                self._append_event({
                    "event": "quality_decision",
                    "period": period,
                    "firm": firm,
                    "quality": quality,
                    "plans": agent.state.quality_plans,
                    "insights": agent.state.quality_insights,
                })
                return quality
        self._append_event({
            "event": "run_aborted",
            "period": period,
            "firm": firm,
            "stage": "quality",
            "reason": "llm_failed_to_return_valid_quality_after_10_attempts",
        })
        self._save_checkpoint(max(period - 1, 0), completed=False)
        raise RuntimeError(
            f"Aborting run: firm {firm} failed to return a valid quality after 10 attempts "
            f"in round {period}."
        )

    def _call_price_stage(
        self,
        firm: str,
        period: int,
        current_quality: Quality,
        competitor_quality: Quality,
        block_end: int,
        rounds_remaining_after_current: int,
    ) -> float:
        agent = self.agents[firm]
        prompt = agent.build_price_prompt(
            self._history_for_agent(firm),
            current_round=period,
            current_quality=current_quality,
            competitor_quality=competitor_quality,
            block_end_round=block_end,
            rounds_remaining_after_current=rounds_remaining_after_current,
        )
        agent.state.raw_prompts.append({"stage": "price", "period": period, "prompt": prompt})
        self._append_event({
            "event": "price_prompt_built",
            "period": period,
            "firm": firm,
            "prompt": prompt,
        })
        for attempt in range(10):
            raw_response = self._dummy_or_call_llm(
                prompt,
                response_json_schema=agent.price_response_json_schema(),
            )
            agent.state.raw_responses.append({
                "stage": "price",
                "period": period,
                "attempt": attempt + 1,
                "response": raw_response,
            })
            self._append_event({
                "event": "price_llm_response",
                "period": period,
                "firm": firm,
                "attempt": attempt + 1,
                "response": raw_response,
            })
            if raw_response:
                plans, insights, price = agent.parse_price_response(raw_response)
            else:
                plans, insights, price = None, None, None
            if price is not None:
                agent.state.price_plans = plans if plans is not None else agent.state.price_plans
                agent.state.price_insights = insights if insights is not None else agent.state.price_insights
                self._append_event({
                    "event": "price_decision",
                    "period": period,
                    "firm": firm,
                    "price": price,
                    "plans": agent.state.price_plans,
                    "insights": agent.state.price_insights,
                })
                return price
        self._append_event({
            "event": "run_aborted",
            "period": period,
            "firm": firm,
            "stage": "price",
            "reason": "llm_failed_to_return_valid_price_after_10_attempts",
        })
        self._save_checkpoint(max(period - 1, 0), completed=False)
        raise RuntimeError(
            f"Aborting run: firm {firm} failed to return a valid price after 10 attempts "
            f"in round {period}."
        )

    def run(self) -> Dict[str, Any]:
        if self.resume:
            start_period = self._load_checkpoint()
        else:
            self._reset_state()
            start_period = 1

        if start_period > self.n_periods:
            return self._build_summary()

        for period in range(start_period, self.n_periods + 1):
            block_index, block_start, block_end = self._block_bounds(period)
            if self._is_quality_choice_round(period):
                quality_a = self._call_quality_stage("A", period, block_end)
                quality_b = self._call_quality_stage("B", period, block_end)
                self.current_block_qualities = (quality_a, quality_b)
                self.block_history.append({
                    "block_index": block_index,
                    "start_round": block_start,
                    "end_round": block_end,
                    "quality_A": quality_a,
                    "quality_B": quality_b,
                    "quality_pair": quality_pair_label(quality_a, quality_b),
                })
            elif self.current_block_qualities is None:
                raise RuntimeError("Missing current block qualities when resuming mid-block")

            if self.current_block_qualities is None:
                raise RuntimeError("Current block qualities must be set before pricing")
            quality_a, quality_b = self.current_block_qualities
            rounds_remaining_after_current = block_end - period
            price_a = self._call_price_stage(
                "A",
                period,
                current_quality=quality_a,
                competitor_quality=quality_b,
                block_end=block_end,
                rounds_remaining_after_current=rounds_remaining_after_current,
            )
            price_b = self._call_price_stage(
                "B",
                period,
                current_quality=quality_b,
                competitor_quality=quality_a,
                block_end=block_end,
                rounds_remaining_after_current=rounds_remaining_after_current,
            )
            quantity_a, quantity_b, profit_a, profit_b = compute_quality_profits(
                self.preset,
                quality_a,
                quality_b,
                price_a,
                price_b,
            )
            entry = {
                "round": period,
                "block_index": block_index,
                "round_in_block": period - block_start + 1,
                "rounds_until_next_quality_choice": rounds_remaining_after_current,
                "quality_A": quality_a,
                "quality_B": quality_b,
                "quality_pair": quality_pair_label(quality_a, quality_b),
                "price_A": price_a,
                "price_B": price_b,
                "quantity_A": quantity_a,
                "quantity_B": quantity_b,
                "profit_A": profit_a,
                "profit_B": profit_b,
            }
            self.market_history.append(entry)
            self.agents["A"].state.update_round_history(price_a, quantity_a, profit_a)
            self.agents["B"].state.update_round_history(price_b, quantity_b, profit_b)
            self._append_event({
                "event": "round_complete",
                **entry,
            })
            self._save_checkpoint(period)

        summary = self._build_summary()
        self._append_event({
            "event": "run_complete",
            "summary": summary,
        })
        self._save_checkpoint(self.n_periods, completed=True)
        return summary

    @staticmethod
    def _count_pairs(entries: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {pair: 0 for pair in QUALITY_PAIRS}
        for entry in entries:
            counts[entry["quality_pair"]] += 1
        return counts

    @staticmethod
    def _frequency_map(counts: Dict[str, int]) -> Dict[str, float]:
        total = sum(counts.values())
        if total == 0:
            return {pair: 0.0 for pair in counts}
        return {pair: counts[pair] / total for pair in counts}

    @staticmethod
    def _average(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.mean(values))

    def _conditional_pair_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        stats: Dict[str, Dict[str, Optional[float]]] = {}
        for pair in QUALITY_PAIRS:
            entries = [entry for entry in self.market_history if entry["quality_pair"] == pair]
            stats[pair] = {
                "avg_price": self._average(
                    [(entry["price_A"] + entry["price_B"]) / 2.0 for entry in entries]
                ),
                "avg_total_profit": self._average(
                    [entry["profit_A"] + entry["profit_B"] for entry in entries]
                ),
                "count": float(len(entries)),
            }
        return stats

    def _build_summary(self) -> Dict[str, Any]:
        final_history = self.market_history[-50:]
        avg_price_a = float(np.mean([entry["price_A"] for entry in final_history]))
        avg_price_b = float(np.mean([entry["price_B"] for entry in final_history]))
        avg_total_profit = float(
            np.mean([entry["profit_A"] + entry["profit_B"] for entry in final_history])
        )

        round_counts = self._count_pairs(self.market_history)
        round_frequencies = self._frequency_map(round_counts)
        last_100_round_counts = self._count_pairs(self.market_history[-100:])
        last_100_round_frequencies = self._frequency_map(last_100_round_counts)
        block_counts = self._count_pairs(self.block_history)
        block_frequencies = self._frequency_map(block_counts)
        last_10_block_counts = self._count_pairs(self.block_history[-10:])
        last_10_block_frequencies = self._frequency_map(last_10_block_counts)

        ll_nash_total_profit = self.static_benchmarks.stage_game_outcomes["LL"].nash_total_profit
        lh_joint_total_profit = self.static_benchmarks.stage_game_outcomes["LH"].joint_total_profit
        if lh_joint_total_profit == ll_nash_total_profit:
            profit_coordination_index = 0.0
        else:
            profit_coordination_index = (
                avg_total_profit - ll_nash_total_profit
            ) / (lh_joint_total_profit - ll_nash_total_profit)

        stage_game_outcomes = {
            pair: asdict(outcome)
            for pair, outcome in self.static_benchmarks.stage_game_outcomes.items()
        }

        return {
            "experiment_mode": self.experiment_mode,
            "prompt_family": self.prompt_family,
            "quality_preset": self.quality_preset,
            "n_periods": self.n_periods,
            "quality_block_length": self.quality_block_length,
            "avg_price_A": avg_price_a,
            "avg_price_B": avg_price_b,
            "avg_total_profit": avg_total_profit,
            "profit_coordination_index": profit_coordination_index,
            "static_quality_nash_pair": self.static_benchmarks.quality_nash_pair,
            "static_joint_optimum_pair": self.static_benchmarks.joint_optimum_pair,
            "static_LL_nash_total_profit": ll_nash_total_profit,
            "static_LH_joint_total_profit": lh_joint_total_profit,
            "quality_pair_counts": round_counts,
            "quality_pair_frequencies": round_frequencies,
            "last_100_round_quality_pair_counts": last_100_round_counts,
            "last_100_round_quality_pair_frequencies": last_100_round_frequencies,
            "block_quality_pair_counts": block_counts,
            "block_quality_pair_frequencies": block_frequencies,
            "last_10_block_quality_pair_counts": last_10_block_counts,
            "last_10_block_quality_pair_frequencies": last_10_block_frequencies,
            "conditional_quality_pair_stats": self._conditional_pair_stats(),
            "stage_game_outcomes": stage_game_outcomes,
            "run_history": self.market_history,
            "block_history": self.block_history,
        }
