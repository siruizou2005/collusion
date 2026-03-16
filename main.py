"""Command-line entry point for the collusion simulation."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from simulation import CollusionSimulation


def parse_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run collusion simulation experiments.")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Path to CSV file to save run-level results.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per treatment/alpha combination.")
    parser.add_argument("--noise_levels", type=str, default="none,low,medium,high",
                        help="Comma-separated list of noise levels: none,low,medium,high.")
    parser.add_argument("--alphas", type=str, default="1,3.2,10",
                        help="Comma-separated list of alpha values to test.")
    parser.add_argument("--prompt_families", type=str, default="P0",
                        help="Comma-separated list of prompt families: P0,P1,P2.")
    parser.add_argument("--n_periods", type=int, default=300,
                        help="Number of periods per run.")
    parser.add_argument("--cycle_effect_shares", type=str, default="0",
                        help="Comma-separated cycle effect shares where max-min = share * mean.")
    parser.add_argument("--cycle_period", type=int, default=150,
                        help="Periods per full cosine cycle.")
    parser.add_argument("--cycle_baseline", type=float, default=1.0,
                        help="Positive baseline market factor around which the cycle oscillates.")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="LLM model name to use for agent calls.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Paper default is 1.0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for per-run checkpoints and event logs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume unfinished runs from checkpoint files when available.")
    parser.add_argument("--session_tag", type=str, default=None,
                        help="Optional checkpoint session tag. Defaults to the current timestamp.")
    args = parser.parse_args(argv)
    return vars(args)


def get_noise_sigma(level: str) -> float:
    """Map a string noise level to a lognormal sigma value."""
    mapping = {
        "none": 0.0,
        "low": 0.05,
        "medium": 0.15,
        "high": 0.3,
    }
    if level not in mapping:
        raise ValueError(f"Unknown noise level: {level}")
    return mapping[level]


def parse_cycle_effect_shares(raw_value: str) -> List[float]:
    shares = [float(x.strip()) for x in raw_value.split(",") if x.strip()]
    if not shares:
        raise ValueError("At least one cycle effect share must be provided.")
    for share in shares:
        if share < 0:
            raise ValueError(f"cycle effect share must be non-negative, got {share}")
    return shares


def _config_root(base_dir: Path,
                 prompt_family: str,
                 noise_label: str,
                 alpha: float,
                 cycle_effect_share: float,
                 cycle_period: int,
                 cycle_baseline: float) -> Path:
    return (
        base_dir
        / f"prompt_{prompt_family}"
        / f"noise_{noise_label}"
        / f"cycle_{cycle_effect_share:g}"
        / f"period_{cycle_period}"
        / f"baseline_{cycle_baseline:g}"
        / f"alpha_{alpha:g}"
    )


def _latest_session_dir(run_root: Path) -> Optional[Path]:
    if not run_root.exists():
        return None
    candidates = [
        path for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("session_")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def resolve_run_dir(config_root: Path,
                    run_idx: int,
                    session_tag: str,
                    resume: bool) -> Path:
    run_root = config_root / f"run_{run_idx:03d}"
    if resume:
        tagged_dir = run_root / f"session_{session_tag}"
        if tagged_dir.exists():
            return tagged_dir
        existing = _latest_session_dir(run_root)
        if existing is not None:
            return existing
    run_dir = run_root / f"session_{session_tag}"
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint directory already exists: {run_dir}. "
            "Provide a different --session_tag or use --resume."
        )
    return run_dir


def main(argv: Optional[List[str]] = None) -> None:
    params = parse_args(argv)
    random.seed(params["seed"])
    np.random.seed(params["seed"])

    prompt_families = [x.strip() for x in params["prompt_families"].split(",") if x.strip()]
    for prompt_family in prompt_families:
        if prompt_family not in {"P0", "P1", "P2"}:
            raise ValueError(f"Unknown prompt family: {prompt_family}")
    model = params["model"]
    temperature = params["temperature"]
    n_periods = params["n_periods"]
    cycle_effect_shares = parse_cycle_effect_shares(params["cycle_effect_shares"])
    cycle_period = params["cycle_period"]
    cycle_baseline = params["cycle_baseline"]
    alphas = [float(x) for x in params["alphas"].split(",")]
    noise_levels = params["noise_levels"].split(",")
    runs_per_comb = params["runs"]
    checkpoint_root = Path(params["checkpoint_dir"])
    session_tag = params["session_tag"] or datetime.now().strftime("%Y%m%dT%H%M%S")

    results = []
    for prompt_family in prompt_families:
        for cycle_effect_share in cycle_effect_shares:
            for noise in noise_levels:
                sigma = get_noise_sigma(noise)
                for alpha in alphas:
                    config_root = _config_root(
                        checkpoint_root,
                        prompt_family,
                        noise,
                        alpha,
                        cycle_effect_share,
                        cycle_period,
                        cycle_baseline,
                    )
                    for run_idx in range(runs_per_comb):
                        run_dir = resolve_run_dir(
                            config_root,
                            run_idx,
                            session_tag=session_tag,
                            resume=params["resume"],
                        )
                        print(
                            f"Running: noise={noise}, prompt={prompt_family}, model={model}, "
                            f"alpha={alpha}, cycle_share={cycle_effect_share}, "
                            f"cycle_period={cycle_period}, temperature={temperature}, "
                            f"run={run_idx+1}/{runs_per_comb}, session={run_dir.name}"
                        )
                        sim = CollusionSimulation(
                            prompt_family,
                            sigma,
                            alpha,
                            n_periods=n_periods,
                            model=model,
                            temperature=temperature,
                            cycle_effect_share=cycle_effect_share,
                            cycle_period=cycle_period,
                            cycle_baseline=cycle_baseline,
                            checkpoint_path=str(run_dir / "checkpoint.json"),
                            event_log_path=str(run_dir / "events.jsonl"),
                            resume=params["resume"],
                        )
                        summary = sim.run()
                        summary["noise_level"] = noise
                        summary["cycle_effect_share"] = cycle_effect_share
                        summary["cycle_period"] = cycle_period
                        summary["cycle_baseline"] = cycle_baseline
                        summary["session_tag"] = run_dir.name.replace("session_", "", 1)
                        summary["run_idx"] = run_idx
                        run_dir.mkdir(parents=True, exist_ok=True)
                        (run_dir / "summary.json").write_text(
                            json.dumps(summary, ensure_ascii=True, indent=2),
                            encoding="utf-8",
                        )
                        results.append(summary)

    df = pd.DataFrame(results)
    out_path = params["output"]
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
