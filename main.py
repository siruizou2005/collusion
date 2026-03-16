"""Command-line entry point for the collusion simulation."""

from __future__ import annotations

import argparse
import json
import random
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
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="LLM model name to use for agent calls.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Paper default is 1.0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for per-run checkpoints and event logs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume unfinished runs from checkpoint files when available.")
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
    alphas = [float(x) for x in params["alphas"].split(",")]
    noise_levels = params["noise_levels"].split(",")
    runs_per_comb = params["runs"]
    checkpoint_root = Path(params["checkpoint_dir"])

    results = []
    for prompt_family in prompt_families:
        for noise in noise_levels:
            sigma = get_noise_sigma(noise)
            for alpha in alphas:
                for run_idx in range(runs_per_comb):
                    run_dir = (
                        checkpoint_root
                        / f"prompt_{prompt_family}"
                        / f"noise_{noise}"
                        / f"alpha_{alpha:g}"
                        / f"run_{run_idx:03d}"
                    )
                    print(
                        f"Running: noise={noise}, prompt={prompt_family}, model={model}, "
                        f"alpha={alpha}, temperature={temperature}, run={run_idx+1}/{runs_per_comb}"
                    )
                    sim = CollusionSimulation(
                        prompt_family,
                        sigma,
                        alpha,
                        n_periods=n_periods,
                        model=model,
                        temperature=temperature,
                        checkpoint_path=str(run_dir / "checkpoint.json"),
                        event_log_path=str(run_dir / "events.jsonl"),
                        resume=params["resume"],
                    )
                    summary = sim.run()
                    summary["noise_level"] = noise
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
