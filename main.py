"""Command-line entry point for the collusion simulation."""

from __future__ import annotations

import argparse
import random
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
    parser.add_argument("--cycle_amplitudes", type=str, default="0,0.1,0.3,0.5",
                        help="Comma-separated list of cycle amplitude values.")
    parser.add_argument("--alphas", type=str, default="1,3.2,10",
                        help="Comma-separated list of alpha values to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args(argv)
    return vars(args)


def main(argv: Optional[List[str]] = None) -> None:
    params = parse_args(argv)
    random.seed(params["seed"])
    np.random.seed(params["seed"])

    prompt_family = "P0C"
    alphas = [float(x) for x in params["alphas"].split(",")]
    amplitudes = [float(x) for x in params["cycle_amplitudes"].split(",")]
    runs_per_comb = params["runs"]

    results = []
    for amplitude in amplitudes:
        for alpha in alphas:
            for run_idx in range(runs_per_comb):
                print(f"Running: cycle_amplitude={amplitude}, prompt={prompt_family}, alpha={alpha}, run={run_idx+1}/{runs_per_comb}")
                sim = CollusionSimulation(prompt_family, alpha, cycle_amplitude=amplitude)
                summary = sim.run()
                summary["run_idx"] = run_idx
                results.append(summary)

    df = pd.DataFrame(results)
    out_path = params["output"]
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
