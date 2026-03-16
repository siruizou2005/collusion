# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project with all source files at the root. `main.py` is the CLI entry point that runs experiment grids and writes CSV output. `simulation.py` coordinates each repeated-game run, `agents.py` builds and parses agent prompts, `economics.py` contains demand and profit calculations, `llm_client.py` handles OpenAI-compatible API calls and dummy fallback behavior, and `prompts.py` stores prompt templates. Use `test_out.csv` only as example output; do not treat it as source data. Ignore `__pycache__/`.

## Build, Test, and Development Commands
Set up a local environment before editing:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas requests
```

Run a normal experiment sweep:

```bash
python3 main.py --output results.csv
```

Run a faster smoke check while developing:

```bash
python3 main.py --runs 1 --noise_levels none --alphas 1 --output /tmp/smoke.csv
```

Use Python’s compiler as the current syntax check:

```bash
python3 -m py_compile main.py simulation.py agents.py economics.py llm_client.py prompts.py
```

## Coding Style & Naming Conventions
Follow the existing style: 4-space indentation, module-level docstrings, type hints on public functions, and `snake_case` for variables and functions. Keep classes in `PascalCase` (`CollusionSimulation`, `PricingAgent`) and constants in `UPPER_SNAKE_CASE` (`PROMPT_P0`). Prefer small, single-purpose functions and keep prompt text changes isolated to `prompts.py`.

## Testing Guidelines
There is no dedicated `tests/` suite yet. For now, treat `py_compile` plus a one-run CLI smoke test as the minimum check before opening a PR. When adding tests, place them in a new `tests/` directory and name files `test_<module>.py`; target deterministic logic first, especially `economics.py`, response parsing in `agents.py`, and argument handling in `main.py`.

## Commit & Pull Request Guidelines
Git history is not available in this workspace snapshot, so no repository-specific commit convention can be inferred. Use short imperative commit subjects such as `Add parser guard for empty responses`. PRs should describe the scenario changed, list commands run, and include sample CSV output or logs when behavior changes.

## Configuration & Security
`llm_client.py` reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment. Do not hardcode credentials or commit generated result files containing sensitive experiment data.
