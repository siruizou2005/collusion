"""LLM communication utilities for the collusion simulation."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional

import requests

_ENV_LOADED = False


def _load_local_env() -> None:
    """Load a local .env file once if present.

    Existing environment variables take precedence.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()

    _ENV_LOADED = True


def call_llm(prompt: str,
             model: str = "gpt-3.5-turbo",
             max_tokens: int = 1024,
             temperature: float = 1.0,
             api_base: Optional[str] = None,
             api_key: Optional[str] = None,
             retries: int = 3,
             timeout: int = 60) -> Optional[str]:
    """Call an OpenAI-compatible chat completion endpoint with the given prompt.

    Parameters
    ----------
    prompt : str
        The full user prompt including any prefix and instructions.
    model : str, optional
        The model name. Defaults to 'gpt-3.5-turbo'.
    max_tokens : int, optional
        Maximum tokens in the response. Defaults to 1024.
    temperature : float, optional
        Sampling temperature. Defaults to 1.0.
    api_base : str, optional
        Base URL for the API. Falls back to OPENAI_BASE_URL env var.
    api_key : str, optional
        API key. Falls back to OPENAI_API_KEY env var.
    retries : int, optional
        Number of retry attempts. Defaults to 3.
    timeout : int, optional
        Timeout in seconds per request. Defaults to 60.

    Returns
    -------
    Optional[str]
        The LLM response content, or None if all retries fail.
    """
    _load_local_env()
    if api_base is None:
        api_base = os.environ.get("OPENAI_BASE_URL")
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_base or not api_key:
        print("[WARN] OPENAI_BASE_URL or OPENAI_API_KEY environment variables are not set."
              " Falling back to dummy agent.")
        return None

    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    return content
                else:
                    print(f"[ERROR] Unexpected response format: {data}")
            else:
                print(f"[ERROR] API returned status {response.status_code}: {response.text}")
        except Exception as exc:
            print(f"[ERROR] LLM call failed on attempt {attempt + 1}/{retries}: {exc}")
        time.sleep(2 ** attempt)
    return None


def dummy_price_strategy(min_price: float, max_price: float) -> float:
    """Fallback strategy used when no API key is provided.

    Returns a random price within the provided bounds.
    """
    return round(random.uniform(min_price, max_price), 2)
