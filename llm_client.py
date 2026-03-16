"""LLM communication utilities for the collusion simulation."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

_ENV_LOADED = False

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


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


@lru_cache(maxsize=16)
def _get_client(api_key: str, timeout: int) -> genai.Client:
    """Create and cache a Gemini-native client."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout * 1000),
    )

def _extract_response_text(response: object, structured: bool = False) -> Optional[str]:
    """Normalize a Gemini response to text or canonical JSON."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="there are non-text parts in the response:.*",
        )
        if structured:
            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                if hasattr(parsed, "model_dump_json"):
                    return parsed.model_dump_json()
                return json.dumps(parsed, ensure_ascii=True, separators=(",", ":"))
        text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def call_llm(prompt: str,
             model: str = "gemini-3-flash-preview",
             max_tokens: int = 1024,
             temperature: float = 1.0,
             api_base: Optional[str] = None,
             api_key: Optional[str] = None,
             response_schema: Optional[Any] = None,
             response_json_schema: Optional[dict[str, Any]] = None,
             retries: int = 3,
             timeout: int = 60) -> Optional[str]:
    """Call Gemini native `generate_content()` with optional structured output.

    Parameters
    ----------
    prompt : str
        The full prompt including instructions and market context.
    model : str, optional
        Gemini model name. Defaults to 'gemini-3-flash-preview'.
    max_tokens : int, optional
        Maximum output tokens. Defaults to 1024.
    temperature : float, optional
        Sampling temperature. Defaults to 1.0.
    api_base : str, optional
        Unused for Gemini native calls. Kept for backward compatibility.
    api_key : str, optional
        API key. Falls back to GEMINI_API_KEY, GOOGLE_API_KEY, then OPENAI_API_KEY.
    response_schema : Any, optional
        Native Gemini structured schema, such as a Pydantic model class.
    response_json_schema : dict, optional
        JSON schema for structured output. When provided, the response is requested
        as `application/json`.
    retries : int, optional
        Number of retry attempts. Defaults to 3.
    timeout : int, optional
        Kept for backward compatibility; Gemini client manages HTTP timeouts internally.

    Returns
    -------
    Optional[str]
        The LLM response text, or None if all retries fail.
    """
    _load_local_env()
    if api_key is None:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    if not api_key:
        print(
            "[WARN] GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY is not set. "
            "Falling back to dummy agent."
        )
        return None

    request_max_tokens = max_tokens
    if model.startswith("gemini-") and max_tokens < 128:
        request_max_tokens = 128

    for attempt in range(retries):
        try:
            client = _get_client(api_key, timeout)
            config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": request_max_tokens,
            }
            if response_schema is not None:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = response_schema
                config["thinking_config"] = {"thinking_budget": 0}
            if response_json_schema is not None:
                config["response_mime_type"] = "application/json"
                config["response_json_schema"] = response_json_schema
                config["thinking_config"] = {"thinking_budget": 0}
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            content = _extract_response_text(
                response,
                structured=response_schema is not None or response_json_schema is not None,
            )
            if content is not None:
                return content
            print(f"[ERROR] Unexpected response format: {response}")
        except Exception as exc:
            print(f"[ERROR] LLM call failed on attempt {attempt + 1}/{retries}: {exc}")
        time.sleep(2 ** attempt)
    return None
