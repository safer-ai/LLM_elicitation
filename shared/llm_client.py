#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM API client management and async call handling.

Handles initialization of API clients (Anthropic/OpenAI),
rate limiting, and making API calls with proper error handling.
Supports extended thinking for Anthropic models.
"""

import asyncio
import time
import logging
from typing import Union, List, Optional, Any, Dict
from asyncio import Semaphore
from dataclasses import dataclass, field

try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic.types import Message
    from anthropic.lib.streaming import AsyncMessageStream
except ImportError:
    anthropic = None
    AsyncAnthropic = None
    Message = None
    AsyncMessageStream = None

try:
    import openai
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
except ImportError:
    openai = None
    AsyncOpenAI = None
    ChatCompletion = None

logger = logging.getLogger(__name__)

_call_timestamps: List[float] = []

# Google's OpenAI-compatible endpoint for the Gemini API. We reach Gemini
# through the OpenAI Python SDK by pointing it at this base URL with a Gemini
# API key, which avoids adding a separate Google SDK dependency.
GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Allowed values for the unified, provider-agnostic reasoning dial. Each
# value is mapped per-provider in `make_api_call`:
#   - Anthropic: maps to a `thinking={"type":"enabled","budget_tokens":N}` block,
#     or omitted entirely for "off".
#   - OpenAI (gpt-5*, o-series): maps to the `reasoning_effort` request param;
#     "off" -> "minimal" since reasoning models always reason internally.
#   - Google Gemini (via OpenAI-compat): same `reasoning_effort` param.
REASONING_EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")


@dataclass
class ThinkingSettings:
    """Legacy Anthropic extended-thinking settings.

    Retained for backward compatibility with callers (e.g. the calibration
    sub-projects) that build `LLMSettings(..., thinking=ThinkingSettings(...))`
    directly. New callers should set `LLMSettings.reasoning_effort` instead;
    when both are present, `reasoning_effort` wins.
    """
    enabled: bool = False
    budget_tokens: int = 10000


@dataclass
class LLMSettings:
    """Settings related to the LLM and API interaction."""
    model: str
    temperature: float = 0.8
    max_concurrent_calls: int = 5
    rate_limit_calls: int = 45
    rate_limit_period: int = 60
    # Provider-agnostic reasoning/thinking knob. One of REASONING_EFFORT_VALUES.
    # If left at the default "off", `make_api_call` falls back to the legacy
    # `thinking` block below so existing configs still work.
    reasoning_effort: str = "off"
    thinking: ThinkingSettings = field(default_factory=ThinkingSettings)


# --- Provider detection ---

def _provider_for_model(model: str) -> str:
    """Infer API provider from model name.

    Returns one of {'anthropic', 'openai', 'google'}. Raises ValueError if the
    name doesn't look like a known provider's model.
    """
    m = model.lower().strip()
    if 'claude' in m:
        return 'anthropic'
    if 'gemini' in m:
        return 'google'
    if 'gpt-' in m or any(m.startswith(p) for p in ('o1', 'o2', 'o3', 'o4', 'o5')):
        return 'openai'
    raise ValueError(
        f"Could not infer API provider from model name: '{model}'. "
        f"Expected name containing 'claude', 'gemini', 'gpt-', or starting with 'o1'..'o5'."
    )


# Back-compat alias for the previous private name.
_provider_for_model = provider_for_model


def parse_reasoning_effort(
    llm_settings_raw: Dict[str, Any],
    *,
    logger_: Optional[logging.Logger] = None,
) -> str:
    """Resolve the unified reasoning_effort value from a raw YAML llm_settings dict.

    Accepts both the new field (`reasoning_effort: "low" | "medium" | "high" |
    "minimal" | "off"`) and the legacy block (`thinking: {enabled, budget_tokens}`),
    translating the latter into one of the new tier values with a deprecation
    warning so old configs keep working.

    When both forms appear, `reasoning_effort` wins and a warning is logged.
    Returns one of REASONING_EFFORT_VALUES.
    """
    log = logger_ if logger_ is not None else logger

    reasoning_effort_raw = llm_settings_raw.get("reasoning_effort")
    legacy_thinking_raw = llm_settings_raw.get("thinking")

    if reasoning_effort_raw is not None:
        if not isinstance(reasoning_effort_raw, str):
            raise TypeError(
                "'llm_settings.reasoning_effort' must be a string, one of "
                f"{list(REASONING_EFFORT_VALUES)}."
            )
        effort = reasoning_effort_raw.strip().lower()
        if effort not in REASONING_EFFORT_VALUES:
            raise ValueError(
                f"'llm_settings.reasoning_effort' must be one of "
                f"{list(REASONING_EFFORT_VALUES)}, got {reasoning_effort_raw!r}."
            )
        if legacy_thinking_raw is not None:
            log.warning(
                "Both 'reasoning_effort' and the legacy 'thinking' block are set in "
                "llm_settings; using 'reasoning_effort' and ignoring 'thinking'."
            )
        return effort

    if isinstance(legacy_thinking_raw, dict):
        enabled = bool(legacy_thinking_raw.get("enabled", False))
        budget_tokens = int(legacy_thinking_raw.get("budget_tokens", 4000))
        if not enabled:
            effort = "off"
        elif budget_tokens < 3000:
            effort = "low"
        elif budget_tokens < 10000:
            effort = "medium"
        else:
            effort = "high"
        log.warning(
            "Config uses the legacy 'thinking: {enabled, budget_tokens}' block. "
            "This is deprecated in favour of the unified 'reasoning_effort' field. "
            f"Translated to reasoning_effort='{effort}'. "
            "Please replace `thinking: ...` with `reasoning_effort: \"%s\"` "
            "in your config.",
            effort,
        )
        return effort

    if legacy_thinking_raw is not None:
        raise ValueError("'llm_settings.thinking' must be a dictionary.")

    return "off"


# --- Reasoning effort mapping ---
#
# Anthropic exposes an explicit token budget; OpenAI / Gemini-via-OpenAI-compat
# expose a categorical `reasoning_effort` parameter. The helpers below map
# the unified config field per provider.

_ANTHROPIC_BUDGET_BY_EFFORT: Dict[str, int] = {
    "minimal": 1024,
    "low":     2048,
    "medium":  8000,
    "high":    16000,
}


def _effective_reasoning_effort(settings: LLMSettings) -> str:
    """Resolve the active reasoning effort, preferring the unified field.

    If `settings.reasoning_effort` is set to anything other than "off", use
    that. Otherwise, translate the legacy `thinking` block: when disabled,
    "off"; when enabled, bucket `budget_tokens` into low/medium/high.
    """
    if settings.reasoning_effort and settings.reasoning_effort != "off":
        return settings.reasoning_effort
    thinking = getattr(settings, "thinking", None)
    if thinking and getattr(thinking, "enabled", False):
        budget = int(getattr(thinking, "budget_tokens", 4000) or 0)
        if budget < 3000:
            return "low"
        if budget < 10000:
            return "medium"
        return "high"
    return "off"


def _anthropic_thinking_param(effort: str) -> Optional[Dict[str, Any]]:
    """Return an Anthropic `thinking` request parameter dict, or None for "off"."""
    if effort == "off":
        return None
    budget = _ANTHROPIC_BUDGET_BY_EFFORT.get(effort)
    if budget is None:
        logger.warning(
            "Unrecognised reasoning_effort=%r for Anthropic; treating as 'off'.", effort,
        )
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _is_openai_compat_reasoning_model(model: str, provider: str) -> bool:
    """Heuristic: does this model accept the `reasoning_effort` request parameter?"""
    m = model.lower()
    if provider == "openai":
        return m.startswith("gpt-5") or any(m.startswith(p) for p in ("o1", "o3", "o4", "o5"))
    if provider == "google":
        return ("gemini-3" in m) or ("gemini-2.5" in m)
    return False


def _openai_compat_reasoning_effort(effort: str) -> Optional[str]:
    """Map unified effort to the OpenAI/Gemini `reasoning_effort` request value.

    Returns None if the parameter should be omitted. "off" maps to "minimal"
    because OpenAI reasoning models always reason internally; we can only
    minimise the budget, not disable it.
    """
    if effort == "off":
        return "minimal"
    if effort in ("minimal", "low", "medium", "high"):
        return effort
    logger.warning(
        "Unrecognised reasoning_effort=%r for OpenAI-compat call; omitting parameter.", effort,
    )
    return None


def initialize_client(
    api_key_anthropic: Optional[str],
    api_key_openai: Optional[str],
    model: str,
    api_key_gemini: Optional[str] = None,
) -> Union[AsyncAnthropic, AsyncOpenAI]:
    """Initialises the appropriate asynchronous API client based on model name.

    Returns an AsyncAnthropic for Claude models, an AsyncOpenAI for OpenAI
    models, and an AsyncOpenAI pointed at Google's OpenAI-compat endpoint for
    Gemini models. `api_key_gemini` is keyword-only-by-convention so existing
    three-arg callers (e.g. the calibration sub-projects) remain unaffected.

    Raises:
        ImportError: If the required API library is not installed.
        ValueError: If the API provider cannot be inferred or API key is missing.
    """
    provider = _provider_for_model(model)

    if provider == 'anthropic':
        if not anthropic:
            raise ImportError("Anthropic provider selected, but 'anthropic' library is not installed. Run: pip install anthropic")
        if not api_key_anthropic:
            raise ValueError("Anthropic provider selected, but API key is missing in config.")
        logger.info(f"Initializing Anthropic client for model: {model}")
        return AsyncAnthropic(api_key=api_key_anthropic)

    if provider == 'openai':
        if not openai:
            raise ImportError("OpenAI provider selected, but 'openai' library is not installed. Run: pip install openai")
        if not api_key_openai:
            raise ValueError("OpenAI provider selected, but API key is missing in config.")
        logger.info(f"Initializing OpenAI client for model: {model}")
        return AsyncOpenAI(api_key=api_key_openai)

    if provider == 'google':
        if not openai:
            raise ImportError(
                "Google (Gemini) provider selected. Gemini is reached via the OpenAI-compatible "
                "endpoint, but 'openai' library is not installed. Run: pip install openai"
            )
        if not api_key_gemini:
            raise ValueError("Google (Gemini) provider selected, but 'api_key_gemini' is missing.")
        logger.info(
            f"Initializing OpenAI-compat client for Gemini model '{model}' "
            f"(base_url={GEMINI_OPENAI_COMPAT_BASE_URL})"
        )
        return AsyncOpenAI(api_key=api_key_gemini, base_url=GEMINI_OPENAI_COMPAT_BASE_URL)

    raise ValueError(f"Unsupported API provider for model: {model}")


async def _rate_limit_wait(semaphore: Semaphore, settings: LLMSettings):
    """Acquires semaphore and waits if the rate limit is hit."""
    global _call_timestamps
    async with semaphore:
        while True:
            current_time = time.time()
            _call_timestamps = [ts for ts in _call_timestamps if current_time - ts < settings.rate_limit_period]

            if len(_call_timestamps) < settings.rate_limit_calls:
                _call_timestamps.append(current_time)
                break
            else:
                time_since_oldest = current_time - _call_timestamps[0]
                wait_time = settings.rate_limit_period - time_since_oldest
                wait_time = max(0.1, wait_time + 0.05)
                logger.debug(f"Rate limit hit ({len(_call_timestamps)} calls/{settings.rate_limit_period}s). Waiting {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)


def _get_final_text_anthropic(response: Union[Message, AsyncMessageStream, Any]) -> str:
    """Extracts the primary text content from an Anthropic API response, skipping thinking blocks."""
    if not response:
        logger.warning("Received empty response object from Anthropic.")
        return ""
    try:
        if isinstance(response, Message):
            if response.content and isinstance(response.content, list):
                text_parts = []
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "text":
                        text_parts.append(getattr(block, 'text', ''))
                    elif hasattr(block, 'type') and block.type == "thinking":
                        logger.debug("Found 'thinking' block in response, skipping for final text extraction.")

                if not text_parts:
                    logger.warning(f"Could not find any 'text' type content blocks in Anthropic Message content: {response.content}")
                    return ""

                return "\n".join(text_parts).strip()
        else:
            logger.warning(f"Received unexpected response type from Anthropic: {type(response)}")
            return str(response)

    except Exception as e:
        logger.error(f"Error extracting text from Anthropic response: {e}. Response type: {type(response)}", exc_info=True)
        return ""


def _get_final_text_openai(response: Union[ChatCompletion, Any]) -> str:
    """Extracts the primary text content from an OpenAI ChatCompletion API response.

    Surfaces a loud warning when ``finish_reason == "length"`` so a truncated
    response (which silently corrupts downstream rationale parsing) is never
    invisible.
    """
    if not response:
        logger.warning("Received empty response object from OpenAI.")
        return ""
    try:
        if openai and isinstance(response, ChatCompletion):
            if response.choices and isinstance(response.choices, list) and len(response.choices) > 0:
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                    reasoning_tokens = None
                    if usage is not None:
                        details = getattr(usage, "completion_tokens_details", None)
                        if details is not None:
                            reasoning_tokens = getattr(details, "reasoning_tokens", None)
                    logger.warning(
                        "OpenAI/Gemini response was truncated by max_completion_tokens "
                        "(finish_reason='length', completion_tokens=%s, reasoning_tokens=%s). "
                        "Downstream parsing may receive an incomplete <rationale> block.",
                        completion_tokens, reasoning_tokens,
                    )
                if choice.message and hasattr(choice.message, 'content'):
                    return choice.message.content or ""
            logger.warning(f"Could not find message content in OpenAI ChatCompletion choices: {response.choices}")
            return ""
        else:
            logger.warning(f"Received unexpected response type from OpenAI: {type(response)}")
            return str(response)

    except Exception as e:
        logger.error(f"Error extracting text from OpenAI response: {e}. Response type: {type(response)}", exc_info=True)
        return ""


async def make_api_call(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    settings: LLMSettings,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 20000
) -> str:
    """
    Makes an asynchronous call to the configured LLM API (Anthropic or OpenAI),
    handling rate limiting, errors, and response text extraction.
    Supports extended thinking for Anthropic models when configured.

    Returns:
        The extracted text content from the LLM response, or an
        error message string starting with "Error:" if the call failed.
    """
    await _rate_limit_wait(semaphore, settings)

    model = settings.model
    provider = _provider_for_model(model)
    effort = _effective_reasoning_effort(settings)

    logger.debug(
        f"Making API call: Provider={provider}, Model={model}, "
        f"Temp={settings.temperature}, MaxTokens={max_tokens}, ReasoningEffort={effort!r}"
    )

    try:
        if provider == 'anthropic' and isinstance(client, AsyncAnthropic):
            params: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": settings.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            thinking_param = _anthropic_thinking_param(effort)
            if thinking_param is not None:
                logger.debug(
                    "Anthropic extended thinking enabled "
                    f"(reasoning_effort={effort!r}, "
                    f"budget_tokens={thinking_param['budget_tokens']})."
                )
                params["thinking"] = thinking_param

            response = await client.messages.create(**params)
            logger.debug(f"Anthropic API call successful. Raw response type: {type(response)}")
            return _get_final_text_anthropic(response)

        # OpenAI and Gemini both go through the OpenAI Python SDK; Gemini uses
        # the same chat.completions surface via Google's OpenAI-compat
        # endpoint. Both surfaces use `max_completion_tokens` (the legacy
        # `max_tokens` is rejected by GPT-5 / o-series).
        elif provider in ('openai', 'google') and isinstance(client, AsyncOpenAI):
            messages_payload = []
            if system_prompt:
                messages_payload.append({"role": "system", "content": system_prompt})
            messages_payload.append({"role": "user", "content": user_prompt})

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages_payload,
                "max_completion_tokens": max_tokens,
                "temperature": settings.temperature,
            }

            if _is_openai_compat_reasoning_model(model, provider):
                effort_value = _openai_compat_reasoning_effort(effort)
                if effort_value is not None:
                    request_kwargs["reasoning_effort"] = effort_value
                    logger.debug(
                        "%s reasoning model: sending reasoning_effort=%r "
                        "(effective effort=%r).",
                        provider, effort_value, effort,
                    )
            elif effort != "off":
                logger.debug(
                    "Model %r is not a known reasoning-capable %s model; "
                    "ignoring reasoning_effort=%r.",
                    model, provider, effort,
                )

            response = await client.chat.completions.create(**request_kwargs)
            if response.usage:
                logger.debug(
                    f"{provider} API call successful. Usage: "
                    f"Prompt={response.usage.prompt_tokens}, "
                    f"Completion={response.usage.completion_tokens}, "
                    f"Total={response.usage.total_tokens}"
                )
            return _get_final_text_openai(response)

        else:
            err_msg = f"Mismatch between inferred provider '{provider}' and client type '{type(client)}'."
            logger.error(err_msg)
            return "Error: Client/Provider mismatch"

    except (anthropic.APIConnectionError if anthropic else Exception, openai.APIConnectionError if openai else Exception) as e:
        logger.error(f"API Connection Error: {e}", exc_info=True)
        return "Error: API Connection Error"
    except (anthropic.RateLimitError if anthropic else Exception, openai.RateLimitError if openai else Exception) as e:
        logger.warning(f"API Rate Limit Error encountered: {e}. Consider adjusting rate limit settings in config.")
        return "Error: API Rate Limit Error"
    except (anthropic.AuthenticationError if anthropic else Exception, openai.AuthenticationError if openai else Exception) as e:
        logger.error(f"API Authentication Error: {e}. Check your API key.")
        return "Error: API Authentication Error"
    except (anthropic.BadRequestError if anthropic else Exception, openai.BadRequestError if openai else Exception) as e:
        logger.error(f"API Bad Request Error: {e}", exc_info=True)
        return f"Error: API Bad Request - {e}"
    except Exception as e:
        logger.error(f"Unexpected error during API call: {type(e).__name__} - {e}", exc_info=True)
        return f"Error: Unexpected error - {str(e)}"
