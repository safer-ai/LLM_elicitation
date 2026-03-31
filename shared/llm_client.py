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
from typing import Union, List, Optional, Any
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


@dataclass
class ThinkingSettings:
    """Settings for Anthropic extended thinking."""
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
    thinking: ThinkingSettings = field(default_factory=ThinkingSettings)


def initialize_client(api_key_anthropic: Optional[str], api_key_openai: Optional[str], model: str) -> Union[AsyncAnthropic, AsyncOpenAI]:
    """
    Initialises the appropriate asynchronous API client based on model name.

    Raises:
        ImportError: If the required API library is not installed.
        ValueError: If the API provider cannot be inferred or API key is missing.
    """
    model_lower = model.lower()

    if 'claude' in model_lower:
        if not anthropic:
            raise ImportError("Anthropic provider selected, but 'anthropic' library is not installed. Run: pip install anthropic")
        if not api_key_anthropic:
            raise ValueError("Anthropic provider selected, but API key is missing in config.")
        logger.info(f"Initializing Anthropic client for model: {model}")
        return AsyncAnthropic(api_key=api_key_anthropic)

    elif 'gpt-' in model_lower or 'o1' in model_lower or 'o3' in model_lower or 'o4' in model_lower:
        if not openai:
            raise ImportError("OpenAI provider selected, but 'openai' library is not installed. Run: pip install openai")
        if not api_key_openai:
            raise ValueError("OpenAI provider selected, but API key is missing in config.")
        logger.info(f"Initializing OpenAI client for model: {model}")
        return AsyncOpenAI(api_key=api_key_openai)

    else:
        raise ValueError(f"Could not infer API provider from model name: {model}")


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
    """Extracts the primary text content from an OpenAI ChatCompletion API response."""
    if not response:
        logger.warning("Received empty response object from OpenAI.")
        return ""
    try:
        if openai and isinstance(response, ChatCompletion):
            if response.choices and isinstance(response.choices, list) and len(response.choices) > 0:
                choice = response.choices[0]
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
    max_tokens: int = 6000
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
    model_lower = model.lower()
    is_anthropic = 'claude' in model_lower
    is_openai = 'gpt-' in model_lower or 'o1' in model_lower or 'o3' in model_lower or 'o4' in model_lower

    logger.debug(f"Making API call: Model={model}, Temp={settings.temperature}, MaxTokens={max_tokens}")

    try:
        if is_anthropic and isinstance(client, AsyncAnthropic):
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": settings.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            if settings.thinking.enabled:
                logger.debug("Extended thinking enabled. Adding 'thinking' parameter to API call.")
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": settings.thinking.budget_tokens
                }

            response = await client.messages.create(**params)
            logger.debug(f"Anthropic API call successful. Raw response type: {type(response)}")
            return _get_final_text_anthropic(response)

        elif is_openai and isinstance(client, AsyncOpenAI):
            messages_payload = []
            if system_prompt:
                messages_payload.append({"role": "system", "content": system_prompt})
            messages_payload.append({"role": "user", "content": user_prompt})

            response = await client.chat.completions.create(
                model=model,
                messages=messages_payload,
                max_tokens=max_tokens,
                temperature=settings.temperature,
            )
            if response.usage:
                logger.debug(f"OpenAI API call successful. Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")
            return _get_final_text_openai(response)

        else:
            err_msg = f"Mismatch between inferred provider and client type '{type(client)}'."
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
