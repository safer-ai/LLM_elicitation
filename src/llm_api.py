# src/llm_api.py

import yaml
import asyncio
import time
import logging
from typing import Union, List, Dict, Optional, Any
from asyncio import Semaphore

# Conditional imports for API clients - will raise ImportError later if needed but not installed
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

# Import necessary configuration
from config import AppConfig, LLMSettings, load_config

logger = logging.getLogger(__name__)

# Module-level state for rate limiting timestamps (shared across calls)
# Alternative: Could be part of a class if more state is needed.
_call_timestamps: List[float] = []

# --- Client Initialization ---

def initialize_client(config: AppConfig) -> Union[AsyncAnthropic, AsyncOpenAI]:
    """
    Initializes the appropriate asynchronous API client based on config.

    Args:
        config: The loaded AppConfig containing settings and API keys.

    Returns:
        An initialized AsyncAnthropic or AsyncOpenAI client instance.

    Raises:
        ImportError: If the required API library (anthropic or openai) is not installed.
        ValueError: If the API provider cannot be inferred or the required API key is missing.
    """
    provider = config.inferred_api_provider # This already checks if provider can be inferred

    if provider == 'anthropic':
        if not anthropic:
            raise ImportError("Anthropic provider selected, but 'anthropic' library is not installed. Run: pip install anthropic")
        if not config.api_key_anthropic:
            # This check is also done in load_config, but double-checking is safe
            raise ValueError("Anthropic provider selected, but 'api_key_anthropic' is missing in config.")
        logger.info(f"Initializing Anthropic client for model: {config.llm_settings.model}")
        return AsyncAnthropic(api_key=config.api_key_anthropic)

    elif provider == 'openai':
        if not openai:
            raise ImportError("OpenAI provider selected, but 'openai' library is not installed. Run: pip install openai")
        if not config.api_key_openai:
            raise ValueError("OpenAI provider selected, but 'api_key_openai' is missing in config.")
        logger.info(f"Initializing OpenAI client for model: {config.llm_settings.model}")
        return AsyncOpenAI(api_key=config.api_key_openai)

    else:
        # Should be caught by config.inferred_api_provider, but as a fallback
        raise ValueError(f"Unsupported or undetectable API provider for model: {config.llm_settings.model}")


# --- Rate Limiting Logic ---

async def _rate_limit_wait(semaphore: Semaphore, settings: LLMSettings):
    """
    Acquires semaphore and waits if the rate limit is hit.

    Args:
        semaphore: The asyncio Semaphore to control concurrency.
        settings: LLMSettings containing rate limit parameters.

    Returns:
        None. Blocks asynchronously until a call slot is available.
    """
    global _call_timestamps
    async with semaphore:
        while True:
            current_time = time.time()
            # Remove timestamps older than the rate limit period
            _call_timestamps = [ts for ts in _call_timestamps if current_time - ts < settings.rate_limit_period]

            if len(_call_timestamps) < settings.rate_limit_calls:
                _call_timestamps.append(current_time)
                # Successfully acquired slot within rate limit
                break
            else:
                # Rate limit hit, calculate wait time for the oldest call to expire
                time_since_oldest = current_time - _call_timestamps[0]
                wait_time = settings.rate_limit_period - time_since_oldest
                # Add a small buffer to avoid race conditions
                wait_time = max(0.1, wait_time + 0.05)
                logger.debug(f"Rate limit hit ({len(_call_timestamps)} calls/{settings.rate_limit_period}s). Waiting {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)
                # Loop continues to re-check after waiting


# --- Response Text Extraction ---

def _get_final_text_anthropic(response: Union[Message, AsyncMessageStream, Any]) -> str:
    """Extracts the primary text content from an Anthropic API response."""
    if not response:
        logger.warning("Received empty response object from Anthropic.")
        return ""
    try:
        if isinstance(response, Message):
            if response.content and isinstance(response.content, list):
                # --- MODIFICATION START ---
                # Combine all 'text' blocks. This handles both cases:
                # 1. Standard response: A single 'text' block.
                # 2. Thinking response: Multiple blocks, we extract and join only the 'text' ones.
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
        # Handle potential streaming response (needs adaptation if streaming is used)
        elif anthropic and isinstance(response, AsyncMessageStream):
             logger.warning("Received Anthropic stream object - text extraction might be incomplete or incorrect without full stream handling.")
             # Attempt basic extraction if possible, but proper stream handling is needed elsewhere
             # This part is placeholder - DO NOT rely on it for actual streaming.
             try:
                 # This is NOT the correct way to handle streams fully
                 if hasattr(response, '_current_message') and response._current_message:
                      return _get_final_text_anthropic(response._current_message)
             except Exception: pass # Ignore errors in placeholder stream handling
             return "[Streaming response - requires specific handling]"
        else:
             logger.warning(f"Received unexpected response type from Anthropic: {type(response)}")
             return str(response) # Fallback to string representation

    except Exception as e:
        logger.error(f"Error extracting text from Anthropic response: {e}. Response type: {type(response)}", exc_info=True)
        return ""

def _get_final_text_openai(response: Union[ChatCompletion, Any]) -> str:
    """Extracts the primary text content from an OpenAI ChatCompletion API response."""
    if not response:
        logger.warning("Received empty response object from OpenAI.")
        return ""
    try:
        # Handle standard ChatCompletion object
        if openai and isinstance(response, ChatCompletion):
            if response.choices and isinstance(response.choices, list) and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and hasattr(choice.message, 'content'):
                    return choice.message.content or "" # Return content or empty string if None
            logger.warning(f"Could not find message content in OpenAI ChatCompletion choices: {response.choices}")
            return ""
        else:
             logger.warning(f"Received unexpected response type from OpenAI: {type(response)}")
             return str(response) # Fallback

    except Exception as e:
        logger.error(f"Error extracting text from OpenAI response: {e}. Response type: {type(response)}", exc_info=True)
        return ""

# --- Main API Call Function ---

async def make_api_call(
    client: Union[AsyncAnthropic, AsyncOpenAI],
    semaphore: Semaphore,
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 6000 # Default max tokens, can be overridden
) -> str:
    """
    Makes an asynchronous call to the configured LLM API (Anthropic or OpenAI),
    handling rate limiting, errors, and response text extraction.

    Args:
        client: The initialized async API client (Anthropic or OpenAI).
        semaphore: The asyncio Semaphore for concurrency control.
        config: The AppConfig object with settings.
        system_prompt: The system prompt or context for the LLM.
        user_prompt: The main user query or instruction.
        max_tokens: The maximum number of tokens to generate in the response.

    Returns:
        The extracted text content from the LLM response, or an
        error message string starting with "Error:" if the call failed.
    """
    await _rate_limit_wait(semaphore, config.llm_settings)

    provider = config.inferred_api_provider
    model = config.llm_settings.model
    temp = config.llm_settings.temperature

    logger.debug(f"Making API call: Provider={provider}, Model={model}, Temp={temp}, MaxTokens={max_tokens}")
    # Avoid logging full prompts unless necessary for debugging sensitive info
    logger.debug(f"System Prompt (start): {system_prompt[:100]}...")
    logger.debug(f"User Prompt (start): {user_prompt[:100]}...")

    try:
        # --- Anthropic API Call ---
        if provider == 'anthropic' and isinstance(client, AsyncAnthropic):
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }

            if config.llm_settings.thinking.enabled:
                logger.debug("Extended thinking enabled. Adding 'thinking' parameter to API call.")
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": config.llm_settings.thinking.budget_tokens
                    }
                
            response = await client.messages.create(**params)

            # Note: Anthropic usage info might be slightly different or evolve
            # logger.debug(f"Anthropic API call successful. Usage: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}")
            logger.debug(f"Anthropic API call successful. Raw response type: {type(response)}")
            return _get_final_text_anthropic(response)

        # --- OpenAI API Call ---
        elif provider == 'openai' and isinstance(client, AsyncOpenAI):
            messages_payload = []
            if system_prompt:
                messages_payload.append({"role": "system", "content": system_prompt})
            messages_payload.append({"role": "user", "content": user_prompt})

            response = await client.chat.completions.create(
                model=model,
                messages=messages_payload,
                max_tokens=max_tokens,
                temperature=temp,
                # Other potential params like top_p could be added from config if needed
            )
            if response.usage:
                logger.debug(f"OpenAI API call successful. Usage: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")
            else:
                 logger.debug(f"OpenAI API call successful. Usage info not available in response.")
            return _get_final_text_openai(response)

        # --- Should not happen due to client initialization logic ---
        else:
            err_msg = f"Mismatch between inferred provider '{provider}' and client type '{type(client)}'."
            logger.error(err_msg)
            return f"Error: Client/Provider mismatch"

    # --- API Error Handling ---
    # Catch specific errors for both libraries where possible
    except (anthropic.APIConnectionError if anthropic else Exception, openai.APIConnectionError if openai else Exception) as e:
        logger.error(f"API Connection Error: {e}", exc_info=True)
        return "Error: API Connection Error"
    except (anthropic.RateLimitError if anthropic else Exception, openai.RateLimitError if openai else Exception) as e:
        logger.warning(f"API Rate Limit Error encountered: {e}. Consider adjusting rate limit settings in config.")
        # Optional: add a small delay here before returning, maybe caller handles retry logic
        # await asyncio.sleep(5)
        return "Error: API Rate Limit Error"
    except (anthropic.AuthenticationError if anthropic else Exception, openai.AuthenticationError if openai else Exception) as e:
        logger.error(f"API Authentication Error: {e}. Check your API key.")
        return "Error: API Authentication Error"
    except (anthropic.BadRequestError if anthropic else Exception, openai.BadRequestError if openai else Exception) as e:
        # These often indicate issues with the prompt, model parameters, or model access
        logger.error(f"API Bad Request Error: {e}", exc_info=True)
        return f"Error: API Bad Request - {e}"
    except (anthropic.APIStatusError) as e: # More specific Anthropic error
         logger.error(f"Anthropic API Status Error {e.status_code}: {e.response}", exc_info=True)
         return f"Error: API Status Error {e.status_code}"
    # Catch remaining OpenAI specific errors if needed, though many inherit from BadRequestError
    # except openai.APIStatusError as e: ... # If needed

    # --- General Error Handling ---
    except Exception as e:
        logger.error(f"Unexpected error during API call: {type(e).__name__} - {e}", exc_info=True)
        return f"Error: Unexpected error - {str(e)}"


# --- Test Execution Block ---
if __name__ == "__main__":
    print("--- Running LLM API Tests ---")

    async def run_test():
        try:
            # Assuming config.yaml is in the project root
            config = load_config("config.yaml")
            print(f"Config loaded. Provider: {config.inferred_api_provider}, Model: {config.llm_settings.model}")

            # Check if necessary key exists before initializing
            key_present = False
            if config.inferred_api_provider == 'anthropic' and config.api_key_anthropic:
                key_present = True
            elif config.inferred_api_provider == 'openai' and config.api_key_openai:
                key_present = True

            if not key_present:
                print(f"ERROR: API key for provider '{config.inferred_api_provider}' not found in config.yaml. Skipping API call test.")
                return

            client = initialize_client(config)
            print(f"Client initialized: {type(client)}")

            # Create a semaphore for the test
            semaphore = Semaphore(config.llm_settings.max_concurrent_calls)

            # Define dummy prompts
            system_p = "You are a helpful assistant."
            user_p = "Explain the concept of asynchronous programming in Python in one sentence."

            print(f"\nMaking test API call to {config.llm_settings.model}...")
            result = await make_api_call(client, semaphore, config, system_p, user_p, max_tokens=6000)

            print("\n--- API Call Result ---")
            if result.startswith("Error:"):
                print(f"FAILED: {result}")
            else:
                print("SUCCESS:")
                print(result)
            print("-----------------------")

        except (FileNotFoundError, yaml.YAMLError, ValueError, TypeError, ImportError) as e:
            print(f"\nERROR during setup or API call: {e}")
            logger.exception("Error in API test block:")
        except Exception as e:
            print(f"\nUNEXPECTED ERROR during testing: {e}")
            logger.exception("Unexpected error in API test block:")

    # Run the async test function
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"Error running asyncio test: {e}") # Catch errors during asyncio.run itself

    print("\n--- LLM API Tests Complete ---")