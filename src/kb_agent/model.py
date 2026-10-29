"""Create the OpenRouter model and bind the agent's tools."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openrouter import ChatOpenRouter

from kb_agent.schemas import ACTION_SCHEMAS, Limits


class ConfigurationError(ValueError):
    """Invalid local settings, without exposing credentials."""


def load_configuration(env_path: str | Path = ".env") -> tuple[dict, Limits]:
    load_dotenv(env_path, override=False)
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
    provider = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    if provider != "openrouter":
        raise ConfigurationError("This project uses OpenRouter. Set LLM_PROVIDER=openrouter.")
    if base_url.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ConfigurationError("Remove the previous provider URL; use https://openrouter.ai/api/v1.")
    
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key or not model:
        raise ConfigurationError("Configure LLM_MODEL and LLM_API_KEY (or OPENROUTER_API_KEY) in .env.")
    interval = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "4"))
    if not 0 <= interval <= 60:
        raise ConfigurationError("LLM_MIN_INTERVAL_SECONDS must be between 0 and 60.")
    
    config = {
        "provider": provider,
        "api_key": key,
        "model": model,
        "base_url": base_url,
        "timeout": 45,
        "max_retries": 0,
        "request_interval": interval,
    }
    
    limits = Limits(
        max_model_calls=int(os.getenv("MAX_MODEL_CALLS", "10")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "8")),
    )
    return config, limits


def create_model(config: dict, *, client=None):
    interval = config.get("request_interval", 4)
    limiter = None
    if interval > 0:
        limiter = InMemoryRateLimiter(
            requests_per_second=1 / interval,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )

    model = ChatOpenRouter(
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=int(config.get("timeout", 45) * 1000),  # OpenRouter SDK uses milliseconds.
        max_retries=0,
        temperature=0,
        rate_limiter=limiter,
        openrouter_provider={"require_parameters": True},
        client=client,  # Optional SDK client for offline tests.
    )

    # Preserve the tool names expected by the graph's execute() function.
    tools = []
    for name, schema in ACTION_SCHEMAS.items():
        tool = convert_to_openai_tool(schema)
        tool["function"]["name"] = name
        tools.append(tool)

    # Explicitly disable the underlying SDK's per-request retries as well.
    return model.bind_tools(tools, tool_choice="required", retries=None)
