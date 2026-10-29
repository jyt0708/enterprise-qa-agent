"""Configuration for an external, tool-capable chat model. No fake runtime mode."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from kb_agent.schemas import ACTION_SCHEMAS, Limits


def load_configuration(env_path: str | Path = ".env") -> tuple[dict, Limits]:
    load_dotenv(env_path, override=False)
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL") or None
    provider = os.getenv("LLM_PROVIDER").strip().lower()
    
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError(f"Configure LLM_API_KEY in .env.")
    
    config = {
        "provider": provider,
        "api_key": key,
        "model": model,
        "base_url": base_url,
        "timeout": 45,
        "max_retries": 0,
        "request_interval": float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "4" if provider == "openrouter" else "0")),
    }
    
    limits = Limits(
        max_model_calls=int(os.getenv("MAX_MODEL_CALLS", "10")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "8")),
    )
    return config, limits


def create_model(config: dict):
    options = dict(config)
    provider = options.pop("provider", "openrouter")
    interval = options.pop("request_interval", 4)
    
    if interval:
        options["rate_limiter"] = InMemoryRateLimiter(requests_per_second=1 / interval, check_every_n_seconds=0.1, max_bucket_size=1)
  
    if provider == "openrouter":
        options.update(use_responses_api=False, stream_usage=False, temperature=0)
        options["extra_body"] = {"provider": {"require_parameters": True}}
    
    schemas = []
    for name, schema in ACTION_SCHEMAS.items():
        tool = convert_to_openai_tool(schema)
        tool["function"]["name"] = name
        schemas.append(tool)
        
    binding = {"parallel_tool_calls": False} 
    return ChatOpenAI(**options).bind_tools(schemas, tool_choice="required", **binding)
