"""
Configuration module for LLM models and Azure OpenAI integration.
Supports the pattern from your example with get_model_args().
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())


class GPTModel(Enum):
    """Enumeration of supported GPT models."""
    GPT4 = "gpt-4o-mini"
    GPT35TURBO = "gpt-35-turbo"
    GPT4OMINI = "gpt-4o-mini"


@dataclass
class ModelArgs:
    """Model configuration arguments for Azure OpenAI."""
    deployment_id: str
    azure_openai_version: str
    azure_openai_endpoint: str
    azure_openai_key: str
    model_name: str


def get_model_args(model: GPTModel) -> ModelArgs:
    """
    Get model arguments for the specified GPT model.
    
    Args:
        model: The GPT model to get arguments for
        
    Returns:
        ModelArgs: Configuration for the specified model
        
    Raises:
        ValueError: If required environment variables are missing
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    version = os.getenv("AZURE_OPENAI_VERSION", "2024-02-15-preview")
    
    if not endpoint or not key:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set in environment variables"
        )
    
    # Map models to their deployment names
    deployment_mapping = {
        GPTModel.GPT4: os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4", "gpt-4o-mini"),
        GPTModel.GPT35TURBO: os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT35", "gpt-35-turbo"),
        GPTModel.GPT4OMINI: os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4OMINI", "gpt-4o-mini"),
    }
    
    deployment_id = deployment_mapping.get(model)
    if not deployment_id:
        raise ValueError(f"No deployment configured for model {model}")
    
    return ModelArgs(
        deployment_id=deployment_id,
        azure_openai_version=version,
        azure_openai_endpoint=endpoint,
        azure_openai_key=key,
        model_name=model.value
    )


def get_use_local_db() -> bool:
    """Check if local database should be used instead of Cosmos DB."""
    return os.getenv("USE_LOCAL_DB", "false").lower() in ("true", "1", "yes")


def get_environment() -> str:
    """Get the current environment (development, production, etc.)."""
    return os.getenv("ENVIRONMENT", "development")


def get_log_level() -> str:
    """Get the logging level."""
    return os.getenv("LOG_LEVEL", "INFO")


# Legacy compatibility for existing code
def get_legacy_model_config() -> dict:
    """Get legacy model configuration for backward compatibility."""
    return {
        "url": os.getenv("GPT4_URL", ""),
        "key": os.getenv("GPT4_KEY", ""),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    }
