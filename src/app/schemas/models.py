"""
This module defines the data models for the application using Pydantic.

Classes:
    Judge: Represents a Judge with an id, model, url, and metaprompt.
    Assemble: Represents an Assemble with an id, list of judges, and roles.
"""

import json
from typing import List, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Judge(BaseModel):
    """
    Represents a Judge with an id, model, url, and metaprompt.

    Attributes:
        id (str): The unique identifier for the judge.
        model (HttpUrl): The model name of the judge.
        metaprompt (str): The metaprompt for the judge.
    """

    id: str = Field(..., description="Judge ID")
    name: str = Field(..., description="Judge Name", max_length=100)
    model: HttpUrl = Field(..., description="Model URL")
    # Accept either a JSON string or a dict with 'text' and 'json' keys; will store as JSON string.
    metaprompt: str = Field(
        ...,
        description="Judge System Prompt (stored as JSON string with 'text' and 'json' keys)",
        max_length=50000,
    )

    @field_validator("metaprompt", mode="before")
    def coerce_metaprompt(cls, v):  # noqa: D401
        """Allow dict input and serialize; validate structure either way."""
        # If already a dict, validate shape then serialize
        if isinstance(v, dict):
            if not ("text" in v and "json" in v):
                raise ValueError("metaprompt dict must contain 'text' and 'json' keys")
            try:
                return json.dumps(v)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("metaprompt dict is not serializable") from exc
        # If it's not a string, reject
        if not isinstance(v, str):
            raise ValueError("metaprompt must be either a JSON string or a dict with 'text' and 'json' keys")
        return v

    @field_validator("model")
    def model_must_be_azure_url(cls, v):  # noqa: D401
        # Placeholder for stricter validation if needed.
        return v

    @field_validator("metaprompt")
    def metaprompt_must_be_json_serializable(cls, v):  # noqa: D401
        try:
            json_obj = json.loads(v)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("metaprompt must be JSON serializable") from exc
        if not (isinstance(json_obj, dict) and "text" in json_obj and "json" in json_obj):
            raise ValueError("metaprompt must contain a JSON object with 'text' and 'json' keys")
        return v


class Assembly(BaseModel):
    """
    Represents an Assemble with an id, list of judges, and roles.

    Attributes:
        id (int): The unique identifier for the assemble.
        judges (List[Judge]): A list of Judge objects.
        roles (List[str]): A list of roles.
    """

    id: str = Field(..., description="Judge Assembly ID")
    judges: List[str] = Field(..., description="List of Judge IDs in this assembly")
    roles: List[str] = Field(..., description="Judge Roles ID")

    @field_validator("roles")
    def roles_must_not_exceed_length(cls, v):  # noqa: D401
        for role in v:
            if len(role) > 600:
                raise ValueError("each role must have at most 600 characters")
        return v
