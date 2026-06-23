"""Pydantic models for the dashboard control API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    suite_name: str = "general_qa"
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    threshold: float = 0.85
    repeat_factor: int = 2
    delay_between_ms: int = 100
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    batch_id: str = ""
    cache_mode: str = "warm"
    status: str = "complete"


class BatchConfig(BaseModel):
    batch_id: str
    description: str = ""
    base: dict[str, Any] = {}
    matrix: dict[str, list[Any]] = {}
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    skip_existing: bool = True


class ApplyThresholdRequest(BaseModel):
    suite_name: str | None = None
    model: str | None = None
    cache_mode: str = "cold"


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    model: str = "gpt-4o-mini"


class FlagRequest(BaseModel):
    flagged: bool = True
