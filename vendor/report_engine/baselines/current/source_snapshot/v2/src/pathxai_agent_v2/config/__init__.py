"""Typed, layered configuration for pathxai_agent_v2.

Precedence (lowest -> highest):

    defaults.yaml  <  user YAML (path=...)  <  environment variables  <  kwargs

Nothing in the pipeline should read an environment variable or hardcode a path
directly; everything flows through :class:`Settings`.
"""

from __future__ import annotations

from .settings import (
    ContextBudgetConfig,
    EvalConfig,
    KnowledgeConfig,
    LeakageConfig,
    LLMConfig,
    PathsConfig,
    ReinspectionConfig,
    ReportConfig,
    RoiConfig,
    SegmentationConfig,
    SessionConfig,
    Settings,
    load_settings,
)

__all__ = [
    "Settings",
    "load_settings",
    "PathsConfig",
    "LLMConfig",
    "SegmentationConfig",
    "RoiConfig",
    "ReinspectionConfig",
    "ContextBudgetConfig",
    "KnowledgeConfig",
    "SessionConfig",
    "LeakageConfig",
    "ReportConfig",
    "EvalConfig",
]
