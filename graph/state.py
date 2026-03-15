"""
Pipeline state definition for Atlas Realty agent system.
Every node reads from and writes to this shared state.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator


def _last_wins(a, b):
    """Reducer: last write wins (for scalar fields)."""
    return b


def _append(a, b):
    """Reducer: append new items to a list."""
    if a is None:
        a = []
    if b is None:
        return a
    if isinstance(b, list):
        return a + b
    return a + [b]


class PipelineState(TypedDict):
    # ── Feature identity ──────────────────────────────────────────────────
    feature_id: Annotated[str, _last_wins]
    feature_name: Annotated[str, _last_wins]

    # ── Shared context (read-only for most nodes) ─────────────────────────
    platform_state: Annotated[str, _last_wins]   # contents of platform-state.md
    features_json: Annotated[str, _last_wins]    # contents of features.json

    # ── Event log (append-only) ───────────────────────────────────────────
    event_log: Annotated[list[dict], _append]

    # ── Stage outputs (set once by the responsible node) ─────────────────
    feature_spec: Annotated[Optional[str], _last_wins]
    arch_constraints: Annotated[Optional[str], _last_wins]
    design_spec: Annotated[Optional[str], _last_wins]
    security_checklist: Annotated[Optional[str], _last_wins]
    arch_spec: Annotated[Optional[str], _last_wins]
    dev_backend_output: Annotated[Optional[str], _last_wins]
    dev_frontend_output: Annotated[Optional[str], _last_wins]
    code_review: Annotated[Optional[str], _last_wins]
    test_report: Annotated[Optional[str], _last_wins]
    performance_report: Annotated[Optional[str], _last_wins]
    docs_output: Annotated[Optional[str], _last_wins]
    deploy_report: Annotated[Optional[str], _last_wins]

    # ── Files actually written to the repo ───────────────────────────────
    files_written_backend: Annotated[list[str], _append]
    files_written_frontend: Annotated[list[str], _append]

    # ── Human gate I/O ────────────────────────────────────────────────────
    human_feedback: Annotated[Optional[str], _last_wins]
    gate_decision: Annotated[Optional[str], _last_wins]   # "APPROVED" | "REJECTED" | "CHANGES"

    # ── Pipeline control ──────────────────────────────────────────────────
    current_stage: Annotated[str, _last_wins]
    retry_code_review: Annotated[int, _last_wins]
    retry_testing: Annotated[int, _last_wins]
    escalated: Annotated[bool, _last_wins]
    pipeline_complete: Annotated[bool, _last_wins]
