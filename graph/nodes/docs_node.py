"""
Docs Agent node — Stage 8.
Produces all documentation, updates platform-state.md, archives event log.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
    STATE_DIR, read_platform_state, read_event_log,
)
from llm.client import call_agent
from graph.state import PipelineState


def docs_node(state: PipelineState) -> dict:
    """
    Stage 8: Docs Agent.
    Produces api-docs, component-docs, feature-guide, developer-notes.
    Updates platform-state.md. Archives event log.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("docs")
    context = build_context_block(
        state,
        include_docs=[
            "feature-spec", "arch-spec", "design-spec",
            "security-checklist", "code-review", "test-report", "performance-report",
        ],
    )

    backend_code = state.get("dev_backend_output", "")
    frontend_code = state.get("dev_frontend_output", "")

    # ── 1. API Docs ──────────────────────────────────────────────────────
    api_docs_msg = f"""{context}

Backend implementation:
{backend_code}

Write **api-docs.md**: document every new/modified API endpoint.
For each: method, path, auth required, request schema (JSON), response schema (JSON),
example request, example response. Written for a developer who has never seen this codebase.
"""
    api_docs = call_agent("docs", system_prompt, api_docs_msg, max_tokens=3000)
    save_output(feature_id, feature_name, "api-docs", api_docs)

    # ── 2. Component Docs ────────────────────────────────────────────────
    component_docs_msg = f"""{context}

Frontend implementation:
{frontend_code}

Write **component-docs.md**: document every new component (purpose, props, states, usage example)
and every modified component (what changed and why).
"""
    component_docs = call_agent("docs", system_prompt, component_docs_msg, max_tokens=3000)
    save_output(feature_id, feature_name, "component-docs", component_docs)

    # ── 3. Feature Guide ─────────────────────────────────────────────────
    feature_guide_msg = f"""{context}

Write **feature-guide.md**: plain-English guide to what this feature does,
how a user interacts with it, and any edge cases or limitations.
No technical jargon — written for a product manager or end user.
"""
    feature_guide = call_agent("docs", system_prompt, feature_guide_msg, max_tokens=2000)
    save_output(feature_id, feature_name, "feature-guide", feature_guide)

    # ── 4. Developer Notes ───────────────────────────────────────────────
    dev_notes_msg = f"""{context}

Write **developer-notes.md**: how would a developer extend or modify this feature in future?
Key files, patterns established, gotchas, and extension points.
"""
    dev_notes = call_agent("docs", system_prompt, dev_notes_msg, max_tokens=2000)
    save_output(feature_id, feature_name, "developer-notes", dev_notes)

    # ── 5. Update platform-state.md ──────────────────────────────────────
    platform_state_msg = f"""{context}

The current platform-state.md is above. You must UPDATE it by appending this feature's
completed status. Add to:
1. Features Completed section — summary of what was built
2. DB Schema section — if DB changed
3. Components Built section — new components
4. Existing API Endpoints — new endpoints
5. Active ADRs — new ADRs
6. Known Issues / Tech Debt — anything flagged in reviews

Return the COMPLETE UPDATED platform-state.md content (not a diff — the full file).
"""
    updated_platform_state = call_agent("docs", system_prompt, platform_state_msg, max_tokens=6000)

    # Write to disk
    platform_state_path = STATE_DIR / "platform-state.md"
    platform_state_path.write_text(updated_platform_state)

    # ── 6. Archive event log ─────────────────────────────────────────────
    current_event_log = read_event_log()
    safe_name = feature_name.lower().replace(" ", "-")
    save_output(feature_id, feature_name, "event-log-archive", current_event_log)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reset_log = f"""# Atlas Realty — Agent Pipeline Event Log

## Current Feature
- Feature ID: (next feature — to be set by PM Agent)
- Pipeline Started: {now}
- Current Stage: STAGE_1 — Awaiting PM Agent

## Event Log
### {now} SYSTEM — Pipeline Reset After {feature_id} Completion
- {feature_name} completed successfully
- platform-state.md updated
- Event log archived to output/{feature_id}-{safe_name}/event-log-archive.md
- Pipeline ready for next feature
"""
    (STATE_DIR / "event-log.md").write_text(reset_log)

    # ── 7. Update features.json ──────────────────────────────────────────
    features_path = STATE_DIR / "features.json"
    if features_path.exists():
        data = json.loads(features_path.read_text())
        # Support both {"features": [...]} and flat [...]
        feature_list = data["features"] if isinstance(data, dict) and "features" in data else data
        for f in feature_list:
            if f.get("id") == feature_id:
                f["status"] = "COMPLETE"
                f["completedAt"] = now
        features_path.write_text(json.dumps(data, indent=2))

    event = make_event(
        "DOCS_AGENT",
        "STAGE_8_COMPLETE — All Docs Written + Platform State Updated",
        [
            f"Feature: {feature_id} — {feature_name}",
            "api-docs.md saved",
            "component-docs.md saved",
            "feature-guide.md saved",
            "developer-notes.md saved",
            "platform-state.md updated",
            "event-log archived",
            "features.json updated — status: COMPLETE",
            "Pipeline ready for Gate 5 (human final review)",
        ],
    )
    append_event_to_file(event)

    docs_summary = (
        f"DOCS COMPLETE — All documentation saved. "
        f"platform-state.md updated. Event log archived. "
        f"Pipeline ready for Gate 5 (human final review)."
    )

    return {
        "docs_output": docs_summary,
        "platform_state": updated_platform_state,
        "current_stage": "GATE_5_FINAL_REVIEW",
        "event_log": [event],
    }
