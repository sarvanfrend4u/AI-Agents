"""
Context loader for Atlas Realty agent system.
Reads shared state files from disk and provides helpers for building
the context block that every agent prepends to its user message.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
REPO_ROOT = BASE_DIR.parent           # atlas-realty/
STATE_DIR = BASE_DIR / os.getenv("STATE_DIR", "state")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "output")
TEMPLATES_DIR = BASE_DIR / os.getenv("TEMPLATES_DIR", "templates")
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"   # canonical prompt location


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_platform_state() -> str:
    """Read agent-system/state/platform-state.md."""
    path = STATE_DIR / "platform-state.md"
    return path.read_text() if path.exists() else "(platform-state.md not found)"


def read_event_log() -> str:
    """Read agent-system/state/event-log.md."""
    path = STATE_DIR / "event-log.md"
    return path.read_text() if path.exists() else "(event-log.md not found)"


def read_active_feature() -> dict:
    """Read agent-system/state/active_feature.json as a dict."""
    path = STATE_DIR / "active_feature.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_features_json() -> str:
    """Read agent-system/state/features.json as raw text."""
    path = STATE_DIR / "features.json"
    return path.read_text() if path.exists() else "{}"


def get_feature_list() -> list:
    """Return the list of feature dicts from features.json."""
    raw = read_features_json()
    try:
        data = json.loads(raw)
    except Exception:
        return []
    # Support both {"features": [...]} and [...]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "features" in data:
        return data["features"]
    return []


def read_template(name: str) -> str:
    """Read a template file by name (e.g. 'feature-spec')."""
    path = TEMPLATES_DIR / f"{name}.md"
    return path.read_text() if path.exists() else f"(template {name}.md not found)"


# Mapping from internal agent key → .claude/agents/ filename
_AGENT_FILE_MAP = {
    "pm":           "pm-agent.md",
    "arch":         "arch-agent.md",
    "design":       "design-agent.md",
    "security":     "security-agent.md",
    "dev_backend":  "dev-backend-agent.md",
    "dev_frontend": "dev-frontend-agent.md",
    "code_review":  "code-review-agent.md",
    "test":         "test-agent.md",
    "performance":  "performance-agent.md",
    "docs":         "docs-agent.md",
}


def read_prompt(agent_name: str) -> str:
    """
    Read the system prompt from .claude/agents/<agent>.md.
    Strips the YAML frontmatter block (--- ... ---) and returns only the prompt body.
    """
    filename = _AGENT_FILE_MAP.get(agent_name, f"{agent_name}.md")
    path = AGENTS_DIR / filename
    if not path.exists():
        return f"(agent prompt not found: {path})"

    content = path.read_text()

    # Strip YAML frontmatter: everything between first --- and second ---
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].lstrip("\n")

    return content


def read_output(feature_id: str, doc_name: str) -> str:
    """Read a previously saved output document for a feature."""
    # Try exact folder match first, then glob for feature-id prefix
    exact = OUTPUT_DIR / feature_id / f"{doc_name}.md"
    if exact.exists():
        return exact.read_text()
    # search for folder starting with feature_id
    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(feature_id):
            candidate = folder / f"{doc_name}.md"
            if candidate.exists():
                return candidate.read_text()
    return f"({doc_name}.md not found for feature {feature_id})"


def save_output(feature_id: str, feature_name: str, doc_name: str, content: str) -> Path:
    """
    Save an output document to agent-system/output/[FEATURE-ID]-[name]/.
    Returns the path written.
    """
    safe_name = feature_name.lower().replace(" ", "-").replace("/", "-")
    folder = OUTPUT_DIR / f"{feature_id}-{safe_name}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc_name}.md"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def build_context_block(state: dict, include_docs: Optional[List[str]] = None) -> str:
    """
    Build a context block to prepend to any agent's user message.
    Contains: current date, feature info, platform state, event log,
    and any requested output documents.
    """
    feature_id = state.get("feature_id", "UNKNOWN")
    feature_name = state.get("feature_name", "Unknown Feature")
    current_stage = state.get("current_stage", "UNKNOWN")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Agent Context Block",
        f"**Date:** {now}",
        f"**Feature:** {feature_id} — {feature_name}",
        f"**Current Stage:** {current_stage}",
        "",
        "---",
        "",
        "## Platform State",
        state.get("platform_state") or read_platform_state(),
        "",
        "---",
        "",
        "## Event Log (full history)",
        read_event_log(),
        "",
    ]

    # Attach any requested output docs from state
    doc_map = {
        "feature-spec":      "feature_spec",
        "arch-constraints":  "arch_constraints",
        "design-spec":       "design_spec",
        "security-checklist":"security_checklist",
        "arch-spec":         "arch_spec",
        "code-review":       "code_review",
        "test-report":       "test_report",
        "performance-report":"performance_report",
    }

    for doc_name in (include_docs or []):
        state_key = doc_map.get(doc_name)
        content = state.get(state_key) if state_key else None
        if not content:
            content = read_output(feature_id, doc_name)
        lines += [f"---", f"", f"## {doc_name}", content, ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event log helpers
# ---------------------------------------------------------------------------

def make_event(who: str, event_type: str, details: list[str]) -> dict:
    """Create a structured event dict for appending to state.event_log."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "who": who,
        "event_type": event_type,
        "details": details,
    }


def append_event_to_file(event: dict) -> None:
    """Append an event to the on-disk event-log.md."""
    path = STATE_DIR / "event-log.md"
    ts = event["timestamp"][:16].replace("T", " ")
    lines = [f"\n### {ts} {event['who']} — {event['event_type']}"]
    for detail in event.get("details", []):
        lines.append(f"- {detail}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
