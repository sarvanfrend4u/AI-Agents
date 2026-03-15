"""
Backend Dev Agent node — Stage 6 (parallel with Frontend).
Implements all backend changes: FastAPI endpoints, DB migrations, Python logic.
Writes code directly to the repo via file_tools.
"""

from pathlib import Path
from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from context.file_tools import (
    extract_code_blocks, write_repo_files,
    read_repo_file, extract_file_paths_from_arch_spec, REPO_ROOT,
)
from llm.client import call_agent
from graph.state import PipelineState


def dev_backend_node(state: PipelineState) -> dict:
    """
    Stage 6 (parallel): Backend Dev Agent.
    1. Reads existing backend files for context.
    2. Calls Claude to generate code per arch-spec.
    3. Parses code blocks from output.
    4. Writes files directly to the repo.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]
    retry = state.get("retry_code_review", 0)
    arch_spec = state.get("arch_spec", "")
    code_review_feedback = state.get("code_review", "")

    # ── Read existing repo files for context ─────────────────────────────
    existing_files = _read_existing_backend_files(arch_spec)

    system_prompt = read_prompt("dev_backend")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-spec", "security-checklist"],
    )

    retry_section = ""
    if retry > 0 and code_review_feedback:
        retry_section = f"""
---

## Code Review Feedback (Retry {retry})

The previous implementation was reviewed and has blockers. Fix EVERY blocker:

{code_review_feedback}

Fix all blockers. Do not change anything already approved.
"""

    existing_section = ""
    if existing_files:
        parts = ["\n---\n\n## Existing Repo Files (read before writing)\n"]
        for path, content in existing_files.items():
            parts.append(f"### `{path}`\n```\n{content[:5000]}\n```\n")
        existing_section = "\n".join(parts)

    user_message = f"""{context}
{existing_section}
{retry_section}
---

## Your Task — Backend Implementation

Implement ALL backend changes specified in arch-spec.md for {feature_id} — {feature_name}.

Rules:
- FastAPI + psycopg2 only (no SQLAlchemy ORM)
- Route handlers that use psycopg2 MUST be `def` (sync) — psycopg2 is a blocking library; async def with blocking IO will freeze the event loop. FastAPI runs sync routes in a thread pool automatically.
- All public functions must have docstrings
- Type hints everywhere
- Parameterised queries ONLY — NO string formatting in SQL
- Do NOT touch any frontend files
- Do NOT add deployment config
- Output the COMPLETE file — no truncation, no "... (rest of file unchanged)" shortcuts

For EACH file to create or modify, output EXACTLY this format:

### `path/to/file.py`
```python
[complete file content — not a diff, the full file]
```

Then for SQL migrations:

### `data/migrations/NNN_description.sql`
```sql
[complete SQL]
```

Output all files then confirm which acceptance criteria are covered.
"""

    output = call_agent("dev_backend", system_prompt, user_message, max_tokens=16000)

    # ── Parse and write files to repo ────────────────────────────────────
    blocks = extract_code_blocks(output)
    # Only write backend files (exclude frontend paths)
    backend_blocks = [
        b for b in blocks
        if not b["path"].startswith("frontend/")
        and not b["path"].startswith("node_modules/")
    ]
    written = write_repo_files(backend_blocks)

    # Save full output as reference doc
    save_output(feature_id, feature_name, "dev-backend-output", output)

    files_summary = "\n".join(f"  - {p}" for p in written) if written else "  (no files parsed from output)"

    event = make_event(
        "DEV_BACKEND_AGENT",
        f"STAGE_6_BACKEND_COMPLETE — Retry {retry}",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Files written to repo: {len(written)}",
        ] + [f"  {p}" for p in written],
    )
    append_event_to_file(event)

    return {
        "dev_backend_output": output,
        "files_written_backend": written,
        "event_log": [event],
    }


def _read_existing_backend_files(arch_spec: str) -> dict[str, str]:
    """Read existing backend files mentioned in arch-spec so Claude has current context."""
    paths = extract_file_paths_from_arch_spec(arch_spec)
    # Filter to backend files only
    backend_paths = [
        p for p in paths
        if p.startswith("backend/") or p.startswith("data/")
    ]

    # Always include main.py as context
    always_include = ["backend/main.py"]
    all_paths = list(dict.fromkeys(always_include + backend_paths))  # dedupe, preserve order

    result = {}
    for p in all_paths:
        content = read_repo_file(p)
        if not content.startswith("(file not found"):
            result[p] = content
    return result
