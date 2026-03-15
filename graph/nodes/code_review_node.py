"""
Code Review Agent node — Stage 6 (after both Dev Agents complete).
Reviews the ACTUAL files written to the repo, not just LLM output text.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from context.file_tools import read_repo_files, extract_file_paths_from_arch_spec
from llm.client import call_agent
from graph.state import PipelineState

MAX_REVIEW_RETRIES = 3


def code_review_node(state: PipelineState) -> dict:
    """
    Stage 6 (after dev): Code Review Agent.
    Reads the actual files written to the repo for a ground-truth review.
    Returns verdict: APPROVED, CHANGES_REQUIRED, or ESCALATE.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]
    retry = state.get("retry_code_review", 0)
    arch_spec = state.get("arch_spec", "")

    # ── Read actual files from disk ───────────────────────────────────────
    written_backend = state.get("files_written_backend") or []
    written_frontend = state.get("files_written_frontend") or []
    all_written = list(dict.fromkeys(written_backend + written_frontend))

    # Also check arch-spec for any files that should have been written
    spec_paths = extract_file_paths_from_arch_spec(arch_spec)
    all_paths = list(dict.fromkeys(all_written + spec_paths))

    actual_files = read_repo_files(all_paths) if all_paths else {}

    system_prompt = read_prompt("code_review")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-spec", "design-spec", "security-checklist"],
    )

    # Build file listing section
    if actual_files:
        file_parts = [f"\n---\n\n## Actual Files Written to Repo (Cycle {retry + 1})\n"]
        for path, content in actual_files.items():
            lang = "python" if path.endswith(".py") else (
                "typescript" if path.endswith((".ts", ".tsx")) else (
                    "sql" if path.endswith(".sql") else ""
                )
            )
            # Truncate very large files but show enough to review
            display = content if len(content) < 6000 else content[:6000] + "\n... (truncated)"
            file_parts.append(f"### `{path}`\n```{lang}\n{display}\n```\n")
        files_section = "\n".join(file_parts)
    else:
        files_section = "\n**WARNING: No files were written to the repo. This is a blocker.**\n"

    missing = [p for p in spec_paths if p not in actual_files]
    missing_section = ""
    if missing:
        missing_section = f"\n**Files in arch-spec but NOT written:** {', '.join(missing)}\n"

    user_message = f"""{context}
{files_section}
{missing_section}
---

## Your Task — Code Review (Cycle {retry + 1})

You are reviewing the ACTUAL files written to the repo above (not generated text).

Check against:
1. arch-spec.md compliance — every file specified was written, content matches spec
2. design-spec.md compliance — layout, states, interactions match
3. security-checklist.md compliance — every mitigation is implemented
4. Code quality — no `any` types, docstrings, type hints, async def, no hardcoded values
5. CSS rules — no `position:relative` on `.price-pin`, no undocumented CSS vars
6. Nothing out-of-scope was written

Format:

# Code Review — {feature_name} ({feature_id})
**Cycle:** {retry + 1}
**Verdict:** APPROVED / CHANGES_REQUIRED / ESCALATE

## Blockers (must fix before proceeding)
- [file:line] — [issue] — [exact fix required]

## Suggestions (non-blocking)
- [observation]

## Approved Items
- [list]

If retry is already at {retry} and equals {MAX_REVIEW_RETRIES - 1}: set ESCALATE.
"""

    output = call_agent("code_review", system_prompt, user_message, max_tokens=4096)

    save_output(feature_id, feature_name, "code-review", output)

    verdict = "CHANGES_REQUIRED"
    output_upper = output.upper()
    if "ESCALATE" in output_upper or retry >= MAX_REVIEW_RETRIES - 1:
        verdict = "ESCALATE"
    elif "**VERDICT:** APPROVED" in output_upper or (
        "APPROVED" in output_upper and "CHANGES_REQUIRED" not in output_upper
    ):
        verdict = "APPROVED"

    event = make_event(
        "CODE_REVIEW_AGENT",
        f"STAGE_6_CODE_REVIEW — Cycle {retry + 1} — {verdict}",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Reviewed {len(actual_files)} actual repo files",
            f"Verdict: {verdict}",
        ],
    )
    append_event_to_file(event)

    return {
        "code_review": output,
        "gate_decision": verdict,
        "retry_code_review": retry if verdict == "APPROVED" else retry + 1,
        "escalated": verdict == "ESCALATE",
        "event_log": [event],
    }
