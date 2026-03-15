"""
Frontend Dev Agent node — Stage 6 (parallel with Backend).
Implements all frontend changes: React components, Zustand state, TypeScript types.
Writes code directly to the repo via file_tools.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from context.file_tools import (
    extract_code_blocks, write_repo_files,
    read_repo_file, extract_file_paths_from_arch_spec,
)
from llm.client import call_agent
from graph.state import PipelineState


def dev_frontend_node(state: PipelineState) -> dict:
    """
    Stage 6 (parallel): Frontend Dev Agent.
    1. Reads existing frontend files for context.
    2. Calls Claude to generate TypeScript/React/Tailwind code per arch-spec + design-spec.
    3. Parses code blocks from output.
    4. Writes files directly to the repo.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]
    retry = state.get("retry_code_review", 0)
    arch_spec = state.get("arch_spec", "")
    code_review_feedback = state.get("code_review", "")

    # ── Read existing repo files for context ─────────────────────────────
    existing_files = _read_existing_frontend_files(arch_spec)

    system_prompt = read_prompt("dev_frontend")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-spec", "design-spec", "security-checklist"],
    )

    retry_section = ""
    if retry > 0 and code_review_feedback:
        retry_section = f"""
---

## Code Review Feedback (Retry {retry})

Fix EVERY blocker listed. Do not change approved items.

{code_review_feedback}
"""

    existing_section = ""
    if existing_files:
        parts = ["\n---\n\n## Existing Repo Files (read before writing)\n"]
        for path, content in existing_files.items():
            parts.append(f"### `{path}`\n```typescript\n{content[:5000]}\n```\n")
        existing_section = "\n".join(parts)

    user_message = f"""{context}
{existing_section}
{retry_section}
---

## Your Task — Frontend Implementation

Implement ALL frontend changes specified in arch-spec.md and design-spec.md
for {feature_id} — {feature_name}.

Rules:
- TypeScript strict mode — NO `any` types
- Tailwind CSS only — no inline styles
- MapLibre GL JS — never Mapbox
- Zustand for state
- Every component and hook must have a JSDoc comment
- Component files: PascalCase. Hook files: camelCase with `use` prefix.
- NEVER set `position: relative` on `.price-pin`
- Do NOT touch any backend or migration files
- Output COMPLETE files — no truncation, no "... rest unchanged" shortcuts. The file parser requires the full content.

You MUST output ALL of the following files (no skipping):
1. `frontend/types/listing.ts` — full file with vastuTier additions only (no Phase 3 fields)
2. `frontend/store/mapStore.ts` — full file with vastuCompliant only (remove vastuFilter if present)
3. `frontend/lib/api.ts` — full file; use params.append() for facing, correct param names (min_price/max_price)
4. `frontend/components/Vastu/VastuBadge.tsx` — NEW component per design-spec
5. `frontend/components/Filters/FilterBar.tsx` — add Vastu Friendly toggle chip
6. `frontend/components/Listing/ListingCard.tsx` — add VastuBadge
7. `frontend/components/Listing/ListingSheet.tsx` — add Vastu detail section
8. `frontend/components/Map/MapPopupCard.tsx` — add VastuBadge

For EACH file output EXACTLY this format (required for file parser):

### `frontend/path/to/Component.tsx`
```typescript
[complete file content — full file, not a diff]
```

Output all 8 files in order, then confirm which acceptance criteria are covered.
"""

    output = call_agent("dev_frontend", system_prompt, user_message, max_tokens=16000)

    # ── Parse and write files to repo ────────────────────────────────────
    blocks = extract_code_blocks(output)
    # Only write frontend files
    frontend_blocks = [
        b for b in blocks
        if b["path"].startswith("frontend/")
        and not b["path"].startswith("frontend/node_modules/")
    ]
    written = write_repo_files(frontend_blocks)

    save_output(feature_id, feature_name, "dev-frontend-output", output)

    event = make_event(
        "DEV_FRONTEND_AGENT",
        f"STAGE_6_FRONTEND_COMPLETE — Retry {retry}",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Files written to repo: {len(written)}",
        ] + [f"  {p}" for p in written],
    )
    append_event_to_file(event)

    return {
        "dev_frontend_output": output,
        "files_written_frontend": written,
        "event_log": [event],
    }


def _read_existing_frontend_files(arch_spec: str) -> dict[str, str]:
    """Read existing frontend files mentioned in arch-spec + key shared files."""
    paths = extract_file_paths_from_arch_spec(arch_spec)
    frontend_paths = [p for p in paths if p.startswith("frontend/")]

    # Always give Claude these key files for pattern consistency
    always_include = [
        "frontend/types/listing.ts",
        "frontend/store/mapStore.ts",
        "frontend/lib/api.ts",
        "frontend/app/globals.css",
    ]
    all_paths = list(dict.fromkeys(always_include + frontend_paths))

    result = {}
    for p in all_paths:
        content = read_repo_file(p)
        if not content.startswith("(file not found"):
            result[p] = content
    return result
