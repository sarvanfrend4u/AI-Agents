"""
Architecture Agent — Pass 2 (Stage 5).
Has design-spec + security-checklist in hand.
Produces arch-spec.md (full implementation blueprint) + adr.md.
Gate 4 follows — last human gate before development.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from llm.client import call_agent
from graph.state import PipelineState


def arch_spec_node(state: PipelineState) -> dict:
    """
    Stage 5: Architecture Agent Pass 2.
    Produces arch-spec.md and adr.md.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("arch")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-constraints", "design-spec", "security-checklist"],
    )

    user_message = f"""{context}

---

## Your Task — Pass 2: Full Architecture Spec

You now have the complete picture: feature spec, constraints, design spec,
and security checklist. Produce **arch-spec.md** — the implementation blueprint
that Dev Agents will follow exactly.

### arch-spec.md must contain:

1. **Files to create** — exact file paths (relative to repo root)
2. **Files to modify** — exact file paths + which sections change
3. **DB changes** — exact SQL (ALTER TABLE / CREATE TABLE)
4. **New API endpoints** — method, path, request schema, response schema
5. **New TypeScript types** — exact interface definitions
6. **Zustand store changes** — new fields and actions with types
7. **Component specifications** — for each new component:
   - File path
   - Props interface
   - Key logic (pseudocode if complex)
   - Which existing components it imports from
8. **Data flow** — how data moves from DB → API → store → component
9. **Security implementations** — exact implementation for each checklist item

### Also produce adr.md:
An Architecture Decision Record for any non-obvious architectural decision
made in this feature. Format per the ADR template.

Be specific. Dev Agents will implement EXACTLY what you write here — nothing more.
"""

    # Arch spec
    arch_output = call_agent("arch", system_prompt, user_message, max_tokens=6000)
    save_output(feature_id, feature_name, "arch-spec", arch_output)

    # ADR
    adr_user_message = f"""{context}

---

## Your Task — ADR

Write **adr.md** for {feature_id} — {feature_name}.

Document any architectural decision that is non-obvious, involves a trade-off,
or establishes a new pattern for the codebase.

If there are no significant architectural decisions (e.g. feature only adds
a new column and a small UI badge), write a brief ADR stating "No significant
architectural decisions — follows existing patterns" and list the patterns followed.
"""
    adr_output = call_agent("arch", system_prompt, adr_user_message, max_tokens=2000)
    save_output(feature_id, feature_name, "adr", adr_output)

    event = make_event(
        "ARCH_AGENT",
        "STAGE_5_COMPLETE — Arch Spec + ADR Written",
        [
            f"Feature: {feature_id} — {feature_name}",
            "arch-spec.md saved",
            "adr.md saved",
            "Awaiting Gate 4 (last human gate before development)",
        ],
    )
    append_event_to_file(event)

    return {
        "arch_spec": arch_output,
        "current_stage": "GATE_4_AWAITING",
        "event_log": [event],
    }
