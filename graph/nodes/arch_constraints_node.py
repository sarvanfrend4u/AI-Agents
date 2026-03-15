"""
Architecture Agent — Pass 1 (Stage 3).
Reads the approved feature spec and produces arch-constraints.md.
This is a lightweight pass: what CAN'T be done, what must be respected.
Full arch-spec comes after Design + Security at Stage 5.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from llm.client import call_agent
from graph.state import PipelineState


def arch_constraints_node(state: PipelineState) -> dict:
    """
    Stage 3: Architecture Agent Pass 1.
    Produces arch-constraints.md — boundaries for Design and Security agents.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("arch")
    context = build_context_block(state, include_docs=["feature-spec"])

    user_message = f"""{context}

---

## Your Task — Pass 1: Architecture Constraints

The feature spec has been approved. Now produce **arch-constraints.md**.

This is NOT the full architecture spec. It is the set of constraints and
non-negotiables that the Design and Security agents must work within.

Cover:
1. **What cannot be changed** — existing patterns, DB schema constraints, z-index order, layout zones
2. **What must be used** — specific libraries, existing components, API patterns
3. **What is forbidden** — paid APIs, new CSS modules, position:relative on .price-pin, etc.
4. **DB impact** — does this feature need a new column/table? Flag schema changes here.
5. **API impact** — new endpoints needed? Must follow existing FastAPI pattern.
6. **State impact** — new Zustand fields needed?
7. **Security constraints** — auth requirements, input validation needed?

Be specific and reference platform-state.md section by section.
Write arch-constraints.md now.
"""

    output = call_agent("arch", system_prompt, user_message, max_tokens=3000)

    save_output(feature_id, feature_name, "arch-constraints", output)

    event = make_event(
        "ARCH_AGENT",
        "STAGE_3_COMPLETE — Arch Constraints Written",
        [
            f"Feature: {feature_id} — {feature_name}",
            "arch-constraints.md saved",
            "Design + Security agents can now run in parallel",
        ],
    )
    append_event_to_file(event)

    return {
        "arch_constraints": output,
        "current_stage": "STAGE_4_DESIGN_SECURITY",
        "event_log": [event],
    }
