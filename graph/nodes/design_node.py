"""
Design Agent node — Stage 4 (parallel with Security).
Produces design-spec.md: layout, component states, interactions, ARIA labels.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from llm.client import call_agent
from graph.state import PipelineState


def design_node(state: PipelineState) -> dict:
    """
    Stage 4 (parallel): Design Agent.
    Produces design-spec.md.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("design")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-constraints"],
    )

    user_message = f"""{context}

---

## Your Task — Design Spec

Produce **design-spec.md** for {feature_id} — {feature_name}.

Cover every new or modified UI surface:

### For each new component:
1. **Component name and file path** (follow PascalCase convention)
2. **Layout** — exact positioning within the layout zones from platform-state.md
3. **4 UI states** — empty, loading, error, active (with exact content for each)
4. **Interactions** — what happens on click, hover, keyboard
5. **ARIA labels** — required for accessibility
6. **Responsive behaviour** — mobile vs desktop differences
7. **Animation/transition** — if any

### For each modified component:
- What specifically changes and why
- Which states are affected

### Visual design rules:
- Use only existing CSS variables from platform-state.md
- Do not introduce new CSS variables unless absolutely necessary (justify if you do)
- Follow existing z-index order: nav=30, sheets=30, comparebar=25, controls=20
- Do not move any layout zone (map, panel, nav)

Write design-spec.md now.
"""

    output = call_agent("design", system_prompt, user_message, max_tokens=4096)

    save_output(feature_id, feature_name, "design-spec", output)

    event = make_event(
        "DESIGN_AGENT",
        "STAGE_4_COMPLETE — Design Spec Written",
        [
            f"Feature: {feature_id} — {feature_name}",
            "design-spec.md saved",
            "Waiting for Security Agent to complete",
        ],
    )
    append_event_to_file(event)

    return {
        "design_spec": output,
        "event_log": [event],
    }
