"""
PM Agent node — Stage 1 & 2.
Runs RICE analysis, proposes feature, discusses with human (Gate 1),
then writes the feature spec (Gate 2 approves before proceeding).
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file, read_features_json, get_feature_list,
)
from llm.client import call_agent
from graph.state import PipelineState


def pm_node(state: PipelineState) -> dict:
    """
    Stage 1: PM Agent.
    - Reads all features, runs RICE analysis.
    - Proposes feature to human (Gate 1).
    - After approval, writes feature-spec.md (Gate 2 follows).
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("pm")
    context = build_context_block(state)

    user_message = f"""{context}

---

## Your Task

You are now running as the PM Agent for feature: {feature_id} — {feature_name}

The human has already selected this feature for the current pipeline run.
Your job now is:

1. Review the feature list and platform state to understand what's already built.
2. Perform a RICE analysis for {feature_id} — {feature_name}.
3. Write a complete Feature Spec using the template format.
4. The spec will be reviewed by the human at Gate 2.

Feature list (first 30 features from features.json):
{str(get_feature_list()[:30])}

Write the feature spec now. Use the template structure:
- Feature ID, Name, Status
- Problem Statement
- Proposed Solution
- Acceptance Criteria (numbered list — be specific and testable)
- Out of Scope
- RICE Score
- Dependencies
- Risks

Save the output as the feature spec for this pipeline run.
"""

    output = call_agent("pm", system_prompt, user_message, max_tokens=4096)

    # Save to disk
    save_output(feature_id, feature_name, "feature-spec", output)

    # Log event
    event = make_event(
        "PM_AGENT",
        "STAGE_1_COMPLETE — Feature Spec Written",
        [
            f"Feature: {feature_id} — {feature_name}",
            "feature-spec.md saved",
            "Awaiting Gate 2 (human approval of spec)",
        ],
    )
    append_event_to_file(event)

    return {
        "feature_spec": output,
        "current_stage": "GATE_2_AWAITING",
        "event_log": [event],
    }
