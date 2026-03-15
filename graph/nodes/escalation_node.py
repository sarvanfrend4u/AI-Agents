"""
Escalation node — triggered when a retry limit is hit.
Pauses the pipeline via interrupt() for human decision.
"""

from langgraph.types import interrupt
from context.loader import make_event, append_event_to_file
from graph.state import PipelineState


def escalation_node(state: PipelineState) -> dict:
    """
    Escalation: called when Code Review or Test Agent exceeds retry limit.
    Human must decide: approve as-is, request rewrite, or abort.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]
    stage = state.get("current_stage", "UNKNOWN")
    code_review = state.get("code_review", "")
    test_report = state.get("test_report", "")

    payload = {
        "gate": "ESCALATION",
        "title": f"ESCALATION — {stage} — Human intervention required",
        "description": (
            f"Feature {feature_id} — {feature_name} has hit the retry limit.\n"
            "Automated agents could not resolve the issues. Human decision required."
        ),
        "code_review_summary": code_review[-1500:] if code_review else "(none)",
        "test_report_summary": test_report[-1500:] if test_report else "(none)",
        "options": [
            "APPROVED — accept current state and proceed to testing",
            "REWRITE — send back to dev agents with specific instructions",
            "ABORT — stop this pipeline run",
        ],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "ESCALATION_RESOLVED",
        [
            f"Decision: {human_input}",
            f"Feature: {feature_id} — {feature_name}",
        ],
    )
    append_event_to_file(event)

    decision = "APPROVED"
    if "REWRITE" in human_input.upper():
        decision = "REWRITE"
    elif "ABORT" in human_input.upper():
        decision = "ABORT"

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "escalated": False,         # reset after human resolves
        "retry_code_review": 0,
        "retry_testing": 0,
        "current_stage": f"ESCALATION_RESOLVED_{decision}",
        "event_log": [event],
    }
