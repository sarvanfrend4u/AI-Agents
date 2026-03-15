"""
Human gate nodes for Atlas Realty agent pipeline.
Each gate uses LangGraph's interrupt() to pause execution and wait for human input.
run.py handles the CLI prompt and resumes the graph with Command(resume=...).
"""

from langgraph.types import interrupt
from context.loader import make_event, append_event_to_file
from graph.state import PipelineState


# ---------------------------------------------------------------------------
# Gate 1 — Feature Selection Approval
# ---------------------------------------------------------------------------

def gate_1_node(state: PipelineState) -> dict:
    """
    Gate 1: Human approves or rejects the feature proposed by PM Agent.
    Interrupt payload includes the RICE analysis and proposed feature.
    """
    payload = {
        "gate": 1,
        "title": "Gate 1 — Feature Selection Approval",
        "description": (
            "The PM Agent has selected a feature for this pipeline run.\n"
            "Review the RICE analysis and approve, reject, or request changes."
        ),
        "feature_id": state["feature_id"],
        "feature_name": state["feature_name"],
        "feature_spec_preview": (state.get("feature_spec") or "")[:2000],
        "options": ["APPROVED", "REJECTED", "CHANGES — <your feedback>"],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "GATE_1_DECISION",
        [f"Decision: {human_input}", f"Feature: {state['feature_id']} — {state['feature_name']}"],
    )
    append_event_to_file(event)

    decision = "APPROVED" if "APPROVED" in human_input.upper() else (
        "REJECTED" if "REJECTED" in human_input.upper() else "CHANGES"
    )

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "current_stage": "GATE_1_COMPLETE",
        "event_log": [event],
    }


# ---------------------------------------------------------------------------
# Gate 2 — Feature Spec Approval
# ---------------------------------------------------------------------------

def gate_2_node(state: PipelineState) -> dict:
    """
    Gate 2: Human approves the feature spec written by PM Agent.
    """
    payload = {
        "gate": 2,
        "title": "Gate 2 — Feature Spec Approval",
        "description": (
            "The PM Agent has written the feature spec.\n"
            "Review acceptance criteria, scope, and RICE score.\n"
            "Approve to proceed to Architecture, or request changes."
        ),
        "feature_spec": state.get("feature_spec") or "(no spec yet)",
        "options": ["APPROVED", "CHANGES — <your feedback>"],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "GATE_2_DECISION",
        [f"Decision: {human_input}"],
    )
    append_event_to_file(event)

    decision = "APPROVED" if "APPROVED" in human_input.upper() else "CHANGES"

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "current_stage": "GATE_2_COMPLETE",
        "event_log": [event],
    }


# ---------------------------------------------------------------------------
# Gate 3 — Design + Security Review (optional lightweight gate)
# ---------------------------------------------------------------------------

def gate_3_node(state: PipelineState) -> dict:
    """
    Gate 3: Human reviews design spec and security checklist before arch spec.
    """
    payload = {
        "gate": 3,
        "title": "Gate 3 — Design & Security Review",
        "description": (
            "The Design Agent and Security Agent have completed their work.\n"
            "Review the design spec and security checklist before full architecture."
        ),
        "design_spec_preview": (state.get("design_spec") or "")[:2000],
        "security_checklist_preview": (state.get("security_checklist") or "")[:1500],
        "options": ["APPROVED", "CHANGES — <your feedback>"],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "GATE_3_DECISION",
        [f"Decision: {human_input}"],
    )
    append_event_to_file(event)

    decision = "APPROVED" if "APPROVED" in human_input.upper() else "CHANGES"

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "current_stage": "GATE_3_COMPLETE",
        "event_log": [event],
    }


# ---------------------------------------------------------------------------
# Gate 4 — Architecture Spec Approval (last gate before development)
# ---------------------------------------------------------------------------

def gate_4_node(state: PipelineState) -> dict:
    """
    Gate 4: Human approves the full architecture spec. Last gate before coding.
    """
    payload = {
        "gate": 4,
        "title": "Gate 4 — Architecture Spec Approval (last gate before development)",
        "description": (
            "The Architecture Agent has produced the full implementation blueprint.\n"
            "Review the arch-spec carefully — Dev Agents will implement EXACTLY this.\n"
            "This is the last gate before autonomous development begins."
        ),
        "arch_spec_preview": (state.get("arch_spec") or "")[:3000],
        "options": ["APPROVED", "CHANGES — <your feedback>"],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "GATE_4_DECISION",
        [f"Decision: {human_input}", "Development will begin after this gate"],
    )
    append_event_to_file(event)

    decision = "APPROVED" if "APPROVED" in human_input.upper() else "CHANGES"

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "current_stage": "GATE_4_COMPLETE",
        "event_log": [event],
    }


# ---------------------------------------------------------------------------
# Gate 5 — Final Review (feature complete)
# ---------------------------------------------------------------------------

def gate_5_node(state: PipelineState) -> dict:
    """
    Gate 5: Final human review. Feature is complete — ship or hold.
    """
    payload = {
        "gate": 5,
        "title": "Gate 5 — Final Review",
        "description": (
            f"Feature {state['feature_id']} — {state['feature_name']} is complete.\n"
            "All tests pass. All docs written. Platform state updated.\n"
            "Approve to mark SHIPPED or hold for additional review."
        ),
        "test_report_preview": (state.get("test_report") or "")[:1000],
        "performance_report_preview": (state.get("performance_report") or "")[:500],
        "docs_output": state.get("docs_output") or "",
        "deploy_report": state.get("deploy_report") or "",
        "options": ["SHIPPED", "HOLD — <reason>"],
    }

    human_input: str = interrupt(payload)

    event = make_event(
        "HUMAN",
        "GATE_5_FINAL_DECISION",
        [f"Decision: {human_input}", f"Feature: {state['feature_id']} — {state['feature_name']}"],
    )
    append_event_to_file(event)

    decision = "SHIPPED" if "SHIPPED" in human_input.upper() else "HOLD"

    return {
        "human_feedback": human_input,
        "gate_decision": decision,
        "pipeline_complete": decision == "SHIPPED",
        "current_stage": "COMPLETE" if decision == "SHIPPED" else "GATE_5_HOLD",
        "event_log": [event],
    }
