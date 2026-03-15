"""
Conditional edge routing functions for the Atlas Realty agent pipeline.
Each function inspects state and returns the next node name.
"""

from graph.state import PipelineState


def route_after_gate_1(state: PipelineState) -> str:
    """After Gate 1: APPROVED → PM spec | REJECTED → end | CHANGES → PM again."""
    decision = state.get("gate_decision", "")
    if "APPROVED" in decision.upper():
        return "pm_node"          # PM writes the feature spec
    elif "REJECTED" in decision.upper():
        return "end"
    else:
        return "gate_1_node"      # loop back for revision


def route_after_gate_2(state: PipelineState) -> str:
    """After Gate 2 (spec approved): proceed to arch constraints."""
    decision = state.get("gate_decision", "")
    if "APPROVED" in decision.upper():
        return "arch_constraints_node"
    else:
        return "pm_node"          # PM revises spec


def route_after_gate_3(state: PipelineState) -> str:
    """After Gate 3 (design + security reviewed): proceed to arch spec."""
    decision = state.get("gate_decision", "")
    if "APPROVED" in decision.upper():
        return "arch_spec_node"
    else:
        # Route back to design (simplest — re-run both in parallel would need Send())
        return "design_node"


def route_after_gate_4(state: PipelineState) -> str:
    """After Gate 4 (arch approved): begin development."""
    decision = state.get("gate_decision", "")
    if "APPROVED" in decision.upper():
        return "dev_launch_node"
    else:
        return "arch_spec_node"   # arch agent revises


def route_after_code_review(state: PipelineState) -> str:
    """
    After Code Review:
    - APPROVED → deploy to dev first, then test
    - CHANGES_REQUIRED → back to dev agents (if retries remain)
    - ESCALATE → human intervention
    """
    decision = state.get("gate_decision", "")
    escalated = state.get("escalated", False)

    if escalated or "ESCALATE" in decision.upper():
        return "escalation_node"
    elif "APPROVED" in decision.upper():
        return "deploy_node"       # deploy first, then test
    else:
        return "dev_launch_node"   # retry dev agents


def route_after_deploy(state: PipelineState) -> str:
    """
    After deploy: always proceed to testing.
    Even if deploy was SKIPPED (Docker not running), tests proceed
    and will report the Docker-not-running message gracefully.
    """
    current = state.get("current_stage", "")
    if "BUILD_FAILED" in current:
        return "escalation_node"        # build failure needs human attention
    return "test_launch_node"


def route_after_testing(state: PipelineState) -> str:
    """
    After Test Agent completes (performance always passes through):
    - PASS → docs
    - FAIL → back to dev agents (if retries remain)
    - ESCALATE → human
    """
    decision = state.get("gate_decision", "")
    escalated = state.get("escalated", False)

    if escalated or "ESCALATE" in decision.upper():
        return "escalation_node"
    elif "PASS" in decision.upper() or "APPROVED" in decision.upper():
        return "docs_node"
    else:
        return "dev_launch_node"   # retry dev agents


def route_after_gate_5(state: PipelineState) -> str:
    """After final gate: SHIPPED → end | HOLD → stay."""
    decision = state.get("gate_decision", "")
    if "SHIPPED" in decision.upper():
        return "end"
    else:
        return "gate_5_node"      # human reviews again


def route_after_escalation(state: PipelineState) -> str:
    """After human resolves escalation: resume from review."""
    decision = state.get("gate_decision", "")
    if "APPROVED" in decision.upper():
        return "test_launch_node"
    elif "REWRITE" in decision.upper():
        return "dev_launch_node"
    else:
        return "end"
