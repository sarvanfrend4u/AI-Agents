"""
Pipeline assembly for Atlas Realty agent system.
Builds the LangGraph StateGraph with all nodes, edges, and parallel fan-outs.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import Send

load_dotenv(Path(__file__).parent.parent / ".env")

from graph.state import PipelineState

# Nodes
from graph.nodes.pm_node import pm_node
from graph.nodes.arch_constraints_node import arch_constraints_node
from graph.nodes.design_node import design_node
from graph.nodes.security_node import security_node
from graph.nodes.arch_spec_node import arch_spec_node
from graph.nodes.dev_backend_node import dev_backend_node
from graph.nodes.dev_frontend_node import dev_frontend_node
from graph.nodes.code_review_node import code_review_node
from graph.nodes.test_node import test_node
from graph.nodes.performance_node import performance_node
from graph.nodes.docs_node import docs_node
from graph.nodes.escalation_node import escalation_node
from graph.nodes.deploy_node import deploy_node

# Gates
from graph.gates.interrupts import (
    gate_1_node, gate_2_node, gate_3_node, gate_4_node, gate_5_node
)

# Routers
from graph.edges.routers import (
    route_after_gate_1,
    route_after_gate_2,
    route_after_gate_3,
    route_after_gate_4,
    route_after_code_review,
    route_after_deploy,
    route_after_testing,
    route_after_gate_5,
    route_after_escalation,
)


# ---------------------------------------------------------------------------
# Fan-out routers (used in add_conditional_edges, NOT as nodes)
# These return list[Send] — valid only as conditional edge functions.
# ---------------------------------------------------------------------------

def dev_parallel_fan_out(state: PipelineState) -> list[Send]:
    """Fan out to Backend and Frontend Dev Agents in parallel."""
    return [
        Send("dev_backend_node", state),
        Send("dev_frontend_node", state),
    ]


def test_perf_fan_out(state: PipelineState) -> list[Send]:
    """Fan out to Test and Performance Agents in parallel."""
    return [
        Send("test_node", state),
        Send("performance_node", state),
    ]


def design_security_fan_out(state: PipelineState) -> list[Send]:
    """Fan out to Design and Security Agents in parallel."""
    return [
        Send("design_node", state),
        Send("security_node", state),
    ]


# ---------------------------------------------------------------------------
# Launch nodes — thin passthrough nodes that precede each fan-out.
# Nodes must return dicts; Send() routing happens in the conditional edge.
# ---------------------------------------------------------------------------

def dev_launch_node(state: PipelineState) -> dict:
    """Precedes dev fan-out. Returns state update so LangGraph treats it as a node."""
    return {"current_stage": "STAGE_6"}


def test_launch_node(state: PipelineState) -> dict:
    """Precedes test/perf fan-out. Returns state update so LangGraph treats it as a node."""
    return {"current_stage": "STAGE_7_READY"}


# ---------------------------------------------------------------------------
# Sync nodes (collect parallel outputs before proceeding)
# ---------------------------------------------------------------------------

def sync_design_security(state: PipelineState) -> dict:
    """
    Called after both Design and Security complete.
    Both outputs are already merged into state via reducers.
    This node just advances the stage.
    """
    return {"current_stage": "STAGE_4_COMPLETE"}


def sync_dev_agents(state: PipelineState) -> dict:
    """Called after both Dev Agents complete. Advances to Code Review."""
    return {"current_stage": "STAGE_6_DEV_COMPLETE"}


def sync_test_perf(state: PipelineState) -> dict:
    """Called after both Test and Performance complete. Advance to docs."""
    return {"current_stage": "STAGE_7_COMPLETE"}


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    """
    Assemble and compile the full Atlas Realty agent pipeline graph.
    Returns a compiled LangGraph graph ready to stream.
    """
    g = StateGraph(PipelineState)

    # ── Register all nodes ────────────────────────────────────────────────
    g.add_node("gate_1_node",              gate_1_node)
    g.add_node("pm_node",                  pm_node)
    g.add_node("gate_2_node",              gate_2_node)
    g.add_node("arch_constraints_node",    arch_constraints_node)
    g.add_node("design_node",              design_node)
    g.add_node("security_node",            security_node)
    g.add_node("sync_design_security",     sync_design_security)
    g.add_node("gate_3_node",              gate_3_node)
    g.add_node("arch_spec_node",           arch_spec_node)
    g.add_node("gate_4_node",              gate_4_node)
    g.add_node("dev_launch_node",          dev_launch_node)   # thin node before dev fan-out
    g.add_node("dev_backend_node",         dev_backend_node)
    g.add_node("dev_frontend_node",        dev_frontend_node)
    g.add_node("sync_dev_agents",          sync_dev_agents)
    g.add_node("code_review_node",         code_review_node)
    g.add_node("test_launch_node",         test_launch_node)  # thin node before test/perf fan-out
    g.add_node("test_node",               test_node)
    g.add_node("performance_node",         performance_node)
    g.add_node("sync_test_perf",           sync_test_perf)
    g.add_node("deploy_node",              deploy_node)
    g.add_node("docs_node",               docs_node)
    g.add_node("gate_5_node",              gate_5_node)
    g.add_node("escalation_node",          escalation_node)

    # ── Entry point ───────────────────────────────────────────────────────
    g.set_entry_point("gate_1_node")

    # ── Stage 1: Gate 1 → PM → Gate 2 ────────────────────────────────────
    g.add_conditional_edges("gate_1_node", route_after_gate_1, {
        "pm_node":   "pm_node",
        "gate_1_node": "gate_1_node",
        "end":       END,
    })
    g.add_edge("pm_node", "gate_2_node")
    g.add_conditional_edges("gate_2_node", route_after_gate_2, {
        "arch_constraints_node": "arch_constraints_node",
        "pm_node":               "pm_node",
    })

    # ── Stage 3: Arch Constraints → Stage 4 parallel fan-out ─────────────
    # design_security_fan_out returns list[Send] — used as conditional edge, not node
    g.add_conditional_edges("arch_constraints_node", design_security_fan_out)
    g.add_edge("design_node",   "sync_design_security")
    g.add_edge("security_node", "sync_design_security")
    g.add_edge("sync_design_security", "gate_3_node")

    # ── Stage 4: Gate 3 → Arch Spec → Gate 4 ─────────────────────────────
    g.add_conditional_edges("gate_3_node", route_after_gate_3, {
        "arch_spec_node": "arch_spec_node",
        "design_node":    "design_node",
    })
    g.add_edge("arch_spec_node", "gate_4_node")
    g.add_conditional_edges("gate_4_node", route_after_gate_4, {
        "dev_launch_node": "dev_launch_node",
        "arch_spec_node":  "arch_spec_node",
    })

    # ── Stage 6: dev_launch_node → parallel fan-out → Code Review ─────────
    # dev_parallel_fan_out returns list[Send] — conditional edge from launch node
    g.add_conditional_edges("dev_launch_node", dev_parallel_fan_out)
    g.add_edge("dev_backend_node",  "sync_dev_agents")
    g.add_edge("dev_frontend_node", "sync_dev_agents")
    g.add_edge("sync_dev_agents", "code_review_node")

    g.add_conditional_edges("code_review_node", route_after_code_review, {
        "deploy_node":     "deploy_node",      # APPROVED → deploy first
        "dev_launch_node": "dev_launch_node",  # retry dev agents
        "escalation_node": "escalation_node",
    })

    # ── Deploy → test_launch_node → parallel fan-out ──────────────────────
    g.add_conditional_edges("deploy_node", route_after_deploy, {
        "test_launch_node": "test_launch_node",
        "escalation_node":  "escalation_node",
    })

    # ── Stage 7: test_launch_node → parallel fan-out ──────────────────────
    # test_perf_fan_out returns list[Send] — conditional edge from launch node
    g.add_conditional_edges("test_launch_node", test_perf_fan_out)
    g.add_edge("test_node",        "sync_test_perf")
    g.add_edge("performance_node", "sync_test_perf")

    g.add_conditional_edges("sync_test_perf", route_after_testing, {
        "docs_node":       "docs_node",
        "dev_launch_node": "dev_launch_node",  # retry dev agents
        "escalation_node": "escalation_node",
    })

    # ── Stage 8: Docs → Gate 5 ───────────────────────────────────────────
    g.add_edge("docs_node", "gate_5_node")
    g.add_conditional_edges("gate_5_node", route_after_gate_5, {
        "end":         END,
        "gate_5_node": "gate_5_node",
    })

    # ── Escalation ───────────────────────────────────────────────────────
    g.add_conditional_edges("escalation_node", route_after_escalation, {
        "test_launch_node": "test_launch_node",
        "dev_launch_node":  "dev_launch_node",
        "end":              END,
    })

    # ── Compile ───────────────────────────────────────────────────────────
    # Try Postgres checkpointer first; fall back to in-memory
    checkpointer = _build_checkpointer()
    return g.compile(checkpointer=checkpointer)


def _build_checkpointer():
    """Build checkpointer. Uses Postgres if URL set, else in-memory."""
    pg_url = os.getenv("POSTGRES_CHECKPOINTER_URL", "")
    if pg_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return PostgresSaver.from_conn_string(pg_url)
        except Exception as e:
            print(f"[pipeline] Postgres checkpointer failed ({e}), using in-memory.")

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
