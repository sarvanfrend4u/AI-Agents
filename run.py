#!/usr/bin/env python3
"""
Atlas Realty — Agent Pipeline CLI
===================================
Standalone entry point for the multi-agent SDLC pipeline.
Runs entirely independently of Claude Code.

Usage:
    python run.py                          # start new pipeline (asks for feature)
    python run.py --feature F099           # start with specific feature ID
    python run.py --resume <thread_id>     # resume a paused pipeline
    python run.py --list                   # list all pipeline runs
    python run.py --status <thread_id>     # show status of a run
"""

from __future__ import annotations
import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Ensure agent-system/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from langgraph.types import Command

from context.loader import (
    read_active_feature, read_features_json, get_feature_list,
    STATE_DIR, make_event, append_event_to_file,
)
from graph.pipeline import build_pipeline

console = Console()

BANNER = """
[bold blue]
╔═══════════════════════════════════════════╗
║     Atlas Realty — Agent Pipeline         ║
║     Autonomous SDLC Orchestrator          ║
╚═══════════════════════════════════════════╝
[/bold blue]
"""

STAGE_LABELS = {
    "GATE_1":                 "Gate 1 — Feature Selection",
    "STAGE_1":                "Stage 1 — PM Agent",
    "GATE_2_AWAITING":        "Gate 2 — Spec Approval",
    "STAGE_3":                "Stage 3 — Arch Constraints",
    "STAGE_4_DESIGN_SECURITY":"Stage 4 — Design + Security (parallel)",
    "GATE_3":                 "Gate 3 — Design & Security Review",
    "STAGE_5":                "Stage 5 — Architecture Spec",
    "GATE_4_AWAITING":        "Gate 4 — Arch Spec Approval",
    "STAGE_6":                "Stage 6 — Development (parallel)",
    "STAGE_6_DEV_COMPLETE":   "Stage 6 — Code Review",
    "STAGE_7_READY":          "Stage 7 — Deploy complete, running tests",
    "STAGE_7_BUILD_FAILED":   "Stage 7 — Build failed",
    "STAGE_7_COMPLETE":       "Stage 7 — Tests + Performance complete",
    "STAGE_8":                "Stage 8 — Documentation",
    "GATE_5_FINAL_REVIEW":    "Gate 5 — Final Review",
    "COMPLETE":               "Complete",
}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Atlas Realty Agent Pipeline",
    )
    parser.add_argument("--feature", help="Feature ID to run (e.g. F099)")
    parser.add_argument("--resume", metavar="THREAD_ID", help="Resume a paused pipeline")
    parser.add_argument("--list", action="store_true", help="List pipeline runs")
    parser.add_argument("--status", metavar="THREAD_ID", help="Show run status")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip startup confirmation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_feature() -> tuple[str, str]:
    """Prompt human to select a feature from features.json."""
    features = get_feature_list()
    available = [f for f in features if f.get("status") in ("NOT_STARTED", None, "BACKLOG")]

    if not available:
        console.print("[yellow]No features available to run.[/yellow]")
        sys.exit(0)

    table = Table(title="Available Features", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Priority")

    for f in available[:20]:  # show first 20
        table.add_row(
            f.get("id", ""),
            f.get("name", ""),
            f.get("category", ""),
            str(f.get("priority", "")),
        )

    console.print(table)

    feature_id = Prompt.ask("\nEnter Feature ID to run").strip().upper()
    match = next((f for f in features if f.get("id") == feature_id), None)
    if not match:
        console.print(f"[red]Feature {feature_id} not found.[/red]")
        sys.exit(1)

    feature_name = match.get("name", feature_id)
    console.print(f"\n[green]Selected:[/green] {feature_id} — {feature_name}\n")
    return feature_id, feature_name


# ---------------------------------------------------------------------------
# Gate handler — called when pipeline pauses at interrupt()
# ---------------------------------------------------------------------------

def handle_gate(interrupt_data: dict) -> str:
    """
    Display gate information and collect human decision.
    Returns the human's input string.
    """
    gate_num = interrupt_data.get("gate", "?")
    title = interrupt_data.get("title", f"Gate {gate_num}")
    description = interrupt_data.get("description", "")
    options = interrupt_data.get("options", [])

    console.print()
    console.print(Panel(
        f"[bold]{title}[/bold]\n\n{description}",
        style="bold yellow",
        expand=False,
    ))

    # Print any preview content
    for key, label in [
        ("feature_spec", "Feature Spec"),
        ("feature_spec_preview", "Feature Spec Preview"),
        ("arch_spec_preview", "Architecture Spec Preview"),
        ("design_spec_preview", "Design Spec Preview"),
        ("security_checklist_preview", "Security Checklist Preview"),
        ("test_report_preview", "Test Report Preview"),
        ("performance_report_preview", "Performance Report Preview"),
        ("code_review_summary", "Code Review Summary"),
        ("test_report_summary", "Test Report Summary"),
        ("docs_output", "Docs Summary"),
        ("deploy_report", "Deploy Report"),
    ]:
        content = interrupt_data.get(key)
        if content:
            console.print(f"\n[bold cyan]── {label} ──[/bold cyan]")
            console.print(Markdown(content[:3000]))

    console.print()
    if options:
        console.print("[bold]Options:[/bold]")
        for i, opt in enumerate(options, 1):
            console.print(f"  {i}. {opt}")

    console.print()
    human_input = Prompt.ask("[bold green]Your decision[/bold green]").strip()

    # Allow shorthand: "1" → first option, "2" → second, etc.
    if human_input.isdigit() and 1 <= int(human_input) <= len(options):
        human_input = options[int(human_input) - 1]
        console.print(f"[dim]→ {human_input}[/dim]")

    return human_input


# ---------------------------------------------------------------------------
# Event display
# ---------------------------------------------------------------------------

def display_event(event: dict) -> None:
    """Print a pipeline event update to the console."""
    stage = event.get("current_stage", "")
    label = STAGE_LABELS.get(stage, stage)

    if stage:
        console.print(f"[dim]▶[/dim] [cyan]{label}[/cyan]")


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(feature_id: str, feature_name: str, thread_id: Optional[str] = None) -> None:
    """Run (or resume) the pipeline for a given feature."""
    pipeline = build_pipeline()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[bold]Thread ID:[/bold] [dim]{thread_id}[/dim]")
    console.print(f"[bold]Feature:[/bold]   {feature_id} — {feature_name}")
    console.print("[dim]Save this thread ID to resume if paused.[/dim]\n")

    # Initial state
    from context.loader import read_platform_state
    initial_state = {
        "feature_id":        feature_id,
        "feature_name":      feature_name,
        "platform_state":    read_platform_state(),
        "features_json":     read_features_json(),
        "event_log":         [],
        "feature_spec":      None,
        "arch_constraints":  None,
        "design_spec":       None,
        "security_checklist":None,
        "arch_spec":         None,
        "dev_backend_output":None,
        "dev_frontend_output":None,
        "code_review":       None,
        "test_report":        None,
        "performance_report": None,
        "deploy_report":      None,
        "docs_output":        None,
        "files_written_backend":  [],
        "files_written_frontend": [],
        "human_feedback":    None,
        "gate_decision":     None,
        "current_stage":     "GATE_1",
        "retry_code_review": 0,
        "retry_testing":     0,
        "escalated":         False,
        "pipeline_complete": False,
    }

    # Archive previous event log (if any) so agents start with a clean slate.
    # Archived logs are preserved in output/{feature_id}-{name}/ for auditing.
    event_log_path = STATE_DIR / "event-log.md"
    if event_log_path.exists() and event_log_path.stat().st_size > 0:
        safe_name = feature_name.lower().replace(" ", "-").replace("/", "-")
        archive_dir = Path(__file__).parent / "output" / f"{feature_id}-{safe_name}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archive_path = archive_dir / f"event-log-archive-{ts}.md"
        archive_path.write_text(event_log_path.read_text())

    # Start fresh event log for this run
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    event_log_path.write_text(
        f"# Atlas Realty — Agent Pipeline Event Log\n\n"
        f"_Pipeline run started {now_str} — {feature_id} — {feature_name}_\n\n"
    )

    # Update active_feature.json
    active_feature_path = STATE_DIR / "active_feature.json"
    active_feature = {
        "id":           feature_id,
        "name":         feature_name,
        "status":       "IN_PROGRESS",
        "threadId":     thread_id,
        "startedAt":    datetime.now(timezone.utc).isoformat(),
        "currentStage": "GATE_1",
        "retries":      {"codeReview": 0, "testing": 0},
    }
    active_feature_path.write_text(json.dumps(active_feature, indent=2))

    # Log pipeline start
    start_event = make_event(
        "ORCHESTRATOR",
        "PIPELINE_STARTED",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Thread ID: {thread_id}",
            "Stage 1: Gate 1 — Feature Selection",
        ],
    )
    append_event_to_file(start_event)

    _stream_until_complete(pipeline, initial_state, config)


def resume_pipeline(thread_id: str) -> None:
    """Resume a paused pipeline by thread ID."""
    pipeline = build_pipeline()
    config = {"configurable": {"thread_id": thread_id}}

    graph_state = pipeline.get_state(config)
    if not graph_state.next:
        console.print("[yellow]No interrupted state found for this thread.[/yellow]")
        return

    console.print(f"\n[bold]Resuming thread:[/bold] [dim]{thread_id}[/dim]")
    _stream_until_complete(pipeline, None, config, is_resume=True)


def _stream_until_complete(pipeline, initial_state, config, is_resume: bool = False) -> None:
    """Stream the pipeline, handling human gates, until complete or interrupted."""
    import_input = initial_state if not is_resume else Command(resume="RESUME")

    while True:
        try:
            for event in pipeline.stream(import_input, config, stream_mode="values"):
                display_event(event)
            # Stream exhausted — check if interrupted
            import_input = None
        except Exception as e:
            console.print(f"[red]Pipeline error: {e}[/red]")
            raise

        graph_state = pipeline.get_state(config)

        if not graph_state.next:
            # Pipeline complete
            console.print()
            console.print(Panel(
                "[bold green]Pipeline complete![/bold green]\n\n"
                "All documentation has been saved to agent-system/output/\n"
                "platform-state.md has been updated.",
                style="green",
            ))
            break

        # Check for interrupts (human gates)
        if graph_state.tasks:
            for task in graph_state.tasks:
                if task.interrupts:
                    for intr in task.interrupts:
                        human_input = handle_gate(intr.value)
                        # Resume with human decision
                        for event in pipeline.stream(
                            Command(resume=human_input), config, stream_mode="values"
                        ):
                            display_event(event)
                    # After processing all interrupts, loop to check state again
                    import_input = None
                    break
            else:
                # No interrupts found but next nodes exist — stream again
                import_input = None
                continue
        else:
            break


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    console.print(BANNER)
    args = parse_args()

    if args.list:
        # TODO: list runs from checkpointer
        console.print("[yellow]--list requires a Postgres checkpointer. Set POSTGRES_CHECKPOINTER_URL.[/yellow]")
        return

    if args.status:
        # TODO: show status from checkpointer
        console.print("[yellow]--status requires a Postgres checkpointer. Set POSTGRES_CHECKPOINTER_URL.[/yellow]")
        return

    if args.resume:
        resume_pipeline(args.resume)
        return

    # New run
    if args.feature:
        features = get_feature_list()
        match = next((f for f in features if f.get("id") == args.feature), None)
        if match:
            feature_id = match["id"]
            feature_name = match.get("name", args.feature)
        else:
            feature_id = args.feature
            feature_name = args.feature
    else:
        feature_id, feature_name = select_feature()

    if not args.yes:
        if not Confirm.ask(f"\nStart pipeline for [bold]{feature_id} — {feature_name}[/bold]?"):
            console.print("[dim]Aborted.[/dim]")
            return

    run_pipeline(feature_id, feature_name)


if __name__ == "__main__":
    main()
