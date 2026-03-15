"""
Deploy node — runs after code review approval.
Rebuilds only the changed Docker services and waits for them to be healthy.
Gives the human a live URL to review the feature before tests run.
"""

from context.loader import make_event, append_event_to_file
from context.file_tools import (
    docker_is_running, docker_rebuild, docker_exec,
    wait_for_healthy, REPO_ROOT,
)
from graph.state import PipelineState


def deploy_node(state: PipelineState) -> dict:
    """
    Rebuild and restart only the Docker services that had files changed.
    Backend changes → rebuild backend.
    Frontend changes → rebuild frontend.
    Both changed → rebuild both.

    If Docker is not running, skips gracefully and logs a warning.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    written_backend = state.get("files_written_backend") or []
    written_frontend = state.get("files_written_frontend") or []

    # Decide which services to rebuild
    services = []
    if written_backend:
        services.append("backend")
    if written_frontend:
        services.append("frontend")
    if not services:
        services = ["backend", "frontend"]  # fallback: rebuild everything

    log_lines: list[str] = []
    verdict = "SKIPPED"

    # ── Check Docker is available ─────────────────────────────────────────
    if not docker_is_running():
        log_lines.append("Docker is not running or docker compose is unavailable.")
        log_lines.append("Start the dev stack with: docker compose up -d")
        log_lines.append("Then run: docker compose up --build -d " + " ".join(services))
        verdict = "SKIPPED"
    else:
        # ── Rebuild ───────────────────────────────────────────────────────
        log_lines.append(f"Rebuilding: {', '.join(services)}")
        result = docker_rebuild(services, timeout=300)

        if result["success"]:
            log_lines.append("Build succeeded.")
            verdict = "DEPLOYED"

            # Wait for health checks
            for svc in services:
                healthy = wait_for_healthy(svc, retries=12, delay=5)
                status = "healthy" if healthy else "started (no health check)"
                log_lines.append(f"  {svc}: {status}")

            # Quick smoke test: hit backend /health if it exists
            if "backend" in services:
                smoke = docker_exec("backend", ["curl", "-sf", "http://localhost:8000/health"])
                if smoke["success"]:
                    log_lines.append("  Backend /health: OK")
                else:
                    log_lines.append("  Backend /health: not found (OK if no health endpoint)")

        else:
            verdict = "BUILD_FAILED"
            log_lines.append("Build FAILED. See error below.")
            log_lines.append(result["stderr"][-1500:] if result["stderr"] else "(no stderr)")

    # ── Build report ──────────────────────────────────────────────────────
    urls = []
    if "backend" in services:
        urls += [
            "Backend API:  http://localhost:8000",
            "API Docs:     http://localhost:8000/docs",
        ]
    if "frontend" in services:
        urls += ["Frontend app: http://localhost:3000"]

    files_changed_summary = (
        "\n".join(f"  backend: {p}" for p in written_backend) +
        "\n".join(f"  frontend: {p}" for p in written_frontend)
    ) or "  (none recorded)"

    report = f"""# Deploy Report — {feature_name} ({feature_id})
**Verdict:** {verdict}
**Services rebuilt:** {', '.join(services)}

## Files that triggered this deploy
{files_changed_summary}

## Build Log
{chr(10).join(log_lines)}

## Live URLs (if deployed)
{chr(10).join(urls) if urls else '(none)'}
"""

    event = make_event(
        "DEPLOY",
        f"DEV_DEPLOY — {verdict}",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Services: {', '.join(services)}",
            f"Verdict: {verdict}",
        ] + urls,
    )
    append_event_to_file(event)

    return {
        "deploy_report": report,
        "current_stage": "STAGE_7_READY" if verdict in ("DEPLOYED", "SKIPPED") else "STAGE_7_BUILD_FAILED",
        "event_log": [event],
    }
