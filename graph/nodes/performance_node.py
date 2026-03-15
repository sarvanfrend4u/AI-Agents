"""
Performance Agent node — Stage 7 (parallel with Test).
Runs lightweight checks: API latency, bundle size, DB query time.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from llm.client import call_agent
from graph.state import PipelineState


def performance_node(state: PipelineState) -> dict:
    """
    Stage 7 (parallel): Performance Agent.
    Analyses arch-spec for performance risks and writes a performance report.
    Note: actual measurements require the app to be running — this node
    produces the test plan and analysis; a separate shell step runs curl/psql.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("performance")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-spec"],
    )

    backend_code = state.get("dev_backend_output", "(no backend output)")
    frontend_code = state.get("dev_frontend_output", "(no frontend output)")

    user_message = f"""{context}

---

## Implementation to Analyse

### Backend Implementation
{backend_code}

### Frontend Implementation
{frontend_code}

---

## Your Task — Performance Report

Analyse the implementation for {feature_id} — {feature_name} against targets:

| Metric | Target |
|---|---|
| API endpoint p99 latency | < 200ms |
| Frontend initial render | < 100ms for new components |
| JS bundle size increase | < 10KB per feature |
| DB query time | < 50ms for simple queries |

### Analysis Areas:

1. **API Latency** — Review new endpoints in arch-spec. Are there N+1 queries?
   Missing indexes? Heavy computation in request path?

2. **Bundle Size** — How many new components/imports? Estimate KB impact.
   Are there any large library imports that should be lazy-loaded?

3. **DB Query** — Review any new SQL queries. Are indexes used? Is EXPLAIN needed?
   Write the EXPLAIN ANALYZE command for the DBA to run.

4. **Map Rendering** — If MapCanvas was modified, flag for human visual check.

### For each concern found:
- State whether it's within target or a FLAG
- Provide specific recommendation if flagged

### Commands to run (for human to execute):
```bash
# API latency check
for i in {{1..20}}; do curl -o /dev/null -s -w "%{{time_total}}\\n" http://localhost:8000/[endpoint]; done

# DB query
docker compose exec db psql -U postgres -d atlasrealty -c "EXPLAIN ANALYZE [query]"
```

Write performance-report.md now.
"""

    output = call_agent("performance", system_prompt, user_message, max_tokens=3000)

    save_output(feature_id, feature_name, "performance-report", output)

    verdict = "PASS"
    if "FLAG" in output.upper():
        verdict = "FLAG"

    event = make_event(
        "PERFORMANCE_AGENT",
        f"STAGE_7_PERFORMANCE — Verdict: {verdict}",
        [
            f"Feature: {feature_id} — {feature_name}",
            f"Verdict: {verdict}",
            "performance-report.md saved",
        ],
    )
    append_event_to_file(event)

    return {
        "performance_report": output,
        "event_log": [event],
    }
