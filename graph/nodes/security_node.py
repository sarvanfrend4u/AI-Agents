"""
Security Agent node — Stage 4 (parallel with Design).
Produces security-checklist.md based on OWASP Top 10 review of the feature.
"""

from context.loader import (
    build_context_block, read_prompt, save_output,
    make_event, append_event_to_file,
)
from llm.client import call_agent
from graph.state import PipelineState


def security_node(state: PipelineState) -> dict:
    """
    Stage 4 (parallel): Security Agent.
    Produces security-checklist.md.
    """
    feature_id = state["feature_id"]
    feature_name = state["feature_name"]

    system_prompt = read_prompt("security")
    context = build_context_block(
        state,
        include_docs=["feature-spec", "arch-constraints"],
    )

    user_message = f"""{context}

---

## Your Task — Security Checklist

Produce **security-checklist.md** for {feature_id} — {feature_name}.

### OWASP Top 10 Review
For each relevant OWASP category, assess whether this feature introduces risk:
1. Injection (SQL, command, template)
2. Broken Authentication
3. Sensitive Data Exposure
4. XML External Entities (XXE)
5. Broken Access Control
6. Security Misconfiguration
7. Cross-Site Scripting (XSS)
8. Insecure Deserialization
9. Using Components with Known Vulnerabilities
10. Insufficient Logging & Monitoring

### Checklist Format
For each risk identified:
- **Risk:** [description]
- **Likelihood:** Low / Medium / High
- **Required Mitigation:** [exact implementation required — be specific]
- **Test required:** [what the Test Agent must verify]

### Overall Risk Level
Assign: Low / Medium / High — and justify.

If this feature has NO security implications (e.g. purely visual), state that
explicitly and explain why.

Write security-checklist.md now.
"""

    output = call_agent("security", system_prompt, user_message, max_tokens=3000)

    save_output(feature_id, feature_name, "security-checklist", output)

    event = make_event(
        "SECURITY_AGENT",
        "STAGE_4_COMPLETE — Security Checklist Written",
        [
            f"Feature: {feature_id} — {feature_name}",
            "security-checklist.md saved",
            "Waiting for Design Agent to complete",
        ],
    )
    append_event_to_file(event)

    return {
        "security_checklist": output,
        "event_log": [event],
    }
