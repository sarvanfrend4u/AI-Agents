# Security Checklist — {{FEATURE_NAME}} ({{FEATURE_ID}})

**Version:** v1
**Date:** {{DATE}}
**Status:** COMPLETE
**Prepared by:** Security Agent

---

## Authentication Requirements

- [ ] Requires user to be logged in: Yes / No
- [ ] Role required: None / User / Agent / Admin
- [ ] Notes: [any auth-specific notes]

---

## Data Sensitivity

- [ ] Personally Identifiable Information (PII) involved: Yes / No
- [ ] Financial data involved: Yes / No
- [ ] Location data involved: Yes / No
- [ ] User behaviour data collected: Yes / No
- [ ] **Overall data sensitivity level: Low / Medium / High**

---

## Input Validation Rules

| Input Field | Validation Required | Rule |
|---|---|---|
| [field name] | Yes / No | [e.g. must be string, max 100 chars, no SQL chars] |
| [field name] | Yes / No | [validation rule] |

---

## Rate Limiting

- [ ] Rate limiting required on new API endpoint: Yes / No
- [ ] Recommended limit: [X requests per minute per IP]
- [ ] Notes: [any rate limiting notes]

---

## API Exposure

- [ ] New public API endpoint created: Yes / No
- [ ] Endpoint requires authentication: Yes / No
- [ ] Risk of data over-exposure: Yes / No
- [ ] Notes: [any API security notes]

---

## OWASP Top 10 Check

| Risk | Applicable | Mitigation |
|---|---|---|
| SQL Injection | Yes / No | [mitigation or "N/A"] |
| XSS | Yes / No | [mitigation or "N/A"] |
| Broken Auth | Yes / No | [mitigation or "N/A"] |
| Sensitive Data Exposure | Yes / No | [mitigation or "N/A"] |
| Broken Access Control | Yes / No | [mitigation or "N/A"] |

---

## Overall Risk Level

**Risk Level: Low / Medium / High**

**Justification:** [Why this risk level — one sentence]

---

## Security Requirements for Dev Agents

[Specific security requirements the developers must implement — be explicit.
Example: "Sanitise all user inputs before passing to DB query", "Do not expose internal DB IDs in API response"]
