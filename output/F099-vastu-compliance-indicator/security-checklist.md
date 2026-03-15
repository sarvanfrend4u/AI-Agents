# Atlas Realty — Security Agent
## F099 — Vastu Compliance Indicator
### Security Checklist

---

**Reading context silently before proceeding.**

Context confirms:
- F099 adds one new query parameter (`facing: List[str]`) to an existing GET endpoint
- Vastu tier computation is entirely client-side (ADR-F099-001: ACCEPTED)
- No authentication exists on this platform (public listing search)
- No PII, no financial transactions, no user accounts
- Backend change: allowlist validation + SQL `IN` clause on `facing` parameter
- Frontend change: lookup table, display badge, filter toggle — all read-only derived from existing data

---

## SECURITY CHECKLIST — F099 Vastu Compliance Indicator

**Feature:** F099 — Vastu Compliance Indicator
**Platform:** Atlas Realty (Chennai, India)
**Stage:** 4 — Design & Security (parallel)
**Date:** 2026-03-15
**Security Agent Version:** 1

---

## Section 1 — Feature Risk Classification

| Dimension | Assessment | Reasoning |
|---|---|---|
| User input collected | **Yes — 1 field** | `facing` multi-value query param |
| PII involved | **No** | Compass directions only |
| Financial data involved | **No** | Read-only filter |
| Authentication changes | **No** | Platform has no auth |
| New external API calls | **No** | All data already in DB |
| New database writes | **No** | Read-only `SELECT` |
| New database columns | **No** | Uses existing `facing` column |
| Client-side computation | **Yes** | Vastu tier lookup (static, no network) |
| New file serving / uploads | **No** | |

**Overall Risk Level: MEDIUM**

**Justification:** The feature introduces one new user-controlled query parameter that directly influences SQL query construction. Although the data is non-sensitive and the platform has no auth layer, unsanitised query parameters feeding into SQL `IN` clauses are a classic injection vector. The risk is Medium — not High (no PII, no auth, no writes) — but the SQL construction path requires specific, verifiable mitigations. All other aspects of this feature (client-side Vastu computation, CSS, badge display) carry no security risk.

---

## Section 2 — Authentication & Authorisation

**Auth requirement:** None. Platform has no authentication. `GET /listings` is a public endpoint. F099 does not change this.

**Required action:** None — no auth gates to add or modify.

**Authorisation concern flagged:** The route path change from `/listings` to `/api/listings` (Blocker #3) must be reverted. This is not a security issue per se, but an unauthorised structural change that could silently break URL validation or proxy rules if any are added in future. Revert to `@app.get("/listings")`.

---

## Section 3 — Input Validation

### 3.1 `facing` Query Parameter

This is the **only new user-controlled input** in F099. It is the primary security surface.

**Field:** `facing` (multi-value, `List[str]`, query parameter on `GET /listings`)

**Valid values (exact):** Must be one of exactly these 8 strings — case-sensitive:
```
North, South, East, West, North-East, North-West, South-East, South-West
```

**Required validation rules — all mandatory:**

| Rule | Requirement | Why |
|---|---|---|
| **Allowlist only** | Each value in the list must be a member of `VALID_FACING_VALUES` frozenset. Reject anything not in the set. | Prevents SQL injection; prevents unexpected query behaviour |
| **Type enforcement** | FastAPI must declare parameter as `facing: List[str] = Query(default=[])`. Do not accept as a single comma-joined string. | Enforces multi-value semantics; prevents `"East,North-East"` bypass |
| **Empty list allowed** | `facing=[]` (parameter absent) must return all listings unfiltered. This is the default state. | No filter = all results is correct product behaviour |
| **Max cardinality** | Reject any request supplying more than 8 `facing` values (there are only 8 valid directions). Return HTTP 400. | Prevents degenerate inputs; trivial to enforce |
| **Rejection response** | If any value fails allowlist check: return `HTTP 400` with body `{"detail": "Invalid facing value: '<value>'"}`. Do not echo back the raw input in a way that could reflect injected content. | Prevents partial-match confusion; safe error messaging |
| **No stripping/coercing** | Do not silently strip invalid values and continue. Reject the whole request. | Fail loudly — silent coercion masks client bugs |

**Required implementation pattern (Backend Dev Agent must follow exactly):**

```python
VALID_FACING_VALUES: frozenset[str] = frozenset({
    "North", "South", "East", "West",
    "North-East", "North-West", "South-East", "South-West"
})

@app.get("/listings")
def get_listings(
    facing: List[str] = Query(default=[]),
    # ... other params
):
    # VALIDATION MUST OCCUR BEFORE ANY SQL CONSTRUCTION
    if len(facing) > 8:
        raise HTTPException(status_code=400, detail="Too many facing values")
    for value in facing:
        if value not in VALID_FACING_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid facing value: '{value}'"
            )
    # Only reach SQL construction after full validation passes
```

**Required SQL construction pattern:**

```python
if facing:
    placeholders = ",".join(["%s"] * len(facing))
    where_clauses.append(f"facing IN ({placeholders})")
    params.extend(facing)  # psycopg2 parameterised — NOT f-string interpolation
```

**Prohibited patterns (Dev Agent must not use):**

```python
# NEVER DO THIS — SQL injection via f-string:
query += f"AND facing IN ({','.join(facing)})"

# NEVER DO THIS — injection via string formatting:
query += "AND facing IN ('%s')" % "','".join(facing)
```

---

## Section 4 — OWASP Top 10 Assessment

### 4.1 Injection (SQL, command, template)

**Risk:** Yes — relevant and primary risk for this feature.

**Attack scenario:** Attacker sends `GET /listings?facing=East' OR '1'='1`. If the `facing` value is interpolated directly into the SQL string rather than parameterised, this modifies the query logic.

**Likelihood:** Medium (simple to attempt; backend is public; no auth barrier)

**Required mitigations:**
1. Allowlist validation (Section 3.1) — must execute before any SQL construction
2. Parameterised query using psycopg2 `%s` placeholders — the allowlist is defence-in-depth, but parameterisation is the primary SQL injection control and must be present regardless
3. Validation must reject on first invalid value — do not process partial lists

**Test required:**
- Send `?facing=East' OR '1'='1` → must return HTTP 400, not 200 with extra results
- Send `?facing=North; DROP TABLE listings--` → must return HTTP 400
- Send `?facing=East,North` (comma-joined, wrong format) → must return HTTP 400 (invalid value)
- Send `?facing=east` (lowercase) → must return HTTP 400 (case-sensitive allowlist)
- Send `?facing=East&facing=North` (correct multi-value) → must return HTTP 200 with filtered results

---

### 4.2 Broken Authentication

**Risk:** Not applicable.

**Reasoning:** Platform has no authentication. F099 does not add auth, remove auth, change session handling, or introduce any credentialed endpoints. No change to assess.

---

### 4.3 Sensitive Data Exposure

**Risk:** Low — requires attention but no new mitigation needed.

**Assessment:** The `facing` field is property metadata (compass direction of a listing). This is not PII. It does not reveal user identity, location, or behaviour. The API response contains no new sensitive fields for F099 — `facing` was already being returned in listing responses per the existing schema.

**One minor note:** The HTTP 400 error response echoes back the invalid value in the `detail` field (e.g., `"Invalid facing value: '<attacker_input>'""`). This is low risk since the value is bounded by HTTP query string length limits and is reflected in a JSON field, not rendered as HTML. However, the Dev Agent must ensure the error message does not echo values longer than 50 characters to prevent response bloat from oversized inputs.

**Required mitigation:** Truncate or cap echoed input in error messages:
```python
safe_value = value[:50]  # cap at 50 chars before echoing in error detail
raise HTTPException(status_code=400, detail=f"Invalid facing value: '{safe_value}'")
```

**Test required:** Send `?facing=` followed by 1000-character string → must return HTTP 400 with truncated or generic error message, not a 1000-character echo.

---

### 4.4 XML External Entities (XXE)

**Risk:** Not applicable.

**Reasoning:** F099 uses no XML. No XML parsing occurs anywhere in this feature. JSON only.

---

### 4.5 Broken Access Control

**Risk:** Not applicable.

**Reasoning:** Platform has no access control model. All listing data is public. F099 does not introduce any role-gated content, user-specific data, or resource ownership checks. The `facing` filter returns a subset of public listings — no data is gated.

---

### 4.6 Security Misconfiguration

**Risk:** Low — one specific concern related to Blocker #3.

**Concern:** The unauthorised route path change from `/listings` to `/api/listings` (Code Review Blocker #3) is a misconfiguration risk vector. If the platform later adds an API gateway, reverse proxy, or CORS policy scoped to specific paths, an undocumented route prefix change can cause the new route to fall outside security controls configured for the old path.

**Required mitigation:** Revert route to `@app.get("/listings")` as established in platform-state.md. Do not introduce `/api/` prefix without Orchestrator approval and platform-state.md update.

**Test required:** `GET /listings?facing=East` must return HTTP 200. `GET /api/listings?facing=East` must return HTTP 404 (route does not exist).

---

### 4.7 Cross-Site Scripting (XSS)

**Risk:** Low — one area warrants explicit confirmation.

**Reasoning:** Vastu tier badges display fixed strings from a static lookup table (`"Excellent"`, `"Good"`, `"Neutral"`). These strings are hardcoded in `frontend/lib/vastu.ts` — they are never derived from API response data or user input. No user-supplied content is rendered into the DOM for this feature.

**One area to confirm:** The `facing` value from the API response (e.g., `"North-East"`) is used as an argument to `getVastuInfo(facing)` — it is looked up in a table, not rendered directly. The lookup result is a hardcoded string. This is safe.

**Required mitigation:** Dev Agent must confirm that no component directly renders `listing.facing` as raw HTML (e.g., `dangerouslySetInnerHTML`). It should only pass `listing.facing` to `getVastuInfo()` and render the returned hardcoded label string.

**Test required:** Confirm `VastuBadge`, `ListingCard`, `ListingSheet`, and `MapPopupCard` use `getVastuInfo(listing.facing).label` (a hardcoded string), not `listing.facing` directly, for any displayed text.

---

### 4.8 Insecure Deserialization

**Risk:** Not applicable.

**Reasoning:** F099 does not introduce object serialization, pickle, JWT, or any deserialization of complex objects. The API returns JSON with string and number fields. The `facing` field is a plain string. No deserialization attack surface introduced.

---

### 4.9 Using Components with Known Vulnerabilities

**Risk:** Not applicable.

**Reasoning