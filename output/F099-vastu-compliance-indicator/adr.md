# Reading All Required Context Files

I'll silently read all mandatory context files before proceeding.

**Context read complete.** Key findings:

- **active_feature.json:** F099, Pass 2 (Stage 5)
- **event-log.md:** GATE_3_COMPLETE — Design + Security approved, now at Stage 5
- **platform-state.md:** F099 incomplete, 7 blockers from Code Review Cycle 1; arch-spec, design-spec, security-checklist all exist
- **Existing ADR:** ADR-F099-001 already accepted — Vastu computation location (client-side). I must not contradict this.
- **arch-constraints.md:** Exists from Pass 1 — confirms no DB changes, `facing` column already exists, multi-value query param pattern
- **arch-spec.md:** Exists from Pass 2 — covers all layers. However, given the Code Review blockers, I need to write a revised ADR that captures decisions made during rework, including the resolution of the architectural questions raised by the blockers.
- **security-checklist.md:** Confirms SQL injection risk on `facing` param; requires allowlist + parameterised IN clause
- **design-spec.md:** Covers all 5 missing UI components
- **Past ADRs:** Only ADR-F099-001 exists — client-side computation (ACCEPTED)
- **listing.ts, mapStore.ts, api.ts, filters.ts, main.py, migrations:** Read. No migrations applied for F099. Existing `/listings` route. psycopg2 sync driver throughout. All existing routes are `def` (sync).

---

The task asks me to write `adr.md` for F099 in the context of Pass 2 / Stage 5. The existing ADR-F099-001 is already accepted. My job now is to:

1. Preserve ADR-F099-001 (do not contradict)
2. Document **additional** architectural decisions that emerged from the rework cycle — specifically those raised by the 7 blockers that have architectural significance
3. For blockers that are pure Dev Agent execution failures (no architectural dimension), note them as such
4. Follow `agent-system/templates/adr.md` format exactly

The decisions with genuine architectural significance are:

- **Route path discipline** (Blocker #3): `/listings` vs `/api/` prefix — establishes a pattern for all future routes
- **Sync vs async route handlers** (Blocker #7): `def` vs `async def` with blocking psycopg2 — establishes a pattern for all future FastAPI routes
- **Multi-value query param serialisation** (Blocker #2): `append()` vs `set()` — establishes the pattern for all future multi-value filter params
- **Type interface discipline** (Blocker #4): Keeping `listing.ts` in sync with actual DB schema — establishes a governance pattern
- **State hygiene** (Blocker #5): Single field per filter, no deprecated state — establishes a store pattern

ADR-F099-001 (client-side computation) is already written and accepted. I will not rewrite it — I will reference it and write the remaining decisions as ADR-F099-002 through ADR-F099-006.

---

```markdown
# Architecture Decision Records — F099 Vastu Compliance Indicator
**Feature:** F099 — Vastu Compliance Indicator  
**Stage:** Stage 5 — Architecture Pass 2 (Rework Cycle 1)  
**Author:** Architecture Agent  
**Date:** 2026-03-15  
**Saved to:** agent-system/output/F099-vastu-compliance-indicator/adr.md

---

## ADR Index

| ADR | Title | Status |
|---|---|---|
| ADR-F099-001 | Vastu Tier Computation Location (Client-Side) | ACCEPTED (prior pipeline run — preserved, not modified) |
| ADR-F099-002 | FastAPI Route Path Convention (No `/api/` Prefix) | ACCEPTED |
| ADR-F099-003 | Synchronous Route Handlers with psycopg2 | ACCEPTED |
| ADR-F099-004 | Multi-Value Query Parameter Serialisation (`append()` Pattern) | ACCEPTED |
| ADR-F099-005 | Listing Interface Discipline (Schema-Bound Type Definition) | ACCEPTED |
| ADR-F099-006 | Zustand Filter State Hygiene (One Field Per Filter) | ACCEPTED |

---

## ADR-F099-001: Vastu Tier Computation Location (Client-Side)

**Status:** ACCEPTED  
**Date:** 2026-03-15 (original pipeline run)  
**Preserved:** This ADR was written and accepted in the prior pipeline run. It is reproduced
here in full for completeness and is not modified.

---

### Context

F099 requires displaying a Vastu compliance tier (excellent / good / neutral) for each
listing. The tier derives entirely from the `facing` field, which already exists on every
listing in the DB. The mapping is a static 8-entry lookup table (compass direction → tier).

The question was: where should this computation run?

### Decision

**Vastu tier is computed entirely on the frontend** via a static lookup table in
`frontend/lib/vastu.ts`. The backend provides the raw `facing` field (already present in
every listing response) and optionally filters by specific `facing` values when the Vastu
filter is active. The backend never computes, stores, or returns a `vastu_tier` field.

### Rationale

- The mapping is a static 8-entry lookup table — no algorithmic complexity, no data joins,
  no DB reads beyond what is already fetched
- Eliminates the need for a new DB migration, a new API response field, and server-side
  computation for a presentation-layer concern
- Future changes to Vastu tier rules (e.g., regional variations in v2) require only a
  frontend deployment, not a backend migration + deployment
- Consistent with the platform principle that the backend is a data layer; presentation
  logic lives in the frontend

### Alternatives Considered

**Option B — Server computes `vastu_tier` per listing:**
Backend adds `vastu_tier` to the JSON response. Rejected — adds API complexity and server
computation for a static lookup that is cheaper to run on the client.

**Option C — Precompute `vastu_tier` in DB as a generated column:**
Add a `vastu_tier` column (or generated expression) to `listings`. Rejected — DB migrations
for presentation-layer logic violate separation of concerns; no query benefit since the
filter operates on `facing` values directly.

### Implications

- `frontend/lib/vastu.ts` is the single source of truth for Vastu tier rules
- The server-side filter uses `facing` direction values (e.g., `?facing=East&facing=North-East`),
  not `vastu_tier` strings — this keeps the backend allowlist simple and direction-based
- `vastuTier` is never in the API response; it is always derived on the frontend from
  `listing.facing`

---

## ADR-F099-002: FastAPI Route Path Convention (No `/api/` Prefix)

**Status:** ACCEPTED  
**Date:** 2026-03-15  
**Triggered by:** Code Review Blocker #3 — Dev Agent changed route from `/listings`
to `/api/listings` without architectural authorisation

---

### Context

During F099 development (Cycle 1), the Dev Agent (Backend) changed the existing route
decorator from `@app.get("/listings")` to `@app.get("/api/listings")`. This was not
specified in the arch-spec and was not authorised by any ADR or Orchestrator decision.

The frontend `api.ts` was simultaneously updated to call `${API_URL}/api/listings`,
making the change self-consistent within the feature — but creating a breaking change
against the documented platform API (platform-state.md documents `GET /listings`),
and against any future proxy, gateway, or test infrastructure configured against the
original path.

### Decision

**Atlas Realty FastAPI routes use no path prefix.** The application root handles all
routes directly. The `GET /listings` route path is authoritative and must not be changed.

Specifically:
- Route decorators are `@app.get("/listings")`, `