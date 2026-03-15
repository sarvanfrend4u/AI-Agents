I will silently read all mandatory context files before producing any output.

**Context read complete.** Key findings:

- F099 is in **Rework Cycle 1** — existing arch-spec, design-spec, security-checklist, and feature-spec all exist
- All 7 Code Review blockers are Dev Agent execution failures against an already-correct architecture — but the arch-spec I wrote in Pass 1 was incomplete in specificity (it described *what* to build but not precisely *how*, which contributed to Dev Agent failures)
- **ADR-F099-001 (ACCEPTED):** Client-side Vastu computation — must not contradict
- **No DB changes required** — uses existing `facing` column
- **Route path:** Must remain `GET /listings` (not `/api/listings`) — Blocker #3
- **Sync route:** Must be `def` not `async def` with psycopg2 — Blocker #7
- The design-spec was truncated in the context provided but contains sufficient detail for the VastuBadge component; I have the full security-checklist
- The previous arch-spec lacked explicit file-by-file implementation instructions, contributing to Dev Agent scope violations and omissions

This Pass 2 arch-spec must be maximally specific — leaving no room for Dev Agent interpretation that caused the 7 blockers.

---

```markdown
# Architecture Specification — F099 Vastu Compliance Indicator
**Feature:** F099 — Vastu Compliance Indicator  
**Pipeline Stage:** Stage 5 — Architecture Pass 2 (Rework Cycle 1)  
**Author:** Architecture Agent  
**Date:** 2026-03-15  
**Status:** COMPLETE — Full implementation blueprint for Dev Agents  
**Saved to:** agent-system/output/F099-vastu-compliance-indicator/arch-spec.md

---

## 0. Rework Context and Scope

This document supersedes the prior arch-spec.md for F099.
It was produced after Code Review Cycle 1 identified 7 blockers, all of which
were Dev Agent execution failures. This spec is written with maximum specificity
to eliminate interpretation gaps.

### Blockers Being Addressed

| # | Blocker | Resolution in This Spec |
|---|---|---|
| #1 | 5 UI component files not created | §9: exact file-by-file implementation specs |
| #2 | `params.set()` instead of `params.append()` in api.ts | §7.2: exact code to write |
| #3 | Route path changed to `/api/listings` unauthorised | §3.1: route must be `/listings` |
| #4 | 16 out-of-scope fields in listing.ts | §5: exact interface — only what is specified here |
| #5 | Dual `vastuFilter` + `vastuCompliant` state in mapStore.ts | §6: exact store shape — nothing else |
| #6 | Backend file truncated | §3: complete backend spec — no truncation permitted |
| #7 | `async def` with blocking psycopg2 | §3.1: route must be `def` (sync) |

### What This Spec Does NOT Change

- ADR-F099-001 (ACCEPTED): Vastu tier computed client-side — unchanged
- DB schema: no changes, uses existing `facing` column — unchanged
- `frontend/lib/vastu.ts`: already written correctly — do not modify
- `frontend/app/globals.css`: already written correctly — do not modify
- Vastu tier mapping: East/North-East = excellent; North/North-West/West = good;
  South/South-East/South-West = neutral — unchanged

### Dev Agent Rule (Non-Negotiable)

**Write exactly what this spec says. Nothing more. Nothing less.**

- Do not add fields not listed in §5 (Listing interface)
- Do not add state not listed in §6 (Zustand store)
- Do not add utility functions not listed in §7
- Do not change route paths from what §3 specifies
- Do not add `async` to route handlers
- Do not add Phase 3 / future features to any file

---

## 1. Files Modified vs. Created

### 1.1 Files to Create (New)

| File Path | What It Is | Section |
|---|---|---|
| `frontend/components/Vastu/VastuBadge.tsx` | Reusable Vastu tier badge component | §9.1 |

### 1.2 Files to Modify (Existing)

| File Path | What Changes | Section |
|---|---|---|
| `frontend/types/listing.ts` | Add `VastuTier` type + `vastuTier?: VastuTier` to `Listing` | §5 |
| `frontend/store/mapStore.ts` | Add `vastuCompliant: boolean` + `setVastuCompliant` action | §6 |
| `frontend/lib/api.ts` | Fix `facing` serialisation; add `vastuCompliant` → `facing` translation | §7 |
| `frontend/components/Filters/FilterBar.tsx` | Add "Vastu Friendly" toggle chip | §9.2 |
| `frontend/components/Listing/ListingCard.tsx` | Add VastuBadge render | §9.3 |
| `frontend/components/Listing/ListingSheet.tsx` | Add Vastu detail section | §9.4 |
| `frontend/components/Map/MapPopupCard.tsx` | Add VastuBadge render | §9.5 |
| `backend/main.py` | Add `facing` filter parameter + allowlist validation | §3 |

### 1.3 Files to Leave Untouched

| File Path | Reason |
|---|---|
| `frontend/lib/vastu.ts` | Already correctly written |
| `frontend/app/globals.css` | Vastu CSS variables already added correctly |
| `frontend/constants/filters.ts` | No changes needed for F099 |
| `data/migrations/*` | No DB changes for F099 |

---

## 2. Database Changes

**None required.**

The `facing` column (TEXT) already exists in the `listings` table and is
populated with the 8 compass direction values by the seed migration.
No ALTER TABLE, no new columns, no new indexes, no new migrations.

---

## 3. Backend: `backend/main.py`

### 3.1 Route Specification

**Route:** `GET /listings` (unchanged — do NOT rename to `/api/listings`)  
**Handler type:** `def` (synchronous) — do NOT use `async def`  
**Reason:** psycopg2 is a blocking library. FastAPI runs sync `def` handlers
in a threadpool automatically. Using `async def` with blocking I/O freezes
the event loop.

### 3.2 New Import Required

```python
from typing import List
from fastapi import Query
```

`List` and `Query` must be imported. If already imported for other parameters,
do not re-import — just confirm they are present.

### 3.3 New Constant: `VALID_FACING_VALUES`

Add this constant at module level (near other constants, before route handlers):

```python
VALID_FACING_VALUES: frozenset = frozenset({
    "North", "South", "East", "West",
    "North-East", "North-West", "South-East", "South-West"
})
```

Do not use a regular `set` — `frozenset` is immutable and appropriate for a
constant lookup table. Do not use a list — membership testing on frozenset
is O(1).

### 3.4 Route Handler Signature

Add `facing` as a new parameter to the existing `get_listings` handler.
Do not change any existing parameter names, types, or defaults.

```python
@app.get("/listings")
def get_listings(
    # --- existing parameters (do not change these) ---
    min_price: int = None,
    max_price: int = None,
    min_beds: int = None,
    max_beds: int = None,
    min_baths: int = None,
    max_baths: int = None,
    property_type: str = None,
    location: str = None,
    builder: str = None,
    bbox: str = None,
    # --- new parameter for F099 ---
    facing: List[str] = Query(default=[]),
):
```

**Critical:** `facing: List[str] = Query(default=[])` — the `Query(default=[])`
wrapper is required for FastAPI to correctly parse multi-value query parameters
(`?facing=East&facing=North`). Without `Query()`, FastAPI will not parse
repeated parameters as a list.

### 3.5 Validation Block

This validation block must appear as the **first executable statement** inside
the route handler body, before any SQL string construction:

```python
    # --- F099: facing filter validation ---
    if len(facing) > 8:
        raise HTTPException(
            status_code=400,
            detail="Too many facing values supplied"
        )
    for value in facing:
        if value not in VALID_FACING_VALUES:
            safe_value = value[:50]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid facing value: '{safe_value}'"
            )
    # --- end validation ---
```

**Why before SQL:** If validation runs after SQL string construction begins,
a malformed value could partially affect query assembly before being caught.
Validate first, build query second — always.

**Why `value[:50]`:** Caps the echoed value at 50 characters to prevent
response bloat from oversized inputs (security-checklist §4.3 requirement).

### 3.6 SQL Construction for `facing` Filter

After the validation block, in the section of the handler that builds the
`WHERE` clause, add the following. Do not modify existing WHERE clause logic:

```python
    # Add facing filter to WHERE clause (after existing filter logic)
    if facing:
        placeholders = ",".join(["%s"] * len(facing))
        where_clauses.append(f"facing IN ({placeholders})")
        params.extend(facing)
```

**Critical:** `params.extend(facing)` passes the values as psycopg2 parameters.
Do NOT interpolate facing values directly into the SQL string.

**Prohibited patterns** — Dev Agent must not use these:

```python
# PROHIBITED — SQL injection via f-string:
query += f"AND facing IN ({','.join(facing)})"

# PROHIBITED — SQL injection via string formatting:
query += "AND facing IN ('%s')" % "','".join(facing)

# PROHIBITED — SQL injection via concatenation:
query += "AND facing IN ('" + "','".join(facing) + "')"
```

### 3.7 Complete Handler Structure (pseudocode)

To ensure the validation and SQL construction are in the correct sequence:

```python
@app.get("/listings")
def get_listings(
    # ... all params including facing: List[str] = Query(default=[])
):
    # STEP 1: Validate facing (must be first)
    [validation block from §3.5]

    # STEP 2: Build WHERE clause
    where_clauses = []
    params = []

    # ... existing filter conditions (min_price, max_price, etc.) ...

    # STEP 3: Add facing to WHERE clause
    [SQL construction from §3.6]

    # STEP 4: Assemble and execute query
    query = "SELECT ... FROM listings"
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    # ... execute with psycopg2 cursor ...
    # ... return results ...
```

**The Dev Agent Backend must deliver the complete `backend/main.py` file.**
Truncated output is not acceptable. Code Review will reject truncated files.

---

## 4. API Contract

### 4.1 Endpoint

```
GET /listings
```

**Base URL (local):** `http://localhost:8000`
**Full URL example:** `http://localhost:8000/listings?facing=East&facing=North-East`

### 4.2 All Query Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_price` | integer | null | Minimum price in INR |
| `max_price` | integer | null | Maximum price in INR |
| `min_beds` | integer | null | Minimum bedrooms |
| `max_beds` | integer | null | Maximum bedrooms |
| `min_baths` | integer | null | Minimum bathrooms |
| `max_baths` | integer | null | Maximum bathrooms |
| `property_type` | string | null | One of the valid property type values |
| `location` | string | null | Neighbourhood name |
| `builder` | string | null | Builder name |
| `bbox` | string | null | Bounding box: `"minLng,minLat,maxLng,maxLat"` |
| `facing` | string[] | `[]` | **[NEW]** Multi-value. One or more of the 8 valid compass directions |

### 4.3 `facing` Parameter — Valid Values

```
"North" | "South" | "East" | "West" |
"North-East" | "North-West" | "South-East" | "South-West"
```

Case-sensitive. Any other value returns HTTP 400.

### 4.4 `facing` Parameter — Multi-Value Encoding

Correct (FastAPI `List[str]` semantics):
```
GET /listings?facing=East&facing=North-East&facing=North
```

Incorrect (must not be used — backend will return 400):
```
GET /listings?facing=East,North-East,North
```

### 4.5 Response Schema

The response schema is unchanged from the existing endpoint. Every listing
object already includes `facing` as a field. No new fields are added to
the response for F099.

```typescript
// Existing response — one object per listing
{
  id: string,
  title: string,
  price: number,
  beds: number,
  baths: number,
  area_sqft: number,
  address: string,
  neighborhood: string,
  city: string,
  lat: number,
  lng: number,
  property_type: string,
  builder: string | null,
  floor_no: number | null,
  total_floors: number | null,
  furnishing: string | null,
  facing: string | null,       // ← already present; Vastu tier derived from this
  parking: number | null,
  age_years: number | null,
  plot_area_sqft: number | null,
  rera_id: string | null,
  created_at: string
}
```

### 4.6 Error Responses

| HTTP Status | Condition | Body |
|---|---|---|
| `400 Bad Request` | Any `facing` value not in allowlist | `{"detail": "Invalid facing value: '<value>'"}`  |
| `400 Bad Request` | More than 8 `facing` values supplied | `{"detail": "Too many facing values supplied"}` |
| `500 Internal Server Error` | DB connection failure | `{"detail": "Internal server error"}` (existing behaviour) |

---

## 5. TypeScript Types: `frontend/types/listing.ts`

### 5.1 Exact Changes Required

**Add** the `VastuTier` type. **Add** `vastuTier` optional field to the
`Listing` interface. **Remove** all fields not listed below.

### 5.2 `VastuTier` Type (add this — new)

```typescript
export type VastuTier = 'excellent' | 'good' | 'neutral';
```

Add this at the top of the file, before the `Listing` interface.

### 5.3 `Listing` Interface (complete — exact)

The `Listing` interface must contain **exactly** these fields. Any field not
in this list must be removed. The Dev Agent added 16 out-of-scope fields in
the previous cycle — all must be removed.

```typescript
export interface Listing {
  id: string;
  title: string;
  price: number;
  beds: number;
  baths: number;
  area_sqft: number;
  address: string;
  neighborhood: string;
  city: string;
  lat: number;
  lng: number;
  property_type: string;
  builder: string | null;
  floor_no: number | null;
  total_floors: number | null;
  furnishing: string | null;
  facing: string | null;
  parking: number | null;
  age_years: number | null;
  plot_area_sqft: number | null;
  rera_id: string | null;
  created_at: string;
  // F099: client-side computed field — populated by api.ts after fetch
  vastuTier?: VastuTier;
}
```

**Explanation of `vastuTier` as optional:**  
`vastuTier` is a client-side derived field, not returned by the API.
It is computed in `api.ts` after the fetch and attached to each listing object
before being returned to the store. It is optional because listings with
`facing: null` will not have a `vastuTier` value.

**Fields explicitly excluded** (do not add these — they are not in the DB
schema and not in scope for F099):

`project_name`, `possession_status`, `possession_year`, `maintenance_monthly`,
`balconies`, `power_backup`, `water_supply`, `amenities`, `lift`,
`servant_room`, `gated_community`, `plot_width_ft`, `plot_length_ft`,
`road_facing_width_ft`, `plot_approval`, `corner_plot`, `listed_by`,
`vastu_compliant` (boolean), `formatPriceFull` (function)

### 5.4 Utility Functions (keep existing, do not add new ones)

Keep the existing `formatPrice` function unchanged. Do not add `formatPriceFull`
or any other utility functions not already in the pre-F099 file.

---

## 6. Zustand Store: `frontend/store/mapStore.ts`

### 6.1 Exact Changes Required

**Add** `vastuCompliant: boolean` to the `filters` object.  
**Add** `setVastuCompliant` action.  
**Remove** `vastuFilter`, `VastuFilter` type, and `setVastuFilter` — these
were added in error in the previous cycle. There is no prior Vastu feature
to be backward-compatible with.

### 6.2 Complete Store Shape (exact)

The store must have **exactly** this shape for the `filters` object.
No additional filter fields are to be added:

```typescript
filters: {
  priceMin: number | null;
  priceMax: number | null;
  beds: number | null;
  baths: number | null;
  location: string | null;
  propertyType: string | null;
  builder: string | null;
  vastuCompliant: boolean;   // F099 — default: false
}
```

### 6.3 `setVastuCompliant` Action (exact)

```typescript
setVastuCompliant: (value: boolean) => void;
```

Implementation inside the Zustand `create` call:

```typescript
setVastuCompliant: (value) =>
  set((state) => ({
    filters: { ...state.filters, vastuCompliant: value },
  })),
```

### 6.4 Initial State

```typescript
filters: {
  priceMin: null,
  priceMax: null,
  beds: null,
  baths: null,
  location: null,
  propertyType: null,
  builder: null,
  vastuCompliant: false,   // F099 default: off
},
```

### 6.5 `resetFilters` Action

The `resetFilters` action must reset `vastuCompliant` to `false`:

```typescript
resetFilters: () =>
  set((state) => ({
    filters: {
      priceMin: null,
      priceMax: null,
      beds: null,
      baths: null,
      location: null,
      propertyType: null,
      builder: null,
      vastuCompliant: false,
    },
  })),
```

### 6.6 What NOT to Include

Do not add:
- `vastuFilter` field of any type
- `VastuFilter` type definition
- `setVastuFilter` action
- `@deprecated` annotations on any Vastu-related state
- Any other Vastu-related state beyond `vastuCompliant: boolean`

The `countActiveFilters()` helper added in the previous cycle is acceptable
to keep if already present in the file, but must not count deprecated fields.
If `vastuCompliant` is `true`, it counts as 1 active filter.

---

## 7. API Client: `frontend/lib/api.ts`

### 7.1 Function: `fetchListings`

The `fetchListings` function is the only function that changes for F099.

### 7.2 Filters Parameter Type

The filters parameter must accept `vastuCompliant: boolean` from the store.
The function signature accepts the full filters object from the Zustand store:

```typescript
interface FetchListingsFilters {
  priceMin?: number | null;
  priceMax?: number | null;
  beds?: number | null;
  baths?: number | null;
  location?: string | null;
  propertyType?: string | null;
  builder?: string | null;
  vastuCompliant?: boolean;
}
```

### 7.3 URL Construction: `vastuCompliant` → `facing` Translation

When `vastuCompliant` is `true`, the function must translate it into the
`VASTU_FRIENDLY_FACING_VALUES` set and append each value as a separate
`facing` query parameter.

```typescript
import { VASTU_FRIENDLY_FACING_VALUES } from '@/lib/vastu';

// Inside fetchListings, after building existing params:

if (filters.vastuCompliant) {
  // Translate vastuCompliant boolean into facing multi-value filter
  VASTU_FRIENDLY_FACING_VALUES.forEach((direction) => {
    params.append('facing', direction);
  });
}
```

**`VASTU_FRIENDLY_FACING_VALUES`** is exported from `frontend/lib/vastu.ts`
(already written). It contains: `['East', 'North-East', 'North', 'North-West', 'West']`

**Critical:** Use `params.append()` not `params.set()`.  
`params.set('facing', ...)` would overwrite previous values, and calling it
multiple times would still only keep the last value.  
`params.append('facing', direction)` adds each direction as a separate
query string entry: `?facing=East&facing=North-East&facing=North&facing=North-West&facing=West`

This matches FastAPI's `List[str] = Query(default=[])` parsing.

### 7.4 `vastuTier` Population After Fetch

After the API response is received and parsed, before returning the listings
array, compute and attach `vastuTier` to each listing:

```typescript
import { getVastuTier } from '@/lib/vastu';

// After const listings = await response.json():
const listingsWithVastu = listings.map((listing: Listing) => ({
  ...listing,
  vastuTier: listing.facing ? getVastuTier(listing.facing) ?? undefined : undefined,
}));

return listingsWithVastu;
```

**`getVastuTier`** is exported from `frontend/lib/vastu.ts` (already written).
It accepts a facing string and returns `VastuTier | null`.

The `?? undefined` converts null (unrecognised facing value) to undefined,
which is correct for the optional `vastuTier?: VastuTier` field.

### 7.5 What NOT to Add

Do not add:
- A raw `facing` filter path in `fetchListings` (the store uses `vastuCompliant`
  boolean, not raw facing strings — the translation happens in `api.ts`)
- Any new exported functions from `api.ts` beyond what already exists