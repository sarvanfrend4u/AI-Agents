# Code Review — Vastu Compliance Indicator (F099)
**Cycle:** 1
**Verdict:** CHANGES REQUIRED

---

## Blockers (must fix before proceeding)

### BLOCKER 1 — `backend/main.py` — File Truncated (Security Unverifiable)

- **`backend/main.py`:~line 130+`** — The file is truncated at `get_listings` docstring. The complete route body — including the validation block (`VALID_FACING_VALUES` check), the WHERE clause construction, and the parameterised SQL execution — is not present in the submitted code.
- **Security-checklist §3.1 requires:** allowlist validation loop before SQL construction, parameterised `IN` clause, rejection on first invalid value, truncation of echoed value to 50 chars.
- **None of these can be verified** because the route body was not submitted.
- **Exact fix required:** Submit the complete, untruncated `backend/main.py`. No `... (truncated)` comments are permitted. The entire file must be present including the full `get_listings` body through to `return JSONResponse(...)`.

---

### BLOCKER 2 — `frontend/store/mapStore.ts` — File Truncated + Out-of-Scope Fields

- **`frontend/store/mapStore.ts`:~line 210+** — The file is truncated mid-`resetFilters` body (`propert...`). The complete store implementation — including `resetFilters` completion, `setVastuCompliant`, `setBuilderFilter`, `compareListings`, `addToCompare`, `removeFromCompare`, `clearCompare`, `mapLoaded`, `setMapLoaded` — is not present.
- **Cannot verify** that `vastuCompliant` is correctly reset to `false` in `resetFilters`, or that deprecated `vastuFilter` / `setVastuFilter` were not included later in the file.
- **Exact fix required:** Submit the complete, untruncated `frontend/store/mapStore.ts`.

- **`frontend/store/mapStore.ts`:lines ~14–31`** — `ActiveFilters` interface contains out-of-scope fields not authorised by arch-spec §6.2:
  - `facing: string[]` — arch-spec §6.2 does not include a raw `facing` array in the store; the store uses `vastuCompliant: boolean` and `api.ts` handles the translation. A raw `facing` field in the store duplicates the concern and creates dual-state confusion (the same problem as Cycle 0's `vastuFilter`).
  - `furnishing: string[]` — not in scope for F099; not in the arch-spec §6.2 store shape.
  - `possessionStatus: string[]` — not in scope for F099; not in DB schema; Phase 3 field.
  - `budgetKey: string | null` — not in the arch-spec §6.2 store shape for F099.
  - `neighborhoods: string[]` — not in arch-spec §6.2.
  - `propertyTypes: string[]` — not in arch-spec §6.2.
  - `beds: number[]` — arch-spec §6.2 defines `beds: number | null`, not an array.
- **Arch-spec §0 "Dev Agent Rule"** explicitly states: "Do not add state not listed in §6 (Zustand store)."
- **Exact fix required:** `ActiveFilters` must contain exactly the fields listed in arch-spec §6.2:
  ```typescript
  { priceMin, priceMax, beds: number | null, baths: number | null,
    location: string | null, propertyType: string | null,
    builder: string | null, vastuCompliant: boolean }
  ```
  All other fields (`facing`, `furnishing`, `possessionStatus`, `budgetKey`, `neighborhoods`, `propertyTypes`, `beds[]`) must be removed.

- **`frontend/store/mapStore.ts`:lines ~60–75`** — `countActiveFilters()` references out-of-scope fields (`budgetKey`, `neighborhoods`, `propertyTypes`, `beds[]`, `furnishing`, `possessionStatus`, `facing`). If these fields are removed from `ActiveFilters` as required, this function must be updated to only reference the fields that exist in the corrected interface.

---

### BLOCKER 3 — `frontend/lib/api.ts` — Out-of-Scope Parameters Serialised

- **`frontend/lib/api.ts`:lines ~65–95`** — `ListingFilters` interface and `fetchListings` body serialise multiple out-of-scope parameters not authorised by arch-spec §7:
  - `neighborhoods` → `params.set("neighborhoods", ...)` — not in arch-spec; not a valid backend parameter per platform-state.md
  - `propertyTypes` → `params.set("property_types", ...)` — not a valid backend parameter per API contract §4.2
  - `builders` / `builderMode` → `params.set("builders", ...)` / `params.set("builder_mode", ...)` — not in F099 scope; not authorised parameters per API contract §4.2
  - `furnishing` → `params.set("furnishing", ...)` — Phase 3 field, not in DB schema
  - `possessionStatus` → `params.set("possession_status", ...)` — Phase 3 field, not in DB schema
  - `beds` as array → `params.set("beds", beds.join(","))` — arch-spec §4.2 defines `min_beds`/`max_beds` integers, not an array of BHK values
- These calls will either be silently ignored by the backend (causing confusion) or cause 422 Unprocessable Entity errors if the backend validates parameter names.
- **Arch-spec §7.5:** "Do not add a raw `facing` filter path in `fetchListings`... Do not add any new exported functions from `api.ts` beyond what already exists." The same principle applies to not adding parameters beyond what the spec authorises.
- **Exact fix required:** `ListingFilters` and `fetchListings` must only serialise the parameters listed in arch-spec §4.2: `bbox` (west/south/east/north), `min_price`, `max_price`, `min_beds`, `max_beds`, `min_baths`, `max_baths`, `property_type`, `location`, `builder`, and `facing` (via `vastuCompliant` translation or raw). Remove all other parameter serialisation.

---

### BLOCKER 4 — Five UI Component Files Not Submitted

- **`frontend/components/Filters/FilterBar.tsx`** — Not submitted. Required by arch-spec §1.2 and design-spec §2. Must add "Vastu Friendly" toggle chip.
- **`frontend/components/Listing/ListingCard.tsx`** — Not submitted. Required by arch-spec §1.2 and design-spec §3. Must integrate `VastuBadge` (size sm).
- **`frontend/components/Listing/ListingSheet.tsx`** — Not submitted. Required by arch-spec §1.2 and design-spec §4. Must add Vastu detail section.
- **`frontend/components/Map/MapPopupCard.tsx`** — Not submitted. Required by arch-spec §1.2 and design-spec §5. Must integrate `VastuBadge` (size sm).
- **Exact fix required:** All four files must be submitted in full. This is the core visible UI of F099. The feature is invisible to users without these files.

*(Note: `VastuBadge.tsx` was submitted and is reviewed under Approved Items.)*

---

### BLOCKER 5 — `frontend/lib/api.ts` — `price_min`/`price_max` Parameter Name Mismatch

- **`frontend/lib/api.ts`:lines ~58–62`** — The function serialises:
  ```typescript
  params.set("price_min", filters.priceMin.toString());
  params.set("price_max", filters.priceMax.toString());
  ```
  But the existing backend parameter names per platform-state.md and arch-spec §4.2 are `min_price` and `max_price`.
- **Exact fix required:** Change `"price_min"` → `"min_price"` and `"price_max"` → `"max_price"` to match the existing backend signature.

---

### BLOCKER 6 — `frontend/store/mapStore.ts` — `BuilderFilter` / `previewScreenPos` / Non-Spec Store Shape

- **`frontend/store/mapStore.ts`:lines ~35–130`** — The store shape submitted contains many fields not listed in arch-spec §6 and not authorised for F099:
  - `previewListing: Listing | null` and `setPreviewListing` — not in arch-spec §6
  - `previewScreenPos` and `setPreviewScreenPos` — not in arch-spec §6
  - `listingCount: number` and `setListingCount` — not in arch-spec §6
  - `BuilderFilter` interface and `builderFilter` state — not in arch-spec §6
  - `compareListings`, `addToCompare`, `removeFromCompare`, `clearCompare` — not in arch-spec §6
  - `mapLoaded` / `setMapLoaded` — not in arch-spec §6
- The arch-spec §6 store shape for F099 is a targeted addition to an **existing** store. The Dev Agent appears to have rewritten the entire store from scratch rather than adding only `vastuCompliant: boolean` and `setVastuCompliant` to the existing file.
- **Exact fix required:** The Dev Agent must not rewrite the entire store. The only permitted changes to `mapStore.ts` are:
  1. Add `vastuCompliant: boolean` (default `false`) to the existing `filters` or `ActiveFilters` object
  2. Add `setVastuCompliant: (value: boolean) => void` action
  3. Ensure `resetFilters` sets `vastuCompliant` back to `false`
  4. Remove `vastuFilter`, `VastuFilter` type, and `setVastuFilter` if they were present from Cycle 0
  - All other existing store fields must be preserved exactly as they were before F099

---

## Suggestions (non-blocking)

- **`frontend/components/Vastu/VastuBadge.tsx`:line ~95`** — The tooltip uses `animate-in fade-in duration-150` which requires the `tailwindcss-animate` plugin. If this plugin is not installed, the classes are silently ignored (tooltip still shows, just without the fade). Worth confirming the plugin is in `tailwind.config.ts` before merge, but not a blocker since degradation is graceful.

- **`frontend/lib/api.ts`:line ~107`** — The type narrowing `const raw: unknown = await response.json()` followed by `raw as Listing[]` cast is correct in intent but the `as Listing[]` cast at line ~118 bypasses the `unknown` safety. A proper guard (checking `Array.isArray(raw)` is present at line ~113, which is good) would be strengthened by also spot-checking at least one known field (e.g. `'id' in raw[0]`) before casting. Low-priority suggestion.

- **`frontend/store/mapStore.ts`** — `countActiveFilters` is a reasonable addition (noted as acceptable in arch-spec §6.6). Once the out-of-scope fields are removed and the function is corrected to only reference the spec-compliant `ActiveFilters` fields, it can be kept.

- **`frontend/types/listing.ts`** — Clean and correct. Minor note: `VastuTier` is imported from `@/lib/vastu` rather than re-declared locally. This is the correct pattern and avoids type duplication.

---

## Approved Items

- **`frontend/types/listing.ts`** — Fully compliant. `Listing` interface matches arch-spec §5.3 exactly. All 16 out-of-scope Phase 3 fields from Cycle 0 have been correctly removed. `vastuTier?: VastuTier` is present and optional. `VastuTier` correctly imported from `@/lib/vastu`. `formatPrice` and `PROPERTY_TYPE_LABELS` preserved unchanged. No `vastu_compliant: boolean` field (correctly absent). No `formatPriceFull`. ✅

- **`frontend/components/Vastu/VastuBadge.tsx`** — Fully compliant with design-spec §1 and arch-spec §9.1. All checked items:
  - `VastuBadgeProps` interface matches spec exactly (`tier`, `size`, `showTooltip`, `className`) ✅
  - `TIER_ICON`, `TIER_LABEL`, `TIER_TOOLTIP`, `TIER_BG`, `SIZE_CLASSES` all match design-spec §1.2 ✅
  - `role="img"` + `aria-label={getVastuAriaLabel(tier)}` on outer wrapper ✅
  - `tabIndex={0}` for keyboard focus ✅
  - Badge `<span>` has `aria-hidden="true"` ✅
  - Icon `<span>` has `aria-hidden="true"` ✅
  - Tooltip arrow implemented as inline `<span>` with border trick ✅
  - `position: relative` is on the outer `<div>` wrapper, NOT on the badge `<span>` — CSS rules not violated ✅
  - No `position: relative` on `.price-pin` ✅
  - No `dangerouslySetInnerHTML` ✅
  - No `any` types ✅
  - JSDoc comment on exported function ✅
  - File is PascalCase ✅
  - `"use client"` directive present ✅
  - `showTooltip && tooltipVisible` guard prevents tooltip render when prop is false ✅
  - All 4 UI states handled: component returns a render (active), parent guards with `listing.vastuTier &&` (empty/loading/error) per design-spec §1.4 ✅

- **`frontend/lib/api.ts` — Vastu-specific logic** — The facing/Vastu serialisation block is correct:
  - `vastuCompliant` → `VASTU_FRIENDLY_FACING_VALUES.forEach(d => params.append("facing", d))` ✅
  - Raw `facing` fallback also uses `params.append()` not `params.set()` ✅ (Blocker #2 from Cycle 0 resolved)
  - `vastuCompliant` takes precedence over raw `facing` via the if/else structure ✅
  - `vastuTier` population via `getVastuTier(listing.facing) ?? undefined` pattern is correct ✅
  - URL construction uses `/listings` (not `/api/listings`) ✅ (Blocker #3 from Cycle 0 resolved)
  - `Array.isArray(raw)` guard before cast ✅
  - JSDoc comment on exported function ✅
  - No `any` types in the Vastu-specific code ✅

- **`backend/main.py` — reviewable portion** — The following is verified from the visible code:
  - Route declared as `def get_listings(...)` (synchronous) — not `async def` ✅ (Blocker #7 from Cycle 0 resolved)
  - Route path is `@app.get("/listings")` — not `/api/listings` ✅ (Blocker #3 from Cycle 0 resolved)
  - `VALID_FACING_VALUES: frozenset[str]` constant declared at module level with all 8 valid values ✅
  - `facing: List[str] = Query(default=[])` in handler signature ✅
  - Explanatory comment in handler documenting why `def` not `async def` is used ✅
  - `List` and `Query` imported from correct modules ✅
  - `get_connection` and `wait_for_db` have docstrings with Args/Returns ✅
  - No hardcoded DB credentials (uses `os.environ.get`) ✅

---

## Summary

**6 blockers identified. 4 from omission (UI files not submitted, files truncated), 2 from out-of-scope additions.**

The Vastu-specific logic that was submitted (`VastuBadge.tsx`, `types/listing.ts`, the facing serialisation in `api.ts`, the `VALID_FACING_VALUES` constant and route signature in `backend/main.py`)