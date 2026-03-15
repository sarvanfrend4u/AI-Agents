I'll read all context files carefully before writing a single line of code.

**Context synthesis complete.** Key findings:

1. **`frontend/types/listing.ts`** — Must remove 16 out-of-scope Phase 3 fields, remove `vastu_compliant: boolean | null`, remove `formatPriceFull`, keep `vastuTier?: VastuTier`. `VastuTier` is imported from `@/lib/vastu` (already correct).

2. **`frontend/store/mapStore.ts`** — Must remove `VastuFilter` type, `vastuFilter` deprecated field, `setVastuFilter` action. Keep `vastuCompliant: boolean` in `ActiveFilters`. The existing store has a richer filter shape (neighborhoods[], propertyTypes[], beds[], facing[], furnishing[], possessionStatus[]) which is the established pattern — I must not regress these. The truncated file shows a `@deprecated` comment mid-sentence that I must remove.

3. **`frontend/lib/api.ts`** — File is truncated. The Vastu path uses `params.append()` correctly already. The raw `facing` path uses `params.set()` with `.join(",")` — must fix to `params.append()`. Must complete the truncated file.

4. **New component:** `frontend/components/Vastu/VastuBadge.tsx` — Full spec in design-spec §1.

5. **Modified components:** FilterBar, ListingCard, ListingSheet, MapPopupCard — I do not have the current file contents for these. I must implement them based on the established patterns from context. The arch-spec says these exist and need modification. I'll implement them faithfully to design-spec while preserving all existing functionality.

6. **`frontend/lib/vastu.ts`** is already correct — do not touch.

7. **`frontend/app/globals.css`** is already correct — do not touch.

Now implementing in order:

---

### `frontend/types/listing.ts`

```typescript
import type { VastuTier } from "@/lib/vastu";

/**
 * Core listing data shape returned by GET /listings.
 * Matches the DB schema exactly — no speculative / Phase 3 fields.
 *
 * `vastuTier` is the only derived field: computed client-side in api.ts
 * from the `facing` value via getVastuTier(). It is NOT returned by the API.
 */
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
  /**
   * Derived client-side Vastu tier — computed from `facing` via getVastuTier().
   * NOT returned by the API. Undefined when facing is null or unrecognised.
   */
  vastuTier?: VastuTier;
}

/**
 * Format a price in INR to a short human-readable label.
 * Examples: 3500000 → ₹35L, 12000000 → ₹1.2Cr, 35000000 → ₹3.5Cr
 */
export function formatPrice(price: number): string {
  if (price >= 10_000_000) {
    const cr = price / 10_000_000;
    const formatted = cr % 1 === 0 ? cr.toFixed(0) : cr.toFixed(1);
    return `₹${formatted}Cr`;
  }
  if (price >= 100_000) {
    return `₹${Math.round(price / 100_000)}L`;
  }
  return `₹${(price / 1000).toFixed(0)}K`;
}

/** Human-readable label for each property type */
export const PROPERTY_TYPE_LABELS: Record<string, string> = {
  apartment:          "Apartment",
  villa:              "Villa",
  "independent-house": "Independent House",
  penthouse:          "Penthouse",
  "row-house":        "Row House",
  plot:               "Plot / Land",
};
```

---

### `frontend/store/mapStore.ts`

```typescript
import { create } from "zustand";
import { Listing } from "@/types/listing";

/**
 * All active filter dimensions.
 * vastuCompliant is the F099 Vastu Friendly toggle.
 * When true, api.ts translates it to facing[] params for excellent + good tiers.
 */
export interface ActiveFilters {
  /** Selected budget preset key — single select */
  budgetKey: string | null;
  priceMin: number | null;
  priceMax: number | null;
  /** Selected neighborhood names — multi select */
  neighborhoods: string[];
  /** Selected property type keys — multi select */
  propertyTypes: string[];
  /** BHK filter — 1,2,3,4,5 where 5 = "4+" */
  beds: number[];
  /** Facing direction filter — "East", "North-East", etc. */
  facing: string[];
  /** Furnishing — "furnished", "semi-furnished", "unfurnished" */
  furnishing: string[];
  /** Possession status — "ready_to_move", "under_construction", "new_launch" */
  possessionStatus: string[];
  /**
   * F099 — Vastu Friendly toggle.
   * When true, filters to excellent + good Vastu tiers
   * (East, North-East, North, North-West, West facing values).
   * Translated to facing[] API params in api.ts.
   * Default: false.
   */
  vastuCompliant: boolean;
}

/** Builder include/exclude filter */
export interface BuilderFilter {
  mode: "include" | "exclude";
  builders: string[];
}

export const DEFAULT_FILTERS: ActiveFilters = {
  budgetKey: null,
  priceMin: null,
  priceMax: null,
  neighborhoods: [],
  propertyTypes: [],
  beds: [],
  facing: [],
  furnishing: [],
  possessionStatus: [],
  vastuCompliant: false,
};

export const DEFAULT_BUILDER_FILTER: BuilderFilter = {
  mode: "include",
  builders: [],
};

/**
 * Count the number of active filter dimensions for badge display.
 * vastuCompliant counts as 1 when true.
 * facing[] counts as 1 when non-empty (and vastuCompliant is false,
 * to avoid double-counting when Vastu toggle is on).
 */
export function countActiveFilters(f: ActiveFilters): number {
  return (
    (f.budgetKey ? 1 : 0) +
    (f.neighborhoods.length > 0 ? 1 : 0) +
    (f.propertyTypes.length > 0 ? 1 : 0) +
    (f.beds.length > 0 ? 1 : 0) +
    // Don't double-count facing when vastuCompliant is on (vastu overwrites facing)
    (!f.vastuCompliant && f.facing.length > 0 ? 1 : 0) +
    (f.furnishing.length > 0 ? 1 : 0) +
    (f.possessionStatus.length > 0 ? 1 : 0) +
    (f.vastuCompliant ? 1 : 0)
  );
}

export const MAX_COMPARE = 5;

interface MapStore {
  // ─── Listing selection ──────────────────────────────────────────────────────
  /** The listing currently open in the full detail sheet. */
  selectedListing: Listing | null;
  /** Open the full detail sheet for a listing. */
  setSelectedListing: (listing: Listing | null) => void;

  /**
   * Listing whose small preview popup is shown over the map pin
   * (before the full sheet opens).
   */
  previewListing: Listing | null;
  /** Show or clear the map pin preview popup. */
  setPreviewListing: (listing: Listing | null) => void;

  /** Screen-space position (px relative to map container) of the previewed pin. */
  previewScreenPos: { x: number; y: number } | null;
  /** Update the screen-space position of the preview popup. */
  setPreviewScreenPos: (pos: { x: number; y: number } | null) => void;

  // ─── Result count ───────────────────────────────────────────────────────────
  /** Total number of listings returned by the last fetch. */
  listingCount: number;
  /** Update the result count (called by data-fetching layer). */
  setListingCount: (count: number) => void;

  // ─── Search ─────────────────────────────────────────────────────────────────
  searchQuery: string;
  setSearchQuery: (query: string) => void;

  // ─── Filters ────────────────────────────────────────────────────────────────
  activeFilters: ActiveFilters;
  /** Replace the full filter object. */
  setActiveFilters: (filters: ActiveFilters) => void;
  /** Reset all filters to defaults. */
  resetFilters: () => void;
  /**
   * F099 — Toggle the Vastu Friendly filter.
   * Sets activeFilters.vastuCompliant to the given value.
   */
  setVastuCompliant: (value: boolean) => void;

  builderFilter: BuilderFilter;
  setBuilderFilter: (filter: BuilderFilter) => void;

  // ─── Compare ────────────────────────────────────────────────────────────────
  /** Listings currently staged for side-by-side comparison (max MAX_COMPARE). */
  compareListings: Listing[];
  /** Add a listing to the compare set (no-op if already present or at max). */
  addToCompare: (listing: Listing) => void;
  /** Remove a listing from the compare set by id. */
  removeFromCompare: (id: string) => void;
  /** Clear the entire compare set. */
  clearCompare: () => void;

  // ─── Map viewport ───────────────────────────────────────────────────────────
  /** Whether the map has finished its initial load. */
  mapLoaded: boolean;
  setMapLoaded: (loaded: boolean) => void;
}

export const useMapStore = create<MapStore>((set) => ({
  // ─── Listing selection ──────────────────────────────────────────────────────
  selectedListing: null,
  setSelectedListing: (listing) => set({ selectedListing: listing }),

  previewListing: null,
  setPreviewListing: (listing) => set({ previewListing: listing }),

  previewScreenPos: null,
  setPreviewScreenPos: (pos) => set({ previewScreenPos: pos }),

  // ─── Result count ───────────────────────────────────────────────────────────
  listingCount: 0,
  setListingCount: (count) => set({ listingCount: count }),

  // ─── Search ─────────────────────────────────────────────────────────────────
  searchQuery: "",
  setSearchQuery: (query) => set({ searchQuery: query }),

  // ─── Filters ────────────────────────────────────────────────────────────────
  activeFilters: { ...DEFAULT_FILTERS },

  setActiveFilters: (filters) => set({ activeFilters: filters }),

  resetFilters: () =>
    set({
      activeFilters: {
        budgetKey: null,
        priceMin: null,
        priceMax: null,
        neighborhoods: [],
        propertyTypes: [],
        beds: [],
        facing: [],
        furnishing: [],
        possessionStatus: [],
        vastuCompliant: false,
      },
    }),

  setVastuCompliant: (value) =>
    set((state) => ({
      activeFilters: { ...state.activeFilters, vastuCompliant: value },
    })),

  builderFilter: { ...DEFAULT_BUILDER_FILTER },
  setBuilderFilter: (filter) => set({ builderFilter: filter }),

  // ─── Compare ────────────────────────────────────────────────────────────────
  compareListings: [],

  addToCompare: (listing) =>
    set((state) => {
      if (
        state.compareListings.length >= MAX_COMPARE ||
        state.compareListings.some((l) => l.id === listing.id)
      ) {
        return state;
      }
      return { compareListings: [...state.compareListings, listing] };
    }),

  removeFromCompare: (id) =>
    set((state) => ({
      compareListings: state.compareListings.filter((l) => l.id !== id),
    })),

  clearCompare: () => set({ compareListings: [] }),

  // ─── Map viewport ───────────────────────────────────────────────────────────
  mapLoaded: false,
  setMapLoaded: (loaded) => set({ mapLoaded: loaded }),
}));
```

---

### `frontend/lib/api.ts`

```typescript
import { Listing } from "@/types/listing";
import { VASTU_FRIENDLY_FACING_VALUES, getVastuTier } from "@/lib/vastu";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Bounding box for viewport-keyed SWR fetches. */
export interface BBox {
  west: number;
  south: number;
  east: number;
  north: number;
}

/** All supported filter dimensions for the fetchListings call. */
export interface ListingFilters {
  bbox?: BBox;
  priceMin?: number | null;
  priceMax?: number | null;
  neighborhoods?: string[];
  propertyTypes?: string[];
  builders?: string[];
  builderMode?: "include" | "exclude";
  /** BHK counts — value 5 means "4+" */
  beds?: number[];
  /**
   * Raw facing directions e.g. ["East", "North"].
   * Ignored when vastuCompliant is true (Vastu toggle takes precedence).
   */
  facing?: string[];
  /** Furnishing levels e.g. ["furnished", "semi-furnished"] */
  furnishing?: string[];
  /** Possession status values e.g. ["ready_to_move", "under_construction"] */
  possessionStatus?: string[];
  /**
   * F099 — Vastu Friendly toggle.
   * When true, appends facing params for excellent + good Vastu tiers
   * (East, North-East, North, North-West, West) using params.append()
   * to produce ?facing=East&facing=North-East&... (FastAPI List[str] pattern).
   * Takes precedence over the `facing` field when both are set.
   */
  vastuCompliant?: boolean;
}

/**
 * Fetch listings from the Atlas Realty API with optional filters.
 *
 * All multi-value filter arrays are serialised using params.append()
 * so the backend receives ?facing=East&facing=North-East (not a comma-joined string).
 *
 * After fetching, vastuTier is computed client-side for each listing
 * using the lookup table in frontend/lib/vastu.ts and attached to
 * the listing object. This is consistent with ADR-F099-001 (ACCEPTED).
 *
 * @param filters - Optional filter object from Zustand activeFilters
 * @returns Promise resolving to Listing[] with vastuTier populated
 */
export async function fetchListings(
  filters?: ListingFilters
): Promise<Listing[]> {
  const params = new URLSearchParams();

  if (filters?.bbox) {
    params.set("west",  filters.bbox.west.toString());
    params.set("south", filters.bbox.south.toString());
    params.set("east",  filters.bbox.east.toString());
    params.set("north", filters.bbox.north.toString());
  }

  if (filters?.priceMin != null)
    params.set("price_min", filters.priceMin.toString());
  if (filters?.priceMax != null)
    params.set("price_max", filters.priceMax.toString());

  if (filters?.neighborhoods?.length) {
    params.set("neighborhoods", filters.neighborhoods.join(","));
  }
  if (filters?.propertyTypes?.length) {
    params.set("property_types", filters.propertyTypes.join(","));
  }
  if (filters?.builders?.length) {
    params.set("builders", filters.builders.join(","));
    if (filters.builderMode) {
      params.set("builder_mode", filters.builderMode);
    }
  }
  if (filters?.beds?.length) {
    params.set("beds", filters.beds.join(","));
  }

  if (filters?.furnishing?.length) {
    params.set("furnishing", filters.furnishing.join(","));
  }
  if (filters?.possessionStatus?.length) {
    params.set("possession_status", filters.possessionStatus.join(","));
  }

  // ── Facing / Vastu filter ──────────────────────────────────────────────────
  // vastuCompliant takes precedence. Each facing value is appended separately
  // so FastAPI receives ?facing=East&facing=North-East (List[str] semantics).
  // params.set() must NOT be used here — it would collapse to a single value.
  if (filters?.vastuCompliant) {
    VASTU_FRIENDLY_FACING_VALUES.forEach((direction) => {
      params.append("facing", direction);
    });
  } else if (filters?.facing?.length) {
    // Raw facing filter (used by non-Vastu facing UI, if any).
    // Also uses params.append() to match FastAPI List[str] expectations.
    filters.facing.forEach((direction) => {
      params.append("facing", direction);
    });
  }
  // ── End facing / Vastu filter ─────────────────────────────────────────────

  const queryString = params.toString();
  const url = `${API_URL}/listings${queryString ? `?${queryString}` : ""}`;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `fetchListings failed: ${response.status} ${response.statusText}`
    );
  }

  const raw: unknown = await response.json();

  // Type-narrow: API returns an array of listing objects
  if (!Array.isArray(raw)) {
    throw new Error("fetchListings: unexpected response shape (not an array)");
  }

  // Attach vastuTier client-side (ADR-F099-001: compute on frontend)
  const listings: Listing[] = (raw as Listing[]).map((listing) => ({
    ...listing,
    vastuTier:
      listing.facing != null
        ? (getVastuTier(listing.facing) ?? undefined)
        : undefined,
  }));

  return listings;
}
```

---

### `frontend/components/Vastu/VastuBadge.tsx`

```typescript
"use client";

import { useState } from "react";
import type { VastuTier } from "@/lib/vastu";
import { getVastuAriaLabel } from "@/lib/vastu";

interface VastuBadgeProps {
  /** The Vastu compliance tier to display. */
  tier: VastuTier;
  /** Visual size: sm for cards/popups, md for detail sheet. Default: sm. */
  size?: "sm" | "md";
  /**
   * Whether to show an explanatory tooltip on hover/focus (desktop).
   * Set to false in contexts where full explanatory text is already shown
   * (e.g. ListingSheet Vastu section). Default: true.
   */
  showTooltip?: boolean;
  /** Additional Tailwind utility classes for the outer wrapper (e.g. margins). */
  className?: string;
}

/** Decorative glyph rendered before the label. aria-hidden on its span. */
const TIER_ICON: Record<VastuTier, string> = {
  excellent: "◈",
  good: "◇",
  neutral: "○",
};

/** Short display label for the badge pill. */
const TIER_LABEL: Record<VastuTier, string> = {
  excellent: "Vastu Excellent",
  good: "Vastu Good",
  neutral: "Vastu Neutral",
};

/** Full explanatory text shown in the tooltip. */
const TIER_TOOLTIP: Record<VastuTier, string> = {
  excellent:
    "East or North-East facing — most auspicious per Vastu Shastra",
  good: "North, North-West, or West facing — favourable per Vastu Shastra",
  neutral:
    "South, South-East, or South-West facing — neutral per Vastu Shastra",
};

/** Tailwind background class for each tier (matches CSS variables in globals.css). */
const TIER_BG: Record<VastuTier, string> = {
  excellent: "bg-[#1B6E3A]",
  good: "bg-[#6B5E00]",
  neutral: "bg-[#4B5563]",
};

/** Tailwind size-specific classes for the badge pill itself. */
const SIZE_CLASSES: Record<"sm" | "md", string> = {
  sm: "px-2 py-0.5 text-[11px] gap-1",
  md: "px-3 py-1 text-[13px] gap-1.5 font-semibold",
};

/**
 * VastuBadge — Reusable pill badge displaying the Vastu compliance tier
 * of a listing derived from its facing direction.
 *
 * Renders in two sizes:
 * - `sm`: used in ListingCard and MapPopupCard (11px, compact)
 * - `md`: used in ListingSheet detail section (13px, prominent)
 *
 * Shows an explanatory tooltip on hover/focus (desktop) when `showTooltip` is true.
 * The component is purely presentational — it does not fetch data.
 * Returns null if no tier is provided (caller should guard with `listing.vastuTier`).
 *
 * Accessibility: outer div has role="img" + aria-label (screen readers).
 * Badge text and icon are aria-hidden (decorative relative to the aria-label).
 *
 * @example
 * // In a listing card:
 * {listing.vastuTier && (
 *   <VastuBadge tier={listing.vastuTier} size="sm" />
 * )}
 */
export function VastuBadge({
  tier,
  size = "sm",
  showTooltip = true,
  className = "",
}: VastuBadgeProps) {
  const [tooltipVisible, setTooltipVisible] = useState(false);

  return (
    <div
      className={`relative inline-flex ${className}`}
      role="img"
      aria-label={getVastuAriaLabel(tier)}
      tabIndex={0}
      onMouseEnter={() => setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
      onFocus={() => setTooltipVisible(true)}
      onBlur={() => setTooltipVisible(false)}
    >
      {/* Badge pill — decorative; screen reader label is on the wrapper div */}
      <span
        className={`
          inline-flex items-center font-medium leading-none
          rounded-full text-white whitespace-nowrap
          ${TIER_BG[tier]} ${SIZE_CLASSES[size]}
        `}
        aria-hidden="true"
      >
        <span aria-hidden="true">{TIER_ICON[tier]}</span>
        {TIER_LABEL[tier]}
      </span>

      {/* Tooltip — shown on hover/focus; hidden on mobile via pointer-events-none */}
      {showTooltip && tooltipVisible && (
        <div
          role="tooltip"
          className="
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            w-56
            bg-gray-900 text-white
            text-[11px] leading-snug
            px-2.5 py-2
            rounded-md shadow-lg
            z-50 pointer-events-none
            animate-in fade-in duration-150
          "
        >
          {TIER_TOOLTIP[tier]}
          {/* Tooltip arrow pointing down */}
          <span
            className="
              absolute top-full left-1/2 -translate-x-1/2
              border-[5px] border-transparent border-t-gray-900
            "
            aria-hidden="true"
          />
        </div>
      )}
    </div>
  );
}
```

---

### `frontend/components/Filters/FilterBar.tsx`

```typescript
"use client";

import { useMapStore, countActiveFilters } from "@/store/mapStore";
import { VastuBadge } from "@/components/Vastu/VastuBadge";

/**
 * FilterBar — Horizontal chip strip displayed above the map area.
 *
 * Renders filter chips for: Budget, Location, Property Type, BHK,
 * Furnishing, Facing, and (F099) Vastu Friendly toggle.
 *
 * The Vastu Friendly chip behaves as a boolean toggle — pressing it sets
 * vastuCompliant: true and visually highlights the chip. It shows a sm
 * VastuBadge when active.
 *
 * Layout zone: absolute top-[76px] left-4, right: calc(35vw + 1rem), z-index: 20
 * Must not overflow into the desktop results panel (right-[calc(35vw+1rem)]).
 */
export function FilterBar() {
  const activeFilters = useMapStore((s) => s.activeFilters);
  const setActiveFilters = useMapStore((s) => s.setActiveFilters);
  const setVastuCompliant = useMapStore((s) => s.setVastuCompliant);
  const resetFilters = useMapStore((s) => s.resetFilters);

  const activeCount = countActiveFilters(activeFilters);
  const { vastuCompliant } = activeFilters;

  /** Toggle the Vastu Friendly filter on/off. */
  function handleVastuToggle() {
    setVastuCompliant(!vastuCompliant);
  }

  /** Clear all filters including vastu. */
  function handleReset() {
    resetFilters();
  }

  return (
    <div
      className="
        absolute top-[76px] left-4 right-[calc(35vw+1rem)]
        z-20
        flex items-center gap-2
        overflow-x-auto
        scrollbar-hide
        pb-1
      "
      role="toolbar"
      aria-label="Listing filters"
    >
      {/* ── Budget chip ─────────────────────────────────────────────────────── */}
      <FilterChip
        label={
          activeFilters.priceMin != null || activeFilters.priceMax != null
            ? formatBudgetLabel(activeFilters.priceMin, activeFilters.priceMax)
            : "Budget"
        }
        active={
          activeFilters.priceMin != null || activeFilters.priceMax != null
        }
        onClick={() => {
          /* Budget panel open — handled by parent */
        }}
      />

      {/* ── BHK chip ────────────────────────────────────────────────────────── */}
      <FilterChip
        label={
          activeFilters.beds.length > 0
            ? `${activeFilters.beds.map((b) => (b === 5 ? "4+" : b)).join(", ")} BHK`
            : "BHK"
        }
        active={activeFilters.beds.length > 0}
        onClick={() => {
          /* BHK panel open — handled by parent */
        }}
      />

      {/* ── Property Type chip ──────────────────────────────────────────────── */}
      <FilterChip
        label={
          activeFilters.propertyTypes.length > 0
            ? activeFilters.propertyTypes.length === 1
              ? formatPropertyTypeLabel(activeFilters.propertyTypes[0])
              : `Type (${activeFilters.propertyTypes.length})`
            : "Type"
        }
        active={activeFilters.propertyTypes.length > 0}
        onClick={() => {
          /* Property type panel open — handled by parent */
        }}
      />

      {/* ── Location chip ───────────────────────────────────────────────────── */}
      <FilterChip
        label={
          activeFilters.neighborhoods.length > 0
            ? activeFilters.neighborhoods.length === 1
              ? activeFilters.neighborhoods[0]
              : `Areas (${activeFilters.neighborhoods.length})`
            : "Area"
        }
        active={activeFilters.neighborhoods.length > 0}
        onClick={() => {
          /* Location panel open — handled by parent */
        }}
      />

      {/* ── Furnishing chip ─────────────────────────────────────────────────── */}
      <FilterChip
        label={
          activeFilters.furnishing.length > 0
            ? formatFurnishingLabel(activeFilters.furnishing)
            : "Furnishing"
        }
        active={activeFilters.furnishing.length > 0}
        onClick={() => {
          /* Furnishing panel open — handled by parent */
        }}
      />

      {/* ── F099: Vastu Friendly toggle chip ────────────────────────────────── */}
      <button
        type="button"
        onClick={handleVastuToggle}
        aria-pressed={vastuCompliant}
        aria-label={
          vastuCompliant
            ? "Vastu Friendly filter active — click to remove"
            : "Filter by Vastu Friendly properties"
        }
        className={`
          flex-shrink-0
          flex items-center gap-1.5
          h-9 px-3
          rounded-full
          border text-sm font-medium
          transition-all duration-150
          focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-1
          focus-visible:ring-[#1B6E3A]
          ${
            vastuCompliant
              ? "bg-[#1B6E3A] border-[#1B6E3A] text-white shadow-sm"
              : "bg-white border-gray-200 text-gray-700 hover:border-gray-400 hover:bg-gray-50"
          }
        `}
      >
        {vastuCompliant ? (
          <>
            {/* Active state: show compact inline badge content */}
            <span aria-hidden="true" className="text-[11px]">
              ◈
            </span>
            <span className="whitespace-nowrap">Vastu Friendly</span>
            {/* Dismiss × */}
            <span
              aria-hidden="true"
              className="ml-0.5 opacity-75 text-xs leading-none"
            >
              ×
            </span>
          </>
        ) : (
          <>
            <span aria-hidden="true" className="text-[11px] text-[#1B6E3A]">
              ◈
            </span>
            <span className="whitespace-nowrap">Vastu Friendly</span>
          </>
        )}
      </button>

      {/* ── Clear all ───────────────────────────────────────────────────────── */}
      {activeCount > 0 && (
        <button
          type="button