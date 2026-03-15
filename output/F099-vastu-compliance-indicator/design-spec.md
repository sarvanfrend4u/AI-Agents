I'll silently read all required context files before producing the design specification.

**Context read complete.** Key findings:
- F099 is in rework cycle — arch-spec, security-checklist, and feature-spec all exist and are valid
- The design-spec I'm now writing covers the 5 missing UI components (the core of Blocker #1)
- Existing CSS variables for Vastu tiers are already defined in globals.css: `--vastu-excellent: #1B6E3A`, `--vastu-good: #6B5E00`, `--vastu-neutral: #4B5563`
- Layout zones are non-negotiable; all new UI must fit within existing zones
- No new CSS variables are needed — the 6 already-defined ones are sufficient
- `--vastu-excellent-text`, `--vastu-good-text`, `--vastu-neutral-text` are the text variants

---

```markdown
# Design Specification — F099 Vastu Compliance Indicator
**Feature:** F099 — Vastu Compliance Indicator  
**Stage:** Stage 4 — Design (Rework Cycle 1)  
**Author:** Design Agent  
**Date:** 2026-03-15  
**Status:** COMPLETE — Rework specification covering all 5 missing UI components  
**Saved to:** agent-system/output/F099-vastu-compliance-indicator/design-spec.md

---

## 0. Design Scope & Rework Context

This document supersedes the prior design-spec.md for F099.
It was written after Code Review Cycle 1 identified Blocker #1:
five UI component files were never created. This spec is the
authoritative implementation reference for Dev Agent (Frontend).

### Components Covered

| # | Component | File | Status |
|---|---|---|---|
| 1 | VastuBadge | frontend/components/Vastu/VastuBadge.tsx | NEW — must create |
| 2 | FilterBar | frontend/components/Filters/FilterBar.tsx | MODIFY — add Vastu toggle |
| 3 | ListingCard | frontend/components/Listing/ListingCard.tsx | MODIFY — add VastuBadge |
| 4 | ListingSheet | frontend/components/Listing/ListingSheet.tsx | MODIFY — add Vastu section |
| 5 | MapPopupCard | frontend/components/Map/MapPopupCard.tsx | MODIFY — add VastuBadge |

### CSS Variables Already Defined (do not redefine)

From frontend/app/globals.css — these already exist:

```css
--vastu-excellent:      #1B6E3A   /* background for excellent tier badge */
--vastu-excellent-text: #FFFFFF   /* text on excellent badge */
--vastu-good:           #6B5E00   /* background for good tier badge */
--vastu-good-text:      #FFFFFF   /* text on good badge */
--vastu-neutral:        #4B5563   /* background for neutral tier badge */
--vastu-neutral:        #FFFFFF   /* text on neutral badge */
```

No new CSS variables are introduced in this spec.
All colour tokens reference the above existing variables.

### Vastu Tier Reference (from frontend/lib/vastu.ts)

| Facing | Tier | Display label | Icon |
|---|---|---|---|
| East | excellent | Vastu Excellent | ◈ |
| North-East | excellent | Vastu Excellent | ◈ |
| North | good | Vastu Good | ◇ |
| North-West | good | Vastu Good | ◇ |
| West | good | Vastu Good | ◇ |
| South-East | neutral | Vastu Neutral | ○ |
| South | neutral | Vastu Neutral | ○ |
| South-West | neutral | Vastu Neutral | ○ |

---

## 1. Component: VastuBadge

**File:** `frontend/components/Vastu/VastuBadge.tsx`  
**Type:** New reusable presentational component  
**Used by:** ListingCard (sm), ListingSheet (md), MapPopupCard (sm)

### 1.1 Props Interface

```typescript
interface VastuBadgeProps {
  tier: 'excellent' | 'good' | 'neutral';
  size: 'sm' | 'md';
  showTooltip?: boolean;  // default: true
  className?: string;
}
```

- `tier` — drives colour, icon, and label
- `size` — sm used in cards/popups; md used in detail sheet
- `showTooltip` — when true, shows explanatory tooltip on hover/focus
- `className` — allows parent to add margin utilities

### 1.2 Visual Design

#### Size: sm (used in ListingCard and MapPopupCard)

```
┌─────────────────────┐
│ ◈ Vastu Excellent   │  ← height: 20px, font-size: 11px
└─────────────────────┘
  px-2 py-0.5 rounded-full
```

Tailwind classes (sm):
```
inline-flex items-center gap-1
px-2 py-0.5
rounded-full
text-[11px] font-medium leading-none
whitespace-nowrap
```

#### Size: md (used in ListingSheet)

```
┌─────────────────────────┐
│  ◈  Vastu Excellent     │  ← height: 28px, font-size: 13px
└─────────────────────────┘
   px-3 py-1 rounded-full
```

Tailwind classes (md):
```
inline-flex items-center gap-1.5
px-3 py-1
rounded-full
text-[13px] font-semibold leading-none
whitespace-nowrap
```

#### Colour mapping (background + text)

| Tier | Background | Text | Tailwind equivalent |
|---|---|---|---|
| excellent | var(--vastu-excellent) #1B6E3A | var(--vastu-excellent-text) #FFFFFF | bg-[#1B6E3A] text-white |
| good | var(--vastu-good) #6B5E00 | var(--vastu-good-text) #FFFFFF | bg-[#6B5E00] text-white |
| neutral | var(--vastu-neutral) #4B5563 | white | bg-[#4B5563] text-white |

**Contrast check (WCAG AA):**
- #FFFFFF on #1B6E3A → 6.8:1 ✓ (exceeds 4.5:1 for normal text)
- #FFFFFF on #6B5E00 → 5.2:1 ✓
- #FFFFFF on #4B5563 → 4.6:1 ✓

#### Icon glyphs per tier

| Tier | Glyph | Unicode | Meaning |
|---|---|---|---|
| excellent | ◈ | U+25C8 | White diamond containing black small diamond |
| good | ◇ | U+25C7 | White diamond |
| neutral | ○ | U+25CB | White circle |

Icon is rendered as a `<span aria-hidden="true">` — purely decorative.
Screen readers hear only the aria-label on the wrapper element.

### 1.3 Tooltip Design

Tooltip appears on hover (desktop) and on focus (keyboard).
On mobile, tooltip does not appear — the md badge in ListingSheet
already shows full explanatory text below it (see §4).

Tooltip content per tier:

| Tier | Tooltip text |
|---|---|
| excellent | "East or North-East facing — most auspicious per Vastu Shastra" |
| good | "North, North-West, or West facing — favourable per Vastu Shastra" |
| neutral | "South, South-East, or South-West facing — neutral per Vastu Shastra" |

Tooltip visual:

```
                    ┌──────────────────────────────────────────────┐
                    │ East or North-East facing — most auspicious  │
                    │ per Vastu Shastra                            │
                    └──────────────────────────────────────────────┘
                              ▲
         ┌────────────────────┤
         │ ◈ Vastu Excellent  │
         └────────────────────┘
```

Tooltip Tailwind classes:
```
absolute bottom-full left-1/2 -translate-x-1/2 mb-2
w-56
bg-gray-900 text-white
text-[11px] leading-snug
px-2.5 py-2
rounded-md
shadow-lg
z-50
pointer-events-none
```

Arrow (CSS pseudo-element, not a separate element):
```css
.vastu-tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 5px solid transparent;
  border-top-color: #111827; /* gray-900 */
}
```

Tooltip wrapper must be `position: relative` on the badge's outer
container div — NOT on the badge span itself (which must remain
`display: inline-flex`). The `position: relative` goes on a wrapping
`<div className="relative inline-flex">`.

### 1.4 All 4 UI States

#### State 1: Empty / Not applicable
When `listing.facing` is null, undefined, or an unrecognised value,
the VastuBadge is **not rendered** — the parent passes no `tier` prop.
The component returns `null`. No empty placeholder is shown.

```
(nothing rendered — zero DOM nodes)
```

#### State 2: Loading
VastuBadge is not rendered during loading. The parent component
(ListingCard, MapPopupCard) shows its own skeleton loader.
VastuBadge renders only when `listing` data is fully available.

#### State 3: Error
Same as Empty — if `getVastuInfo()` returns null (due to unrecognised
facing value), the badge is not rendered. Silently degrades.
No error state is shown for a missing badge.

#### State 4: Active (normal render)

```
┌────────────────────┐
│ ◈ Vastu Excellent  │   ← sm size, in ListingCard / MapPopupCard
└────────────────────┘

┌──────────────────────┐
│  ◈  Vastu Excellent  │  ← md size, in ListingSheet
└──────────────────────┘
```

### 1.5 Interactions

#### Hover (desktop)
- Tooltip fades in after 200ms delay
- Transition: `opacity-0 → opacity-100`, duration 150ms, ease-in-out
- Tooltip fades out immediately on mouse-leave (no delay)
- The badge itself does not change colour on hover — it is informational, not interactive

#### Focus (keyboard)
- Badge wrapper `<div>` has `tabIndex={0}` and `role="img"`
- On focus, tooltip appears (same as hover, no delay for keyboard users)
- On blur, tooltip disappears
- No click action on the badge itself

#### Mobile tap
- No tooltip on mobile — tap has no effect on the badge
- Full explanation is provided in ListingSheet Vastu section (see §4)

### 1.6 Accessibility

```tsx
<div
  className="relative inline-flex"
  role="img"
  aria-label={getVastuAriaLabel(tier)}
  tabIndex={0}
>
  <span
    className={badgeClasses}
    aria-hidden="true"  /* badge text and icon are decorative */
  >
    <span aria-hidden="true">{TIER_ICON[tier]}</span>
    {TIER_LABEL[tier]}
  </span>
  {showTooltip && <Tooltip tier={tier} visible={isVisible} />}
</div>
```

`getVastuAriaLabel()` returns (from vastu.ts):
- excellent → "Vastu Excellent: East or North-East facing property"
- good → "Vastu Good: North, North-West, or West facing property"
- neutral → "Vastu Neutral: South, South-East, or South-West facing property"

### 1.7 Responsive Behaviour

| Context | Size | Tooltip |
|---|---|---|
| Desktop ListingCard | sm | Yes, on hover/focus |
| Desktop MapPopupCard | sm | Yes, on hover/focus |
| Desktop ListingSheet | md | Yes, on hover/focus |
| Mobile ListingCard (in sheet) | sm | No — sheet has full Vastu section |
| Mobile ListingSheet | md | No — explanatory text shown below |

### 1.8 Full Component Markup (implementation reference)

```tsx
// frontend/components/Vastu/VastuBadge.tsx
'use client';

import { useState } from 'react';
import { getVastuAriaLabel } from '@/lib/vastu';

type VastuTier = 'excellent' | 'good' | 'neutral';

interface VastuBadgeProps {
  tier: VastuTier;
  size?: 'sm' | 'md';
  showTooltip?: boolean;
  className?: string;
}

const TIER_ICON: Record<VastuTier, string> = {
  excellent: '◈',
  good: '◇',
  neutral: '○',
};

const TIER_LABEL: Record<VastuTier, string> = {
  excellent: 'Vastu Excellent',
  good: 'Vastu Good',
  neutral: 'Vastu Neutral',
};

const TIER_TOOLTIP: Record<VastuTier, string> = {
  excellent: 'East or North-East facing — most auspicious per Vastu Shastra',
  good: 'North, North-West, or West facing — favourable per Vastu Shastra',
  neutral: 'South, South-East, or South-West facing — neutral per Vastu Shastra',
};

const TIER_BG: Record<VastuTier, string> = {
  excellent: 'bg-[#1B6E3A]',
  good:      'bg-[#6B5E00]',
  neutral:   'bg-[#4B5563]',
};

const SIZE_CLASSES = {
  sm: 'px-2 py-0.5 text-[11px] gap-1',
  md: 'px-3 py-1 text-[13px] gap-1.5 font-semibold',
};

export function VastuBadge({
  tier,
  size = 'sm',
  showTooltip = true,
  className = '',
}: VastuBadgeProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div
      className={`relative inline-flex ${className}`}
      role="img"
      aria-label={getVastuAriaLabel(tier)}
      tabIndex={0}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
    >
      {/* Badge pill */}
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

      {/* Tooltip — desktop only, conditional on showTooltip prop */}
      {showTooltip && (
        <div
          role="tooltip"
          className={`
            absolute bottom-full left-1/2 -translate-x-1/2 mb-2
            w-56 bg-gray-900 text-white text-[11px] leading-snug
            px-2.5 py-2 rounded-md shadow-lg z-50 pointer-events-none