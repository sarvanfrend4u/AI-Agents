# Design Spec — {{FEATURE_NAME}} ({{FEATURE_ID}})

**Version:** v1
**Date:** {{DATE}}
**Status:** DRAFT → APPROVED
**Prepared by:** Design Agent

---

## User Flow

[Step-by-step description of how the user interacts with this feature from start to finish]

1. User does X
2. System shows Y
3. User does Z
4. Outcome

---

## Component Breakdown

### New Components

| Component | File Path | Description |
|---|---|---|
| ComponentName | frontend/components/... | What it does |

### Modified Components

| Component | File Path | What Changes |
|---|---|---|
| ComponentName | frontend/components/... | What changes and why |

---

## Desktop Layout

```
[ASCII or text mockup of desktop layout — 1280px+]
Example:
+------------------+-------------------+
|   MAP AREA       |  LISTING CARD     |
|                  |  [ Vastu badge ]  |
|                  |  ₹ 85L  3BHK      |
+------------------+-------------------+
```

---

## Mobile Layout

```
[ASCII or text mockup of mobile layout — <768px]
Example:
+---------------------------+
|  LISTING CARD             |
|  [ Vastu badge ]          |
|  ₹ 85L  3BHK  East facing |
+---------------------------+
```

---

## States

### Empty State
[What does the user see when there is no data to show?]

### Loading State
[What does the user see while data is loading?]

### Error State
[What does the user see if something goes wrong?]

### Active / Success State
[What does the user see when the feature is working correctly?]

---

## Interaction Patterns

[Hover effects, click behaviours, animations, transitions — be specific]

---

## Visual Design Tokens

[Colours, typography, spacing — reference existing Tailwind classes and CSS variables only]

Example:
- Badge background: `bg-green-100`
- Badge text: `text-green-800`
- Badge border: `border border-green-300`

---

## Accessibility Notes

- ARIA labels required: [list them]
- Keyboard navigation: [describe expected behaviour]
- Colour contrast: [confirm WCAG AA compliance]

---

## Notes for Dev Agents

[Anything specific the frontend developer needs to know that is not obvious from the mockups]
