# Architecture Constraints — {{FEATURE_NAME}} ({{FEATURE_ID}})

**Version:** v1
**Date:** {{DATE}}
**Status:** COMPLETE
**Prepared by:** Architecture Agent (Pass 1)

---

## Modules Touched

- [ ] search
- [ ] listings
- [ ] ai
- [ ] agents
- [ ] users
- [ ] admin
- [ ] shared/database
- [ ] shared/auth
- [ ] frontend/components/Map
- [ ] frontend/components/Listing
- [ ] frontend/components/Filters
- [ ] frontend/store
- [ ] frontend/types
- [ ] frontend/lib/api

---

## Existing Patterns to Follow

[List specific files, functions, or patterns already in the codebase that this feature must follow.
Example: "Follow the filter pattern in frontend/constants/filters.ts — add new filter type here first"]

---

## Constraints for Design Agent

[What the Design Agent must respect: existing layout zones, component boundaries, z-index rules, responsive breakpoints, etc.
Example: "Filter chips live in FilterBar.tsx — do not create a new UI zone for this"]

---

## Constraints for Security Agent

[Data sensitivity notes, auth boundaries, any external API exposure flags the Security Agent must assess.
Example: "This feature reads from the DB but writes nothing — read-only risk profile"]

---

## Risk Flags

[Any architectural concerns raised at this early stage.
Example: "None" or "The facing field in DB has inconsistent casing — normalise before logic runs"]

---

## Notes

[Anything else Design and Security agents need to know before they start]
