# Architecture Spec — {{FEATURE_NAME}} ({{FEATURE_ID}})

**Version:** v1
**Date:** {{DATE}}
**Status:** DRAFT → APPROVED
**Prepared by:** Architecture Agent (Pass 2)

---

## Database Schema Changes

### New Tables
```sql
-- None OR SQL here
```

### Modified Tables
```sql
-- ALTER TABLE statements here OR "None"
```

### New Indexes
```sql
-- CREATE INDEX statements here OR "None"
```

### Migration File
- File: `data/migrations/XXX_description.sql`
- Contents: [describe what the migration does]

---

## API Endpoints

### New Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| GET | /api/v1/... | Yes / No | [what it returns] |

### Modified Endpoints

| Method | Path | What Changes |
|---|---|---|
| GET | /api/v1/... | [description of change] |

### Request Schema
```json
{
  "field": "type — description"
}
```

### Response Schema
```json
{
  "field": "type — description"
}
```

---

## Backend Service Layer

### New Functions

| Function | File | Description |
|---|---|---|
| function_name() | backend/api/... | What it does |

### Modified Functions

| Function | File | What Changes |
|---|---|---|
| function_name() | backend/api/... | What changes |

---

## Frontend Component Architecture

### New Components

| Component | File | Props | Description |
|---|---|---|---|
| ComponentName | frontend/components/... | prop: type | What it renders |

### Modified Components

| Component | File | What Changes |
|---|---|---|
| ComponentName | frontend/components/... | What changes |

---

## State Management Changes

[Changes to Zustand store in frontend/store/mapStore.ts — or "None"]

```typescript
// New fields added to store shape, if any
```

---

## Type Changes

[Changes to frontend/types/listing.ts or other type files — or "None"]

```typescript
// New or modified TypeScript interfaces
```

---

## API Client Changes

[Changes to frontend/lib/api.ts — or "None"]

```typescript
// New or modified fetch functions
```

---

## Third-Party Libraries

[New open-source libraries required — must be free and open-source only]

| Library | Purpose | License | Install Command |
|---|---|---|---|
| library-name | what it does | MIT / Apache / etc. | npm install library-name |

---

## Performance Considerations

[Caching strategy, query optimisation, lazy loading, memoisation — or "None required"]

---

## Error Handling Strategy

[How errors are caught at each layer, surfaced to the user, and logged]

---

## Notes for Dev Agents

[Specific implementation guidance — anything not obvious from the spec above]

**Tech stack reminder (non-negotiable):**
- Frontend: Next.js 14, TypeScript, Tailwind CSS, MapLibre GL JS, Zustand
- Backend: FastAPI (Python), psycopg2, PostGIS
- DB: PostgreSQL 15 + PostGIS
- No paid APIs, no proprietary services
- Deployment is OUT OF SCOPE — do not add deployment config
