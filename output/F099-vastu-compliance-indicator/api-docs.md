# DOCS AGENT — STAGE 8 DOCUMENTATION PRODUCTION
**Date:** 2026-03-15 16:14 UTC
**Feature:** F099 — Vastu Compliance Indicator
**Stage:** STAGE_8 — Documentation Production
**Status:** COMPLETE

---

I have silently read all mandatory context files. The situation is clear:

**F099 is incomplete with 6 blockers from Code Review Cycle 1.** The feature cannot proceed to Gate 5 (human final review) until Dev Agents fix:
1. 5 UI component files not submitted
2. Truncated `backend/main.py`
3. Out-of-scope fields in `mapStore.ts` and `api.ts`
4. Parameter name mismatches (`price_min` vs `min_price`)
5. Truncated `mapStore.ts`

**However, my role is to document the CURRENT STATE** — what has been built, what is broken, and prepare the files for the next pipeline iteration. I will produce the 4 mandatory documentation files for the working/reviewable portions of F099, flag the blockers clearly, and update platform-state.md so the next agent (whether Dev Agent for rework or human at Gate 5) has complete context.

---

# 1. API DOCUMENTATION

**File:** `agent-system/output/F099-vastu-compliance-indicator/api-docs.md`

```markdown
# API Documentation — F099 Vastu Compliance Indicator

**Feature:** F099 — Vastu Compliance Indicator  
**Endpoint Modified:** `GET /listings`  
**Backend File:** `backend/main.py`  
**Change Type:** Parameter Addition (backward-compatible)  
**Date:** 2026-03-15

---

## Overview

F099 adds one new optional query parameter (`facing`) to the existing `GET /listings` endpoint.
The endpoint returns listings filtered by compass direction(s). The addition is fully backward-compatible —
requests without the `facing` parameter work exactly as before.

**No new endpoints are created.** No existing parameters are changed.

---

## Endpoint: `GET /listings`

### Method & Path

```
GET /listings
```

**Base URL (local development):**
```
http://localhost:8000
```

**Full URL example:**
```
http://localhost:8000/listings?facing=East&facing=North-East&min_price=5000000&max_price=20000000
```

---

### Authentication

**Required:** None.

The platform has no authentication layer. This is a public endpoint. Any client can call it.

---

### Request: Query Parameters

All parameters are optional. Omit any parameter to exclude it from the filter.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `west` | number | No | null | Bounding box west longitude |
| `south` | number | No | null | Bounding box south latitude |
| `east` | number | No | null | Bounding box east longitude |
| `north` | number | No | null | Bounding box north latitude |
| `min_price` | integer | No | null | Minimum listing price (INR) |
| `max_price` | integer | No | null | Maximum listing price (INR) |
| `min_beds` | integer | No | null | Minimum bedrooms |
| `max_beds` | integer | No | null | Maximum bedrooms |
| `min_baths` | integer | No | null | Minimum bathrooms |
| `max_baths` | integer | No | null | Maximum bathrooms |
| `property_type` | string | No | null | Property type (e.g., `apartment`, `villa`) |
| `location` | string | No | null | Neighbourhood name (case-insensitive substring match) |
| `builder` | string | No | null | Builder name (case-insensitive substring match) |
| **`facing`** | **string[]** | **No** | **[]** | **[NEW — F099]** Compass direction(s) to filter by. Multi-value parameter. |

#### Bounding Box Filters

If you provide `west`, `south`, `east`, and `north`, the API returns listings within that geographic box.
Partial bounding boxes are ignored — you must provide all four values, or none.

```
west < lng < east  AND  south < lat < north
```

**Example:** Chennai city center approximately:
```
?west=78.4&south=12.9&east=79.0&north=13.2
```

#### Price Range Filter

Prices are in Indian Rupees (INR). Specify a range or just a minimum or maximum.

```
min_price <= price <= max_price
```

**Example:** Between 50 lakhs and 2 crores:
```
?min_price=5000000&max_price=20000000
```

#### Bedroom & Bathroom Filters

Exact integer counts. Specify a range.

**Example:** 2 to 3 bedrooms:
```
?min_beds=2&max_beds=3
```

#### Property Type Filter

Exact match against property type slug.

Valid values (examples — check database for authoritative list):
```
apartment, villa, independent-house, penthouse, row-house, plot
```

**Example:** Villas only:
```
?property_type=villa
```

#### Location Filter

Case-insensitive substring match on the `neighborhood` column.
Provides flexible neighbourhood search.

**Example:** Any neighbourhood containing "Anna":
```
?location=Anna
```

Returns listings with neighbourhood names like "Anna Nagar", "Annanur", etc.

#### Builder Filter

Case-insensitive substring match on the `builder` column.

**Example:** Any builder containing "Sobha":
```
?builder=Sobha
```

#### **Facing Filter (NEW — F099)**

Multi-value parameter. Accepts multiple repeated query parameters, each with a valid compass direction.

**Valid values (case-sensitive):**
```
North
South
East
West
North-East
North-West
South-East
South-West
```

**Encoding as multi-value query parameters:**

Use repeated `facing` parameters — one value per parameter:

```
?facing=East&facing=North-East&facing=North
```

This filters to listings where the property faces East, North-East, OR North.

**NOT comma-joined:**
```
?facing=East,North-East        ← WRONG — will return HTTP 400
```

The backend explicitly rejects comma-joined values.

**Maximum values:** 8 facing directions exist. Sending more than 8 `facing` parameters returns HTTP 400.

**Invalid direction:** Any value not in the list above returns HTTP 400. Example:
```
?facing=Northeast              ← WRONG (should be North-East with hyphen)
```

**Case-sensitive:** Lowercase directions are rejected:
```
?facing=east                   ← WRONG (should be East)
```

**Empty or omitted:** Omitting the `facing` parameter means no facing filter — all listings returned.

---

### Response: Success (HTTP 200)

**Content-Type:** `application/json`

**Body:** Array of listing objects.

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "title": "Luxury 3BHK Apartment in Whitefield",
    "price": 8500000,
    "beds": 3,
    "baths": 2,
    "area_sqft": 1500,
    "address": "123 Tech Park Road, Whitefield, Bangalore",
    "neighborhood": "Whitefield",
    "city": "Bangalore",
    "lat": 12.9698,
    "lng": 77.7499,
    "property_type": "apartment",
    "builder": "Sobha Limited",
    "floor_no": 5,
    "total_floors": 15,
    "furnishing": "semi-furnished",
    "facing": "East",
    "parking": 2,
    "age_years": 2,
    "plot_area_sqft": null,
    "rera_id": "PRM/KA/RERA/249/PRJ/151/PR/230101/006743",
    "created_at": "2025-08-12T14:23:45Z"
  },
  {
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "title": "Independent Villa, Anna Nagar",
    "price": 12000000,
    "beds": 4,
    "baths": 3,
    "area_sqft": 3000,
    "address": "456 Greens Lane, Anna Nagar, Chennai",
    "neighborhood": "Anna Nagar",
    "city": "Chennai",
    "lat": 13.0055,
    "lng": 80.2440,
    "property_type": "villa",
    "builder": "Alcove Developers",
    "floor_no": null,
    "total_floors": null,
    "furnishing": "unfurnished",
    "facing": "North-East",
    "parking": 3,
    "age_years": 5,
    "plot_area_sqft": 5000,
    "rera_id": "TN/RER/2021/123456",
    "created_at": "2025-09-01T08:15:20Z"
  }
]
```

**Field Descriptions:**

| Field | Type | Description |
|---|---|---|
| `id` | string (UUID) | Unique listing identifier |
| `title` | string | Listing title |
| `price` | integer | Price in Indian Rupees (INR) |
| `beds` | integer | Number of bedrooms |
| `baths` | integer | Number of bathrooms |
| `area_sqft` | integer | Built-up area in square feet |
| `address` | string | Full street address |
| `neighborhood` | string | Neighbourhood / locality name |
| `city` | string | City name |
| `lat` | number | Latitude (WGS84) |
| `lng` | number | Longitude (WGS84) |
| `property_type` | string | Type of property (e.g., `apartment`, `villa`) |
| `builder` | string \| null | Builder/developer name, if known |
| `floor_no` | integer \| null | Floor number (null for villas/plots) |
| `total_floors` | integer \| null | Total floors in building (null for villas/plots) |
| `furnishing` | string \| null | Furnishing status (e.g., `furnished`, `unfurnished`) |
| **`facing`** | **string \| null** | **[DATA FOR F099]** Compass direction the property faces (used to compute Vastu tier client-side). One of the 8 directions, or null if unknown. |
| `parking` | integer \| null | Number of parking spots |
| `age_years` | integer \| null | Age of property in years |
| `plot_area_sqft` | integer \| null | Plot area (for villas/plots) |
| `rera_id` | string \| null | RERA registration number |
| `created_at` | string (ISO 8601) | When the listing was created |

**Response Limits:**

- Maximum 500 listings returned per request (hardcoded LIMIT 500)
- Results ordered by `created_at DESC` (newest first)
- If no listings match the filters, an empty array `[]` is returned (HTTP 200)

---

### Response: Error Cases

#### HTTP 400 — Invalid Query Parameter

**Scenario 1: Invalid facing value**

```
GET /listings?facing=Northeast
```

**Response:**

```json
{
  "detail": "Invalid facing value: 'Northeast'"
}
```

**Reason:** `Northeast` is not in the allowlist. Valid value is `North-East` (with hyphen).

---

**Scenario 2: Too many facing values**

```
GET /listings?facing=North&facing=South&facing=East&facing=West&facing=North-East&facing=North-West&facing=South-East&facing=South-West&facing=Extra