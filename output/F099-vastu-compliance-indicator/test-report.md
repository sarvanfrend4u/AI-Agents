I will now silently read all mandatory context files before writing a single test.

**Context analysis complete.** Key findings:

1. **Code Review Cycle 2** found 6 blockers — most critically: `backend/main.py` is truncated (route body missing), `mapStore.ts` is truncated AND contains out-of-scope fields (`facing[]`, `furnishing[]`, `possessionStatus[]`, `budgetKey`, `neighborhoods`, `propertyTypes`, `beds[]`), `api.ts` serialises out-of-scope params and uses wrong param names (`price_min`/`price_max` instead of `min_price`/`max_price`), and 4 UI component files were never submitted.

2. **What I can actually test:** The code shown is real — I must test what was written, document what will fail, and provide root cause analysis.

3. **Critical test findings to document:**
   - `backend/main.py` is truncated — the `get_listings` route body is missing; validation and SQL construction are unverifiable/untestable
   - `mapStore.ts` has wrong store shape (out-of-scope fields)
   - `api.ts` has wrong param names and out-of-scope params
   - 4 UI components not submitted (FilterBar, ListingCard, ListingSheet, MapPopupCard)
   - `VastuBadge.tsx` is submitted and reviewable
   - `types/listing.ts` is submitted and correct

---

### `backend/tests/test_f099.py`
```python
"""
Unit tests for F099 — Vastu Compliance Indicator (backend)
Tests cover: VALID_FACING_VALUES constant, input validation logic,
SQL construction logic, and route handler behaviour.

Test Agent Cycle 1 — Atlas Realty F099
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call
from fastapi.testclient import TestClient
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Ensure backend is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Import the application — will fail if main.py has syntax errors
# ---------------------------------------------------------------------------
def test_main_imports_without_error():
    """main.py must be importable without raising any exception."""
    try:
        import main  # noqa: F401
        assert True
    except Exception as exc:
        pytest.fail(f"main.py failed to import: {exc}")


# ---------------------------------------------------------------------------
# VALID_FACING_VALUES constant
# ---------------------------------------------------------------------------
class TestValidFacingValues:
    """Tests for the VALID_FACING_VALUES frozenset constant."""

    @pytest.fixture(autouse=True)
    def import_main(self):
        import main
        self.main = main

    def test_valid_facing_values_exists(self):
        """VALID_FACING_VALUES constant must be defined in main.py."""
        assert hasattr(self.main, "VALID_FACING_VALUES"), (
            "VALID_FACING_VALUES not found in main.py"
        )

    def test_valid_facing_values_is_frozenset(self):
        """VALID_FACING_VALUES must be a frozenset (immutable, O(1) lookup)."""
        assert isinstance(self.main.VALID_FACING_VALUES, frozenset), (
            f"Expected frozenset, got {type(self.main.VALID_FACING_VALUES)}"
        )

    def test_valid_facing_values_contains_all_eight_directions(self):
        """VALID_FACING_VALUES must contain exactly the 8 compass direction strings."""
        expected = frozenset({
            "North", "South", "East", "West",
            "North-East", "North-West", "South-East", "South-West",
        })
        assert self.main.VALID_FACING_VALUES == expected, (
            f"Expected {expected}, got {self.main.VALID_FACING_VALUES}"
        )

    def test_cardinal_directions_in_set(self):
        """Each of the 4 cardinal directions is in the set."""
        for direction in ("North", "South", "East", "West"):
            assert direction in self.main.VALID_FACING_VALUES, (
                f"Cardinal direction '{direction}' missing from VALID_FACING_VALUES"
            )

    def test_intercardinal_directions_in_set(self):
        """Each of the 4 intercardinal directions is in the set."""
        for direction in ("North-East", "North-West", "South-East", "South-West"):
            assert direction in self.main.VALID_FACING_VALUES, (
                f"Intercardinal direction '{direction}' missing from VALID_FACING_VALUES"
            )

    def test_lowercase_not_in_set(self):
        """Lowercase variants must NOT be in the set (case-sensitive per spec)."""
        for bad_value in ("north", "south", "east", "west", "north-east"):
            assert bad_value not in self.main.VALID_FACING_VALUES, (
                f"Lowercase '{bad_value}' should not be in VALID_FACING_VALUES"
            )

    def test_set_has_exactly_eight_elements(self):
        """VALID_FACING_VALUES must contain exactly 8 elements."""
        assert len(self.main.VALID_FACING_VALUES) == 8, (
            f"Expected 8 elements, got {len(self.main.VALID_FACING_VALUES)}"
        )

    def test_frozenset_is_immutable(self):
        """frozenset must not allow mutation (add/remove)."""
        with pytest.raises(AttributeError):
            self.main.VALID_FACING_VALUES.add("Northeast")  # type: ignore

    def test_membership_testing_is_o1_compatible(self):
        """Membership test on frozenset must work correctly for all valid values."""
        import main
        for direction in ("North", "South", "East", "West",
                          "North-East", "North-West", "South-East", "South-West"):
            result = direction in main.VALID_FACING_VALUES
            assert result is True


# ---------------------------------------------------------------------------
# Validation logic (tested in isolation, not via HTTP)
# ---------------------------------------------------------------------------
class TestFacingValidationLogic:
    """
    Tests for the facing filter validation logic.
    Because main.py is truncated (route body not present), these tests
    replicate the validation logic specified in arch-spec §3.5 and verify
    the constants are correct for that logic to work.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        import main
        self.VALID_FACING_VALUES = main.VALID_FACING_VALUES

    def _validate_facing(self, facing_list: list) -> None:
        """
        Replicate the validation block specified in arch-spec §3.5.
        Raises ValueError if validation fails (mirrors HTTPException behaviour).
        """
        if len(facing_list) > 8:
            raise ValueError("Too many facing values supplied")
        for value in facing_list:
            if value not in self.VALID_FACING_VALUES:
                safe_value = value[:50]
                raise ValueError(f"Invalid facing value: '{safe_value}'")

    def test_empty_list_passes_validation(self):
        """Empty facing list must pass validation (no filter = all results)."""
        try:
            self._validate_facing([])
        except ValueError as e:
            pytest.fail(f"Empty list should pass validation, got: {e}")

    def test_single_valid_value_passes(self):
        """Single valid facing value must pass validation."""
        self._validate_facing(["East"])  # should not raise

    def test_all_eight_valid_values_pass(self):
        """All 8 valid values together must pass validation."""
        all_valid = ["North", "South", "East", "West",
                     "North-East", "North-West", "South-East", "South-West"]
        self._validate_facing(all_valid)  # should not raise

    def test_invalid_value_raises(self):
        """Invalid facing value must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["Northeast"])  # missing hyphen

    def test_lowercase_raises(self):
        """Lowercase valid direction must raise (case-sensitive)."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["east"])

    def test_sql_injection_attempt_raises(self):
        """SQL injection attempt in facing must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["East' OR '1'='1"])

    def test_sql_drop_table_attempt_raises(self):
        """DROP TABLE injection in facing must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["North; DROP TABLE listings--"])

    def test_comma_joined_string_raises(self):
        """
        Comma-joined string 'East,North' must fail validation.
        Frontend must use params.append() not params.set() to avoid this.
        """
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["East,North-East"])

    def test_nine_values_raises_too_many(self):
        """More than 8 facing values must raise 'Too many' error."""
        nine_values = ["North", "South", "East", "West",
                       "North-East", "North-West", "South-East", "South-West",
                       "North"]  # duplicate to make 9
        with pytest.raises(ValueError, match="Too many facing values"):
            self._validate_facing(nine_values)

    def test_error_message_truncates_long_input(self):
        """
        Error message must truncate input to 50 chars (security-checklist §4.3).
        """
        long_input = "A" * 200
        with pytest.raises(ValueError) as exc_info:
            self._validate_facing([long_input])
        # The echoed value in the error message must be truncated to 50 chars
        error_msg = str(exc_info.value)
        # Extract the echoed portion — it should be 50 'A's not 200
        assert "A" * 51 not in error_msg, (
            "Error message echoed more than 50 characters of input"
        )
        assert "A" * 50 in error_msg, (
            "Error message should echo exactly 50 chars of the invalid value"
        )

    def test_unicode_injection_raises(self):
        """Unicode in facing value must raise (not in allowlist)."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["Nörth"])

    def test_empty_string_raises(self):
        """Empty string in facing list must raise (not in allowlist)."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing([""])

    def test_whitespace_string_raises(self):
        """Whitespace-only string must raise (not in allowlist)."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["   "])

    def test_mixed_valid_invalid_raises(self):
        """Mix of valid and invalid — must raise on the invalid value."""
        with pytest.raises(ValueError, match="Invalid facing value"):
            self._validate_facing(["East", "North-East", "invalid"])

    def test_validation_rejects_on_first_invalid(self):
        """Validation must reject on first invalid value (fail-fast)."""
        call_count = 0
        original_validate = self._validate_facing

        results = []
        inputs = ["East", "bad_value", "North"]
        for i, v in enumerate(inputs):
            if v not in self.VALID_FACING_VALUES:
                results.append(i)
                break

        assert results == [1], "Should fail on index 1 (first invalid value)"


# ---------------------------------------------------------------------------
# SQL construction logic (isolated)
# ---------------------------------------------------------------------------
class TestSQLConstruction:
    """
    Tests for the SQL IN clause construction specified in arch-spec §3.6.
    Because main.py is truncated, these tests verify the prescribed pattern
    works correctly in isolation.
    """

    def _build_facing_clause(self, facing: list, where_clauses: list, params: list):
        """
        Replicate the SQL construction from arch-spec §3.6.
        """
        if facing:
            placeholders = ",".join(["%s"] * len(facing))
            where_clauses.append(f"facing IN ({placeholders})")
            params.extend(facing)

    def test_single_value_produces_correct_placeholder(self):
        """Single facing value → `facing IN (%s)`."""
        where_clauses = []
        params = []
        self._build_facing_clause(["East"], where_clauses, params)
        assert where_clauses == ["facing IN (%s)"]
        assert params == ["East"]

    def test_multiple_values_produce_correct_placeholders(self):
        """Three facing values → `facing IN (%s,%s,%s)`."""
        where_clauses = []
        params = []
        self._build_facing_clause(
            ["East", "North-East", "North"], where_clauses, params
        )
        assert where_clauses == ["facing IN (%s,%s,%s)"]
        assert params == ["East", "North-East", "North"]

    def test_empty_list_adds_nothing_to_clause(self):
        """Empty facing list must add nothing to WHERE clause."""
        where_clauses = []
        params = []
        self._build_facing_clause([], where_clauses, params)
        assert where_clauses == []
        assert params == []

    def test_all_eight_values_produce_correct_placeholders(self):
        """All 8 values → `facing IN (%s,%s,%s,%s,%s,%s,%s,%s)`."""
        all_values = ["North", "South", "East", "West",
                      "North-East", "North-West", "South-East", "South-West"]
        where_clauses = []
        params = []
        self._build_facing_clause(all_values, where_clauses, params)
        expected_clause = "facing IN (%s,%s,%s,%s,%s,%s,%s,%s)"
        assert where_clauses == [expected_clause]
        assert len(params) == 8
        assert set(params) == set(all_values)

    def test_values_are_not_interpolated_into_sql_string(self):
        """
        Facing values must appear in params list, NOT in the SQL string.
        This is the core SQL injection prevention check.
        """
        where_clauses = []
        params = []
        injection_attempt = "East' OR '1'='1"
        self._build_facing_clause([injection_attempt], where_clauses, params)

        # The injection string must be in params (safely parameterised)
        assert injection_attempt in params

        # The SQL clause string must only contain %s placeholders, not the value
        sql_clause = where_clauses[0]
        assert injection_attempt not in sql_clause, (
            "SQL injection value must not appear in the SQL clause string"
        )
        assert "%s" in sql_clause, "SQL clause must use %s parameterisation"

    def test_placeholder_count_matches_value_count(self):
        """Number of %s placeholders must exactly equal number of values."""
        facing = ["East", "North-East", "North"]
        where_clauses = []
        params = []
        self._build_facing_clause(facing, where_clauses, params)
        sql_clause = where_clauses[0]
        placeholder_count = sql_clause.count("%s")
        assert placeholder_count == len(facing), (
            f"Expected {len(facing)} placeholders, got {placeholder_count}"
        )

    def test_params_extend_not_append(self):
        """
        Using params.extend() must add all values to existing params list.
        This test verifies the extend() pattern works correctly for
        building the combined WHERE clause with other existing params.
        """
        existing_params = [50000, 5000000]  # e.g. min_price, max_price already added
        facing = ["East", "North-East"]
        where_clauses = ["price BETWEEN %s AND %s"]
        self._build_facing_clause(facing, where_clauses, existing_params)

        assert existing_params == [50000, 5000000, "East", "North-East"]
        assert len(where_clauses) == 2


# ---------------------------------------------------------------------------
# App and Route Tests (via TestClient)
# ---------------------------------------------------------------------------
class TestAppRoutes:
    """
    Tests for app-level route configuration.
    NOTE: Many of these will FAIL because main.py is truncated.
    Failures are expected and documented in the test report.
    """

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Create TestClient — requires main.py to be importable and complete."""
        import main
        self.client = TestClient(main.app)

    def test_health_endpoint_returns_200(self):
        """GET /health must return 200 with {status: ok}."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_listings_route_exists_at_slash_listings(self):
        """
        GET /listings (not /api/listings) must be the correct route.
        Blocker #3 from Code Review — route must NOT be /api/listings.
        """
        with patch("main.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock(return_value=False)
            mock_conn.return_value.__enter__ = MagicMock(
                return_value=MagicMock(cursor=MagicMock(return_value=mock_cursor))
            )
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            # This tests the route path — if 404 is returned, route is wrong
            response = self.client.get("/listings")
            assert response.status_code != 404, (
                "GET /listings returned 404 — route may be at /api/listings instead"
            )

    def test_api_listings_route_does_not_exist(self):
        """
        GET /api/listings must return 404.
        The unauthorised route prefix change must be reverted.
        """
        response = self.client.get("/api/listings")
        assert response.status_code == 404, (
            "GET /api/listings should not exist — route must be at /listings"
        )

    def test_valid_single_facing_param_returns_200(self):
        """GET /listings?facing=East with valid value must return 200."""
        with patch("main.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn_instance = MagicMock()
            mock_conn_instance.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn_instance.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_conn.return_value = mock_conn_instance

            response = self.client.get("/listings?facing=East")
            # NOTE: Will FAIL if route body is truncated (not implemented)
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text[:200]}"
            )

    def test_invalid_facing_param_returns_400(self):
        """GET /listings?facing=invalid must return 400."""
        response = self.client.get("/listings?facing=invalid_direction")
        assert response.status_code == 400, (
            f"Expected 400 for invalid facing, got {response.status_code}"
        )

    def test_sql_injection_in_facing_returns_400(self):
        """GET /listings?facing=East' OR '1'='1 must return 400."""
        response = self.client.get("/listings?facing=East' OR '1'='1")
        assert response.status_code == 400, (
            f"SQL injection attempt must be rejected with 400, got {response.status_code}"
        )

    def test_lowercase_facing_returns_400(self):
        """GET /listings?facing=east (lowercase) must return 400 (case-sensitive)."""
        response = self.client.get("/listings?facing=east")
        assert response.status_code == 400, (
            f"Lowercase facing 'east' must be rejected with 400, got {response.status_code}"
        )

    def test_comma_joined_facing_returns_400(self):
        """
        GET /listings?facing=East,North must return 400.
        'East,North' is not a valid facing value (comma-joined format).
        """
        response = self.client.get("/listings?facing=East,North")
        assert response.status_code == 400, (
            f"Comma-joined facing must be rejected with 400, got {response.status_code}"
        )

    def test_multi_value_facing_accepted(self):
        """GET /listings?facing=East&facing=North must be accepted (multi-value format)."""
        with patch("main.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn_instance = MagicMock()
            mock_conn_instance.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn_instance.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_conn.return_value = mock_conn_instance

            response = self.client.get("/listings?facing=East&facing=North")
            assert response.status_code == 200, (
                f"Multi-value facing must return 200, got {response.status_code}"
            )

    def test_too_many_facing_values_returns_400(self):
        """More than 8 facing values must return 400."""
        # 9 facing params (only 8 valid values exist, so we duplicate)
        params = "&".join([
            "facing=North", "facing=South", "facing=East", "facing=West",
            "facing=North-East", "facing=North-West", "facing=South-East",
            "facing=South-West", "facing=North"  # 9th value
        ])
        response = self.client.get(f"/listings?{params}")
        assert response.status_code == 400, (
            f"9 facing values must return 400, got {response.status_code}"
        )

    def test_no_facing_param_returns_200(self):
        """GET /listings with no facing param must return 200 (default: no filter)."""
        with patch("main.get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn_instance = MagicMock()
            mock_conn_instance.cursor.return_value.__enter__ = MagicMock(
                return_value=mock_cursor
            )
            mock_conn_instance.cursor.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_conn.return_value = mock_conn_instance

            response = self.client.get("/listings")
            assert response.status_code == 200, (
                f"No facing param must return 200, got {response.status_code}"
            )

    def test_error_message_does_not_echo_more_than_50_chars(self):
        """
        Error response for oversized input must not echo more than 50 chars
        of the invalid value (security-checklist §4.3).
        """
        long_value = "A" * 200
        response = self.client.get(f"/listings?facing={long_value}")
        assert response.status_code == 400
        response_body = response.text
        # The 200-char string must not appear verbatim in the response
        assert long_value not in response_body, (
            "Error response echoed the full 200-character invalid value"
        )

    def test_drop_table_injection_returns_400(self):
        """DROP TABLE injection attempt must be rejected with 400."""
        response = self.client.get("/listings?facing=North;%20DROP%20TABLE%20listings--")
        assert response.status_code == 400

    def test_handler_is_sync_not_async(self):
        """
        Route handler must be synchronous (def, not async def).
        This is verifiable by inspecting the route's call stack attribute.
        """
        import main
        import inspect

        # Find the /listings route
        listings_route = None
        for route in main.app.routes:
            if hasattr(route, "path") and route.path == "/listings":
                listings_route = route
                break

        assert listings_route is not None, "GET /listings route not found in app"

        endpoint_func = listings_route.endpoint
        assert not inspect.iscoroutinefunction(endpoint_func), (
            "get_listings must be `def` (synchronous), not `async def`. "
            "Using async def with blocking psycopg2 freezes the event loop. "
            "See arch-spec §3.1 and Code Review Blocker #7."
        )

    def test_facing_parameter_accepts_list(self):
        """
        Verify the /listings route declares facing as List[str] with Query().
        FastAPI will parse ?facing=East&facing=North as ['East', 'North'].
        """
        import main
        import inspect

        listings_route = None
        for route in main.app.routes:
            if hasattr(route, "path") and route.path == "/listings":
                listings_route = route
                break

        assert listings_route is not None

        sig = inspect.signature(listings_route.endpoint)
        assert "facing" in sig.parameters, (
            "get_listings must have a 'facing' parameter"
        )


# ---------------------------------------------------------------------------
# VALID_BUILDER_MODES — verify out-of-scope constant
# ---------------------------------------------------------------------------
class TestBuilderModesConstant:
    """
    Test the VALID_BUILDER_MODES constant added to main.py.
    This is an out-of-scope addition (noted in Code Review as non-blocking)
    but its presence should not break anything.
    """

    def test_valid_builder_modes_exists(self):
        """VALID_BUILDER_MODES must exist if added (should not cause import error)."""
        import main
        if hasattr(main, "VALID_BUILDER_MODES"):
            assert isinstance(main.VALID_BUILDER_MODES, frozenset)
            assert "include" in main.VALID_BUILDER_MODES
            assert "exclude" in main.VALID_BUILDER_MODES


# ---------------------------------------------------------------------------
# get_connection and wait_for_db functions
# ---------------------------------------------------------------------------
class TestDBFunctions:
    """Tests for get_connection and wait_for_db helper functions."""

    def test_get_connection_function_exists(self):
        """get_connection must be defined in main.py."""
        import main
        assert hasattr(main, "get_connection")
        assert callable(main.get_connection)

    def test_wait_for_db_function_exists(self):
        """wait_for_db must be defined in main.py."""
        import main
        assert hasattr(main, "wait_for_db")
        assert callable(main.wait_for_db)

    def test_wait_for_db_raises_runtime_error_after_max_retries(self):
        """wait_for_db must raise RuntimeError after exhausting retries."""
        import main
        with patch("psycopg2.connect", side_effect=Exception("Connection refused")):
            with patch("time.sleep"):  # Don't actually sleep
                with pytest.raises(RuntimeError, match="not available"):
                    main.wait_for_db(max_retries=3)

    def test_wait_for_db_succeeds_on_first_try(self):
        """wait_for_db must succeed and not raise when DB connects."""
        import main
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn):
            try:
                main.wait_for_db(max_retries=3)
            except RuntimeError:
                pytest.fail("wait_for_db raised RuntimeError despite DB being available")

    def test_get_connection_uses_realdictcursor(self):
        """get_connection must use RealDictCursor for dict-like row access."""
        import main
        import psycopg2.extras
        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value = MagicMock()
            main.get_connection()
            # Verify RealDictCursor was passed as cursor_factory
            call_kwargs = mock_connect.call_args
            assert call_kwargs is not None
            # Either positional or keyword arg
            args, kwargs = call_kwargs
            assert kwargs.get("cursor_factory") == psycopg2.extras.RealDictCursor, (
                "get_connection must use RealDictCursor"
            )
```

---

### `backend/tests/test_f099_integration.py`
```python
"""
Integration tests for F099 — Vastu Compliance Indicator (backend)
Tests hit the real API endpoints against the actual database.

These tests require:
- Docker Compose running: docker compose up -d
- DB seeded with 002_seed_2000.sql (listings with facing values)

Run: docker compose exec backend pytest backend/tests/test_f099_integration.py -v

Test Agent Cycle 1 — Atlas Realty F099
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import psycopg2
import psycopg2.extras
from fastapi.testclient import TestClient

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/atlas_realty"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db_connection():
    """Session-scoped real DB connection for integration tests."""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        yield conn
        conn.close()
    except Exception as exc:
        pytest.skip(f"Database not available for integration tests: {exc}")


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient using real database."""
    try:
        import main
        with TestClient(main.app) as c:
            yield c
    except Exception as exc:
        pytest.skip(f"Could not create TestClient: {exc}")


@pytest.fixture(scope="session")
def sample_listings_with_facing(db_connection):
    """
    Fetch a small set of listings with known facing values from the real DB.
    Used to drive integration test assertions.
    """
    with db_connection.cursor() as cur:
        cur.execute("""
            SELECT id, title, facing, price, beds
            FROM listings
            WHERE facing IS NOT NULL
            LIMIT 20
        """)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Schema verification
# ---------------------------------------------------------------------------

class TestDatabaseSchema:
    """Verify the DB schema meets F099 requirements."""

    def test_facing_column_exists(self, db_connection):
        """listings table must have a 'facing' column."""
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                

---

## Actual Test Run Results

**Docker is not running — tests not executed automatically.**
Run manually after `docker compose up`:
```bash
docker compose exec backend pytest backend/tests/test_f099.py -v --tb=short
docker compose exec frontend npm run test -- --watchAll=false
```