"""
retrieval/filters.py
---------------------
Phase 6: Filter Construction and RBAC Enforcement

Single responsibility: convert a (role, RetrievalFilter) pair into a
metadata `where` dict that can be passed directly to
SQLiteVectorStore.query_with_filter().

RBAC hierarchy:
    Admin   → ["Public", "Student", "Faculty", "Admin"]
    Faculty → ["Public", "Student", "Faculty"]
    Student → ["Public", "Student"]
    Public  → ["Public"]

metadata where-clause operators used:
    $eq   — exact match for a single value
    $in   — match any value in a list (used for multi-level RBAC)
    $and  — all conditions must match (list of condition dicts)

Design:
    FilterBuilder is the public API.
    build() returns the where dict or None (no filter = Admin on empty RetrievalFilter).
    Downstream code (DenseSearchBackend) passes this directly to the store.
"""

from __future__ import annotations

from typing import Optional

from retrieval.retrieval_schema import RetrievalFilter, VALID_ROLES


# ---------------------------------------------------------------------------
# RBAC hierarchy
# ---------------------------------------------------------------------------

ACCESS_HIERARCHY: dict[str, list[str]] = {
    "Admin":   ["Public", "Student", "Faculty", "Admin"],
    "Faculty": ["Public", "Student", "Faculty"],
    "Student": ["Public", "Student"],
    "Public":  ["Public"],
}


def get_allowed_levels(role: str) -> list[str]:
    """
    Returns the access_level values visible to the given role.
    Falls back to Public-only for any unrecognized role (fail-safe).
    """
    return ACCESS_HIERARCHY.get(role, ACCESS_HIERARCHY["Public"])


# ---------------------------------------------------------------------------
# FilterBuilder
# ---------------------------------------------------------------------------

class FilterBuilder:
    """
    Builds a metadata `where` dict from an (role, RetrievalFilter).

    Usage:
        where = FilterBuilder(role="Student", filters=query.filters).build()
        results = store.query_with_filter(embedding, where, n_results=10)

    The build() method returns None only when Admin queries with no metadata
    filters — in that case the store can skip the where evaluation entirely,
    which is a meaningful performance hint.
    """

    def __init__(self, role: str, filters: Optional[RetrievalFilter] = None) -> None:
        self._role    = role if role in VALID_ROLES else "Public"
        self._filters = filters or RetrievalFilter()

    def build(self) -> Optional[dict]:
        """
        Returns a metadata where dict, or None if no filtering is needed.

        None is only returned for Admin role with no metadata filters — every
        other case produces at least the RBAC access_level constraint.
        """
        conditions: list[dict] = []

        # --- RBAC constraint ---
        allowed = get_allowed_levels(self._role)
        if len(allowed) == 4:
            # Admin sees all levels — omit the constraint entirely
            rbac_condition = None
        elif len(allowed) == 1:
            rbac_condition = {"access_level": {"$eq": allowed[0]}}
        else:
            rbac_condition = {"access_level": {"$in": allowed}}

        if rbac_condition:
            conditions.append(rbac_condition)

        # --- Metadata filters ---
        if self._filters.department:
            conditions.append({"department": {"$eq": self._filters.department}})

        if self._filters.category:
            conditions.append({"category": {"$eq": self._filters.category}})

        if self._filters.doc_id:
            conditions.append({"doc_id": {"$eq": self._filters.doc_id}})

        if self._filters.version:
            conditions.append({"version": {"$eq": self._filters.version}})

        # --- Combine ---
        if not conditions:
            return None              # Admin, no metadata filters → unfiltered
        if len(conditions) == 1:
            return conditions[0]    # Single condition — no $and wrapper needed
        return {"$and": conditions}

    def describe(self) -> dict:
        """Returns a human-readable summary of the active filters (for response echo)."""
        active: dict = {"role": self._role}
        if self._filters.department:
            active["department"] = self._filters.department
        if self._filters.category:
            active["category"] = self._filters.category
        if self._filters.doc_id:
            active["doc_id"] = self._filters.doc_id[:16] + "..."
        if self._filters.version:
            active["version"] = self._filters.version
        return active


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_where_clause(
    role: str,
    department: Optional[str] = None,
    category: Optional[str]   = None,
    doc_id: Optional[str]     = None,
    version: Optional[str]    = None,
) -> Optional[dict]:
    """
    Convenience wrapper for one-liner filter construction without
    instantiating a RetrievalFilter explicitly.

    Used by tests and interactive scripts.
    """
    f = RetrievalFilter(
        department = department,
        category   = category,
        doc_id     = doc_id,
        version    = version,
    )
    return FilterBuilder(role=role, filters=f).build()
