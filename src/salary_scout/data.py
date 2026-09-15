"""Load and clean the jobs dataset from DuckDB.

The raw table has ~27k rows. This module handles the boring-but-important parts:
row-level dedupe on ``requisition_id``, excluding the sensitive
``job_information_json`` blob (it contains third-party user IDs), and pulling a
handful of useful fields out of the nested ``v5_processed_job_data_json`` blob.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "jobs.duckdb"

# Flat columns we keep from the base table. ``job_information_json`` is deliberately
# excluded: it carries hiringcafe user IDs and has no modelling value.
FLAT_COLUMNS = [
    "requisition_id",
    "id",
    "source",
    "board_token",
    "collapse_key",
    "title",
    "core_job_title",
    "job_category",
    "seniority_level",
    "technical_tools",
    "workplace_type",
    "formatted_workplace_location",
    "workplace_countries",
    "latitude",
    "longitude",
    "yearly_min_compensation",
    "yearly_max_compensation",
    "listed_compensation_currency",
    "min_industry_and_role_yoe",
    "company_name",
    "enriched_status",
    "nb_employees",
    "year_founded",
    "estimated_publish_date",
    "is_expired",
    "description",
]

# Fields pulled out of ``v5_processed_job_data_json``: (json key, output column, duckdb type).
V5_FIELDS: list[tuple[str, str, str]] = [
    ("is_compensation_transparent", "is_compensation_transparent", "BOOLEAN"),
    ("listed_compensation_frequency", "listed_compensation_frequency", "VARCHAR"),
    ("hourly_min_compensation", "hourly_min_compensation", "DOUBLE"),
    ("hourly_max_compensation", "hourly_max_compensation", "DOUBLE"),
    ("commitment", "commitment", "VARCHAR"),
    ("security_clearance", "security_clearance", "VARCHAR"),
    ("bachelors_degree_requirement", "bachelors_degree_requirement", "VARCHAR"),
    ("masters_degree_requirement", "masters_degree_requirement", "VARCHAR"),
    ("doctorate_degree_requirement", "doctorate_degree_requirement", "VARCHAR"),
    ("min_management_and_leadership_yoe", "min_management_yoe", "DOUBLE"),
    ("workplace_states", "workplace_states", "VARCHAR"),
    ("visa_sponsorship", "visa_sponsorship", "VARCHAR"),
    ("company_sector_and_industry", "company_sector_and_industry", "VARCHAR"),
    ("requirements_summary", "requirements_summary", "VARCHAR"),
]

# Fields pulled out of ``enriched_company_data_json``.
COMPANY_FIELDS: list[tuple[str, str, str]] = [
    ("organization_type", "organization_type", "VARCHAR"),
    ("hq_country", "hq_country", "VARCHAR"),
    ("stock_exchange", "stock_exchange", "VARCHAR"),
    ("latest_funding_type", "latest_funding_type", "VARCHAR"),
    ("industries", "company_industries", "VARCHAR"),
]


def _json_select(blob: str, fields: list[tuple[str, str, str]]) -> str:
    parts = []
    for key, out, typ in fields:
        expr = f"json_extract_string({blob}, '$.{key}')"
        if typ != "VARCHAR":
            expr = f"TRY_CAST({expr} AS {typ})"
        parts.append(f"{expr} AS {out}")
    return ",\n    ".join(parts)


def build_query(limit: int | None = None) -> str:
    """SQL that dedupes on requisition_id and flattens the JSON fields we care about."""
    flat = ",\n    ".join(FLAT_COLUMNS)
    v5 = _json_select("v5_processed_job_data_json", V5_FIELDS)
    company = _json_select("enriched_company_data_json", COMPANY_FIELDS)
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    return f"""
WITH deduped AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY requisition_id ORDER BY estimated_publish_date DESC
    ) AS rn
    FROM jobs
)
SELECT
    {flat},
    TRY_CAST(estimated_publish_date AS TIMESTAMPTZ) AS publish_ts,
    {v5},
    {company}
FROM deduped
WHERE rn = 1
{limit_clause}
"""


def load_jobs(db_path: Path | str = DEFAULT_DB_PATH, limit: int | None = None) -> pd.DataFrame:
    """Load the deduplicated jobs table with flattened JSON fields.

    Parameters
    ----------
    db_path:
        Path to ``jobs.duckdb``. Opened read-only.
    limit:
        Optional row cap for quick iteration.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(build_query(limit)).df()
    finally:
        con.close()
    for col in ("technical_tools", "workplace_countries"):
        df[col] = df[col].astype("string")
    return df
