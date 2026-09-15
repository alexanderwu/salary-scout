# Data Dictionary — `jobs.duckdb` (table: `jobs`)

Source: job postings scraped from [hiringcafe.com](https://hiringcafe.com). Snapshot profiled on 2026-09-15: **27,098 rows**, 33 columns.

Note: `id` is not a reliable unique row key (26,498 distinct values out of 27,098 rows — a handful of postings appear multiple times, e.g. re-scraped over time). Use `requisition_id` as the primary key; it is unique and non-null for every row.

## Identifiers & source tracking

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `id` | VARCHAR | 0 | 26,498 | Composite ID built as `{source}___{board_token}___{posting-slug-or-numeric-id}`. Mostly unique but a small number of postings recur (re-scrapes). |
| `source` | VARCHAR | 0 | 107 | The ATS (applicant tracking system) platform the posting was scraped from, e.g. `workday`, `grnhse` (Greenhouse), `ashby`, `icims2`, `lever`, `taleo_careersection`, `oraclecloud`, `successfactors`, `smartrecruiters`, `adhoc` (custom/non-standard career sites). |
| `board_token` | VARCHAR | 0 | 7,699 | The employer's identifier/slug within that ATS platform (e.g. Workday tenant, Greenhouse board name). Roughly one per employer-ATS pairing. |
| `apply_url` | VARCHAR | 0 | 26,362 | Direct URL to the job posting / application page. |
| `requisition_id` | VARCHAR | 0 | 27,098 | **Primary key.** hiringcafe's internal unique ID for the posting record (16-char alphanumeric). |
| `collapse_key` | VARCHAR | 0 | 7,780 | SHA-256-style hash used by hiringcafe to group/deduplicate near-identical postings (e.g. the same role reposted, or cross-posted to multiple boards). |

## Job title & classification

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `title` | VARCHAR | 0 | 16,231 | Job title as displayed. Identical to `job_title_raw` for all rows in this snapshot. |
| `job_title_raw` | VARCHAR | 0 | 16,231 | Raw/unprocessed job title as scraped from the source. |
| `core_job_title` | VARCHAR | 0 | 7,107 | Normalized/canonicalized version of the title (e.g. stripping level, location, req numbers) used for grouping similar roles. |
| `job_category` | VARCHAR | 0 | 42 | High-level functional category assigned by hiringcafe's enrichment model. Top values: `Data and Analytics` (14,083), `Software Development` (5,660), `Information Technology` (3,003), `Engineering` (2,394), `Research and Development (R&D)` (740). |
| `seniority_level` | VARCHAR | 0 | 4 | One of: `Mid Level` (11,172), `Senior Level` (11,001), `Entry Level` (4,253), `No Prior Experience Required` (672). |
| `role_type` | VARCHAR | 0 | 1 | Always `Individual Contributor` in this snapshot (no manager/executive roles present). |
| `technical_tools` | VARCHAR (JSON array as text) | 0 | 22,518 | JSON-encoded list of tools/technologies/skills mentioned in the posting, e.g. `["C#",".NET","REST","Azure",...]`. Parse with `json_extract`/`json_transform` or DuckDB's `->` JSON operators. |

## Location & work arrangement

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `workplace_type` | VARCHAR | 0 | 4 | `Remote` (13,635), `Onsite` (7,483), `Hybrid` (5,920), `Field` (60). |
| `formatted_workplace_location` | VARCHAR | 0 | 6,147 | Human-readable location string, e.g. city/state/country or "Remote - US". |
| `workplace_countries` | VARCHAR (JSON array as text) | 0 | 358 | JSON-encoded list of ISO country codes eligible for the role, e.g. `["US"]`, `["US","CA"]`. Dominated by `["US"]` (25,729 rows). |
| `latitude` / `longitude` | DOUBLE | 7,743 (28.6%) | 4,771 / 4,986 | Approximate geocoded coordinates of the primary workplace location. Range roughly lat `[-73.6, 61.2]`, lon `[-158.1, 174.8]` (includes non-US locations and some geocoding outliers). Null when the location couldn't be geocoded (common for fully remote/global postings). |

## Compensation

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `yearly_min_compensation` | DOUBLE | 2,907 (10.7%) | 4,308 | Minimum annualized compensation as listed/normalized by hiringcafe, in `listed_compensation_currency`. Range observed: 1 – 230,216,480 (upper end likely includes equity-heavy outliers/data errors — treat extreme values with caution). |
| `yearly_max_compensation` | DOUBLE | 2,907 (10.7%) | 4,747 | Maximum annualized compensation, same caveats as above (max observed 340,496,000). |
| `listed_compensation_currency` | VARCHAR | 0 (330 empty string) | 14 | ISO currency code. Dominated by `USD` (26,595); others include `CAD`, `EUR`, `GBP`, `INR`, `MXN`, `AUD`, etc. 330 rows have an empty string (currency not specified, typically alongside missing compensation). |
| `min_industry_and_role_yoe` | DOUBLE | 5,043 (18.6%) | 13 | Minimum years of industry/role experience required, as extracted from the posting text. Discrete values (0, 0.5, 1, 2, 3, 4, 5, 6, plus a few fractional outliers). Null when not stated in the posting. |

Note: Both `job_information_json` and `v5_processed_job_data_json` also carry sub-annual compensation breakdowns (`monthly_*`, `weekly_*`, `hourly_*`, etc.) and an `is_compensation_transparent` flag not exposed as their own top-level columns — see below.

## Company / employer

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `company_name` | VARCHAR | 272 (1.0%) | 8,242 | Employer name. |
| `company_website` | VARCHAR | 10,870 (40.1%) | 5,689 | Employer's website or careers-site domain (often the ATS subdomain, e.g. `spektrasystems.zohorecruit.com`, rather than the corporate domain). |
| `enriched_status` | VARCHAR | 52 (0.2%) | 2 | Result of hiringcafe's company-enrichment pipeline: `VALID_COMPANY` (27,028) or `EXCLUDED` (18, e.g. staffing agencies/spam). Null for 52 rows where enrichment wasn't attempted. |
| `nb_employees` | INTEGER | 1,268 (4.7%) | 1,884 | Employer headcount estimate. Range 2 – 2,900,000 (large multinationals included); average ~56,158 is skewed by big employers. |
| `year_founded` | INTEGER | 744 (2.7%) | 208 | Employer founding year. Range 1803 – 2026. |

## Timing

| Column | Type | Nulls | Distinct | Description |
|---|---|---|---|---|
| `estimated_publish_date` | VARCHAR (ISO 8601 timestamp) | 0 | 16,632 | hiringcafe's estimate of when the posting first went live, e.g. `2026-05-08T14:37:18+00:00`. Not necessarily the scrape date. Range observed: 2023-01-05 to 2026-09-15. Cast with `TRY_CAST(estimated_publish_date AS TIMESTAMPTZ)` for date arithmetic. |
| `is_expired` | BOOLEAN | 0 | 2 | Whether hiringcafe has flagged the posting as no longer active/live. `false` (25,140 — active) vs `true` (1,958 — expired). |

## Free text

| Column | Type | Nulls | Description |
|---|---|---|---|
| `description` | VARCHAR | 0 | Full job posting text (HTML or plain text, as scraped). Length ranges 18 – 562,625 characters, averaging ~9,272 chars. |

## Nested JSON blob columns

These three columns store full JSON documents as text (VARCHAR). Use DuckDB's JSON functions (`json_extract`, `->>`,`json_keys`, `read_json`) to query into them rather than reading them as flat columns.

### `job_information_json`
hiringcafe app/user-engagement metadata for the posting. Top-level keys observed: `title`, `job_title_raw`, `viewedByUsers`, `savedFromUsers`, `hiddenFromUsers`, `appliedFromUsers` (arrays of hiringcafe Firebase user IDs).

⚠️ **Contains third-party user identifiers** (hiringcafe account UIDs of people who viewed/saved/applied to the posting through the platform). Treat as sensitive — avoid re-publishing or joining against other identity data; these are not you and not people who consented to inclusion in your dataset.

### `v5_processed_job_data_json`
The richest field — hiringcafe's full "v5" enrichment output for the posting, a superset of most of the flat columns above plus many fields not otherwise exposed, including:
- Education/certification requirements: `associates_degree_requirement`, `bachelors_degree_requirement`, `masters_degree_requirement`, `doctorate_degree_requirement` (+ corresponding `*_fields_of_study`), `is_high_school_required`, `licenses_or_certifications`
- Experience: `min_industry_and_role_yoe`, `min_management_and_leadership_yoe` (+ "not mentioned" flags)
- Location detail: `workplace_cities`, `workplace_counties`, `workplace_states`, `workplace_countries`, `workplace_continents`, and `boundless_*` variants, plus counts (`number_of_workplace_cities`, etc.) and `is_workplace_worldwide_ok`
- Work conditions: `commitment`, `security_clearance`, `position_employer_type`, `workplace_physical_environment`, `oral_communication_level`, `physical_labor_intensity`, `physical_position`, `computer_usage`, `cognitive_demand`, travel requirements, shift/schedule flags (`morning_shift_work`, `evening_shift_work`, `overnight_work`, `on_call_requirement`, `weekend_availability_required`, `overtime_required`, etc.)
- Benefits/perks flags: `visa_sponsorship`, `relocation_assistance`, `military_veterans`, `tuition_reimbursement`, `retirement_plan`, `generous_parental_leave`, `401k_matching`, `generous_paid_time_off`, `four_day_work_week`, `fair_chance`
- Full compensation breakdown across pay periods: `yearly_min/max_compensation`, `monthly_*`, `weekly_*`, `bi-weekly_*`, `daily_*`, `hourly_*`, plus `is_compensation_transparent` and `listed_compensation_frequency`
- Language requirements: `language_requirements`, `num_language_requirements`
- Company snapshot at enrichment time: `company_name`, `company_website`, `company_sector_and_industry`, `company_activities`, `company_tagline`
- `role_activities`, `requirements_summary` (free-text)

### `enriched_company_data_json`
Company profile from hiringcafe's company-enrichment pipeline. Top-level keys: `enriched_at`, `status`, `name`, `homepage_uri`, `hq_country`, `parent_company`, `subsidiaries`, `industries`, `activities`, `nb_employees`, `year_founded`, `tagline`, `organization_type`, `latest_funding_investors`, `latest_funding_type`, `latest_funding_year`, `latest_funding_amount`, `stock_exchange`, `stock_symbol`. Null for 47 rows (0.2%) where company enrichment failed or wasn't run.

## Known data quality notes

- `id` has ~600 duplicate values (postings scraped more than once); dedupe on `requisition_id` for row-level uniqueness.
- Compensation fields have extreme outliers at the top end (into the hundreds of millions) that are almost certainly parsing errors or equity/bonus values misclassified as base salary — filter or winsorize before analysis.
- `company_website` is frequently the ATS-hosted careers subdomain rather than the employer's actual corporate domain.
- `listed_compensation_currency` has 330 rows with an empty string rather than `NULL` when currency isn't specified.
