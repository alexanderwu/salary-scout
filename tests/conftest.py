"""Shared fixtures: a small synthetic frame shaped like the derived table."""

import numpy as np
import pandas as pd
import pytest

from salary_scout.dataset import add_targets, prepare_inputs


@pytest.fixture(scope="session")
def toy() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 120
    lo = rng.integers(60, 180, n) * 1000.0
    df = pd.DataFrame(
        {
            "requisition_id": [f"r{i}" for i in range(n)],
            "collapse_key": [f"g{i // 3}" for i in range(n)],
            "company_name": rng.choice(["Acme", "Globex", "Initech", "Umbrella"], n),
            "publish_ts": pd.date_range("2026-04-01", periods=n, freq="12h", tz="UTC"),
            "title": rng.choice(["Data Engineer", "Senior Analyst", "ML Engineer"], n),
            "core_job_title": rng.choice(["Data Engineer", "Analyst"], n),
            "description": [
                f"<p>We pay ${int(a):,} to ${int(a * 1.3):,}. Python and SQL required.</p>"
                for a in lo
            ],
            "requirements_summary": ["3+ years of experience"] * n,
            "seniority_level": rng.choice(["Entry Level", "Mid Level", "Senior Level"], n),
            "job_category": rng.choice(["Data and Analytics", "Engineering"], n),
            "min_industry_and_role_yoe": rng.choice([1.0, 3.0, 5.0, np.nan], n),
            "min_management_yoe": np.nan,
            "commitment": '["Full Time"]',
            "bachelors_degree_requirement": "Required",
            "masters_degree_requirement": "Not Mentioned",
            "doctorate_degree_requirement": "Not Mentioned",
            "security_clearance": "None",
            "visa_sponsorship": "false",
            "technical_tools": rng.choice(
                ['["Python","SQL"]', '["Java","Kafka","Spark"]', "[]"], n
            ),
            "workplace_type": rng.choice(["Remote", "Onsite", "Hybrid"], n),
            "workplace_states": rng.choice(['["California, US"]', '["Texas, US","Ohio, US"]', "[]"], n),
            "workplace_countries": '["US"]',
            "latitude": rng.choice([37.0, 30.0, np.nan], n),
            "longitude": rng.choice([-122.0, -97.0, np.nan], n),
            "nb_employees": rng.choice([50.0, 5000.0, np.nan], n),
            "year_founded": rng.choice([1999.0, 2015.0, np.nan], n),
            "organization_type": rng.choice(["Private", "Public"], n),
            "company_sector_and_industry": "Information Technology",
            "company_industries": '["Software","Cloud"]',
            "hq_country": "US",
            "source": rng.choice(["workday", "grnhse"], n),
            "yearly_min_compensation": lo,
            "yearly_max_compensation": lo * 1.3,
            "listed_compensation_currency": "USD",
        }
    )
    return add_targets(prepare_inputs(df))
