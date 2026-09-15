import pytest

from salary_scout.data import DEFAULT_DB_PATH, build_query, load_jobs


def test_build_query_mentions_dedupe():
    q = build_query(limit=10)
    assert "PARTITION BY requisition_id" in q
    assert "job_information_json" not in q
    assert "LIMIT 10" in q


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="jobs.duckdb not present")
def test_load_jobs_smoke():
    df = load_jobs(limit=200)
    assert len(df) == 200
    assert df["requisition_id"].is_unique
    assert "is_compensation_transparent" in df.columns
    assert "job_information_json" not in df.columns
