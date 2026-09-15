import pandas as pd

from salary_scout.cleaning import (
    filter_plausible_compensation,
    strip_html,
    strip_salary_mentions,
)


def test_strip_salary_mentions_ranges_and_singletons():
    text = "Pay: $120,000 - $150,000 per year. Bonus up to $10k. USD 95,000. 401(k) match."
    out = strip_salary_mentions(text)
    assert "120,000" not in out
    assert "150,000" not in out
    assert "$10k" not in out
    assert "95,000" not in out
    assert "401(k)" in out


def test_strip_html():
    assert strip_html("<p>Hello&nbsp;<b>world</b></p>") == "Hello world"


def test_filter_plausible_compensation():
    df = pd.DataFrame(
        {
            "yearly_min_compensation": [100_000, 51.8, 90_000, 200_000_000, 100_000, None],
            "yearly_max_compensation": [150_000, 60.7, 100_000, 300_000_000, 500_000, None],
            "listed_compensation_currency": ["USD", "USD", "CAD", "USD", "USD", "USD"],
        }
    )
    kept = filter_plausible_compensation(df)
    assert kept.index.tolist() == [0]
