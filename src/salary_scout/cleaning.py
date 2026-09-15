"""Target cleaning and leakage removal.

Two things every downstream step needs:

1. ``filter_plausible_compensation`` — restrict to USD rows whose annualised
   salary range is physically plausible. The raw data has hourly rates stored as
   yearly figures (``$51.80 - $60.69``) and yearly figures multiplied by 2080 as
   if they were hourly (``$223,479,360``). Rather than repair these, we drop them
   and document the counts.
2. ``strip_salary_mentions`` — remove dollar amounts from description text.
   Two thirds of postings print the exact salary figure in the body, so a text
   model trained on raw descriptions learns to read the answer off the page.
"""

from __future__ import annotations

import re

import pandas as pd

# Plausibility bounds for annualised USD compensation.
MIN_YEARLY = 20_000
MAX_YEARLY = 1_000_000
MAX_SPREAD_RATIO = 3.0

# Money patterns. Order matters: ranges first so the whole span is replaced at once.
# Four shapes of a single amount:
#   comma-grouped, optional currency prefix   $120,000  120,000.00  USD 95,000
#   currency prefix, any digits, optional k   $120000  $95k  US$ 80.5k
#   bare 2-3 digits with a k suffix           100k  62.5k   (as in "100k-150k")
#   bare 5-6 digit integer, optional cents    155000.00  198000  229400USD
# The last shape also eats zip codes and requisition numbers, which carry no pay signal.
_AMOUNT = (
    r"(?:USD|US\$|\$)?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|(?:USD|US\$|\$)\s?\d+(?:\.\d+)?\s?[kK]?"
    r"|\b\d{2,3}(?:\.\d+)?\s?[kK]\b"
    r"|\b\d{5,6}(?:\.\d{1,2})?(?:\s?usd)?\b"
)
_SEP = r"\s*(?:-|–|—|to|and)\s*"
_RANGE = rf"(?:{_AMOUNT}){_SEP}(?:{_AMOUNT})"
# "150-200k": the k applies to both ends, so the first number alone is an amount too.
_RANGE_SHARED_K = rf"\b\d{{2,3}}(?:\.\d+)?{_SEP}\d{{2,3}}(?:\.\d+)?\s?[kK]\b"
_SALARY_RE = re.compile(rf"{_RANGE_SHARED_K}|{_RANGE}|{_AMOUNT}", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

SALARY_TOKEN = " [SALARY] "


def strip_salary_mentions(text: str) -> str:
    """Replace dollar amounts, ``$120k`` style figures and comma-grouped numbers with a token."""
    if not isinstance(text, str):
        return text
    return _SALARY_RE.sub(SALARY_TOKEN, text)


def strip_html(text: str) -> str:
    """Cheap tag removal. Good enough for EDA and bag-of-words features."""
    if not isinstance(text, str):
        return text
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return _WS_RE.sub(" ", text).strip()


def plausibility_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean columns for each plausibility rule, so notebooks can report drop counts."""
    lo, hi = df["yearly_min_compensation"], df["yearly_max_compensation"]
    return pd.DataFrame(
        {
            "has_comp": lo.notna() & hi.notna(),
            "is_usd": df["listed_compensation_currency"] == "USD",
            "min_ok": lo >= MIN_YEARLY,
            "max_ok": hi <= MAX_YEARLY,
            "ordered": lo <= hi,
            "spread_ok": (hi / lo) <= MAX_SPREAD_RATIO,
        },
        index=df.index,
    )


def filter_plausible_compensation(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable USD salary range. See module docstring for the rules."""
    return df[plausibility_mask(df).all(axis=1)].copy()
