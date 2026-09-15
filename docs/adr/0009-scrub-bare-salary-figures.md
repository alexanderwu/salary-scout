# 0009. Extend salary scrubbing to bare figures

Date: 2026-09-15
Status: Accepted (amends 0005)

## Context

ADR 0005 scrubs prefixed and comma-grouped amounts and promised a pipeline check for
residual leakage. The check, `check_salary_leakage` in `salary_scout.features`, looks
for a row's own minimum or maximum figure in every written form (`120,000`, `120000`,
`120k`) across the scrubbed title, description and requirements summary.

On the full cleaned set it found the figure surviving in 1.8% of rows (431 of 23,824).
Two shapes accounted for nearly all of them:

- bare numbers with a `k` suffix: `100k-150k`, `150-200k+`, `$70,000 -100k`
- bare five- or six-digit integers, sometimes with cents or a glued currency:
  `155000.00 - 173500.00 USD per year`, `max rate 157500`, `229400USD`

## Decision

- `strip_salary_mentions` also replaces bare two- or three-digit numbers followed by
  `k`, ranges where a single `k` applies to both ends, and bare five- or six-digit
  integers with optional cents and an optional trailing `usd`.
- The leakage check runs on the derived table and fails above 0.1% of rows. The
  tolerance is a limit, not a target; new leak shapes found later get added to the regex.

## Consequences

- Residual leakage on the full cleaned set is 0.02% (5 rows). What remains is unusual
  formatting that a general rule would over-scrub.
- Five- and six-digit rule also removes zip codes, requisition numbers and headcounts
  written without commas. None of these carry pay signal the metadata blocks lack.
- The `[SALARY]` token becomes more frequent, so its count is a weak "posts a range"
  signal. That is acceptable; it is the same signal as `is_compensation_transparent`.
