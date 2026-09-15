# 0005. Scrub salary figures from text before featurisation

Date: 2026-09-15
Status: Accepted

## Context

Pay-transparency postings print the range in the body. The exact minimum salary figure
appears verbatim in 67% of descriptions. A text model trained on raw descriptions would
learn to read the answer off the page, benchmark scores would be meaningless, and a
"description only" prediction mode would silently become "copy the salary".

## Decision

- Every free-text field used as input (`description`, `title`, `core_job_title`,
  `requirements_summary`) passes through `strip_html` and `strip_salary_mentions`
  before any feature is computed. Dollar amounts, `$120k` style figures, comma-grouped
  numbers, bare figures with a `k` suffix (`100k-150k`, `150-200k`), bare five- or
  six-digit integers with optional cents or a glued `USD` (`155000.00`, `229400USD`),
  and ranges joined by dashes or "to" are replaced with a `[SALARY]` token.
- The feature pipeline re-checks residual leakage (`check_salary_leakage` in
  `salary_scout.features`: the row's own min or max figure in any written form still
  present after scrubbing) and fails if it exceeds 0.1% of rows.
- The phrasing around pay ("plus equity", "pay transparency statement") is kept
  deliberately; it is legitimate signal about the employer.

## Consequences

- After scrubbing, the exact figure survives in 0.02% of the full cleaned set (5 of
  23,824 rows). The first regex caught only prefixed or comma-grouped forms and left
  1.8%; the two bare-number shapes above closed that gap.
- The bare five- or six-digit rule also eats zip codes and requisition numbers, and
  some non-salary numbers (customer counts, revenue) are scrubbed too. Acceptable loss.
- Spelled-out amounts are still not caught. The pipeline check exists to catch them if
  they matter.
