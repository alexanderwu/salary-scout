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
  numbers, and ranges joined by dashes or "to" are replaced with a `[SALARY]` token.
- The feature pipeline re-checks residual leakage (exact figure still present after
  scrubbing) and fails loudly if it exceeds a small tolerance.
- The phrasing around pay ("plus equity", "pay transparency statement") is kept
  deliberately; it is legitimate signal about the employer.

## Consequences

- After scrubbing, the exact figure survives in 0.00% of the cleaned EDA sample.
- Non-comma integers such as `120000` and spelled-out amounts are not caught by the
  current regex. The pipeline check exists to catch them if they matter.
- Some non-salary numbers (customer counts, revenue) are scrubbed too. Acceptable loss.
