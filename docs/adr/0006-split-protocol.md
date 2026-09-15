# 0006. Group splits by collapse key, hold out the newest postings

Date: 2026-09-15
Status: Accepted

## Context

- `collapse_key` groups near-identical postings. Almost 90% of usable postings sit in a
  group with siblings, but only 13% of multi-posting groups share an identical salary.
  Siblings are the same role at different locations or levels: same title, employer,
  and most of the description text.
- A few employers contribute one to two hundred postings each out of nearly 7,000.
- 95% of postings were published between April and September 2026. The data is a
  snapshot, not a time series.

A random row-level split would put siblings on both sides of the boundary and inflate
every score.

## Decision

- Final test set: the most recent 15% of postings by `estimated_publish_date`, with
  whole collapse groups assigned to one side only.
- Model selection: grouped 5-fold cross-validation by `collapse_key` on the remaining data.
- Supplementary evaluation: company-held-out folds, reported separately, to quantify how
  much the model relies on recognising the employer.

## Consequences

- Reported scores are conservative relative to a naive split, and closer to what a user
  of the app experiences.
- The time-ordered holdout cannot measure salary drift; the data span is too short.
- Company-held-out scores will be lower; the report must explain why both are shown.
