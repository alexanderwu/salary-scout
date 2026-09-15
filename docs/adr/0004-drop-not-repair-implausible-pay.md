# 0004. Drop implausible compensation rows rather than repair them

Date: 2026-09-15
Status: Accepted

## Context

Raw annualised compensation runs from 1 dollar to 340 million. The extremes are unit
errors in both directions: hourly rates stored as yearly (`51.80 – 60.69`), and yearly
salaries multiplied by 2080 as if they were hourly (`223,479,360`). There are also
genuine low values for part-time contract annotation work, which is a different labour
market from the full-time salaried postings that make up 97% of the data.

Repairing would mean guessing the direction of each error per row.

## Decision

Keep a row only if all of the following hold (`salary_scout.cleaning`):

| rule | threshold |
|---|---|
| currency | USD |
| yearly minimum | at least 20,000 |
| yearly maximum | at most 1,000,000 |
| spread ratio max/min | at most 3 |
| ordering | min ≤ max |

Report the drop count for each rule in the EDA notebook. Thresholds are module
constants so they can be revisited in one place.

## Consequences

- 367 of 24,191 postings with pay are removed, leaving 23,824.
- The model cannot predict pay for part-time or hourly contract roles below 20k a year.
- Postings that sit outside the envelope for legitimate reasons (very senior roles above
  1M) are excluded; the 99.9th percentile of the raw data is about 1.1M so the loss is small.
