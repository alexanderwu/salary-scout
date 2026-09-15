# 0003. Predict log midpoint and log spread of the USD range

Date: 2026-09-15
Status: Accepted

## Context

Postings advertise a range, not a number. From the EDA (`notebooks/01_eda.ipynb`):

- USD accounts for 98% of postings with pay; other currencies total a few dozen rows each.
- The midpoint is right-skewed in dollars and close to symmetric in log space.
- The ratio max/min sits mostly between 1.2 and 1.6, is 1.0 for 4% of postings, and is
  only weakly correlated with the level (r = 0.19).
- Predicting min and max as two independent regressions can yield max below min.

## Decision

- Scope version one to USD postings. No currency conversion.
- Model two targets in log space:
  - `log_mid = log((min + max) / 2)`, the pay level
  - `log_spread = log(max / min)`, the band width, constrained to be non-negative
- Reconstruct the advertised range from the two predictions, which guarantees ordering.
- Report error on the midpoint in dollars (MAE and MAPE) and the share of test postings
  whose true range overlaps the predicted range.

## Consequences

- One additional model head, or a two-output model, compared with predicting a point.
- The app can show a range with an honest width instead of a single number.
- Non-USD markets are out of scope and must be stated as such in the report and UI.
