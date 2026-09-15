# 0010. Encode employer identity with group-cross-fitted target encoding

Date: 2026-09-15
Status: Accepted

## Context

The training split has 20,257 postings from 6,924 employers, and the EDA showed that
employer explains a large share of pay variance. The `company` block needs a
representation of `company_name` that works for a linear model, a gradient-boosted
model, and a browser model with a small lookup table.

Options considered:

1. One-hot with a frequency floor. Thousands of near-empty columns; employers below the
   floor collapse to nothing.
2. Hashing. Compact, but the model has to learn each bucket from scratch and
   collisions mix unrelated employers.
3. Mean-target encoding. One column, meaningful for every model type, and the
   lookup table is 7k floats.

Target encoding leaks unless it is cross-fitted, and the obvious cross-fit (random
K-fold) is not enough here: nearly 90% of postings have siblings under the same
`collapse_key` with near-identical pay, so a row's encoding would be computed from its
own siblings. scikit-learn's `TargetEncoder` only supports plain K-fold.

## Decision

- `GroupTargetEncoder` in `salary_scout.features` encodes `company_name` as the
  smoothed mean of `log_mid` (smoothing weight 10 toward the global mean) plus
  `log1p` of the employer's training frequency.
- `fit_transform` cross-fits with `GroupKFold` on `collapse_key` (five folds), so each
  training row is encoded from other collapse groups only. `transform` uses the
  full-training encoding. Unknown or masked employers get the prior and frequency zero.
- The encoder takes `collapse_key` as a second input column purely for the cross-fit;
  the column is never a feature and is filled with NaN when absent at inference.
- The whole `FeatureBlocks` transformer is therefore fitted inside every CV fold with
  `y = log_mid`. Both targets share the resulting matrix.

## Consequences

- `fit_transform(train)` and `transform(train)` differ in the company block by design.
  Any comparison of masked and unmasked features must use `transform` for both.
- The browser demo ships a table of about 7k employer values. Employers absent from
  training receive the prior, which is the honest answer.
- Company-held-out folds (ADR 0006) now have a precise meaning: they measure how much
  accuracy comes from this one encoding.
- Option 1 remains available for the small categorical company fields
  (`organization_type`, sector, HQ country, source), which are one-hot encoded.
