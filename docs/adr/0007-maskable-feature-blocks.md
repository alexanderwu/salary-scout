# 0007. Support masked inputs with grouped feature blocks and block dropout

Date: 2026-09-15
Status: Proposed

## Context

The product requirement is a prediction from whatever the user has: description only,
metadata only, or any combination. Explanations must be understandable at the level of
"what kind of information drove this".

Options considered:

1. One model per mask pattern. Simple, but the number of patterns grows fast and the
   models disagree with each other.
2. One model that tolerates missing feature groups, trained with random block dropout
   so it sees masked examples during training.
3. Impute the missing block from the present ones. Adds a second model and hides the
   uncertainty.

## Decision

Option 2. Features are organised into five named blocks that map directly to the UI:

| block | source columns |
|---|---|
| `title` | `title`, `core_job_title` |
| `description` | `description` (scrubbed), `requirements_summary` |
| `role_meta` | `seniority_level`, `job_category`, `min_industry_and_role_yoe`, `commitment`, degree requirements, `security_clearance`, `technical_tools` |
| `location` | `workplace_type`, primary state, `workplace_countries`, latitude/longitude |
| `company` | `company_name`, `nb_employees`, `year_founded`, `organization_type`, `company_industries`, `source` |

A masked block is represented as missing values plus a per-block presence indicator.
During training each block is dropped at random with a fixed probability. Feature
attributions (SHAP) are summed per block for the explanation view.

## Consequences

- One model to train, evaluate, and deploy; the benchmark reports accuracy per mask
  pattern from the same model.
- Block dropout may cost some accuracy on fully specified inputs. The benchmark
  compares against a no-dropout model on the full-input case to measure that cost.
- If dropout proves too costly, option 1 becomes the fallback and this record is
  superseded.
