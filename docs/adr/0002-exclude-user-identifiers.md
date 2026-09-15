# 0002. Exclude third-party user identifiers from all derived data

Date: 2026-09-15
Status: Accepted

## Context

The `job_information_json` column carries arrays of hiring.cafe account IDs for people
who viewed, saved, hid, or applied to each posting. Those people did not consent to
appearing in a public portfolio project, and the field has no modelling value for salary.

## Decision

The loader never selects `job_information_json`. No exported table, notebook output,
model artefact, or application bundle may contain it or anything derived from it.
A unit test asserts the column is absent from the loader query.

## Consequences

- Engagement signals (view and save counts) are unavailable as features. They would in
  any case be leakage-prone, since popularity is observed after the salary is posted.
- Any future feature that needs engagement data requires a new record that addresses
  aggregation and consent.
