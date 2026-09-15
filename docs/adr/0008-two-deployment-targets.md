# 0008. Ship a browser-only demo and a separate service deployment

Date: 2026-09-15
Status: Accepted; the ONNX Runtime choice for the browser is superseded by 0012

## Context

Two audiences: recruiters and casual visitors who should be able to try the model
with zero setup, and a "scalable" version that demonstrates production practice.
A browser can run ONNX models and hashed text featurisers, but a full gradient-boosted
model with a sentence encoder and TreeSHAP is heavy to ship client-side.

## Decision

- **Browser demo.** A compact model exported to ONNX and run with ONNX Runtime Web.
  Text features use a hashing vectoriser so the featuriser can be reimplemented in
  JavaScript without shipping a vocabulary. Explanations use per-block attributions
  that are cheap to compute client-side (linear contributions or precomputed
  approximations). No server, no data leaves the page.
- **Service deployment.** The full model behind a FastAPI service in a Docker image,
  with batch scoring and the same block-masking contract as the browser demo.
- Both consume the same feature-block definitions from `salary_scout` so predictions
  agree on the blocks they share.

## Consequences

- The text featuriser must be hash-based from the first benchmark, or the browser demo
  will need a separate model. This constrains the feature pipeline now.
- The browser model will be less accurate than the service model; the UI must say so.
- Two deployment paths to maintain and document.
