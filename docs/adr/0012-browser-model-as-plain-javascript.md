# 0012. Run the browser model as a plain JavaScript port, not ONNX Runtime

Date: 2026-09-15
Status: Accepted (supersedes the runtime choice in 0008; the two-target split stands)

## Context

ADR 0008 proposed ONNX Runtime Web for the browser demo. When it came to building it,
three facts argued against that:

- The model is two LightGBM boosters over a 119,710-column sparse vector. ONNX's tree
  operator takes dense float32 input and stores float32 thresholds, while LightGBM
  splits on float64. On l2-normalised hashed features (values like 1/sqrt(277)), the
  known threshold-rounding mismatches would make the browser disagree with Python for
  some rows, and the disagreement could not be tested away.
- ONNX Runtime Web is a 10 MB+ WebAssembly download on top of the model, for a demo
  whose model is 6 MB of JSON.
- The featuriser had to be ported to JavaScript regardless (hashing, one-hot,
  imputation, the company table). A tree walker is 40 lines next to that.

The explanation also needed a decision. TreeSHAP in the browser is heavy; ADR 0008
allowed "linear contributions or precomputed approximations".

## Decision

- `salary_scout.export` writes the fitted `FeatureBlocks` as a JSON spec (one entry per
  sub-transformer, with its fitted state) and each booster as flat arrays from
  `dump_model()`, keeping float64 thresholds and LightGBM's missing-value rules.
- `web/js/` reimplements `prepare_inputs`, every transformer family, and the tree
  evaluation in dependency-free ES modules. No build step, no runtime library.
- The contract is a fixture file of 60 raw test postings with the Python predictions
  under every mask pattern and the Python explanations. `web/test/verify.mjs` replays
  them; `tests/test_web.py` runs it under pytest when Node is present. The port must
  match to 1e-9 in log space.
- Explanations are exact Shapley values over the five blocks, computed by masking
  every coalition (32 predictions). The empty coalition is worth the training mean of
  `log_mid`, not an all-masked prediction the model never saw. The Python model offers
  the same method (`SalaryModel.explain(method="coalition")`) next to summed TreeSHAP,
  and the fixtures pin the two implementations together.
- The browser model uses hash widths 2^14 (title) and 2^16 (description), the same
  training settings otherwise, and quotes its own time-holdout error for the exact
  combination of blocks in use (`metrics.json`, 31 combinations).

## Consequences

- Python and the browser agree to floating-point precision on every fixture row, so
  the demo's numbers are the report's numbers with a smaller hash. No "approximately
  the same model" caveat is needed in the UI.
- Any change to `features.py` must be mirrored in `web/js/featurizer.js`; the fixture
  test fails loudly when it is not, which is the intended guard.
- Regex-based cleaning is shared by exporting the Python patterns into the spec; the
  only known divergence is Unicode `\b`/`\d` semantics between Python `re` and
  JavaScript, which no fixture row has hit.
- The exported files (about 7 MB, mostly the two tree files) are committed under
  `web/model/` so the demo can be hosted as static files.
- The service deployment (ADR 0008) is unaffected and still runs the full model.
