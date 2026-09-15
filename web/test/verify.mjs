// Replays web/model/fixtures.json through the JavaScript featuriser and trees and
// compares with the Python outputs. Run with:  node web/test/verify.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { SalaryModel } from "../js/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = join(here, "..", "model");
const load = (f) => JSON.parse(readFileSync(join(modelDir, f), "utf-8"));

const model = new SalaryModel(
  load("featuriser.json"), load("trees_mid.json"), load("trees_spread.json"), load("metrics.json"),
);
const fx = load("fixtures.json");
const masks = model.meta.masks;
const TOL = 1e-9;

let failures = 0;
const fail = (msg) => {
  failures++;
  if (failures <= 25) console.log("FAIL", msg);
};

// 1. Feature vectors: per-block nnz and sum on the full input.
fx.inputs.forEach((raw, i) => {
  const prepared = model.prepare(raw);
  const x = model.featurizer.featurize(prepared);
  for (const [b, [lo, hi]] of Object.entries(model.featurizer.blocks)) {
    let nnz = 0;
    let sum = 0;
    for (let j = lo; j < hi; j++) if (x[j] !== 0) { nnz++; sum += x[j]; }
    const exp = fx.block_stats[i][b];
    if (nnz !== exp.nnz || Math.abs(sum - exp.sum) > 1e-6) {
      fail(`row ${i} block ${b}: nnz ${nnz} vs ${exp.nnz}, sum ${sum.toFixed(6)} vs ${exp.sum.toFixed(6)}`);
    }
  }
});

// 2. Predictions under every benchmark mask pattern.
let maxErr = 0;
for (const [pattern, hidden] of Object.entries(masks)) {
  const masked = new Set(hidden);
  fx.inputs.forEach((raw, i) => {
    const p = model.predict(model.prepare(raw), masked);
    const em = Math.abs(p.logMid - fx.expected[pattern].log_mid[i]);
    const es = Math.abs(p.logSpread - fx.expected[pattern].log_spread[i]);
    maxErr = Math.max(maxErr, em, es);
    if (em > TOL || es > TOL) fail(`row ${i} pattern ${pattern}: log_mid err ${em.toExponential(2)}, log_spread err ${es.toExponential(2)}`);
  });
}

// 3. Coalition Shapley values on the first rows.
const nExplain = fx.explain_coalition.baseline.length;
for (let i = 0; i < nExplain; i++) {
  const ex = model.explain(model.prepare(fx.inputs[i]));
  if (Math.abs(ex.baseline - fx.explain_coalition.baseline[i]) > TOL) fail(`row ${i} baseline`);
  for (const b of model.blockNames) {
    const err = Math.abs(ex.blocks[b] - fx.explain_coalition[b][i]);
    if (err > 1e-8) fail(`row ${i} shapley ${b}: err ${err.toExponential(2)}`);
  }
}

// 4. Metrics lookup.
if (!model.metricsFor(new Set()) || !model.metricsFor(new Set(["title", "description"]))) {
  fail("metrics lookup");
}

const nChecks = fx.inputs.length * (5 + 2 * Object.keys(masks).length) + nExplain * 6 + 1;
console.log(`${nChecks - failures}/${nChecks} checks passed on ${fx.inputs.length} fixture rows; max prediction error ${maxErr.toExponential(2)}`);
process.exit(failures ? 1 : 0);
