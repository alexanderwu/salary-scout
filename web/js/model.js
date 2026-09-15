// The browser model: featuriser + two tree ensembles + block-coalition Shapley values.
// Mirrors salary_scout.models.SalaryModel (predict, explain(method="coalition")).

import { Featurizer } from "./featurizer.js";
import { TreeEnsemble } from "./trees.js";

function factorial(n) {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

/** low, high, mid in dollars from the two targets (ADR 0003). */
export function reconstructRange(logMid, logSpread) {
  const mid = Math.exp(logMid);
  const spread = Math.exp(Math.max(logSpread, 0));
  const low = (2 * mid) / (1 + spread);
  return { low, high: low * spread, mid };
}

export class SalaryModel {
  constructor(featSpec, midSpec, spreadSpec, metrics) {
    this.featurizer = new Featurizer(featSpec);
    this.mid = new TreeEnsemble(midSpec);
    this.spread = new TreeEnsemble(spreadSpec);
    this.metrics = metrics;
    this.meta = featSpec.meta;
    this.blockNames = this.featurizer.blockNames;
    this.baseline = featSpec.meta.baseline_log_mid;
  }

  static async load(baseUrl = "model/") {
    const get = (f) => fetch(baseUrl + f).then((r) => {
      if (!r.ok) throw new Error(`${f}: ${r.status}`);
      return r.json();
    });
    const [feat, mid, spread, metrics] = await Promise.all([
      get("featuriser.json"), get("trees_mid.json"), get("trees_spread.json"), get("metrics.json"),
    ]);
    return new SalaryModel(feat, mid, spread, metrics);
  }

  prepare(raw) {
    return this.featurizer.prepare(raw);
  }

  /** Per-block sparse features, present and masked, so coalitions are cheap to assemble. */
  blockParts(prepared) {
    const parts = {};
    for (const b of this.blockNames) {
      parts[b] = {
        present: this.featurizer.featurizeBlock(b, prepared, false),
        masked: this.featurizer.featurizeBlock(b, prepared, true),
      };
    }
    return parts;
  }

  assemble(parts, presentSet) {
    const x = new Float64Array(this.featurizer.nFeatures);
    for (const b of this.blockNames) {
      for (const [i, v] of presentSet.has(b) ? parts[b].present : parts[b].masked) x[i] = v;
    }
    return x;
  }

  /** Prediction with the given blocks masked. */
  predict(prepared, masked = new Set()) {
    const present = new Set(this.blockNames.filter((b) => !masked.has(b)));
    const x = this.assemble(this.blockParts(prepared), present);
    const logMid = this.mid.predict(x);
    const logSpread = this.spread.predict(x);
    return { logMid, logSpread, ...reconstructRange(logMid, logSpread) };
  }

  /**
   * Exact Shapley values of log_mid over the present blocks (masked blocks get 0).
   * The empty coalition is worth the training-mean baseline. Sum + baseline = log_mid.
   */
  explain(prepared, masked = new Set()) {
    const present = this.blockNames.filter((b) => !masked.has(b));
    const n = present.length;
    const parts = this.blockParts(prepared);
    const value = new Map(); // bitmask over `present` -> log_mid
    for (let m = 0; m < 1 << n; m++) {
      if (m === 0) {
        value.set(0, this.baseline);
        continue;
      }
      const set = new Set(present.filter((_, i) => m & (1 << i)));
      value.set(m, this.mid.predict(this.assemble(parts, set)));
    }
    const phi = {};
    for (const b of this.blockNames) phi[b] = 0;
    present.forEach((b, i) => {
      const bit = 1 << i;
      let sum = 0;
      for (let m = 0; m < 1 << n; m++) {
        if (m & bit) continue;
        let s = 0;
        for (let j = 0; j < n; j++) if (m & (1 << j)) s++;
        const w = (factorial(s) * factorial(n - s - 1)) / factorial(n);
        sum += w * (value.get(m | bit) - value.get(m));
      }
      phi[b] = sum;
    });
    const full = value.get((1 << n) - 1) ?? this.baseline;
    return { baseline: this.baseline, blocks: phi, logMid: full };
  }

  /** Holdout metrics for the combination of present blocks. */
  metricsFor(masked = new Set()) {
    const key = this.blockNames.filter((b) => !masked.has(b)).join("+");
    return this.metrics.by_present_blocks[key] || null;
  }
}
