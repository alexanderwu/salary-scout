// Evaluator for the flattened LightGBM trees written by salary_scout.export.trees_spec.
// Mirrors LightGBM's NumericalDecision: a node's missing_type decides where NaN and
// zero go; otherwise value <= threshold goes left. Regression output is the plain sum
// of leaf values over all trees.

const ZERO_THRESHOLD = 1e-35;

export class TreeEnsemble {
  constructor(spec) {
    this.nFeatures = spec.n_features;
    this.trees = spec.trees.map((t) => ({
      feature: Int32Array.from(t.feature),
      threshold: Float64Array.from(t.threshold),
      left: Int32Array.from(t.left),
      right: Int32Array.from(t.right),
      defaultLeft: Uint8Array.from(t.default_left),
      missing: Uint8Array.from(t.missing),
      leafValue: Float64Array.from(t.leaf_value),
    }));
  }

  static treeValue(t, x) {
    if (t.feature.length === 0) return t.leafValue[0];
    let i = 0;
    for (;;) {
      let v = x[t.feature[i]];
      const mt = t.missing[i];
      if (mt !== 2 && Number.isNaN(v)) v = 0;
      let next;
      if ((mt === 1 && Math.abs(v) <= ZERO_THRESHOLD) || (mt === 2 && Number.isNaN(v))) {
        next = t.defaultLeft[i] ? t.left[i] : t.right[i];
      } else {
        next = v <= t.threshold[i] ? t.left[i] : t.right[i];
      }
      if (next < 0) return t.leafValue[-next - 1];
      i = next;
    }
  }

  predict(x) {
    let s = 0;
    for (const t of this.trees) s += TreeEnsemble.treeValue(t, x);
    return s;
  }
}
