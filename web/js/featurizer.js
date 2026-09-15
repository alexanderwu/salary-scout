// JavaScript port of salary_scout.dataset.prepare_inputs and
// salary_scout.features.FeatureBlocks, driven by web/model/featuriser.json.
//
// prepare(raw)            raw posting fields -> the derived-table columns
// featurizeBlock(...)     one block -> sparse [index, value] pairs, present or masked
// featurize(prepared, m)  all five blocks -> dense Float64Array
//
// Every rule here mirrors a scikit-learn transformer. The fixtures in
// web/model/fixtures.json are the contract: web/test/verify.mjs replays them.

import { hashIndex } from "./hash.js";

// Python's (?u)\b\w\w+\b: maximal runs of two or more Unicode word characters.
const TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;
const NAN_KEY = "__nan__";

function isMissing(v) {
  return v === null || v === undefined || v === "" || (typeof v === "number" && Number.isNaN(v));
}

function toNumber(v) {
  // pandas.to_numeric(errors="coerce"): anything non-numeric becomes NaN.
  if (isMissing(v)) return NaN;
  if (typeof v === "boolean") return v ? 1 : 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

/** salary_scout.dataset.parse_json_list */
export function parseJsonList(value) {
  if (typeof value !== "string" || !value.startsWith("[")) return [];
  let items;
  try {
    items = JSON.parse(value);
  } catch {
    return [];
  }
  if (!Array.isArray(items)) return [];
  return items
    .filter((x) => typeof x === "string" || typeof x === "number")
    .map((x) => String(x).trim())
    .filter((x) => x.length > 0);
}

function codePointLength(s) {
  let n = 0;
  for (const _ of s) n++;
  return n;
}

function sliceCodePoints(s, max) {
  if (max == null) return s;
  let out = "";
  let n = 0;
  for (const ch of s) {
    if (n++ >= max) break;
    out += ch;
  }
  return out;
}

export class Featurizer {
  constructor(spec) {
    this.spec = spec;
    this.nFeatures = spec.n_features;
    this.blocks = spec.blocks; // {block: [start, stop]}
    this.blockNames = Object.keys(spec.blocks);
    this.blockColumns = spec.block_columns;
    this.stopWords = new Set(spec.stop_words);
    this.seniorityOrder = spec.seniority_order;
    this.referenceYear = spec.reference_year;
    this.textColumns = spec.text_columns;
    this.htmlTagRe = new RegExp(spec.cleaning.html_tag_pattern, "g");
    this.salaryRe = new RegExp(spec.cleaning.salary_pattern, "gi");
    this.salaryToken = spec.cleaning.salary_token;
    this.byBlock = {};
    for (const b of this.blockNames) this.byBlock[b] = [];
    for (const t of spec.transformers) this.byBlock[t.block].push(t);
  }

  // -- prepare_inputs ------------------------------------------------------

  /** salary_scout.cleaning.strip_html then strip_salary_mentions. */
  cleanText(text) {
    if (typeof text !== "string") return "";
    let t = text.replace(this.htmlTagRe, " ");
    t = t.replaceAll("&nbsp;", " ").replaceAll("&amp;", "&");
    t = t.replace(/\s+/g, " ").trim();
    return t.replace(this.salaryRe, this.salaryToken);
  }

  /** Raw posting object -> derived columns the blocks read. */
  prepare(raw) {
    const out = { ...raw };
    for (const col of this.textColumns) out[`${col}_clean`] = this.cleanText(raw[col]);
    const states = parseJsonList(raw.workplace_states);
    out.primary_state = states.length ? states[0].replaceAll(", US", "") : null;
    out.n_states = states.length;
    out.is_remote = raw.workplace_type === "Remote" ? 1 : 0;
    return out;
  }

  // -- text ----------------------------------------------------------------

  tokens(text, useStopWords) {
    const lowered = text.toLowerCase();
    const found = lowered.match(TOKEN_RE) || [];
    return useStopWords ? found.filter((t) => !this.stopWords.has(t)) : found;
  }

  textHash(entry, value) {
    const text = sliceCodePoints(typeof value === "string" ? value : "", entry.max_chars);
    const toks = this.tokens(text, entry.stop_words);
    const [lo, hi] = entry.ngram_range;
    const idx = new Set();
    for (let n = lo; n <= hi; n++) {
      for (let i = 0; i + n <= toks.length; i++) {
        const gram = n === 1 ? toks[i] : toks.slice(i, i + n).join(" ");
        idx.add(hashIndex(gram, entry.n_features));
      }
    }
    if (idx.size === 0) return [];
    const v = 1 / Math.sqrt(idx.size); // binary=True then l2 norm
    return Array.from(idx, (j) => [entry.start + j, v]);
  }

  // -- the other families --------------------------------------------------

  onehot(entry, row, masked) {
    const out = [];
    entry.features.forEach((feat, i) => {
      const v = masked ? null : row[entry.columns[i]];
      const key = isMissing(v) ? NAN_KEY : String(v);
      if (key in feat.map) out.push([entry.start + feat.map[key], 1]);
      else if (feat.infrequent !== null && feat.infrequent_values.includes(key))
        out.push([entry.start + feat.infrequent, 1]);
      // unknown category: all zeros (handle_unknown="ignore")
    });
    return out;
  }

  numericValue(entry, col, v) {
    switch (entry.func) {
      case "identity":
        return toNumber(v);
      case "log1p": {
        const x = toNumber(v);
        return Number.isNaN(x) ? NaN : Math.log1p(Math.max(x, 0));
      }
      case "company_age": {
        const age = this.referenceYear - toNumber(v);
        return age < 0 || age > 300 || Number.isNaN(age) ? NaN : age;
      }
      case "seniority": {
        if (isMissing(v)) return NaN;
        const o = this.seniorityOrder[String(v)];
        return o === undefined ? NaN : o;
      }
      default:
        throw new Error(`unknown numeric func ${entry.func}`);
    }
  }

  numeric(entry, row, masked) {
    const out = [];
    const k = entry.columns.length;
    entry.columns.forEach((col, i) => {
      let x = masked ? NaN : this.numericValue(entry, col, row[col]);
      const missing = Number.isNaN(x);
      if (missing) x = entry.medians[i];
      if (x !== 0) out.push([entry.start + i, x]);
      const ind = entry.indicators.indexOf(i);
      if (ind >= 0 && missing) out.push([entry.start + k + ind, 1]);
    });
    return out;
  }

  multihot(entry, value) {
    let items = parseJsonList(value);
    if (entry.lowercase) items = items.map((x) => x.toLowerCase());
    const idx = new Set();
    const nv = entry.vocab.length;
    for (const item of items) {
      const j = entry.vocabIndex.get(item);
      if (j !== undefined) idx.add(entry.start + j);
      else if (entry.n_hash) idx.add(entry.start + nv + hashIndex(item, entry.n_hash));
    }
    const out = Array.from(idx, (j) => [j, 1]);
    if (items.length) out.push([entry.start + nv + entry.n_hash, Math.log1p(items.length)]);
    return out;
  }

  targetEnc(entry, value) {
    const hit = isMissing(value) ? undefined : entry.table[String(value)];
    const [enc, logFreq] = hit || [entry.prior, 0];
    const out = [[entry.start, enc]];
    if (logFreq !== 0) out.push([entry.start + 1, logFreq]);
    return out;
  }

  // -- assembly ------------------------------------------------------------

  /** Sparse [index, value] pairs for one block; `masked` reproduces mask_blocks. */
  featurizeBlock(block, row, masked = false) {
    const out = [];
    for (const entry of this.byBlock[block]) {
      switch (entry.kind) {
        case "presence":
          if (!masked) out.push([entry.start, 1]);
          break;
        case "text_hash":
          if (!masked) out.push(...this.textHash(entry, row[entry.column]));
          break;
        case "log_len": {
          const s = masked || typeof row[entry.column] !== "string" ? "" : row[entry.column];
          const v = Math.log1p(codePointLength(s));
          if (v !== 0) out.push([entry.start, v]);
          break;
        }
        case "onehot":
          out.push(...this.onehot(entry, row, masked));
          break;
        case "numeric":
          out.push(...this.numeric(entry, row, masked));
          break;
        case "multihot":
          if (!entry.vocabIndex) entry.vocabIndex = new Map(entry.vocab.map((v, i) => [v, i]));
          if (!masked) out.push(...this.multihot(entry, row[entry.column]));
          break;
        case "target_enc":
          out.push(...this.targetEnc(entry, masked ? null : row[entry.column]));
          break;
        default:
          throw new Error(`unknown transformer kind ${entry.kind}`);
      }
    }
    return out;
  }

  /** Dense feature vector for a prepared row with the given blocks masked. */
  featurize(row, masked = new Set()) {
    const x = new Float64Array(this.nFeatures);
    for (const b of this.blockNames) {
      for (const [i, v] of this.featurizeBlock(b, row, masked.has(b))) x[i] = v;
    }
    return x;
  }
}
