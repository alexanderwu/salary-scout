// Page logic: form <-> raw posting, prediction, explanation chart, example loader.

import { SalaryModel } from "./model.js";

const form = document.getElementById("posting-form");
const $ = (id) => document.getElementById(id);
const BLOCK_LABELS = {
  title: "Title",
  description: "Description",
  role_meta: "Role metadata",
  location: "Location",
  company: "Company",
};
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const pct = (v, digits = 0) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;

let model = null;
let samples = [];
let currentSample = null;

// -- options from the featuriser spec --------------------------------------

function optionValues(spec, key) {
  // "block:onehot:column" -> categories of that one-hot column; "multihot:name" -> vocab
  const [a, b, c] = key.split(":");
  if (a === "multihot") {
    const entry = spec.transformers.find((t) => t.name === `${b}:${c}`);
    return entry.vocab.map((v) => JSON.stringify([capitalise(v)]));
  }
  const entry = spec.transformers.find((t) => t.name === `${a}:${b}`);
  const i = entry.columns.indexOf(c);
  const feat = entry.features[i];
  const values = Object.keys(feat.map).filter((k) => k !== "__nan__");
  if (feat.infrequent !== null) values.push(...feat.infrequent_values.filter((k) => k !== "__nan__"));
  return values.sort((x, y) => x.localeCompare(y));
}

function capitalise(s) {
  return s.replace(/\b\w/g, (m) => m.toUpperCase());
}

function fillSelects(spec) {
  for (const select of form.querySelectorAll("select[data-options]")) {
    const values = optionValues(spec, select.dataset.options);
    select.append(new Option("—", ""));
    for (const v of values) {
      const label = select.name === "commitment" ? JSON.parse(v).join(", ") : v;
      select.append(new Option(label, v));
    }
  }
  const company = spec.transformers.find((t) => t.kind === "target_enc");
  const list = $("company-names");
  const frag = document.createDocumentFragment();
  for (const name of Object.keys(company.table).sort()) {
    const o = document.createElement("option");
    o.value = name;
    frag.append(o);
  }
  list.append(frag);
}

// -- form <-> raw posting ----------------------------------------------------

const listFields = ["technical_tools", "workplace_countries", "company_industries"];

function csvToJsonList(text) {
  const items = (text || "").split(",").map((s) => s.trim()).filter(Boolean);
  return items.length ? JSON.stringify(items) : null;
}

function jsonListToCsv(value) {
  if (typeof value !== "string" || !value.startsWith("[")) return value || "";
  try {
    return JSON.parse(value).join(", ");
  } catch {
    return "";
  }
}

function readForm() {
  const fd = new FormData(form);
  const raw = {};
  for (const [k, v] of fd.entries()) {
    if (k.startsWith("use-")) continue;
    raw[k] = v === "" ? null : v;
  }
  for (const f of listFields) raw[f] = csvToJsonList(raw[f]);
  // states: the primary state plus placeholders for the count
  const n = Math.max(0, parseInt(raw.n_states ?? "", 10) || (raw.primary_state ? 1 : 0));
  const states = [];
  if (raw.primary_state && n > 0) states.push(`${raw.primary_state}, US`);
  while (states.length < n) states.push("Other, US");
  raw.workplace_states = JSON.stringify(states);
  delete raw.primary_state;
  delete raw.n_states;
  for (const k of ["nb_employees", "year_founded", "latitude", "longitude", "min_industry_and_role_yoe", "min_management_yoe"]) {
    if (raw[k] !== null && raw[k] !== undefined) raw[k] = Number(raw[k]);
  }
  return raw;
}

function maskedBlocks() {
  const masked = new Set();
  for (const b of Object.keys(BLOCK_LABELS)) {
    if (!form.elements[`use-${b}`].checked) masked.add(b);
  }
  return masked;
}

function setField(name, value) {
  const el = form.elements[name];
  if (!el) return;
  if (value === null || value === undefined) {
    el.value = "";
    return;
  }
  if (el.tagName === "SELECT") {
    const v = String(value);
    if (![...el.options].some((o) => o.value === v)) el.append(new Option(v, v));
    el.value = v;
  } else {
    el.value = String(value);
  }
}

function loadSample(sample) {
  currentSample = sample;
  for (const [k, v] of Object.entries(sample)) {
    if (k.startsWith("true_") || k === "requisition_id") continue;
    if (listFields.includes(k)) setField(k, jsonListToCsv(v));
    else if (k !== "workplace_states") setField(k, v);
  }
  let states = [];
  try { states = JSON.parse(sample.workplace_states || "[]"); } catch { /* ignore */ }
  setField("primary_state", states.length ? states[0].replaceAll(", US", "") : null);
  setField("n_states", states.length || null);
  for (const b of Object.keys(BLOCK_LABELS)) form.elements[`use-${b}`].checked = true;
  update();
}

// -- rendering ---------------------------------------------------------------

function renderAttribution(explanation, masked) {
  const blocks = Object.keys(BLOCK_LABELS);
  const effects = blocks.map((b) => (masked.has(b) ? 0 : Math.exp(explanation.blocks[b]) - 1));
  const maxAbs = Math.max(0.05, ...effects.map(Math.abs));
  const W = 340, labelW = 110, rowH = 26, top = 6;
  const plotW = W - labelW - 56;
  const zero = labelW + plotW / 2;
  const scale = (v) => (v / maxAbs) * (plotW / 2);
  const H = top + rowH * blocks.length + 4;
  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<line class="grid" x1="${labelW}" x2="${W - 56}" y1="${H - 2}" y2="${H - 2}"/>`;
  svg += `<line class="axis" x1="${zero}" x2="${zero}" y1="${top}" y2="${H - 2}"/>`;
  const tbody = $("attribution-table").querySelector("tbody");
  tbody.innerHTML = "";
  blocks.forEach((b, i) => {
    const y = top + i * rowH;
    const v = effects[i];
    const isMasked = masked.has(b);
    const w = Math.abs(scale(v));
    const x = v >= 0 ? zero : zero - w;
    const cls = isMasked ? "masked" : v >= 0 ? "raise" : "lower";
    const label = isMasked ? "not used" : pct(v, 1);
    const title = isMasked ? `${BLOCK_LABELS[b]}: switched off` : `${BLOCK_LABELS[b]}: ${label} on the midpoint`;
    svg += `<g class="row"><title>${title}</title>`;
    svg += `<rect class="hit" x="${labelW}" y="${y}" width="${plotW}" height="${rowH}"/>`;
    svg += `<text x="${labelW - 8}" y="${y + rowH / 2 + 4}" text-anchor="end">${BLOCK_LABELS[b]}</text>`;
    if (isMasked) {
      svg += `<rect class="bar masked" x="${zero - 1}" y="${y + 7}" width="2" height="${rowH - 14}"/>`;
    } else {
      svg += `<rect class="bar ${cls}" x="${x}" y="${y + 6}" width="${Math.max(w, 1)}" height="${rowH - 12}" rx="2"/>`;
    }
    svg += `<text class="value" x="${W - 50}" y="${y + rowH / 2 + 4}">${label}</text></g>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${BLOCK_LABELS[b]}</td><td>${label}</td>`;
    tbody.append(tr);
  });
  svg += "</svg>";
  $("attribution").innerHTML = svg;
}

function update() {
  if (!model) return;
  const masked = maskedBlocks();
  for (const fs of form.querySelectorAll("fieldset.block")) {
    fs.classList.toggle("masked", masked.has(fs.dataset.block));
  }
  const raw = readForm();
  const prepared = model.prepare(raw);
  const notes = [];

  if (masked.size === Object.keys(BLOCK_LABELS).length) {
    $("range").textContent = "—";
    $("mid").textContent = "—";
    $("mape").textContent = "—";
    $("overlap").textContent = "—";
    $("pattern-note").textContent = "Switch on at least one block.";
    $("attribution").innerHTML = "";
    $("notes").innerHTML = "";
    return;
  }

  const pred = model.predict(prepared, masked);
  const explanation = model.explain(prepared, masked);
  $("range").textContent = `${usd.format(pred.low)} – ${usd.format(pred.high)}`;
  $("mid").textContent = usd.format(pred.mid);

  const m = model.metricsFor(masked);
  if (m) {
    $("mape").textContent = `±${(m.mape * 100).toFixed(0)}%`;
    $("overlap").textContent = `${(m.overlap * 100).toFixed(0)}%`;
    const used = Object.keys(BLOCK_LABELS).filter((b) => !masked.has(b)).map((b) => BLOCK_LABELS[b].toLowerCase());
    $("pattern-note").textContent = `Measured on ${model.metrics.n_test.toLocaleString()} held-out postings with ${used.join(", ")} as the only inputs. Mean absolute error of the midpoint: ${usd.format(m.mae_usd)}.`;
  }

  renderAttribution(explanation, masked);

  if (currentSample) {
    const t = $("truth");
    t.hidden = false;
    t.textContent = `advertised: ${usd.format(currentSample.true_low)} – ${usd.format(currentSample.true_high)}`;
  } else {
    $("truth").hidden = true;
  }

  const companyEntry = model.featurizer.spec.transformers.find((t) => t.kind === "target_enc");
  if (!masked.has("company")) {
    if (!raw.company_name) notes.push("No employer given, so the company block uses the average employer.");
    else if (!(raw.company_name in companyEntry.table)) notes.push(`"${raw.company_name}" was not in the training data, so it received the average employer's pricing. Employers with a history in the data can move the estimate a lot.`);
    else notes.push(`${raw.company_name} has ${Math.round(Math.expm1(companyEntry.table[raw.company_name][1]))} postings in the training data; its pricing history is part of the estimate.`);
  }
  if (!masked.has("description") && (!raw.description || raw.description.length < 200)) {
    notes.push("The description is short; the description block carries the most signal when it is the full posting.");
  }
  if (pred.logSpread <= 0) notes.push("The spread model predicted a zero-width band; the range is shown as a point.");
  $("notes").innerHTML = notes.map((n) => `<li>${n}</li>`).join("");
}

// -- boot --------------------------------------------------------------------

let timer = null;
form.addEventListener("input", () => {
  currentSample = null;
  clearTimeout(timer);
  timer = setTimeout(update, 120);
});
form.addEventListener("change", (e) => {
  if (e.target.name?.startsWith("use-")) update();
});
form.addEventListener("submit", (e) => e.preventDefault());

$("sample-select").addEventListener("change", (e) => {
  const s = samples[Number(e.target.value)];
  if (s) loadSample(s);
});

async function boot() {
  try {
    const [m, s] = await Promise.all([
      SalaryModel.load("model/"),
      fetch("model/samples.json").then((r) => (r.ok ? r.json() : [])),
    ]);
    model = m;
    samples = s;
    fillSelects(model.featurizer.spec);
    const picker = $("sample-select");
    samples.forEach((row, i) => {
      picker.append(new Option(`${row.title} — ${row.company_name || "unknown employer"}`, String(i)));
    });
    $("loading").hidden = true;
    $("result").hidden = false;
    if (samples.length) {
      picker.value = "0";
      loadSample(samples[0]);
    } else {
      update();
    }
  } catch (err) {
    const el = $("loading");
    el.textContent = `Could not load the model: ${err.message}. Serve this folder over HTTP (for example "python -m http.server" in web/).`;
    el.classList.add("error");
  }
}

boot();
