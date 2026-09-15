# Salary Scout: UI/UX review of the browser demo

Review of `web/` as of commit `485998e`, September 2026. Measured against the users and
the trust ladder in `docs/user_stories.md`, and against the claims the model can actually
support per `docs/report.md`.

Everything quoted below was measured by driving the live page, not read off the source.
Method: `python -m http.server -d web 8765` with Playwright/Chromium at 1280×900 and
390×844, reading the rendered DOM.

## 1. Summary

The demo is well built. It is fast, it has no dependencies, the explanation is a genuine
Shapley decomposition rather than a feature-importance chart, and the accuracy figures
come from `metrics.json` rather than from prose that can drift. Those are not small things
and most of what follows keeps them.

The gap is not polish. It is that **the interface is organised around the model's
structure (five maskable blocks, 40 fields) rather than around the question a user came to
ask**, and that **its confidence is decorative**: the same "±14% typical error" is printed
over an estimate built from a full posting and over an estimate built from nothing at all.

Three findings are severe enough to undermine the project's own honesty claims:

| # | Finding | Evidence |
|---|---|---|
| 1 | An empty form produces a confident estimate with full-input error quoted | Blank every field, leave all blocks on: **\$97,706 – \$129,957, "±14% typical error"** |
| 2 | Out-of-scope inputs return confident numbers | "Barista" and "Plumber" both return **\$117,956 – \$161,129**, the same answer as "asdf asdf" |
| 3 | The bundled examples visibly contradict the headline accuracy | Across all 12: mean absolute midpoint error **21.5%**, and **3 of 12** predicted ranges do not overlap the advertised one, against claims of ±14% and 92% |

Each is fixable without touching the model.

## 2. What already works and should be protected

- **Speed.** The model is usable 240 ms after load on localhost. Editing a field
  re-runs 31 tree-ensemble evaluations for exact Shapley values and the update is
  imperceptible.
- **The attribution view.** Five bars, one per input block, with a hidden data table for
  screen readers and a `role="img"` label on the SVG. This is the best part of the page.
- **Live update on every keystroke.** No submit button. Correct for Marcus, who explores
  rather than looks up.
- **Metrics driven by data.** `metricsFor()` keys into `metrics.json` by block pattern, so
  a refit updates the quoted error automatically.
- **Privacy.** Everything runs in the page. This is a real differentiator and section 5.9
  argues it is under-sold rather than over-sold.
- **Sensible empty state.** Switching off all five blocks says "Switch on at least one
  block" rather than erroring.

## 3. Findings

### 3.1 A blank form is treated as a full-information posting

Clear every field and leave the five toggles on:

```
Estimated advertised range   $97,706 – $129,957
Midpoint                     $113,832
Typical error for these inputs   ±14%
Ranges that overlapped the real one   92%
Measured on 3,567 held-out postings with title, description, role metadata,
location, company as the only inputs.
```

The model is behaving as designed: `metricsFor()` keys on which blocks are *switched on*,
and all five are. But "present with every field blank" is not a state the model was trained
on — training rows always had content when a block was present — and ±14% is the error
measured on postings that had a full title, a full description, and complete metadata.

The app quotes its most flattering accuracy figure over its least informed estimate. For
Dana (`user_stories.md` §3.3) this is the finding that ends the evaluation, and for Priya
it is a number quoted in a screening call that nothing supports.

The two notes that do appear ("No employer given…", "The description is short…") are
correct and far too quiet: body-size list items below the fold of the card.

### 3.2 Nothing marks the edge of the training distribution

Title-only mode, one field, measured:

| Title typed | Estimate | Quoted error |
|---|---|---|
| Barista | \$117,956 – \$161,129 | ±21% |
| Plumber | \$117,956 – \$161,129 | ±21% |
| asdf asdf | \$117,807 – \$160,675 | ±21% |
| Registered Nurse | \$112,982 – \$154,308 | ±21% |
| Neurosurgeon | \$121,378 – \$170,257 | ±21% |
| Chief Executive Officer | \$112,950 – \$153,365 | ±21% |
| Senior Data Engineer | \$137,278 – \$179,791 | ±21% |

The data contains no management roles at all and no roles outside data, analytics,
software, IT and engineering (`report.md` §9). "Barista" and "asdf asdf" returning the same
number is the model correctly saying *I have nothing*, and the interface rendering that as
a currency figure to the dollar.

Note what the attribution panel does here: Barista contributes +14.3%, "asdf asdf" +14.1%,
"Senior Data Engineer" +29.9%, "Staff Machine Learning Engineer" +68.7%. The floor of about
+14% is the block-presence indicator, not the words. So the signal exists in the page
already — an input whose contribution sits at the empty-text floor is an input the model
could not read. Section 5.2 turns that into a warning.

### 3.3 The shipped examples undercut the shipped accuracy

`build_samples()` draws 12 rows uniformly at random from the test split (seed 1). Driving
each one:

| # | Example | Predicted mid | Advertised mid | Error | Ranges overlap |
|---|---|---|---|---|---|
| 0 | Lead AI Engineer — Stjude | 169,710 | 132,080 | +28% | yes |
| 1 | Senior Machine Learning Engineer — Generalmotors | 168,639 | 225,054 | −25% | **no** |
| 2 | Remote Finance Analyst — Turing | 112,955 | 312,000 | −64% | **no** |
| 3 | Data Scientist — Molina | 133,379 | 126,045 | +6% | yes |
| 4 | Staff Business Analyst - Credit — Plaid | 164,196 | 257,100 | −36% | **no** |
| 5 | Associate Analyst — Molina | 64,973 | 53,456 | +22% | yes |
| 6 | Recovery Resolution Consultant — UnitedHealth | 84,496 | 101,400 | −17% | yes |
| 7 | Senior AI/ML Flex Solution Engineer — Snowflake | 169,824 | 190,781 | −11% | yes |
| 8 | Principal AI Engineer II — AbbVie | 179,725 | 205,000 | −12% | yes |
| 9 | Data Security Engineer II — Centene | 91,854 | 89,097 | +3% | yes |
| 10 | Data Quality and Governance Analyst — ManTech | 103,051 | 126,000 | −18% | yes |
| 11 | Enterprise Data Governance Architect | 130,994 | 113,382 | +16% | yes |

Mean absolute error 21.5%, median 18.2%, 9 of 12 ranges overlapping. The card next to them
says ±14% and 92%.

Twelve rows is a small sample and the holdout figures stand — this is an unlucky draw, not
a broken metric. But the product consequence is real: **the app invites the user to check
its work on exactly twelve cases, and on those twelve it looks worse than it claims.** The
default example on load, index 0, is a +28% overshoot, and example 2 is a −64% miss on what
looks like a contract-marketplace posting whose \$208,000–\$416,000 band survived the
cleaning filter.

Two honest fixes, in section 5.5: raise the sample count so the visible distribution
matches the claimed one, and show the hit/miss record rather than leaving the user to
discover it.

### 3.4 The headline range is read as a confidence interval

`$136,441 – $202,980` under the label "estimated advertised range" is, to almost every
reader, "the model is fairly sure the answer is in here". It is not. It is a point estimate
of what the employer would have *printed*, being a midpoint and a band width predicted by
two separate regressions — and the band width is explicitly the weaker target (`report.md`
§7.6: predicted to within about 12%, R² well below the level).

The actual uncertainty lives in a grey box labelled "typical error for these inputs", as a
percentage, detached from the number it qualifies. Nothing on screen connects "±14%" to
"$136,441 – $202,980", and no user will do that arithmetic. This is the single biggest
comprehension gap in the page: rung 3 of the trust ladder is present but not attached.

### 3.5 "Use" toggles teach the wrong mental model

The toggles read as include/exclude filters. They are something stranger and more
interesting: switching a block off tells the model *this posting genuinely has no such
information*, which is a state it was trained on. Blanking the fields while leaving the
block on tells it *this posting has a title, and the title is empty*, which is not.

The hint says so ("a masked block is not 'unknown', it is genuinely absent"), which is
precise and will be read by roughly nobody, and is only legible to a reader who already
knows what masking is. Nothing in the interface makes the two states look different, and
they produce different answers.

### 3.6 The form is the model's schema, not a user's task

Forty fields, all visible, no hierarchy. Latitude and longitude sit next to job title.
"Number of states listed" is a featuriser detail exposed as a question. "Applicant tracking
system" asks a job seeker to know whether the employer uses Workday or Greenhouse — a real
feature in the model, and meaningless as a question to Priya.

The three fields that carry most of the signal — title, description, state — are scattered
across three of the five cards. `report.md` §7.2 measures it: title plus description plus
location is 14.6% MAPE against 13.7% for all five blocks. **The other thirty-odd fields
are worth 0.9 points of MAPE**, and the interface gives the three that matter no more
prominence than HQ country.

### 3.7 Mobile loses the answer

At 390 px the page is 3,950 px tall. The result card is placed first (correct), then never
seen again: editing any field in the form below scrolls the estimate off screen, so the
core interaction — change an input, watch the number move — is impossible on a phone. The
desktop layout handles this with a sticky rail; the mobile layout has no equivalent.

### 3.8 Employer matching is exact and case-sensitive

`raw.company_name in companyEntry.table` is a literal key lookup over 6,592 names.

- `Google` → found, pricing history used.
- `google` → "was not in the training data, so it received the average employer's pricing."
- `Amazon` → genuinely absent.
- The default example ships with employer `Stjude`, which is absent, so **the first thing a
  new visitor reads is a caveat about an unrecognised employer.**

The stored names are also inconsistent enough to make the autocomplete feel broken:
`Jll`, `Uagc`, `Childrensnational`, `Sazerac Company Overview`, `Molina Talent Acquisition`.

There is a strong argument, in section 5.8, for demoting the field entirely: `report.md`
§7.5 found that recognising the employer by name is worth at most 0.002 log MAE.

### 3.9 Smaller findings

- **Typed employer names are written into the DOM unescaped.** `notes.push()` interpolates
  `raw.company_name` into a template string that is assigned to `$("notes").innerHTML`.
  Entering `<img src=x onerror=...>` executes. Today it is self-inflicted on a static
  origin with nothing to steal, so the severity is low — but it becomes a genuine
  reflected-XSS vector the moment any state arrives from the URL, which section 5.10
  proposes. Fix it first, regardless.
- **`aria-live="polite"` wraps the entire results panel,** including the estimate, both
  statistics, the pattern note, the attribution table and the notes list. Every keystroke
  re-announces all of it. It should be scoped to a short summary line and debounced.
- **A dark theme exists but cannot be reached.** `style.css` defines `:root[data-theme="dark"]`
  and `[data-theme="light"]` overrides; nothing in the page ever sets the attribute, so the
  only path is the OS preference. Either wire up a toggle or delete the dead selectors.
- **The 7 MB load has no progress.** "Loading the model (about 7 MB)…" is static text.
  Locally that is 240 ms; on a phone on cellular it is a blank card for a long time.
- **No reset, no copy, no permalink.** Once a user has built an interesting input there is
  nothing to do with it.
- **The advertised-range comparison is a 12 px pill that vanishes on edit.** Any input event
  sets `currentSample = null`, so the most persuasive element on the page disappears the
  moment the user interacts.
- **The training baseline is never shown.** Attributions are "relative to the average
  posting in training", and that average — \$122,060 — appears nowhere, which makes the
  percentages unanchored.
- **No social preview metadata.** For a portfolio piece that gets shared in Slack and on
  LinkedIn, no `og:image` or `og:description` is a missed first impression.

## 4. The shape of the fix

The findings cluster into three problems, and it is worth naming them before the list of
proposals, because most individual fixes serve one of the three:

1. **Confidence is not conditioned on what the user actually gave.** (3.1, 3.2, 3.4)
2. **The app never shows its work at the aggregate level.** It shows one prediction and
   one claimed error, and never the record. (3.3, and the disappearing truth pill in 3.9)
3. **The interface exposes the model's schema instead of the user's task.** (3.5, 3.6, 3.7,
   3.8)

## 5. Proposals

Each has a rough cost in the `web/` codebase. "Export" means `src/salary_scout/export.py`
changes and a re-run, which also means `node web/test/verify.mjs` must still pass.

### 5.1 Treat a blank block as absent, and key the error to what was actually given

**What.** Detect blocks whose fields are all empty. Either auto-switch them off (with a
visible, reversible "switched off because it is empty" state), or keep them on but compute
`metricsFor()` from the *effectively filled* pattern rather than the toggled pattern. Show
the three-state distinction from 3.5 explicitly: **on and filled / on and empty / off**.

**Why.** It is finding 3.1, the most damaging single behaviour in the app, and it is the
difference between "±14%" and the honest "±21%, because all you gave me was a title".

**Pros.**
- Removes an unsupportable claim from the most-read number on the page.
- Costs nothing in model work; `metrics.json` already carries all 31 patterns.
- Makes the toggles self-teaching: a user who clears a field *sees* the block go quiet, and
  learns what masking means without reading the hint.

**Cons.**
- Auto-switching is state changing under the user's hands, which is usually bad. Mitigate
  by making it visibly automatic and one click to override.
- Partially filled blocks stay ambiguous: a title block with a title but no core title is
  "filled", and there is no metric for degrees of fill. The honest framing is that the
  pattern figure is an upper bound on how much you gave it.

**Cost.** Small, `app.js` only. Half a day.

### 5.2 Warn when the input is outside what the model was trained on

**What.** Two complementary checks:

- *Cheap and certain*: scope checks on structured input — a non-US country, a title
  matching management or executive patterns, a job category outside the training mix.
  These are lookups against facts already in `featuriser.json`.
- *Principled*: compare the block's contribution against a **present-but-empty reference**.
  `featurizeBlock()` already supports present and masked variants; a third "present, no
  text" variant gives the +14% floor from 3.2. When the user's text moves the estimate less
  than a few points off that floor, the honest message is "none of the words you typed are
  ones this model learned from" rather than a dollar figure.

**Why.** Finding 3.2. It is also the single highest-value thing for Sam
(`user_stories.md` §3.4): a model that declines to answer is a stronger hiring signal than
one that always answers.

**Pros.**
- Converts the app's worst failure mode into its most convincing feature.
- The reference-prediction approach reuses machinery that is already in `model.js`.
- Directly serves the anti-personas in `user_stories.md` §5, who are otherwise silently
  given wrong answers.

**Cons.**
- It is a heuristic, not a calibrated out-of-distribution test, and it should be worded as
  one. A confident "out of scope" label on a legitimate but unusual title would be worse
  than the current silence.
- The keyword scope check needs a curated list, which is prose that can drift from the data.
  Prefer deriving it from `job_category` values in the featuriser spec where possible.
- A third featurisation variant per block adds an evaluation; negligible at current speed.

**Cost.** Medium. Scope checks are a day in `app.js`; the empty-text reference is a day
across `model.js` and `featurizer.js`, with a new fixture case so `verify.mjs` still
proves the JS matches Python.

**Cheaper variant worth considering first.** Ship a table of the most frequent title tokens
in training (tens of KB) and say plainly: "3 of the 4 words in this title never appeared in
the 20,257 postings this model learned from." No inference, no calibration argument, and it
is the most legible possible statement of the limitation.

### 5.3 Put the scope above the fold

**What.** A short strip under the lede — three or four lines, not a paragraph — stating:
US postings, USD, published April–September 2026, individual contributors, mostly data,
analytics and software, from employers that publish pay. Keep the full "About this model"
disclosure for the detail.

**Why.** Rungs 1, 2 and 6 of the trust ladder are currently in a `<details>` element. Dana
decides in thirty seconds and will not open it; Priya never knows the model has a scope at
all.

**Pros.**
- The cheapest change in this document and it closes the largest honesty gap.
- Wins the persona most likely to reject the tool, by volunteering the limitation first.

**Cons.**
- Competes for the space above the form, and every line there delays the first interaction.
- Risks reading as a disclaimer wall. The mitigation is to write it as what the tool *is*
  rather than what it is not: "trained on 24,000 US postings that published pay, mostly
  data and software roles, mid-2026."

**Cost.** Trivial. `index.html` and `style.css`, an hour.

### 5.4 Attach the uncertainty to the number

**What.** Render the error band with the estimate rather than beside it. Concretely: keep
the advertised-range figure as the headline, and draw a single horizontal strip underneath
showing the ±MAPE interval around the midpoint, labelled "estimates like this are usually
within ±14%". Relabel "ranges that overlapped the real one" — it is a precise metric and an
opaque phrase — as something like "9 times in 10 the real range overlapped ours".

**Why.** Finding 3.4. It is the difference between a user quoting "$136,441 to $202,980" and
a user quoting "about $170k, give or take 15%", and the second is the only defensible one.

**Pros.**
- Makes rung 3 of the trust ladder impossible to skip without adding any text to read.
- A visual band naturally shows the pattern effect: switch off the description and watch
  the uncertainty widen. That teaches block masking better than the hint text does.

**Cons.**
- Two ranges on screen — the advertised band and the error band — is exactly the confusion
  being fixed, if drawn carelessly. They must not look alike; the error band should read as
  a blur or a gradient, not as a second bracket.
- MAPE is a mean absolute percentage error, not a confidence interval, and drawing it as a
  band implies coverage it does not carry. Label it as "typical", never as "90% of the time",
  unless the export starts shipping quantiles.

**Cost.** Small-to-medium, `app.js` and `style.css`. One to two days including the honest
wording.

### 5.5 Show the record, not just the claim

**What.** A compact scoreboard over the bundled examples: predicted versus advertised for
each, hit or miss, and the aggregate. Raise the example count from 12 to 30–40 so the
visible distribution matches the claimed one, and keep the truth comparison on screen when
the user edits an example rather than clearing it on the first keystroke.

**Why.** Finding 3.3, and it is the single most persuasive thing the app can do for every
persona. A user who watches the model get 9 of 12 right, including the ones it got wrong,
trusts the 14% figure in a way no statistic achieves.

**Pros.**
- Turns the unflattering sample draw from a liability into a demonstration of honesty.
- Cheap in bytes: samples are metadata plus description text; 40 rows is a modest increase
  against the 6.9 MB of model already shipped.
- Gives Jordan something to repeat and Sam something to be impressed by.

**Cons.**
- Publishing a visible miss rate invites "it got that one 64% wrong" as the takeaway. The
  counter is that the misses are already visible to anyone who clicks through the examples;
  the choice is whether the app frames them or the user discovers them.
- Example 2 (Turing, \$208,000–\$416,000) looks like a contract-marketplace posting that
  survived the plausibility filter. Shipping it uncommented invites a data-quality question
  with no answer on the page. Either annotate it or re-draw with a filter, and say which.
- Keeping the truth visible during edits means tracking "edited from example N", which
  complicates `currentSample` handling.

**Cost.** Medium. `export.py` for the sample count, `app.js` and `index.html` for the
scoreboard. Two days.

### 5.6 One-click presets that demonstrate the findings

**What.** Three or four buttons above the form: **Everything**, **Title only**,
**No description**, **Hide the employer**. Each sets the toggles and updates the card.

**Why.** The project's most interesting results — the title alone gets within 21%, the
employer name is worth almost nothing — exist only in `docs/blog.md`. A user who clicks
"Title only" and sees the error move from ±14% to ±21% has learned the block-dropout story
in one gesture.

**Pros.**
- Nearly free: the toggles already exist, this is a preset over them.
- Gives the blog post a landing target, which is the whole Jordan persona.
- Makes the five-block structure legible as a *feature* rather than as a form layout.

**Cons.**
- More controls above the form, competing with 5.3 for the same space.
- "Hide the employer" demonstrates a null result, which is interesting to Sam and Dana and
  anticlimactic for Priya. Worth including anyway; it is the most honest thing in the
  project.

**Cost.** Small. `app.js` and `index.html`, half a day.

### 5.7 Attribution as a dollar waterfall from a stated baseline

**What.** Keep the five bars, add the anchor: start at the training baseline of
**\$122,060** ("the average posting in this data"), show each block's contribution in
dollars as well as percent, and end at the estimate.

**Why.** Finding in 3.9. Percentages relative to an unstated baseline are not
interpretable; the same panel with the baseline named becomes a sentence anyone can read:
"an average posting is \$122,060; this title is worth +\$18k, this description +\$19k,
being in this location −\$4k."

**Pros.**
- The numbers are already computed; `baseline_log_mid` is in `featuriser.json` and `meta`
  is already loaded into `SalaryModel`.
- Dollars are what Marcus has to defend to a founder. Percentages are not.
- Makes the Shapley property visible: the parts add up to the whole.

**Cons.**
- Shapley values are additive in log space, so their dollar equivalents are
  order-dependent and will not sum exactly to the estimate. Either present them as a
  multiplicative sequence (honest, slightly harder to read) or state that dollar figures
  are approximate. Do not silently fudge the arithmetic.
- A waterfall is a busier chart than five bars; at 380 px it needs care.

**Cost.** Medium, `app.js` rendering only. One day, plus deciding the log-to-dollar framing.

### 5.8 Rebuild the form around the task

**What.** A fast path of three fields — title, description, state — in a single prominent
card, with everything else behind "Add more detail". Within the advanced area, drop or
relabel the fields that are model artifacts rather than user knowledge: latitude/longitude,
number of states listed, applicant tracking system. Normalise the employer lookup
(case-insensitive, trimmed, punctuation-insensitive) and demote the field with a note
that the name is worth almost nothing.

**Why.** Findings 3.6 and 3.8. Title + description + location is 14.6% MAPE against 13.7%
for everything; the other 30-odd fields buy 0.9 points of accuracy and cost every user the
time to scan past them.

**Pros.**
- Priya's path goes from scanning 40 fields to filling 3.
- Demoting the employer field is supported by the project's own result and is a good story
  rather than a concession.
- Removing "applicant tracking system" from the primary surface removes a question that
  makes the tool look like it is guessing from noise.

**Cons.**
- Hiding fields hides capability, and Marcus genuinely wants headcount and sector.
  Progressive disclosure must be one click and must remember its state.
- The block structure is load-bearing for the masking demo: if the fast path mixes fields
  from three blocks into one card, the five-block story gets harder to tell. The
  fast path should show which blocks it is feeding.
- Normalising employer names changes prediction behaviour for the same typed input, which
  should be noted in the export or the about panel rather than done silently.

**Cost.** Large. This is a re-layout of `index.html` plus form handling in `app.js`. Three
to four days, and the piece most worth prototyping before committing.

### 5.9 Make the privacy claim land where it matters

**What.** Move "nothing you type leaves your browser" from the lede to the description
textarea, where the user is about to paste something they may not want to send anywhere.

**Why.** Priya is pasting from a work laptop; Marcus is pasting an unpublished req. It is a
genuine differentiator against every hosted salary tool and it is currently one clause in a
paragraph most users skim.

**Pros.** Free. Turns a feature of the architecture into a reason to use the product.

**Cons.** Repeating a claim can read as protesting too much. One short line at the point of
paste, not a badge.

**Cost.** Trivial.

### 5.10 Permalink and copy, after fixing the escaping

**What.** Encode the form state in the URL fragment; add "copy summary" producing a few
lines of text with the estimate, the inputs used, and the typical error. **Prerequisite:**
escape user-supplied strings before they reach `innerHTML` (finding 3.9).

**Why.** Marcus needs to paste something into a document; Sam wants to send a link.

**Pros.**
- Makes the tool shareable, which is most of its distribution.
- A copy summary is the natural place to force the caveat to travel with the number: any
  quoted estimate leaves with its error figure and scope attached.

**Cons.**
- A description pasted into a URL is large; either restrict the permalink to the structured
  fields or compress, and say which. Putting a full job description in a URL that gets
  shared is also a privacy footgun that contradicts 5.9 — restrict the permalink to
  structured fields and exclude free text.
- Any URL-driven state turns the unescaped-innerHTML bug into a real vulnerability. Do not
  ship this one before that fix.

**Cost.** Small-to-medium, plus the escaping fix, which should happen regardless and is
twenty minutes.

### 5.11 Mobile: keep the answer on screen

**What.** A compact sticky bar at the top or bottom on narrow viewports showing the range
and the typical error, expanding to the full card on tap.

**Why.** Finding 3.7: on a phone the core interaction does not work at all.

**Pros.** Restores change-an-input-watch-the-number on the devices where a job seeker is
most likely to be reading a posting.

**Cons.** Sticky bars eat scarce vertical space on small screens, and the page is already
3,950 px. Needs to be genuinely compact — one line — and 5.8's progressive disclosure would
cut the page length enough to make it comfortable.

**Cost.** Small, `style.css` and a little `app.js`. One day.

### 5.12 Housekeeping

- Scope `aria-live` to a single concise summary node and debounce announcements.
- Either wire a theme toggle to the `data-theme` attribute the CSS already supports, or
  delete the dead selectors.
- Show real progress during the 7 MB load, or defer `trees_spread.json` until after the
  first midpoint render.
- Add `og:title`, `og:description` and an `og:image` for link previews.
- Add a reset control.

## 6. Suggested sequence

Ordered by honesty-per-hour, not by size.

| Tier | Items | Why first |
|---|---|---|
| **1. Do not ship without** | Escape user strings (3.9), blank-block handling (5.1), scope above the fold (5.3) | These are corrections, not features. Two of the three are under a day each |
| **2. The comprehension fixes** | Uncertainty attached to the number (5.4), out-of-scope warning (5.2), privacy at the paste point (5.9) | This is where "it just clicks" actually happens |
| **3. The persuasion fixes** | Example scoreboard (5.5), presets (5.6), dollar waterfall (5.7) | Turns a working demo into a convincing one |
| **4. The ergonomics** | Form re-layout (5.8), mobile sticky result (5.11), permalink and copy (5.10) | Largest effort, and 5.8 benefits from prototyping against real use |
| **5. Housekeeping** | 5.12 | Whenever |

Tiers 1 and 2 are roughly a week and address every one of the three severe findings.

## 7. Considered and rejected

- **A cost-of-living adjuster.** Tempting and unsupported: the model learns advertised
  bands including their location effect, and layering a COL index on top would double-count
  geography.
- **Negotiation advice ("ask for the top of the band").** Crosses from describing employer
  behaviour to advising an individual, which is exactly the boundary `report.md` §10 says
  the apps will not cross.
- **Letting users submit their own salary data.** Ends the privacy claim, adds a moderation
  problem, and self-reported data is precisely what this project avoided.
- **Accounts, saved searches, history.** No server, and the privacy story is worth more
  than the retention.
- **A chat interface over the model.** Would obscure the block structure, which is the one
  genuinely novel thing the model does.
- **Replacing the range with a single number.** Simpler to read and less honest; the range
  is the target the model was trained on.

## 8. Open questions for the maintainer

1. **Who is the primary persona?** This review assumes Sam and Dana (credibility) rank
   above Priya (utility), because the project is explicitly also a portfolio piece. If
   Priya comes first, 5.8 moves up and 5.5 moves down.
2. **How much load budget is there for more examples?** 5.5 assumes 30–40 samples is
   acceptable against the existing 6.9 MB.
3. **Should the out-of-scope check refuse, or warn?** Refusing is more honest and will
   frustrate a legitimate unusual title. A warning that visibly de-emphasises the number is
   the middle path this review assumes.
4. **Is the service deployment (`handoff.md` step 8) still planned?** Several proposals
   here — quantile bands for 5.4 in particular — are cheaper to do once, in `export.py`,
   if both targets will consume them.
