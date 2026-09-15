# Salary Scout: users, jobs, and what the product must say

Product document, September 2026. Companion to `docs/report.md` (what the model does)
and `docs/ux_review.md` (where the current app falls short of this document).

## 1. Why this document exists

The model works. The report in `docs/report.md` establishes what it predicts, how well,
and where it stops being trustworthy. None of that reaches a person who lands on
[the demo](https://alexanderwu.github.io/salary-scout/) and sees a form.

This document names the people who might land there, what each of them actually wants,
and what they have to understand before the answer is useful rather than merely
plausible. It is the yardstick for UI decisions: a change is good if it moves one of
these people closer to understanding, and suspect if it only adds a control.

The organising claim: **Salary Scout answers one question well, and the product's job is
to make sure nobody mistakes it for a different question.**

> The question: *given a job posting, what salary range would a US employer that
> publishes pay have advertised for it?*
>
> Not: what will this company offer me? Not: what am I worth? Not: what does this role
> pay in my city? Those are different questions with different data behind them.

## 2. What the model can and cannot be asked

Straight from the limitations in `docs/report.md` §9, restated as user-facing scope.
Every persona below is defined partly by whether their question survives this table.

| A user asks | Can the model answer it | Why |
|---|---|---|
| "What range would this posting have advertised?" | Yes, within ~14% with full inputs | The trained task |
| "This posting hides its range. What is it?" | Yes, with a caveat | Non-disclosing employers are an extrapolation; nobody checked the model's homework on them |
| "Is the range on this posting unusual for the role?" | Yes | Comparing a prediction against a printed range is exactly the comparison the test set measures |
| "What would this pay if it were senior instead of mid?" | Yes, directionally | Editing an input and watching the estimate move is the model's strongest interaction |
| "What will Google offer *me* at L5?" | No | Advertised bands, not individual offers; no level taxonomy, no equity, no bonus |
| "What does a nurse / plumber / CEO make?" | No | Individual-contributor data, analytics, software, IT and engineering roles only |
| "What does this pay in London / in CAD?" | No | US postings, USD only |
| "What will this pay in 2028?" | No | A five-month snapshot of mid-2026 |
| "What is the hourly rate for this contract?" | No | Annualised salaried pay between 20k and 1M; everything else was dropped |
| "What is the exact range, to the dollar?" | No | The band width is the weaker of the two targets; read it as plausible, not exact |

The first four rows are the product. The last six are the ways the product gets misused,
and each one is a thing the interface has to say out loud rather than leave to a
collapsed "About" panel.

## 3. Primary personas

### 3.1 Priya, the candidate with a posting in hand

**Who.** Five years as a data engineer, currently employed, casually looking. Lives in a
state without a pay-transparency law, so about a fifth of the postings they care about
print no range at all.

**The moment they arrive.** They have a specific posting open in another tab. It is
interesting, it is a two-hour application, and it does not say what it pays. They do not
want a salary encyclopedia; they want to know whether this posting is worth the two hours.

**What they do.** Paste the description, add the title, pick the state. Look at one
number. Decide.

**What they need to believe.** That the number came from postings like this one, that
"about 14% off" is a real measured quantity and not marketing, and that pasting a job
description into a web page is safe. The last one matters more than it looks: Priya is
job-hunting from a work laptop and will not paste anything into a form that posts to a
server.

**What "clicks" for them.** Seeing the estimate land next to a real posting's actual
advertised range and land close. One worked example where the model was right does more
than any accuracy figure.

**Job stories.**

- When I find a posting with no salary range, I want a defensible estimate in under a
  minute, so I can decide whether to apply without asking anyone.
- When I see a posting *with* a range, I want to know whether that range is low for the
  role as described, so I can decide whether to negotiate or walk.
- When I read an estimate, I want to know how wrong it usually is, so I can quote a
  number in a screening call without being embarrassed by it.

**This works for Priya when:** the fastest path through the app is paste-title-state-done;
the uncertainty is attached to the number rather than parked in a separate statistic; and
"nothing leaves your browser" is visible at the moment they paste, not only in the header.

### 3.2 Marcus, writing a requisition that has to carry a range

**Who.** Engineering manager at a 200-person company, hiring their first analytics
engineer. Legal has told them the posting must carry a range in three of the states
they would hire in. They have no compensation team and one data point: what they pay the
person doing half of this job today.

**The moment they arrive.** A blank req and a band they have to defend to a founder in
one direction and a candidate in the other.

**What they do.** Type the draft title, paste the draft description, set state and
headcount, and then start *editing*. What if the title says Senior. What if it is remote.
What if the requirement is five years instead of three. Marcus is not looking up a number,
they are exploring a surface.

**What they need to believe.** That the tool describes what employers *advertise* rather
than what they pay, because advertising is exactly Marcus's problem. That the comparison
set is companies like theirs. That they can paste an unpublished req without it leaving
the building.

**What "clicks" for them.** Changing one field and watching the estimate and the
attribution bars move together. That is the moment the model stops being an oracle and
becomes an instrument.

**Job stories.**

- When I draft a req, I want a starting band for the role as I have written it, so I am
  not anchoring on the one salary I happen to know.
- When I change the title or the seniority, I want to see what that is worth in dollars,
  so I can trade wording against budget.
- When a founder asks why the band is what it is, I want a breakdown I can paste into a
  document, so the number has a story attached.

**This works for Marcus when:** editing is cheap and the result is always on screen while
editing; the attribution is expressed in dollars from a stated baseline rather than
percentages against an unstated one; and there is something to take away — a link, a copy
button, a paragraph.

### 3.3 Dana, the compensation analyst

**Who.** Works in people operations at a mid-size company, owns the salary bands, has
access to a paid survey that costs five figures and lags by a year.

**The moment they arrive.** Someone sent them the link. Their default posture is
disbelief, and it is well earned: most free salary tools are self-reported, unweighted,
and silently stale.

**What they do.** Go straight for the methodology. How many postings, from where, over
what period, self-reported or scraped, what was excluded. Only then do they try the form,
and they will try it on a role whose true band they already know, to see if it fails.

**What they need to believe.** That the sampling frame is stated (postings from employers
who disclose, mid-2026, mostly data and software), that the holdout protocol is real, and
that the tool volunteers its weaknesses before they find them. Dana is the persona most
likely to be *won* by a limitation clearly stated, and most certainly lost by one they
discover themselves.

**What "clicks" for them.** The disclosure-by-state figure and the sentence "a prediction
for an employer that does not disclose is an extrapolation whose error is not measured
here". That is the sentence that says a careful person built this.

**Job stories.**

- When I evaluate a salary tool, I want the sampling frame and the exclusions on the
  first screen, so I can reject it in thirty seconds if it is not fit for purpose.
- When I test it against a band I already know, I want the tool to tell me when my input
  is outside what it was trained on, so a wrong answer is labelled rather than silent.
- When I use an estimate in a document, I want a citable description of the method, so my
  band survives review.

**This works for Dana when:** the scope statement is above the fold and not inside a
disclosure triangle; the app refuses, or at least flags, inputs outside its training
distribution; and the link to the report is prominent.

### 3.4 Sam, evaluating Alexander

**Who.** A hiring manager or senior engineer with a tab open on this project because it is
on a CV. They have seven minutes and four other candidates.

**The moment they arrive.** Skeptical in a specific way: they have seen a hundred
portfolio projects that are a notebook with a 0.95 R² and no holdout.

**What they do.** Click the demo first, because a working demo is rare. Poke at it for
ninety seconds. If it survives, read the report. If the demo is broken, slow, or obviously
overclaiming, they never open the report.

**What they need to believe.** That the person who built this understood leakage (the
salary is printed in two thirds of the descriptions), understood that a random split would
be wrong here (siblings, employer concentration, time), and knew where to stop.

**What "clicks" for them.** Finding, unprompted, that the app is honest about something it
could have hidden. A model that says "the words you typed are not ones I have seen" is a
stronger hiring signal than one that always answers.

**Job stories.**

- When I open a portfolio demo, I want to understand the problem and see it work in under
  a minute, so I can decide whether the rest is worth reading.
- When I poke at a demo, I want it to behave sensibly at the edges, so I can tell whether
  the author thought past the happy path.
- When I want the detail, I want a direct route from the thing I just clicked to the
  method behind it, so I am not hunting through a repository.

**This works for Sam when:** the demo loads fast, holds up to adversarial input, and every
interesting claim in the UI links to the section of the report that establishes it.

## 4. Secondary personas

**Jordan, the blog reader.** Arrives from "The job title alone predicts pay within about
23%", wants to test that claim in the app in one click, and wants a surprising thing to
repeat to a colleague. The app currently has no route to any of the blog's findings; a
"title only" preset would close most of the distance.

**Alexander, the maintainer.** Needs the demo to stay an honest artifact as the model
changes: error figures that come from `metrics.json` rather than prose, a scope statement
that is edited in one place, and an interface that does not have to be re-justified after
every refit.

## 5. Anti-personas

Naming these matters as much as naming the users, because the interface is what stops
them from getting a confident wrong answer.

| Who | What they will try | What should happen |
|---|---|---|
| Someone negotiating a live offer | "Google, Senior, California" and treats the output as an offer band | Say plainly that this is advertised range, not individual offer, and point at Levels.fyi-style sources |
| A non-US or non-USD job seeker | A London or Toronto posting | Say the model is US and USD only, ideally when a non-US country is entered |
| A manager or executive | "VP of Engineering", "Director of Data" | Say the training data is individual-contributor only. Today this returns a confident number |
| A nurse, teacher, tradesperson | Their own job title | Say the data is data, analytics, software, IT and engineering. Today this returns a confident number |
| An hourly or contract worker | An hourly posting | Say the model covers annual salaried pay from 20k to 1M |

The last three rows are the sharpest product failure in the current app, and
`docs/ux_review.md` §3.1 measures it: "Barista" returns \$117,956 – \$161,129, which is the
same answer the app gives for "asdf asdf", labelled with the same "±21% typical error".

## 6. The trust ladder

A user does not need to understand the model. They need to understand six things, in this
order, and the interface should deliver them in this order rather than all at once.

1. **What the number is.** The range an employer would *print* in a posting. Not an offer,
   not total compensation, not what you are worth.
2. **Where it came from.** About 24,000 US postings that published pay, from April to
   September 2026, mostly data, analytics and software, individual contributors.
3. **How wrong it usually is.** A measured quantity on postings the model never saw, and
   it depends on how much you told it: about 14% with everything, about 21% from a title
   alone.
4. **What drove this particular answer.** Five bars, one per input group, in dollars from
   a stated baseline.
5. **When it does not know.** Inputs outside the training distribution, unknown employers,
   blank blocks, a title made of words that never appeared in training.
6. **What it will never know.** Non-disclosing employers, other countries, other years,
   management roles, individual offers.

Rungs 1, 3 and 5 are the ones users skip and the ones that cause harm when skipped. The
current app states 1 and 2 in a lede that is easy to miss, states 3 as a statistic detached
from the number it qualifies, states 4 well, barely states 5, and hides 6 behind a
disclosure triangle.

## 7. What "it just clicks" would mean

Concretely, per persona, the app clicks when:

- **Priya** gets a usable number from a paste and a title in under a minute, and can tell
  a recruiter "about 150 to 180, and that estimate is usually within 15%" without
  overstating anything.
- **Marcus** changes "Analyst" to "Senior Analyst" and sees both the number and the reason
  move, and leaves with a link they can paste into a doc.
- **Dana** finds the sampling frame before they find a flaw, and sees the app decline to
  answer something it should decline to answer.
- **Sam** understands the problem, the trick (block dropout), and the honesty (scrubbing,
  grouped splits) in ninety seconds of clicking, without reading the report.
- **Jordan** clicks one button, sees the title-only claim demonstrated on a real posting,
  and has something to repeat.

None of those require a new model. All of them are interface.
