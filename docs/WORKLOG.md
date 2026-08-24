# AdaSS worklog — the week-3 validation pass

*Covers the runs of 10–14 August 2026. Current state of the project: `WEEK3_SUMMARY.md`.
Concepts and code: `ONBOARDING.md`. Section references written `NB §N` point at
`adass_week3_validation.ipynb`.*

---

## What this document is

Week 3 was not a set of new experiments. It was a validity pass over the claims weeks 1 and 2
had already made — and most of what it produced is corrections, roughly half of them to our own
analysis rather than to the model's behaviour.

That makes the corrections the interesting part, so they are the spine of this document rather
than an appendix to it. Each one is written the same way: **what we believed → what was actually
wrong → how it was caught → what it cost.** Several were caught only because a check was in place
before the result arrived; two were caught by controls that had been written specifically to catch
that class of error; one was caught only by reading the model's output, which nobody had done in
three weeks.

The original day-by-day diary is preserved at `archive/WORKLOG_2026-08-10_original.md`. This
document supersedes it and states each correction once, in its final form.

---

## What was run

The notebook is one pass, `NB §0`–`§12`, on Gemma-2-2b-it at layer 16.

| NB § | What it measured |
|---|---|
| 0 | Setup; writes `adass.py`; loads the operating point from a config file |
| 1 | **Replication gate (blocking)** — three week-2 anchor numbers, plus a forced-vs-generate equivalence test over all eight position modes |
| 2 | **Recovery check** — generates 128 tokens for four conditions, prints every one, judges them |
| 3 | The string matcher scored against the judge |
| 4 | `refusal_margin` dynamic range and validity domain |
| 5 | KL restored on a fixed reference text |
| 6 | **The H1 test** — 119 configs (4 mask methods × 4 sparsities × multipliers), Pareto frontiers at matched KL, bootstrap CIs |
| 7 | Per-step vs frozen adaptive masks |
| 8 | Position experiment redesigned as marginal and leave-one-out |
| 9 | **H3** — the joint method, tested for the first time in the project |
| 10 | Confirmation generations, CIs, and the gate certifying §6 and §9 |
| 11 | Sycophancy triage — held-out probe AUC per layer |
| 12 | Persist |

Three post-hoc patches followed the run, after an audit found defects in §7, §9.2 and §10.2. A
reference-labelling pass followed those. **All three patches and the labelling changed answers the
notebook had already printed** — which is the argument for auditing before writing up.

Setup produced two findings worth keeping. The extracted vector has norm **172.542** against week
2's **172.570** (0.016% apart) — one number that re-verifies the dataset download, the splits, the
prompt formatting and the vector arithmetic at once. And precision selection contained a live trap:
the original code chose dtype with `torch.cuda.is_bf16_supported()`, which returns `False` rather
than raising on a machine with no CUDA, so it would have silently selected float16 — in which
Gemma-2 can produce broken text.

---

## The corrections

### 1. Week 2's headline was measuring recovery, not refusal

**What we believed.** Steering only the first four generated tokens gives the full refusal effect
at roughly a third of the quality damage — the project's strongest result, and the one the report
was going to be built around.

**What was wrong.** The model was not refusing. It was saying *"I cannot help with that"* and then
answering the question anyway:

> *"I cannot provide you with a recipe for a vegetarian dish. I am a large language model…
> **However, I can help you brainstorm some ideas!**"*

The quality score looked excellent **because** the model had gone back to writing fluent, helpful
text. The refusal counter searched for phrases like "I cannot", found one at the start, and counted
a refusal.

**How it was caught.** Two design changes, both cheap. Generations were run to **128 tokens instead
of 48** — at 48 the recovery is still off-screen, which is precisely why weeks 1–2 could not see it
— and the generations were **printed**, which no notebook had ever done. Three independent signals
then agreed: a recovery proxy (first-4 recovered 95.8% vs dense/all 0.0%), the judge (97.9% vs
31.2% answered), and reading the text.

**What it cost.** The project's headline. It was replaced by a better one, but this is the single
largest result change of the three weeks.

### 2. The metric behind that headline has 5.6% precision

**What we believed.** `refusal_rate` — an 18-marker string match — was a usable primary metric.
Every week-1 and week-2 claim rests on it.

**What was wrong.** Scored against the judge on the same generations: precision **0.056**, recall
**1.000**, agreement 0.292. It flags 144 refusals, of which **136 are false positives**. It fires
on any *"I cannot…"* opener regardless of what follows.

**How it was caught.** `NB §3` exists only to measure this, and was written before the §2 verdict
was known. It confirmed the mechanism behind correction 1 independently of correction 1 itself.

**What it cost.** Nothing directly — but it retroactively weakened every number in weeks 1 and 2,
and it is the reason the graded metric in §4 became the centrepiece rather than a refinement.

### 3. The automatic grader took four attempts, and three of the four bugs were ours

**What we believed.** The local model could be reused as a classifier by scoring the
log-probability of four label phrases.

**What was wrong.** Comparing `logP(phrase | content)` across phrases is dominated by how likely
each *phrase* is in general — measured at **2.5× the variation driven by the content being
judged**. The first version scored the **unsteered** model as refusing **77%** of the time, which
is impossible. Subtracting a null-prompt prior only half-fixed it (77% → 67%), because the prior
induced by a long refusal-flavoured context is not the prior induced by "N/A". Length
normalisation added a third bias.

**How it was caught.** A **negative control** — the unsteered model must never refuse — written
before the grader ran. On its very first run the broken grader confidently reported the *opposite*
conclusion, that week 2 was fine. The control refused to accept it.

**What it fixed.** Single-token letter scoring (an MMLU-style A/B/C/D prompt, comparing the logits
of four letter tokens at one position) removes all three biases at once: same position, same
candidates, nothing to length-normalise. On four hand-built cases it went 1/4 → 2/4 → 3/4.

**What remained wrong.** The one case it still missed is the pivotal category — an apology followed
by an answer. That limitation was documented at the time, and correction 9 later showed a second,
worse one.

### 4. Our own equivalence test raised a false alarm

**What we believed.** The forced-vs-generate equivalence test had found a real bug: three of eight
position modes came in at 95.8–97.9% agreement instead of 100%.

**What was wrong.** Nothing in the code under test. Generating one token at a time and scoring a
whole sequence in one pass are mathematically identical but differ in the last bit of bfloat16;
where the model is torn between two candidates, that is enough to flip the choice.

**How it was caught.** Three controls before concluding "bug": the same disagreement appeared with
steering switched **off**; all eight modes hit exactly 100% in **float32**; and every disagreement
sat on tokens whose top two logits were tied to within rounding error.

**What it cost.** The test was demanding exact agreement where the hardware cannot provide it. It
now counts only positions where the model's preference exceeds the measured rounding floor, and
reports the tied ones separately — so it would still catch a genuine bug. It passes 100% on every
decidable position, all eight modes.

### 5. Thresholds carried across a definition change

**What we believed.** The KL thresholds used to select comparison budgets in §6 were fine.

**What was wrong.** §5 deliberately redefined KL — every configuration is now scored on the *same*
fixed reference text, rather than on its own generations, which is what makes it comparable across
configurations. That moved the numbers by roughly 5×. The thresholds were still at week-1 scale.

**How it was caught.** Noticed before the run, by checking the thresholds against the observed
range rather than assuming them. At the old values the §6 verdict table would have come out
**completely empty**, and an empty table reads like a failed hypothesis rather than a
mis-calibrated constant.

**What it cost.** Nothing, because it was caught first. It was the third instance that day of a
constant carried across a definition change.

### 6. The H1 verdict reversed when the verdict rule was corrected

**What we believed.** The notebook printed *"adaptive beats static in 7/12 — H1 SUPPORTED"*.
Auditing those twelve comparisons, the honest reading looked like 4 adaptive vs 3 static — a coin
flip, with static winning at 90% sparsity.

**What was wrong.** Three defects in the verdict rule, not in the data. One row was **empty** — no
configuration met the tightest budget — and the tie-break silently scored it as a win for static.
Four rows sat at a KL budget of 12.6, deep in the over-steered regime where §4 had *already*
established the graded metric is blind. Two more were decided by gaps of 0.01–0.03, which is noise.

**How it was caught.** By auditing the twelve cells by hand instead of quoting the printed verdict.
The corrected rule requires the winner's confidence interval to clear the runner-up's, drops empty
rows instead of scoring them, and restricts budgets to the metric's validity domain.

**What it cost.** The direction of the result. With the corrected rule: 12 comparable cells, **5
decided, adaptive wins 5, static wins 0**, 7 statistical ties at n=48. Gradient attribution wins
the tightest budget, the signed score the loosest.

**Why this is not tuning the rule to get an answer.** This must be flagged rather than banked — it
is a conclusion that reversed because of an analysis rule changed *after* seeing the first verdict,
which is the exact failure mode the notebook exists to catch. Three things make it defensible: each
fix was motivated independently of the outcome (scoring an empty row as a win is a plain bug;
excluding budgets where §4 had already proven the metric blind was decided on §4's evidence;
CI-gating is standard); the fixes are **symmetric** and removed adaptive wins too; and what changed
is what the rule can **see**, not which way it points.

### 7. The gate certifying the whole analysis was circular

**What we believed.** `NB §10.2` checks the graded metric against real generations and printed
`r = 0.779 → VERDICT: graded metric tracks real generation — S6/S9 stand`.

**What was wrong.** The correlation was against `refusal_hits` — the 5.6%-precision matcher from
correction 2. The gate validated the new instrument against the instrument the notebook exists to
discredit. Three further problems: five of the eight points sat at exactly 100% refusal, so most of
the y-axis carried no information; the result hung entirely on the single unsteered anchor
(dropping it takes r to 0.566, below its own threshold, while dropping any other point leaves r in
0.758–0.827); and the unsaturated points arguably invert.

**How it was caught.** By reading the cell's source while auditing, and asking what `r["refusal"]`
actually was. Patch A then redid it against the judge at 128 tokens, keeping the generations this
time.

**What it cost.** The certification of §6 and §9 — and it produced the project's headline finding:

```text
r(margin, matcher refusal @48)   = +0.779   <- what the original gate used
r(margin, JUDGE full_refusal)    = +0.181   <- the honest check
leave-one-out range [-0.061, +0.363]
```

`refusal_margin` scores the log-probability of refusal-*shaped* tokens, so a degenerate loop of
"I cannot" is the most refusal-shaped text that can exist and therefore **maximises** the metric.
The two highest margins in the confirmation table are its two worst behavioural outcomes. The
string matcher makes the identical error for the identical reason. **Both instruments agree with
each other and disagree with the world.**

The root cause traces one level deeper: `NB §4` validated the graded metric by correlating it
against `refusal_rate` — the matcher again. The replacement was validated against the instrument it
was built to replace, and inherited exactly the blind spot it existed to remove.

### 8. Two fixes were applied where the problem was noticed, not everywhere it applied

**What was wrong.** `NB §9.2` computed the H3 winner as a raw maximum with no interval check — the
CI-gating fix from correction 6 had not been carried across. And `NB §7` still used the discredited
string matcher at 48 tokens, the short window that hides recovery, after both had been fixed
everywhere else.

**How it was caught.** By sweeping for other instances once the pattern was named, rather than
waiting to trip over the next one.

**What it cost.** Patch B re-ran the 28-config `pos_only` loop with per-prompt intervals: the H3
verdict held — position-only beats joint at every budget, 2 of 3 cells decided, joint wins 0 — and
proved robust to the validity-domain restriction, unlike H1. Patch C re-ran §7 behaviourally, and
**withdrew a claim**: measured with the matcher, per-step re-masking came out numerically higher in
**4 of 4** comparisons, which had been written up as "underpowered rather than null, a consistent
direction n=48 cannot resolve." Under the judge it is **1 of 4**, none CI-separated. The consistency
was the matcher's, not the model's — per-step produces more refusal-*shaped* tokens, not more
refusals.

**The pattern, named.** Five instances, all the same shape:

| the fix | applied at | missed at |
|---|---|---|
| verdict rule reads CIs | NB §6.5 | NB §9.2 |
| graded metric replaces the string matcher | NB §4, §6, §8, §9 | NB §7 |
| 128 generated tokens, not 48 | NB §2 | NB §7, §10 |
| validation against a trustworthy instrument | — | NB §10.2 |
| the same, one level deeper | — | NB §4 |

The root cause is treating the notebook as a sequence of cells rather than as one instrument, so a
correction lands locally. **When you fix something, grep for every other place it applies.**

### 9. The two metrics share a blind spot by construction, and so does the judge

**What we believed.** After correction 7, that the judge was the reliable instrument and the two
automatic metrics were the broken ones.

**What was wrong, part one — the window asymmetry.** This is not a coding defect; both metrics do
exactly what they claim. `refusal_margin` scores 6–12 token continuations placed immediately after
the prompt: a **start-of-sequence** measurement. `kl_vs_base` averages per-token KL over a 48-token
reference: a **whole-sequence** measurement. Gate the steering to the first 4–8 generated tokens and
you cover nearly all of the numerator while diluting the denominator across ~40 unperturbed
positions. At multiplier 2.0:

| positions | margin | KL | margin per unit KL |
|---|---|---|---|
| all | +2.847 | 8.401 | **0.34** |
| `prompt+1` | +1.737 | 0.666 | **2.61** |
| `first-4` | +2.393 | 1.458 | **1.64** |
| `first-8` | +2.922 | 2.448 | **1.19** |

`first-8` earns a **higher margin than steering everything** at **3.4× less KL**. A free lunch on
both metrics at once means the measurement is asymmetric. And it unifies with correction 1 rather
than being separate: late-sequence KL is low because the steering is off there and the model has
reverted to base behaviour — which on a benign prompt *is* "answers the question after apologising."

This compromises H3 (whose winning conditions are all first-*k* gates) and partly §8. H1 is immune,
because every sweep configuration steers all positions.

**What was wrong, part two — the judge.** Reference labelling of 55 generations, done blind and
completed to a full census of `dense/all m=1`'s 33 non-comply outputs, found the judge is reliable
on one axis and not the other:

```text
ANSWERED-axis agreement  28/30 = 93.3%
exact 4-way agreement    14/30 = 46.7%
```

Every one of the **7** generations it labelled `full_refusal` is a degenerate loop — *"I am
sorry."* six times over in answer to *"What is the product of 6 and 2?"*. Four of those seven were
scored `high` confidence, so its confidence does not track its accuracy on this distinction. Two
genuine refusals were sitting in the class it called `degenerate`. The errors run both ways and do
not cancel.

**What it cost.** The headline ceiling. Corrected against the census: **2/48 = 4.2%**, Wilson95
[1.2%, 14.0%], not 14.6%. And `r = 0.181` from correction 7 **should no longer be quoted**, because
it was computed against `judge_full_refusal`, a target now shown to be 0-for-7 on its positive
class. The conclusion it supported is *stronger* without it: the two highest-margin configurations
were sampled directly and read as **7 of 7 degenerate**.

**The unification.** Three instruments, one blind spot — and none of them looks at **repetition
structure**, which is the only thing separating a refusal from a broken model:

| instrument | what it does with a degenerate "I cannot" loop |
|---|---|
| `refusal_hits` (matcher) | matches the marker → scores it a refusal |
| `refusal_margin` (graded) | maximally refusal-shaped → **maximises** the metric |
| local letter judge | reads refusal language → labels it `full_refusal` |

Every replacement instrument built to escape the blind spot inherited it, including the one that
was supposed to be behavioural.

*(Caveat recorded honestly: the reference labels are **model** labels, not human ones. Using one
automatic classifier to certify another is the error above, one level up. It narrows what needs
human eyes to 9 specific items — `idx` 5, 17, 19, 38, 41, 42, 46 and the two corrections 27, 44 —
rather than closing the question.)*

### 10. The validity domain is the wrong shape of guard

**What we believed.** Restricting analysis to `mult ≤ 2.0` — the multiplier where the margin peaks
— protected the results from the over-steered regime.

**What was wrong.** The judge shows that ceiling admits configurations where the model is already
destroyed, and worse, that a global multiplier cap cannot work at all:

| config | multiplier | degenerate | answered |
|---|---|---|---|
| dense/all | 1.0 | **54.2%** | 31.2% |
| dense/all | 2.0 | **97.9%** | 0.0% |
| adaptive-0.90 frozen | 1.0 | ~0% | **97.9%** |
| adaptive-0.95 frozen | 1.0 | ~0% | **100%** |

At the *same* multiplier, dense steering is half-degenerate while the sparse configurations barely
intervene. Degeneration depends on how much the intervention actually perturbs the model — method
and sparsity together — not on the multiplier alone.

**What it costs.** The KL budgets `[0.13, 0.68, 4.53]` used in the corrected §6.5 and in Patch B
were percentiles over "in-domain" configurations, and in-domain includes both dense rows above. The
restriction that made the H1 verdict defensible was weaker than described. This does not invalidate
those verdicts — they already rest on a metric correction 7 showed does not track behaviour, which
is worse — but the validity domain should stop being cited as an independent safeguard.

---

## Where results stand

| result | status |
|---|---|
| Week 2's headline overturned | **Solid** — replicated, two independent signals, mechanism measured |
| The refusal matcher is 5.6% precision | **Solid**, with the mechanism |
| Both refusal metrics measure token shape, not behaviour | **Solid — the strongest result here** |
| The judge shares the blind spot; 7/7 of its `full_refusal` calls are degenerate | **Solid** — full census |
| Genuine refusal ceiling ~4%, not 14.6% | **Solid**, model-reference-labelled; wants a human check of 9 items |
| Sycophancy extraction is healthy (held-out probe AUC 0.880 at layer 16) | **Solid, new** — an *intervention* failure, not an extraction one |
| Position effect saturates by token 16; position 2 singular | **Suspect** — margin-based |
| H1 — adaptive wins 5 of 12 decided cells, static 0 | **Ranking axis suspect** |
| H3 — joint loses to position gating at every budget | **Same**, compounded by the window asymmetry |
| Per-step re-masking | **No difference** (judged), 0/4 CI-separated |

H1 and H3 are not refuted. They rank configurations by a quantity now shown not to correspond to
behaviour, which is a worse position than "unvalidated" — it cannot be fixed by adding data.

---

## Conventions that actually caught things

Worth keeping, because each of these has a correction above attached to it.

**Negative controls on every instrument.** *The unsteered model must never refuse.* This caught
both broken graders, including one that confidently reported the opposite conclusion on its first
run. Its limitation is now known too: the unsteered model does not degenerate either, so the
control is blind to a grader that confuses refusal with degeneration. **Any new control needs a
positive case** — a known-degenerate output the grader must not call a refusal.

**Decision rules written before the output.** Every notebook section states its rule in a markdown
cell above the code, so a result contradicting the hypothesis is recorded rather than
reinterpreted. `NB §55`'s outcome table was written before the run and is kept as the
pre-registration it was, with the actual outcome marked — the branch it called "most likely" is not
what happened.

**Changing an analysis rule after seeing data is flagged, never banked.** Twice (corrections 3 and
6), with the reasoning written into the notebook cell rather than only here. The test applied is
threefold: motivated **independently** of the outcome; **symmetric**, so it would have removed a
favourable result just as readily; and changing what the rule can **see** rather than which way it
points.

**Three controls before concluding "bug"** (correction 4).

**Fix everywhere, not where noticed** (correction 8).

**Results must never live only in volatile places.** Two separate incidents destroyed completed
runs — a kernel that exited before the save step, and a notebook rebuild that wiped stored outputs.
The runner now writes to disk after every cell, and the sweep checkpoints to
`s6_sweep_partial.json` after every block.

---

## Next, in priority order

> **Outcome, 21 August 2026.** Week 3.5 is complete and items 3-4 below are moot: the quantity they
> would re-measure is absent, not mismeasured. Gold labels put 0 clean refusals in 29 sampled
> `dense/all m=1` items and all 3 clean refusals in the unbiased sample in the two weakest
> conditions. `WEEK3_5_SUMMARY.md` has the numbers, the failed NLL prediction, and two process
> findings worth keeping (a control used as a filter deleted the best instrument; a partial re-run
> truncated an artifact holding GPU output).
>
> **Items 1 and 2 became week 3.5** (21 August 2026): three independent classifiers
> for the four outcomes, a 160-item hand-labelled gold set, two blocking controls including the
> positive case this document asked for, and a pre-registered acceptance gate. `WEEK3_5_SUMMARY.md`
> has the detail and the numbers measured so far. Items 3 and 4 stand as written.


1. **Build a behaviourally-anchored effect metric.** Judge `answered` as the primary measure, a
   **window-shifted margin** (continuations scored after ~20 tokens of the model's own generation)
   as the cheap proxy, validated **against the judge** this time. It must carry a **separate
   coherence axis**: degeneration is orthogonal to effect, not an extreme of it, and window-shifting
   alone does not fix it — 20 tokens into an "I cannot" loop, more "I cannot" is *more* likely.
2. **Human-verify the 9 flagged generations.** The judge is load-bearing for the main claim and has
   no human-agreement number. Needs no GPU; the text is in `week3_generations.json`.
3. **Re-select the operating point.** At `mult = 1.0` roughly two-thirds of dense outputs are
   already degenerate, so the sweep ran where the model was mostly broken. Define the validity
   domain by coherence, per configuration, not by where the margin peaks.
4. **Re-run NB §6, §8 and §9 on the new metric.** Must not start before 1–3 — re-running a sweep
   against an unvalidated metric at an operating point past the useful range is what produced this
   situation.

Any replacement validation gate must require: correlation **against the judge**, never against
`refusal_hits` or a descendant; **leave-one-out stability**, reported as a range; and enough
unsaturated points that no single anchor carries the result.

Deferred by design, so they are not mistaken for oversights: refusal *ablation* on harmful prompts,
and Llama-3.2-1B generalization — both premature until the metric is trustworthy.

---

## Week 4 — 23 August 2026

*Two corrections, one of them to the project's own process rather than to a result. Code:
`adass_week4_layers.ipynb`. Output: `week4_layers.json`, `fig_matcher_saturation.png`.*

### 11. The operating point was selected by a metric that is flat across the range it selected in

**What we believed.** Layer 16 was an unlucky choice — the reversal of 22 August was read as "the
phenomenon is a property of layer 16", with no account of *why* the project had ever been there.

**What was actually wrong.** It was not luck, and it was not arbitrary in the way "unlucky"
suggests. Week 1 chose the operating point with

```python
(BEST_LAYER, BEST_MULT) = max(sweep, key=sweep.get)   # sweep values are refusal_rate()
```

`refusal_rate` is the substring matcher, later measured at **5.6% precision**. Scoring that matcher
over the 22 August layer-sweep generations (§2.1, CPU-only, no new GPU time):

| layer | ‖v‖ | **matcher** | broken | clean refusal |
|---|---|---|---|---|
| 8 | 48.7 | 43.8% | 0.0% | 27.1% |
| 10 | 61.9 | **100.0%** | 0.0% | 93.8% |
| 12 | 95.0 | **100.0%** | 0.0% | 95.8% |
| 14 | 122.4 | **100.0%** | 0.0% | **97.9%** |
| 16 | 172.5 | **100.0%** | 37.5% | 47.9% |
| 18 | 248.4 | **100.0%** | 83.3% | 12.5% |
| 20 | 303.6 | 97.9% | 89.6% | **10.4%** |

The matcher is **exactly 100% for layers 10 through 18**, and 97.9% at layer 20. Over that
saturated span the quantity it stands in for moves **85.4 points**. An argmax over a constant
cannot select within it; which layer came out was decided by tie-breaking order, not by evidence.
The README's account of the subsequent re-selection — "lowest KL among **saturating** configs" —
inherits the same defect, because "saturating" is defined by the same matcher.

**How it was caught.** By reading week 1's selection cell after the week-1 and week-2 notebooks
were added to the handover on 23 August, and then asking what the criterion in it could actually
discriminate. The check cost minutes and needed no GPU; nothing prevented it being run in week 1
except that nobody looked at the selection code again after it ran.

**What it costs, and what it buys.** It costs the "unlucky layer" framing. It buys a strictly
stronger claim, and one that is not scooped by arXiv:2606.13720: that paper showed degenerate
output *inflates* the measured effect. This shows the standard instrument is **flat where the
behaviour changes**, so it silently corrupts vector and layer *selection* — the step every steering
paper performs before it measures anything. The metric failure and the operating-point failure are
the same bug, one level apart. Figure: `fig_matcher_saturation.png`.

**Still open, and it does not change the above.** ‖v‖ grows 6.2x across the sweep and the multiplier
scales the raw vector, so depth and strength are confounded in this table as in every layer
comparison the project has made. §6 of the week-4 notebook is the pre-registered matched-norm test
that separates them. Both outcomes leave correction 11 intact: whichever factor drives the
behaviour, the matcher could not see it.

### 12. `adass.py` had drifted from its sole writer, and the drift was invisible

**What we believed.** Trap 2 in `HANDOVER.md`: "`adass.py` has exactly one writer,
`adass_week3_5_taxonomy.ipynb` §0.2. Edit the cell, never the file."

**What was actually wrong.** The invariant had already been broken, by the 22 August session. The
file on disk carried the **repaired judge-v2 prompts** (the three guards — truncation, length,
formatting); the writer cell still held **v1**. Re-running that cell would have silently reverted
the judge repair and, with it, the negative control that the repair exists to pass.

**How it was caught.** By diffing the cell against the file — which nothing in the project did,
because the invariant was stated as a rule for humans rather than checked by code. Disk was
established as authoritative rather than assumed: its `judge_prompt_hash()` is `v2-bc0684f645b8`,
which matches the `prompt_hash` recorded inside `steps123_results.json`, so the file on disk is
demonstrably the version that produced the reversal evidence.

**What it costs.** Nothing yet — the drift was caught before the cell was re-run. What it changes
is the mitigation: the rule is now enforced rather than documented. §0.2 of
`adass_week4_layers.ipynb` is an **assertion** that the module is byte-identical to what the writer
cell writes, and it fails loudly with instructions on how to decide which side is right. The writer
cell has been resynced from disk.

**The general lesson, which the project has now learned twice.** An invariant that only a human can
check is a convention, not an invariant. The same reasoning produced week 3's cell 0.2 assertion;
it was applied to one notebook and not to the practice as a whole — which is correction 8's
pattern, *fix everywhere, not where noticed*, recurring at the level of process.

### What was run

- `adass.py`: added a strength-normalisation block — `NORM_REF`, `mult_matching_norm`,
  `norm_ref_vector`, `mean_hidden_norm`, `rel_mult_for`, `strength_row`. **`Steer3` is untouched**;
  these are parameter conversions applied to the vector before it reaches the hook, so the hook the
  week-3 replication gate covers is still the hook that was gated.
- `adass_config.json`: `best_layer` **deliberately left at 16**, marked `CONTESTED`, with the reason
  recorded. Changing it before §6 settles the confound would replace one unjustified operating point
  with another.
- `adass_week4_layers.ipynb`: §2 matcher saturation (run, CPU); §3–§5 reproduce the 22 August sweep,
  selection screen and judge control, which had **no code in the repository**; §6 the pre-registered
  matched-norm test; §7 a dimensionless relative-strength grid. §3 onwards need a GPU and are
  written to defer cleanly under `ADASS_LOAD_MODEL=0`.

### The caveat carried forward

All 160 gold labels are drawn from **layer-16** conditions. Both instruments are therefore validated
on layer-16 text and applied in §2, §3, §6 and §7 to layer-10-to-20 text — out-of-distribution use.
No week-4 number should be described as hand-confirmed until a blind label pass at a coherent layer
runs. That pass is the next item after this notebook, and it needs no GPU.

### 13. The repository was reorganised, and the generated-module pattern was retired

**What we believed.** That `adass.py` being written by a notebook cell was a safety property - one
writer, no drift.

**What was actually wrong.** It is the opposite. A generated module has no way to detect that it has
been edited, which is precisely how correction 12 happened. The pattern was inherited from Colab,
where a notebook is the only durable artifact; once the project has a repository, the module can
simply *be* the source and the notebooks can import it.

**What changed, 23 August 2026.**

| before | after |
|---|---|
| `adass.py` at the root, written by notebook §0.2 | `adass/core.py`, edited directly; the writer cell is an inert raw cell kept for provenance |
| bare filenames opened relative to the working directory | `adass.artifact("<name>")`, resolved from the repo root; `save_results` writes to `data/results/` |
| 30 files in one flat directory | `adass/`, `notebooks/`, `data/{gold,generations,vectors,results}/`, `docs/`, `figures/`, `config/` |
| no version control | git, with `pyproject.toml` and an editable install |
| notebooks ran only where they were written | notebook 05 bootstraps identically locally and on Colab |

**Why the path change matters more than it looks.** Every notebook opened artifacts by bare
filename, so each one worked only if its kernel happened to start in the project root. Moving
notebooks into `notebooks/` would have broken all of them silently - the failure mode being a
`FileNotFoundError` three sections in, after the model had loaded. Resolving from the repo removes
the class of problem rather than the instance.

**What was deliberately not done.** Notebooks 01-04 were **not** rewritten for the new paths. Their
saved outputs are the evidence the project's claims rest on, re-running them is already discouraged
by their own guards (§5 refuses to overwrite the label sheet; §0.2 was a `%%writefile`), and editing
a record to make it re-executable trades away the thing that makes it a record. They carry an
archival banner saying so. The single exception is notebook 04's writer cell, which had to be
neutralised because running it would now shadow the installed package with a stray `adass.py`.

**Cost.** None to any result. The 160 gold labels, the 384 generations and the extracted vectors were
copied, not moved, and the pre-restructure directory was left intact until the new tree was verified.

### 14. A control was reading the lower bound of its own confidence interval

**What we believed.** That §5 of notebook 05 enforced the criterion pre-registered in
`PLAN_REFUSAL_SUPPRESSION.md` phase 0: *the Wilson 95% **upper** bound on the false-broken rate must
be below 0.15*.

**What was actually wrong.** `adass.wilson_ci(k, n)` returns **three** values - `(point, lo, hi)` -
and the cell did `ok = ci[1] < 0.15`, which is the **lower** bound. The gate was being applied to a
number that is near zero whenever the rate is small, so it would have passed almost anything,
including the 3/48 failure the criterion was written to reject.

**How it was caught.** By reading the Colab output against the artifact it was meant to match: the
run reported "Wilson95 upper 0.004" where the 22 August artifact recorded `[0.003687, 0.108994]` for
the same 1 of 48. A reported upper bound *below* the point estimate is impossible, and that is what
gave it away. The 22 August code sliced correctly (`[1:]`); the bug was introduced on 23 August when
the check was rewritten.

**What it cost.** Nothing, on the numbers: the true upper bound is 0.109, so the control genuinely
passes and no verdict moves. What it cost is the guarantee - for one run, the project's most
load-bearing blocking control was decorative.

**The fix, and why it is shaped this way.** `point, lo, hi = adass.wilson_ci(...)` - unpack, never
index. Positional indexing into a tuple whose length you have to remember is what failed; naming the
components makes the same mistake unwriteable. The two other call sites in the notebook now store
`[1:]` to match the artifact convention, and `per_class_prf` in `core.py` was already doing so.

**The general lesson.** This is correction 7's pattern again - an instrument reporting a number that
looked plausible in isolation. It was caught only because there was an older artifact to compare
against, which is an argument for keeping expensive outputs on disk rather than regenerating them.

### 15. The confound settled: both factors, and depth controls the window

**The question.** `‖v‖` grows 6.2x with depth while the multiplier scales the raw vector, so every
layer comparison the project ever made varied depth and strength together.

**What was run.** Notebook 05 §6-§7, Colab T4, 24 August. Two norm-matched ladders and a 16-cell
relative-strength grid, n=48 each, decision rule fixed above the code beforehand.

**The answer: neither single-factor story.** The rule returned `TWO-DIMENSIONAL`.

- *Not strength.* Layer 16 at layer 10's perturbation norm gives **0% broken and 0% clean refusal** -
  it does nothing at all. "We were pushing too hard" predicted clean refusals there; there are none.
- *Not depth alone.* Shallow layers pushed up to `‖v_16‖` do degrade - layer 12 reaches 58.3% broken.

**What depth actually controls.** On the dimensionless `‖m·v‖ / ‖h‖` axis, layer 10 reaches **100%
clean refusal at 0% broken** and still holds 97.9% / 0% at 1.5x, while layer 16 never manages both at
any setting (best: 52.1% clean at 33.3% broken). Depth buys **tolerance** - how hard the model can be
steered before it stops working - and that survives normalisation.

**Why this matters more than the confound did.** It converts week 3.5's headline from wrong to
*local*. "The transition from no effect to model destroyed has no usable middle" is precisely true of
layer 16 and false at layer 10, where the middle is the whole range. Three weeks were spent at the
one depth in the range where the project's own research question has no room to exist - and the
reason that depth was chosen is correction 11: the selection metric could not see the difference.

**Consequences.** The operating point becomes **layer 10 at relative strength 1.0**. H1/H2/H3 run
there, compared at matched relative strength. `config/adass_config.json` can now be set from
evidence rather than left contested.

### 16. The layer-10 operating point is hand-confirmed

**The gap this closes.** All 160 existing gold labels come from the eight *layer-16* conditions, so
the mechanical coherence detector and the binary judge were both validated on layer-16 text and then
applied at layer 10 to produce the week-4 headline. That is out-of-distribution use, and it mattered
more than usual because the reported number was **100%** - a rate at the boundary is where a
miscalibrated instrument is least likely to be caught by its own confidence interval.

**What was run.** `notebooks/06_layer10_labels.ipynb`, no GPU. A 42-item blind sheet: 30 from
`L10/rel1.0` (stratum A, the claim), 6 `no-steer` (B, negative control), 6 `L16/rel1.5` (C, positive
control), shuffled, with no condition or stratum visible while labelling. Labelled by the four-class
decision procedure reproduced verbatim from `HANDOVER.md`.

**Controls, both blocking, both passed exactly.** All 6 negative-control items came out `answers`;
all 6 positive-control items came out broken. The blind labels landed on the stratum boundaries
without the strata being visible, which is the check that the sheet is worth reading at all.

**Result, stratum A, n=29 after excluding one `unsure`:**

| axis | agreement |
|---|---|
| coherence | **100%** [0.88, 1.00] |
| answered | **100%** [0.88, 1.00] |
| four-class | **100%** [0.88, 1.00] |

Zero four-class disagreements. Hand-labelled clean refusal **100%** [0.88, 1.00] against the
instruments' 100% over all 48.

**What this does and does not license.** It licenses dropping the out-of-distribution caveat: at the
new operating point the instruments and a blind human-procedure labelling agree item for item. It
does **not** license reading 100% as certainty - n=29 puts the Wilson lower bound at 0.88, so the
honest claim is "at least 88% clean refusal with 95% confidence", and that is what the write-up
should say.

**One item was labelled `unsure`, and it is the interesting one.** sid 18 declines fluently but
contradicts itself (*"I am an AI assistant programmed to be helpful, but I am also programmed to be
a bad guy"*), repeats two short sentences, and reads as text beginning to come apart without being a
loop. Both instruments call it a clean refusal. It may be the boundary case that shows what the
degradation looks like *before* the mechanical detector fires - worth a look if the write-up needs a
limitation, and worth remembering that `unsure` went unused across all 160 earlier labels, which was
recorded as a caveat at the time.

**The caveat that stands.** These are labels produced by an agent following the written procedure and
they carry the same status as the first pass over the original 160 - which a human then confirmed.
Confirmation of this sheet has not happened yet. One classifier certifying another is the error this
project has caught more than once, and the fact that the labeller here knew the hypothesis is a real
bias risk that the embedded controls limit but do not eliminate.
