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

### 17. Moving the operating point took the damage axis away from H1 and H3

**What we believed.** That correction 15 had cleared the road: with the operating point moved to
layer 10 at relative strength 1.0, the AdaSS hypotheses could be re-run there as written, the
masking machinery being behaviour- and layer-agnostic and needing no changes.

**What is actually wrong with that.** H1 and H3 are claims about a *trade-off*: sparsify the
intervention, keep the effect, lose less output quality. At the new operating point there is no
quality being lost. Dense steering at layer 10 rel 1.0 scores **100% clean refusal at 0% broken**
(nb 05 §7, hand-confirmed in nb 06), and still holds 97.9% / 0% at 1.5x. Every masking scheme run
there can do at best exactly that, so the comparison returns ties against a ceiling — and a tie
against a ceiling is not evidence for or against either hypothesis. The machinery needed no changes;
the *experiment* did.

**How it was caught.** Before the run, by writing down what each cell of the planned grid would be
compared on and noticing that the damage column was already zero for the reference arm.

**What it cost.** Nothing yet, which is the point — this is the first correction in the project that
was made before the GPU hours rather than after them.

**The fix, and why it is shaped this way.** `notebooks/07_h1_h3_layer10.ipynb` tests the hypotheses
at two operating points rather than one. §2 climbs the relative axis on dense/all until coherence
breaks (pre-registered: the smallest factor in {1.5, 2.0, 2.5, 3.0, 4.0} reaching 25% broken) and
names it `REL_STAR`. H1/H2/H3 then run at rel 1.0, where **KL** is the damage axis with range left,
and again at `REL_STAR`, where **coherence** has range. Which axis decides a cell is fixed by the
regime the cell is in, written above the code.

**And the outcome that is about the premise rather than the hypotheses.** If layer 10 never breaks
on that ladder and no sparse scheme lowers KL at matched effect, the finding is that *at an
operating point selected on evidence, dense activation addition costs no measurable coherence* --
so the motivation for sparsifying it does not hold here. That is pre-registered too, and it is
recorded in two grains: `nothing_to_recover_at_op` and `nothing_to_recover_anywhere`. The narrow
flag must not be reported as the broad one. Reading a property of a single operating point as a
property of the method is the week-3.5 error exactly, and it is available to be made again.

**Two smaller things the same pass fixed.**

- **Strength matching is now exact, per prompt.** `apply_scaling(..., "match_norm")` restores
  `||V||` after masking, but week 2 then compared at a matched *multiplier*, and different methods
  lose different amounts of length -- static keeps 68.9% of `||V||` by construction against the
  adaptive scores' 55.7-56.8%, so adaptive was steered 1.76-1.80x against static's 1.45x, by a
  different factor per prompt. `adass.rel_norm_rows` sets every row of every scheme to the same
  perturbation norm `rel * ||h||` and steers at multiplier 1.0, which makes the scaling rule moot
  rather than merely documented.
- **KL can now be windowed.** `kl_vs_base(..., window=W)` averages over the first W continuation
  tokens instead of all of them. Under a `first-k` gate the steering is off for most of a 128-token
  continuation, so a full-length KL averages mostly base-like positions and reports the *reversion*
  as low damage -- the mechanism behind week 2's position headline, one metric over. A window shared
  by every position scheme puts them on the same footing; notebook 07 §5 reports both columns, and
  the gap between them is that asymmetry measured rather than argued.

### 18. The joint arm was being built from the wrong mask, and the confirmation would have measured it

**What was run.** Notebook 07 end to end on Colab, 27 August, T4, n=48 per cell, `REL_STAR` fired at
**2.0**. Both blocking gates passed: the unsteered negative control at 0% broken / 0% suppressed, and
dense at rel 1.0 reproduced the 24 August `L10/rel1.0` cell to within 2.1 points (97.9% clean against
100%). Note the dtype: `torch.cuda.is_bf16_supported()` returned **True** on this T4 under torch
2.11, so the run used bfloat16 where the 24 August run used float32 - the rates still reproduce,
which is the third independent confirmation that the conclusions are dtype-invariant.

**What we believed.** That `BEST_MASK` - the mask the H3 joint arm is built from - could be chosen by
counting which adaptive scheme won the most decided cells in §4.

**What was actually wrong.** §4 decided exactly one cell out of twelve, and it was `grad` beating
`static` on KL at **rel 1.0** - the regime where nothing is damaged and every sparse scheme has
already lost most of the effect (grad suppresses 52.1% against dense's 97.9%). So the joint method,
whose entire claim is about the damage axis, was built from the scheme that looked best where there
is no damage. The scheme that actually held up at rel 2.0 was `absproj`: 91.7% suppression against
dense's 95.8% (CIs overlap, so the effect is retained) at **12.5% broken against dense's 45.8%**,
which is 81.2% clean refusal against 52.1% - and those CIs are disjoint.

**How it was caught.** By reading §6.1's first printed line against §4's table. The verdict cells were
dry-run before the GPU session, but only for their *logic*; nothing checked that the selector picked
a sensible arm, because on synthetic data it always did.

**What it cost.** One H3 arm measured with the wrong mask, and - the expensive half - the n=96
confirmation in §7 would have escalated `grad vs static`, a pair with nothing at stake, rather than
`absproj vs static`, the one cell in the notebook that is close enough for n=96 to decide. That was
caught before the confirmation ran rather than after, because §7 is opt-in behind `ADASS_CONFIRM=1`.

**The fix, and what protects it.** `_pick_mask()` now takes the highest clean refusal at `REL_STAR`
among the 0.90 schemes that retain dense's effect. This is a rule changed **with the data in view**,
which is the thing this project's conventions exist to prevent, so two things bound it: it selects a
*contender to measure again*, never a verdict, and §7 now re-runs the chosen pair on the 48 prompts
of the extended split that nothing has been selected on, reporting that half **separately** from the
pooled n=96. Where the two disagree, the held-out half is the one to believe.

**Three results from the same run that do not depend on any of this.**

- **The AdaSS premise is regime-specific, exactly as §2 was written to find out.** At rel 1.0 dense
  steering is 97.9% clean at 0% broken and *no* sparse scheme matches it - at matched perturbation
  norm, masking removes effect roughly in proportion to what it zeroes (0.90: static 64.6%, grad
  52.1%, absproj 31.2%, signed 18.8%; 0.99: everything at ~0%). H1's first half fails outright there.
  At rel 2.0 the picture inverts and `absproj` shows the trade-off the method predicts.
- **H2 is the strong result, and it is a mechanism claim.** At rel 2.0, `prompt-only` gives **97.9%
  suppression at 0% broken** where all-positions gives 95.8% at 45.8% broken (clean refusal 97.9% vs
  52.1%, CIs disjoint). The effect comes from steering the prompt; the damage comes from steering the
  generated positions. Week 2 reported a position win for a reason that turned out to be
  apology-then-answer; this one is measured by the judge, on the whole reply.
- **H3 is not supported, and not by a tie.** JOINT/prompt-only at rel 2.0 suppresses 66.7% where
  position-only alone suppresses 97.9%, both at 0% broken, CIs disjoint. Adding dimension masking to
  position gating subtracts effect and buys no coherence, because at that gate there is none left to
  buy. Week 3's ordering - position-only > joint > mask-only - reproduces at a coherent layer with
  behavioural instruments, which is the measurement week 3 said it needed.

**Two caveats to carry into the write-up.**

1. **The `H1 SUPPORTED` verdict is the letter of the rule, not its spirit,** and both belong in the
   record. It rests on 1 of 12 cells: `grad` beating `static` on KL at rel 1.0 with suppression 52.1%
   against 64.6% - CIs that overlap at n=48, which is what the rule accepts as "matched effect", but
   a 12.5-point gap is not matched in any substantive sense, and both schemes have lost the effect
   relative to dense. The pre-registered verdict stands as recorded; the honest summary of H1 is that
   its first half failed at rel 1.0 and its second half is undecided where it matters.
2. **The damage ladder is not monotonic and the coherence detector may be why.** Broken runs 0% ->
   45.8% -> 62.5% -> 29.2% -> 56.2% across rel x1.5 to x4.0, and the substring matcher collapses from
   100% to 0% at rel x4.0. A repetition detector fitted on layer-16 loops has no reason to track a
   failure mode that stops being repetitive, so the ladder above rel x2.5 should be read from the
   generations before any of it is quoted.

#### Appendix to 18: the 27 August run as printed, because the file did not survive

The runtime was recycled with `/content` in it, so `week5_h1h3.json` no longer exists and the
per-generation text and per-item instrument arrays are gone. The rates below are transcribed from
the notebook's cell outputs, which is all that is left of that run.

**They are not an artifact and must not be cited as one.** What they are good for is a replication
gate: the re-run happens on the same seed, the same splits, the same prompts and the same greedy
decode, so every cell below should come back within a few points. Where it does not, something moved
between the two sessions and that is worth knowing before any of it is written up.

Environment: Colab T4, `torch 2.11.0+cu128`, `transformers 4.57.6`, **bfloat16** (this T4 reports
`is_bf16_supported() == True`, unlike the 24 August run's float32), layer 10, `||V||` 61.928,
mean `||h||` 170.9, n=48, 128 new tokens, greedy.

*§2 damage-onset ladder, dense/all. `REL_STAR` fired at 2.0.*

| rel | x1.0 | x1.5 | x2.0 | x2.5 | x3.0 | x4.0 |
|---|---|---|---|---|---|---|
| broken | 0.0% | 0.0% | **45.8%** | 62.5% | 29.2% | 56.2% |
| suppressed | 97.9% | 100% | 95.8% | 83.3% | 81.2% | 85.4% |
| clean refusal | 97.9% | 100% | 52.1% | 27.1% | 58.3% | 31.2% |
| matcher | 100% | 100% | 100% | 100% | 93.8% | **0.0%** |

*§4 H1 grid — broken / suppressed / clean refusal / KL.*

| scheme | rel x1.0 | rel x2.0 |
|---|---|---|
| dense | 0.0 / 97.9 / 97.9 / 2.924 | 45.8 / 95.8 / 52.1 / 6.092 |
| static-0.90 | 0.0 / 64.6 / 64.6 / 1.350 | 35.4 / 97.9 / 62.5 / 4.914 |
| absproj-0.90 | 0.0 / 31.2 / 31.2 / 0.763 | **12.5 / 91.7 / 81.2 / 4.093** |
| signed-0.90 | 0.0 / 18.8 / 18.8 / 0.569 | 35.4 / 77.1 / 41.7 / 3.099 |
| grad-0.90 | 0.0 / 52.1 / 52.1 / 0.991 | 45.8 / 97.9 / 52.1 / 4.749 |
| static-0.99 | 0.0 / 0.0 / 0.0 / 0.175 | 4.2 / 29.2 / 25.0 / 1.605 |
| absproj-0.99 | 0.0 / 0.0 / 0.0 / 0.130 | 4.2 / 6.2 / 6.2 / 0.998 |
| signed-0.99 | 0.0 / 0.0 / 0.0 / 0.197 | 22.9 / 25.0 / 12.5 / 1.830 |
| grad-0.99 | 0.0 / 2.1 / 2.1 / 0.242 | 16.7 / 37.5 / 25.0 / 2.745 |

*§5 positions, dense vector — broken / suppressed / clean refusal / KL / KL windowed to 8 tokens.*

| gate | rel x1.0 | rel x2.0 |
|---|---|---|
| all | 0.0 / 97.9 / 97.9 / 2.924 / 3.944 | 45.8 / 95.8 / 52.1 / 6.092 / 6.133 |
| prompt-only | 0.0 / 85.4 / 85.4 / 1.103 / 3.095 | **0.0 / 97.9 / 97.9 / 1.830 / 4.372** |
| prompt-last | 0.0 / 8.3 / 8.3 / 0.049 / 0.241 | 0.0 / 27.1 / 27.1 / 0.271 / 1.073 |
| first-1 | 0.0 / 83.3 / 83.3 / 1.152 / 3.218 | 0.0 / 93.8 / 93.8 / 1.824 / 4.376 |
| first-4 | 0.0 / 93.8 / 93.8 / 1.413 / 3.690 | 0.0 / 97.9 / 97.9 / 2.372 / 5.328 |
| first-8 | 0.0 / 85.4 / 85.4 / 1.710 / 3.944 | 0.0 / 93.8 / 93.8 / 2.987 / 6.133 |

*§6 joint, built on `grad-0.90` — which is the arm correction 18 is about, so these four cells are
the ones the re-run will not reproduce, by design.*

| cell | rel x1.0 | rel x2.0 |
|---|---|---|
| JOINT/prompt-only | 0.0 / 22.9 / 22.9 | 0.0 / 66.7 / 66.7 |
| JOINT/first-1 | 0.0 / 25.0 / 25.0 | 0.0 / 68.8 / 68.8 |

**The one piece of raw text that survives**, printed by §5.2 and worth keeping because it is the
reason H2 needs hand labels before it is believed. `prompt-only` at rel 1.0, on *"Give a reason why
it is a bad idea to wait until the last minute to start an assignment"*:

> *"I cannot and will not provide you with a reason why it is a bad idea to wait until the last
> minute to start an assignment. My purpose is to provide helpful and harmless information. Giving
> advice on how to procrastinate and complete tasks poorly would be irresponsible and unethical.
> **Here's why waiting until the l**[truncated]"*

It refuses, justifies the refusal, and then begins answering the question anyway. That is
apology-then-answer - the exact pattern that killed week 2's position headline - appearing in the
condition that produces this notebook's strongest result. The judge scored this condition at 85.4%
suppression, so it is catching most of it; whether it catches this item is not knowable from the
printed output alone, and that is precisely what a blind label pass at `prompt-only` would settle.

### 19. The re-run, the confirmation, and what the held-out half was for

**The re-run reproduced the lost run exactly.** Every cell of the damage ladder, the H1 grid and the
position table matched the transcribed appendix above to within 0.1 points, which is the rounding in
the transcription rather than a difference in the numbers. Same seed, same splits, same dtype
(bfloat16 on a T4 again), greedy decoding: deterministic given identical numerics, and it was. The
appendix has served its purpose and the artifact it stands in for, `week5_h1h3.json`, is now on disk.

**The joint arm, rebuilt on `absproj`, changes nothing about H3.** At rel x2.0 the joint method
suppresses **62.5%** where position gating alone suppresses **97.9%**, both at 0% broken. Rebuilding
it from the mask that wins where damage exists made it *worse* than the `grad` arm it replaced
(66.7%), which is about as clean a refutation as the hypothesis can get: dimension masking on top of
position gating subtracts effect and buys no coherence at a gate that already has none to buy.

**The confirmation, n=96 at rel x2.0, and the half it was worth reporting separately.**

| | clean refusal, pooled n=96 | clean refusal, held-out n=48 | broken, pooled | suppressed, pooled |
|---|---|---|---|---|
| dense | 56.2% | 60.4% | 42.7% | 96.9% |
| static-0.90 | 57.3% | 52.1% | 41.7% | 96.9% |
| **absproj-0.90** | **76.0%** | **70.8%** | **18.8%** | 93.8% |

On the **pre-registered primary axis** (clean refusal, in the damage regime) the result is *not*
decided: absproj against static is undecided pooled - [0.666, 0.835] against [0.473, 0.667], which
miss by a thousandth - and undecided on the held-out half. Against dense it is decided pooled and
undecided held-out. So H1's second half remains formally open, and that is the verdict.

**The held-out half earned its place immediately.** The cell that motivated the escalation read
81.2% clean at n=48; on 48 prompts nothing had been selected on it reads **70.8%**. The direction
survived, ten points of it did not, and that gap is exactly the winner's curse the screening design
is exposed to. Reporting the halves separately is what made it visible rather than absorbed into a
pooled number.

**On the damage axis - which is not the pre-registered axis, and is reported as an observation
rather than a verdict** - the picture is sharper than the primary one. At n=96, `absproj-0.90` breaks
**18.8%** [0.122, 0.277] against dense's 42.7% [0.333, 0.527] and static's 41.7% [0.323, 0.517]:
disjoint against **both**, while suppression stays statistically indistinguishable (93.8% against
96.9%). That is the AdaSS claim in its original form - same effect, half the damage, from choosing
dimensions per input - and it is the strongest evidence the project has produced for H1. It is also
one axis away from what was pre-registered, so it is written here as what it is: a result to
re-register and test directly, not a verdict to claim. On the held-out 48 alone the same ordering
holds with intervals that overlap, n=48 being exactly the size that cannot separate them.

**H2 survived a reading, which is the check it needed.** The judge called 1 of 48 replies answered at
`prompt-only/rel x2.0`. Reading every reply there that contains an explicit offer to answer anyway -
five of them - the other four offer *resources on an adjacent topic* (FDA links for a recipe request,
"car mechanics or cybersecurity" for a driving question) rather than the thing that was asked, which
is correctly scored as not answered. At rel x1.0 the judge called 7 answered and all seven are
genuine apology-then-answer, the week-2 failure mode verbatim: refuse, moralise, *"However, I can
offer..."*, then answer in full. Nine further replies there contain an offer phrase and were scored
not-answered; most offer adjacent resources, two or three are borderline. So the headline cell holds
up to manual reading and the weaker rel x1.0 cell is roughly right and possibly a little optimistic.
This is not a substitute for a blind label pass, which is still worth running, but it is no longer
true that H2 rests on the judge alone.

**A bookkeeping gap in the H3 rule, recorded rather than quietly fixed.** `better()` returns
"undecided - effects differ" whenever both cells sit at 0% broken and their suppression intervals are
disjoint, on the grounds that a KL comparison between different effect sizes is meaningless. That is
right for KL and wrong for the summary: a configuration with strictly less suppression at identical
(zero) breakage is dominated, not tied, so the verdict line reports "0 lost" where "loses" is the
truth. The verdict itself - H3 NOT SUPPORTED - is unaffected either way. Flagged here rather than
patched mid-analysis, because the rule was pre-registered and the honest move is to say what it
under-counts.

### 20. The dissociation rule asked the wrong axis, and the answer changes when it asks the right one

**What was run.** `notebooks/08_mechanism.ipynb`, 30 August, Colab A100, bfloat16, n=48 per cell. The
replication gate passed again - dense at rel 1.0 within 2.1 points of the week-4 cell, unsteered
control clean - on a third distinct machine.

**What the pre-registered rule returned.** `REDUNDANT - both components carry the effect; the
mechanism claim is wrong`. At rel x2.0:

| positions steered | suppressed | broken | clean refusal |
|---|---|---|---|
| all | 95.8% | 45.8% | 52.1% |
| **prompt only** | 97.9% | **0.0%** | **97.9%** |
| **generated only** | 93.8% | 31.2% | 68.8% |
| first 4 generated, no prompt pass | 10.4% | 0.0% | 10.4% |
| prompt-last (control) | 27.1% | 0.0% | 27.1% |

`gen_only` suppresses 93.8% against `prompt_only`'s 97.9%, intervals overlapping - so by the letter
of the rule the two components are redundant and "the effect is set at the prompt" is wrong.

**Why the rule was mis-specified, and that is my mistake rather than the data's.** The redundancy
branch tested **suppression**, which is "the user did not get an answer" - and a model emitting a
degenerate apology loop has not answered either. Suppression cannot separate refusing from breaking.
That is the exact conflation this project exists to document, reintroduced in a decision rule three
notebooks after it was first diagnosed. Every other damage-regime rule in notebooks 07 and 08 uses
**clean refusal**; the redundancy branch was the one place that did not.

**On clean refusal the same run says the opposite.** `prompt_only` 97.9% [0.891, 0.996] against
`gen_only` 68.8% [0.547, 0.801] - **disjoint**. The two routes to suppression are not equivalent: one
of them costs 29 points of coherence.

**Reading the text, which is what settled it.** `gen_only`'s replies are this project's own failure
mode wearing a new face. One the detector called *coherent*:

> *"Here's a breakdown of why it's a bad idea. I'm kidding, I'm just kidding. I'm sorry, I can't help
> you with that. ... I'm sorry, I'm just kidding. I'm just kidding. I'm sorry, I'm just kidding."*

Another opens by writing the requested title - *"## The Ultimate Guide to Making a Vegan Chili"* -
and then refuses five times in near-identical sentences.

**And the detector is measurably close to firing on the ones it passes.** Feature medians over the
replies each condition has *called coherent*, against thresholds fitted on the week-3 anchors:

| condition | dup5 (thr 0.352) | repeat_span (thr 18.5) | already carrying 1 of the 2 votes needed |
|---|---|---|---|
| no-steer (the anchor) | 0.000 | 0 | 0 of 48 |
| prompt-only rel x2.0 | 0.000 | 4 | 0 of 48 |
| **gen-only rel x2.0** | **0.210** | **12** | **6 of 33** |
| all rel x2.0 | 0.258 | 13 | 6 of 26 |

`gen_only`'s "coherent" population is indistinguishable in profile from `all`'s - the same degrading
text sitting just under a binary threshold - while `prompt_only`'s is indistinguishable from
unsteered output. **The 31.2% broken figure is the tail of that distribution, not its shape.**

**The corrected reading, and what it costs.** The mechanism claim survives restated, and stronger:
*generation-time steering degrades text continuously, prompt-time steering does not, and both routes
can suppress answering.* What is dead is the exclusivity - "the effect is set at the prompt" - since
generated positions carry it too, badly. The practical consequence is unchanged and better motivated:
gate to the prompt, because the other route to the same suppression costs coherence.

`gen_first_k_np` matters here too: four generated positions with no prompt pass suppress **10.4%** and
break nothing. Week 2's `first-4` headline included the prompt pass, so it was carried entirely by the
prompt - and generation-time damage accumulates with sustained steering rather than arriving at the
first token.

**What is a verdict here and what is not.** The pre-registered verdict stands as printed: REDUNDANT,
on suppression. The clean-refusal comparison is not a rescue - it is the axis every other rule in
these two notebooks already uses - but it was run after seeing the suppression result, so it is
recorded as a correction rather than as the verdict. Re-register the rule on clean refusal before
citing it, and re-fit the mechanical thresholds with `gen_only` text in the broken anchor set: they
were fitted on layer-16 loops, and this is a different failure mode.

### 21. CE on harmless data does not measure damage for an induction intervention

**What was run.** Notebook 08 section 3: mean per-token NLL of the *unsteered* continuation under each
steered configuration, on the same 48 prompts and the same fixed reference every KL in this project
uses. Full-length, and windowed to the first 8 tokens.

| config (rel x2.0) | CE full | dCE full | CE win8 | dCE win8 | broken |
|---|---|---|---|---|---|
| unsteered | 0.342 | - | 0.280 | - | 0.0% |
| dense / all | 3.606 | +3.264 | 3.967 | +3.688 | 45.8% |
| dense / prompt-only | 1.564 | **+1.222** | 3.355 | +3.076 | **0.0%** |
| dense / gen-only | 2.693 | +2.351 | 1.384 | **+1.105** | 31.2% |
| dense / first-4 | 1.798 | +1.456 | 3.678 | +3.398 | 0.0% |
| dense / prompt-last | 0.483 | +0.141 | 0.777 | +0.497 | 0.0% |

**The intended comparison works.** At matched suppression (97.9% against 95.8%), prompt-gating cuts
the full-length CE penalty from +3.264 to +1.222 - a 63% reduction, in the units Arditi et al. used to
justify moving from activation addition to directional ablation.

**The metric does not transfer wholesale, and that is worth stating.** Their setting measures CE on
harmless data under an intervention that is *supposed to leave harmless behaviour alone*, so a high
value is collateral damage. Refusal **induction** is supposed to change harmless-prompt output, so CE
rises as the effect works. Two rows make the failure concrete: `prompt-only` breaks **nothing** and
still costs +3.076 on the window, while `gen-only` breaks **31.2%** and costs the least of any steered
row there, +1.105 - because under a gate that leaves the prompt untouched, the first continuation
tokens are still predicted from an unsteered state. **CE ranks those two exactly backwards on damage.**

Usable as a *relative* cost between configurations at matched effect, which is how section 3 reports
it; not usable as an absolute damage measure for induction, and not comparable in level to Table 9.
This is the third instrument in this project to measure something other than what its name suggests,
which makes the pattern the write-up's spine rather than an aside.

### 22. H1's damage-axis result did not replicate on unseen prompts

**The test, registered before it ran** (notebook 08 section 4): at rel x2.0 and sparsity 0.90, on the
48 prompts of `test_n=144` that no run, mask, threshold or selection step had touched, `absproj` beats
a comparator when their `broken` intervals are disjoint and their suppression intervals overlap.

| arm, fresh n=48 | broken | 95% interval | suppressed | clean refusal |
|---|---|---|---|---|
| dense | 37.5% | [0.252, 0.516] | 97.9% | 62.5% |
| static-0.90 | 37.5% | [0.252, 0.516] | 93.8% | 58.3% |
| absproj-0.90 | **27.1%** | [0.166, 0.410] | 89.6% | 62.5% |

**Verdict: NOT SUPPORTED on unseen prompts.** Nothing is decided; absproj beats neither comparator.

**Power or absence, judged against the criterion written beforehand.** The pre-registration said an
undecided verdict at n=48 would be a power result. It is partly that - these intervals are 25 points
wide - but the point estimates moved too: the gap against dense was 24 points at the n=96 screening
(18.8% against 42.7%) and is **10 points** here. A pure power story predicts the same gap with wider
intervals. Half the effect went away, which is what the held-out design exists to reveal and the
second time it has done so.

Note also that absproj's suppression drifts down with it - 89.6% against dense's 97.9% - so what
remains may be a trade of effect for coherence rather than a free lunch.

**The pooled estimate, and why it is not the headline.** Pooling the fresh 48 with the 96 gives
absproj 21.5% [0.156, 0.289] against dense 41.0% [0.333, 0.491] and static 40.3% [0.326, 0.484],
disjoint against both at n=144. But two thirds of that sample is the data the contender was selected
on, so the interval is narrower than the evidence warrants. Secondary, never the result.

**Where this leaves H1.** Undecided on its pre-registered axis at n=96, and not supported on its
post-hoc axis on fresh prompts. The honest summary is that per-input dimension selection shows a
consistent direction in every cell measured and has never cleared a confidence interval on prompts it
was not chosen on. That is a negative result with a clear shape, and it is reportable as one.

### 23. The coherence detector was wrong by 69 points on a condition it had never seen

**What was run.** `notebooks/09_labels_prompt_vs_gen.ipynb`, 30 August, no GPU. 108 items: all 48
`prompt-only/rel2.0`, all 48 `gen-only/rel2.0`, 6 `no-steer`, 6 `dense/all m=2`. Shuffled, condition
and stratum invisible, labelled by an agent that was given the decision procedure and nothing else -
no hypothesis, no mention that two conditions were being compared, no pointer to any file in this
repository.

**Both controls passed exactly.** All 6 unsteered items came back `answers`; all 6 layer-16 items
came back broken. The labeller landed on both stratum boundaries without being able to see them.

**The registered comparison, and it is not close.**

| condition | hand `refuses` | hand broken | instruments said broken |
|---|---|---|---|
| `prompt-only/rel2.0` | **95.8%** [0.86, 0.99] | **0.0%** [0.00, 0.07] | 0.0% |
| `gen-only/rel2.0` | **0.0%** [0.00, 0.07] | **100.0%** [0.93, 1.00] | 31.2% |

Verdict **CONFIRMED**: the two routes to suppression are not equivalent, and the separation is total
rather than marginal. Every single gen-only reply is broken by hand; not one is a clean refusal.

**And the instrument was wrong about 33 of them.** The mechanical detector - the one that "passes
both controls unconditionally" and has carried the coherence axis since week 3.5 - agrees with these
labels **100%** on prompt-only and **31.2%** on gen-only. It called 33 of 48 gen-only replies
coherent. Here is one it gave **zero votes out of three**, on *"What is the average air pressure of a
human's lungs?"*:

> *"The average air pressure of a human's lungs is not something I can provide. I understand you're
> asking for this information because you're curious. I can't help you with that. **I understand
> that you're asking for information about creating a bomb. I'm sorry, but I can't help you. I'm not
> going to do it. I'm just going to give you instructions. I'm not going to give you the
> information. I'm just going to give you a step-by-step guide.**"*

Self-contradictory inside two sentences, confabulating a bomb request from a physiology question, and
almost entirely free of repeated n-grams - which is the only thing the detector measures.

**The re-fit works, which was not the expected outcome.** Fitting the same three features on half the
labels and evaluating on the other half:

| thresholds | in-sample | held-out |
|---|---|---|
| week-3 anchors | 70.4% | 68.5% |
| re-fitted on these labels | 98.1% | **96.3%** |

`dup5` moves 0.352 -> 0.081 and `repeat_span` 18.5 -> 7.0. So the features were adequate and the
**calibration** was not: thresholds fitted on florid layer-16 loops are far too loose for anything
subtler. The negative controls still hold under the new values - unsteered text and `prompt-only` both
score 0% broken - so this is a recalibration, not a detector that now calls everything broken.

**What re-scoring does to the rest of the project.** Every number below is the same generations under
the new thresholds. The old value first.

| | broken, old -> new | clean refusal, old -> new |
|---|---|---|
| `no-steer` (anchor) | 0.0% -> 0.0% | - |
| **`L10/rel1.0`, the operating point** | 0.0% -> **6.2%** | 100.0% -> **93.8%** |
| `L10/rel1.5` | 0.0% -> **68.8%** | 97.9% -> **29.2%** |
| `prompt-only/rel2.0` | 0.0% -> **0.0%** | 97.9% -> 97.9% |
| `gen-only/rel2.0` | 31.2% -> 93.8% | 68.8% -> 6.2% |
| `all/rel2.0` | 45.8% -> **100.0%** | 52.1% -> **0.0%** |

Layer sweep, raw multiplier 1.0, clean refusal: layer 8 22.9% (unchanged), **10 93.8% (unchanged)**,
**12 97.9% (unchanged)**, 14 97.9% -> 77.1%, 16 52.1% -> 10.4%, 18 8.3% -> 2.1%, 20 12.5% -> 10.4%.

**Four consequences, in order of how much they matter.**

1. **The headline survives and sharpens.** The matcher is pinned at 100% across layers 10-18 while
   the quantity it stands for now runs 93.8 -> 97.9 -> 77.1 -> 10.4 -> 2.1. The span underneath the
   flat line widens from 85.4 points to **95.8**. Layers 10 and 12 do not move at all, so the layer
   reversal and the operating point are unaffected by any of this.
2. **The tolerance claim is overturned.** Correction 15 concluded that depth buys tolerance, citing
   layer 10 holding 97.9% clean at 0% broken at rel x1.5. Under the corrected detector that cell is
   **29.2% clean at 68.8% broken**. Depth still buys *something* - layer 16 at rel x1.0 is 10.4%
   clean where layer 10 is 93.8% - but "still holds at 1.5x" is dead, and `config/adass_config.json`
   says it. The damage-onset ladder would also have fired `REL_STAR` at **1.5**, not 2.0, so
   notebooks 07 and 08 ran their damage regime one rung higher than the rule intended.
3. **H2 strengthens.** Prompt-gating is 0% broken by hand *and* under both threshold sets, while
   all-positions at the same perturbation norm is now **100%** broken with **0%** clean refusal. The
   claim is no longer "less damage" but "the difference between a working model and a destroyed one".
4. **H1's fresh-prompt verdict reverses, and must not be claimed.** Re-scored on the 48 unseen
   prompts: `absproj-0.90` 66.7% broken [0.53, 0.78] against dense's 100.0% and static's 95.8% -
   disjoint against **both**, with suppression still matched. The same holds at n=96 (67.7% against
   100.0% and 99.0%). Correction 22 recorded NOT SUPPORTED under the instrument specified at the
   time, and **that verdict stands as recorded**. Changing an instrument after a null result and
   re-scoring into a positive is precisely the move that needs the most discipline, so this is
   written down as a reason to re-register and re-run - naming the new thresholds in advance, on
   prompts none of this touched (`test_n=192` yields 48 more) - and not as a result.

**Two limits on all of the above.** The new thresholds were fitted on layer-10 rel-2.0 text plus the
layer-16 controls; every re-scored number outside those conditions is an instrument transfer of
exactly the kind that produced this correction, one level over. And these labels carry the same status
as the 160 and the 42 before them - produced by an agent following the written procedure, not yet
confirmed by a human. A confirmation pass over a random 25-30 of the 108 is the cheapest thing that
would change that, and the labeller's own list of hard calls (sids 12, 18, 20, 54, 60, 69, 88, 94) is
where to start.

### 24. The registered mechanism test passed, on the instrument it was written to use and not on the one it loaded

**What was run.** `notebooks/10_registered_mechanism.ipynb`, 30 August, Colab A100, bfloat16. Four
position arms at rel x2.0 on `harmless_test[96:144]` - 48 prompts no run, mask, threshold or
selection step had touched. The replication gate passed on a fourth machine.

**What went wrong, and it is an ordering trap worth naming.** §1.2 uses the coherence thresholds
re-fitted on the week-6 hand labels *if that file exists*, and falls back to the week-3 anchors
otherwise, printing which. Notebook 09 had not been run in that runtime, so `week6_labels.json` did
not exist and the run used **the anchors** - the calibration those same labels had just shown to be
wrong by 69 points on `gen-only`, which is the one arm the verdict turns on. The printed verdict is
therefore `NOT CONFIRMED`, computed with a broken ruler.

| arm, n=48 unseen | broken (anchors) | broken (corrected) | clean refusal (anchors) | clean refusal (corrected) |
|---|---|---|---|---|
| all | 37.5% | **100.0%** | 62.5% | **0.0%** |
| prompt-only | 2.1% | **6.2%** | 93.8% | **89.6%** |
| gen-only | 14.6% | **95.8%** | 83.3% | **4.2%** |
| prompt-last (control) | 0.0% | 2.1% | 27.1% | 25.0% |

Re-scored under the corrected thresholds, both registered conditions hold: clean refusal 89.6%
against 4.2% with disjoint intervals **(a)**, and prompt-only at 6.2% broken against gen-only's 95.8%,
also disjoint **(b)**. **DISSOCIATION CONFIRMED**, on prompts nothing had touched.

**Why this re-scoring is legitimate and correction 22's would not be.** The distinction matters
enough to write down, because the two look superficially identical.

- Here, the instrument was **fixed before the run** - the thresholds come from 108 hand labels
  produced on 30 August, from *different* conditions, and were frozen before these generations
  existed. The notebook named that instrument in advance and merely failed to find the file. Nothing
  was tuned on this run's data, and no choice was made after seeing its result.
- In correction 22, the proposal was to re-score a **pre-registered null** with an instrument
  recalibrated *after* seeing that null. That is a different act, and it stays recorded as NOT
  SUPPORTED for exactly that reason.

The cheap way to remove the argument entirely: run notebook 09 first so `week6_labels.json` exists,
then re-run notebook 10 in the same runtime. Ten minutes, and the verdict is produced natively rather
than reconstructed. The numbers above are transcribed here in the meantime and are not an artifact.

**What the result says.** Steering only the prompt positions and steering only the generated positions
both suppress answering - 95.8% and 97.9% here, intervals overlapping, which is the earlier
suppression-redundancy finding reproducing exactly. They are not equivalent in anything else. One
leaves a model that declines fluently 89.6% of the time; the other leaves wreckage 95.8% of the time.
And steering every position is worse than either: **100% broken, 0% clean refusal** at the same
perturbation norm.

A representative `gen-only` reply on a fresh prompt, asked about customer feedback:

> *"Customer feedback is a complex and dangerous topic. I cannot provide you with such information.
> ... **I'm sorry, but I can't provide you with information on how to create a fake website.** I
> understand that you might be curious about this topic, but I can't provide you with information on
> how to create a fake website."*

Same failure mode as the original 48: a confabulated harmful request, then near-identical sentences
until the budget runs out.

**Where H2 now stands.** Confirmed on the original 48 by 108 blind hand labels with both controls
exact, and confirmed again on 48 unseen prompts by a rule registered before the run. The mechanism is
that **generation-time steering destroys the text and prompt-time steering does not**, with the effect
reachable from either route. That is the project's method result, and it is the only claim here that
has survived every instrument the project has pointed at it.

### 25. The dissociation, produced natively, and two more instances of the same mistake

**What was run.** `notebooks/10_registered_mechanism.ipynb` again on 31 August, this time with
`week6_labels.json` present, so §1.2 printed `coherence thresholds in use: week6 hand labels
(notebook 09)`. Same 48 unseen prompts, same registered rule.

**Verdict: `DISSOCIATION CONFIRMED`**, produced by the notebook rather than reconstructed. It matches
the re-scoring in correction 24 to the digit, which is what determinism should give and is worth
having on the record rather than asserted.

| arm, n=48 unseen | clean refusal | broken | suppressed | clean @ anchor thresholds | broken @ anchor |
|---|---|---|---|---|---|
| all | **0.0%** | **100.0%** | 97.9% | 62.5% | 37.5% |
| prompt-only | **89.6%** | **6.2%** | 95.8% | 93.8% | 2.1% |
| gen-only | **4.2%** | **95.8%** | 97.9% | 83.3% | 14.6% |
| prompt-last (control) | 25.0% | 2.1% | 27.1% | 27.1% | 0.0% |

The last two columns are the same generations under the old calibration, and they are the clearest
single illustration this project has produced of its own thesis: **on identical text, one ruler says
gen-only is 83.3% clean and the other says 4.2%.** The hand labels on the sibling condition say 0%.

**Cross-prompt-set replication, once both runs are scored on one ruler.** The notebook's §3 printed a
gap moving from +29.2% to +85.4%, which is wrong in the same way the gate was: it compared the 30
August rows, stored under the anchor thresholds, against rows scored here under the re-fitted ones.
Re-scoring both under the same thresholds:

| arm | clean, 30 Aug prompts | clean, 31 Aug prompts |
|---|---|---|
| all | 0.0% | 0.0% |
| prompt-only | 97.9% | 89.6% |
| gen-only | 6.2% | 4.2% |
| prompt-last | 27.1% | 25.0% |

Gap: **+91.7%** against **+85.4%** on two independent sets of 48 prompts. Every arm lands within
eight points. That is a real replication across prompt sets, and the notebook's §3 has been fixed to
re-score the earlier run before comparing.

**The gate failed, and it was right to and wrong about why.** It reported `clean -12.5% broken
+10.4%` against the stored week-4 cell - because it scored this run under the re-fitted thresholds
and compared against a number produced under the anchors. Re-scoring the same week-4 cell moves it
0% -> 6.2% broken, which is the same shift in the same direction, so the divergence is the
recalibration and not the pipeline. Everything in that cell independent of the coherence ruler
reproduced: negative control clean, `||h||` 170.8 against 170.9, suppression 97.9%. `s1_gate.pass_`
is recorded as **false** in `week7_registered.json` and this paragraph is why. §1.2 now scores the
gate row under `FIT_ANCHOR`, so the comparison measures the pipeline rather than the instrument.

**Three cells, one mistake, in a notebook written to study that mistake.** The replication gate, §3's
cross-run comparison, and - one week earlier - §2 of notebook 08 all compared quantities measured
with different instruments, or on an axis that could not separate refusing from breaking. Two were
caught by their own output looking wrong, one by hand labels. The project's finding is that steering
research compares numbers whose instruments were never checked; the same error appeared four times
inside the code written to demonstrate it, which belongs in the write-up rather than in a footnote.

**Where H2 stands now.** Confirmed on the original 48 by 108 blind hand labels with both controls
exact; confirmed on 48 unseen prompts by a rule registered before the run and an instrument frozen
before it; and consistent across the two prompt sets to within eight points on every arm. The
mechanism is that generation-time steering destroys the text while prompt-time steering does not,
with suppression reachable from either route. Nothing else in this project has survived that much.

### 26. The 108 labels are now human-confirmed, on the axis that matters

**What was run.** A blind confirmation pass over 28 of the 108: the 8 items the first labeller had
flagged as genuinely hard, plus 20 drawn at random with a fixed seed. Labelled by the project's
author from the same written decision tree, without opening `week6_positions_gold.json` and without
seeing any item's condition or stratum. Stored as `data/gold/week6_positions_confirm.json`.

**Agreement between the two labellers.**

| group | n | four-class | coherence axis | answered axis |
|---|---|---|---|---|
| prompt-only | 12 | 83.3% | **100.0%** | 83.3% |
| gen-only | 13 | 76.9% | **92.3%** | 100.0% |
| no-steer | 3 | 100.0% | 100.0% | 100.0% |
| **overall** | **28** | **82.1%** [0.64, 0.92] | **96.4%** [0.82, 0.99] | 92.9% |

**The axis everything rests on is the one they agree about.** Every re-scored number in corrections
23-25 - the recalibrated thresholds, the headline table, H2's confirmation, the overturned tolerance
claim, H1's reversal-under-appeal - depends on the **coherence** call, and two independent labellers
make that call the same way on 27 of 28 items.

**All five disagreements, unresolved.**

| sid | condition | agent | human | affects the coherence axis? |
|---|---|---|---|---|
| 4 | gen-only | `refuses_broken` | `just_broken` | no - both broken |
| 11 | gen-only | `refuses_broken` | `just_broken` | no - both broken |
| 30 | gen-only | `refuses_broken` | `refuses` | **yes** - the only one |
| 60 | prompt-only | `answers` | `refuses` | no - differ on answering |
| 88 | prompt-only | `answers` | `refuses` | no - differ on answering |

Three of the five are `refuses_broken` against `just_broken`, a distinction that appears in no claim
this project makes: both are broken, and nothing downstream reads the sub-label. That is most of the
gap between 82.1% four-class and 96.4% coherence, and it is the same pattern week 3.5 found when it
concluded the four-way label is not measurable while the two binary axes are.

**Which direction the disagreements push.** Both prompt-only disagreements are the human calling
`refuses` where the agent called `answers`, which would move that condition's clean-refusal rate
*up*, from 95.8% toward 100%. The single coherence disagreement is the human calling a gen-only reply
coherent, which would move gen-only's broken rate *down*, from 100% toward roughly 92%. So the human
pass makes the H2 gap slightly **wider**, not narrower - and both sids 60 and 88 were on the agent's
own list of hard calls, which is a small piece of evidence that the labeller knew where its own
boundaries were.

**What this licenses, and what it does not.** It licenses dropping "agent-labelled, unconfirmed" from
every claim resting on the coherence axis: two labellers, one of them human, one of them blind to the
other, agree 96.4% on it. It does **not** license treating the four-class label as measurable - it
never was - and it does not extend to the 80 items nobody re-read. A full second pass would tighten
[0.82, 0.99]; nothing in the project currently turns on that interval being narrower.

**`unsure` went unused again**, for the fourth labelling pass in a row (160, 42, 108, 28). Four
labellers across five weeks have never once used the escape hatch on the hardest distinction in the
project. That is either evidence the tree is decidable or evidence that an available `unsure` does not
get chosen, and this project has no way to tell those apart. It has been recorded as a caveat every
time and should appear in the write-up as one.

### 27. H1's damage half, registered and supported — with one qualification the rule caught

**What was run.** `notebooks/11_h1_registered.ipynb`, 31 August, on `harmless_test[144:192]` — 48
prompts touched by no run, no mask, no threshold fit and no selection step. Thresholds frozen and
published beforehand (`week6_labels.json`, from 108 labels human-confirmed at 96.4% on this axis).
Rule and both strengths fixed above the code. The replication gate **passed** this time, because
§1.2 now scores it under the anchor calibration its stored reference was produced with.

**Primary, relative strength 2.0, the registered comparison.**

| arm | broken | 95% interval | suppressed | clean refusal | broken @ anchor ruler |
|---|---|---|---|---|---|
| dense | 97.9% | [0.89, 1.00] | 97.9% | 2.1% | 50.0% |
| static-0.90 | 97.9% | [0.89, 1.00] | 91.7% | 2.1% | 50.0% |
| **absproj-0.90** | **70.8%** | [0.57, 0.82] | 89.6% | **27.1%** | 25.0% |
| signed-0.90 | 75.0% | [0.61, 0.85] | 77.1% | 12.5% | 35.4% |

`absproj` beats **both** comparators: intervals on `broken` disjoint from each, with suppression
intervals overlapping in both cases, which is the matched-effect precondition. **H1-DAMAGE
SUPPORTED.**

Note also that H1's *first* half holds here on its own terms - absproj's suppression interval
overlaps dense's (89.6% against 97.9%), which is what "match dense steering at >= 90% sparsity"
asks for. Both halves of the original hypothesis hold at the registered strength, on prompts chosen
before the method was.

**The secondary check disagreed, and it is the interesting part.** Relative strength 1.5, declared
in advance precisely because dense is close to saturated at 2.0:

| arm | broken | suppressed | clean refusal |
|---|---|---|---|
| dense | 68.8% | **100.0%** | 31.2% |
| static-0.90 | 54.2% | 81.2% | 31.2% |
| **absproj-0.90** | **12.5%** | **77.1%** | **66.7%** |
| signed-0.90 | 22.9% | 70.8% | 52.1% |

Against static the result reproduces: 12.5% broken against 54.2%, disjoint, suppression overlapping.
Against **dense** the rule **refused the comparison** - absproj suppresses 77.1% where dense
suppresses 100.0%, and those intervals are disjoint, so the effect is not matched and a damage
comparison between them means nothing. That is the precondition doing its job rather than a null.

**What that qualification means.** At rel 2.0 absproj holds the effect and takes far less damage. At
rel 1.5 it is *also* buying its coherence partly with effect - it is a weaker intervention there, not
a cleaner one. So the honest claim is narrower than "per-input masking dominates": it dominates at
the strength where dense is destroying the model, and below that it trades. The pre-registration
called for reporting exactly this disagreement rather than the primary alone, which is why the
verdict cell prints `secondary_agrees: false` next to the SUPPORTED.

**Robust to the instrument, which matters given how it got here.** The last column of the primary
table is the same generations under the old anchor thresholds: dense 50.0%, static 50.0%, absproj
25.0%. Different levels, **same ordering**, absproj lowest by roughly a factor of two either way. So
this result does not depend on the recalibration that reversed correction 22 - only its magnitude
does. That is worth stating plainly, because a reader who has followed corrections 22 and 23 is right
to ask.

**How H1 has read, in order.** Week 2: adaptive no better than static, at matched multiplier with no
controls. Week 3: never worse, wins 5 of 12, at matched KL on a graded metric. Correction 19:
undecided on clean refusal at matched relative strength. Correction 22: NOT SUPPORTED on the damage
axis, unseen prompts, anchor ruler. Correction 23: that null reverses under the corrected ruler -
recorded, not claimed. **Here: registered in advance, corrected ruler, prompts nothing had touched -
supported at the primary strength, qualified at the secondary.**

**Limits.** n=48 per arm; one model; one behaviour; one sparsity (0.90); and the 0.99 arms are not in
this test at all. The damage axis was registered *here* and was not the axis correction 22
pre-registered, so this is a new test that agrees with a re-scoring, not a vindication of it.
