# Week 3.5 summary - a validated four-class outcome classifier

> **ITS CONCLUSION IS OVERTURNED. Read `HANDOVER.md` first.** The headline below - "the quantity week
> 4 was going to re-measure is not there" - is true of **layer 16 only**, which is the single layer
> weeks 1 to 3.5 used. A layer sweep on 22 August found clean, coherent refusals at **93.8% (layer
> 10), 95.8% (12) and 97.9% (14) with zero breakage**. The classifier work, the controls, the hand
> labels and the process findings below all stand; the claim that the phenomenon is absent does not.


> Section references written `NB §N` point at the week-3 notebook; `§N` alone points at
> `adass_week3_5_taxonomy.ipynb`.

*Written 21 August 2026, covering the runs of the same day. Status: **complete. The gate says no-go,
and the reason is not the instrument.** Concepts and code: `ONBOARDING.md`. Prior work on output
quality: `RELATED_WORK.md`. How the labels were produced: `LABELLING-INSTRUCTIONS.md`.*

---

## Headline

**The quantity week 4 was going to re-measure is not there to measure.** Three classifiers were built
and scored against 160 hand-confirmed labels. Two of the two axes work - "is it broken" and "was it
answered" both reach ~91%. The class the project actually needs, a **coherent genuine refusal**, is
identified by nothing: the best precision across four candidate instruments is 0.12.

And the labels say why. At the operating point the whole project ran on, `dense/all` at multiplier 1,
**every sampled generation is a broken refusal and none is a clean one.** The only clean refusals in
the unbiased sample come from the two *weakest* interventions, which otherwise mostly just answer.

So steering harder does not produce more refusal. It produces more breakage. There is no
coherent-and-refusing regime for this vector on benign prompts, which means a better metric cannot
rescue H1 or H3 here - the phenomenon is absent, not mismeasured.

## The design: four classes are two axes

|  | answered | not answered |
|---|---|---|
| **coherent** | `answers` | `refuses` |
| **broken** | (rare) | `refuses_broken` if a refusal is expressed, else `just_broken` |

Every classifier produces the axes; `adass.four_class_from_axes` derives the label. Nothing votes on
four names. This was also the diagnosis of the incumbent judge's failure - a four-way discrimination
in one shot, 0 for 7 on the only pair that mattered.

## The three approaches, scored against the gold set

n=140 (strata A and B, duplicates and `unsure` excluded). Ceiling from the 20 duplicates: 100%,
CI [0.84, 1.00] - but see the caveats, this number does not mean what it would for a human labeller.

| approach | four-class | coherence axis | answered axis | `refuses` precision | kappa | controls |
|---|---|---|---|---|---|---|
| **binary judge** | **69.3%** [.61,.76] | **91.4%** [.86,.95] | **91.4%** | 0.12 | 0.522 | FAIL |
| combined (pre-registered) | 63.6% [.55,.71] | 85.0% [.78,.90] | 90.7% | 0.06 | 0.437 | PASS |
| internal (NLL + window margin) | 40.0% [.32,.48] | 69.3% [.61,.76] | 70.7% | 0.00 | 0.192 | FAIL |
| mechanical (gzip, dup-5gram, repeat span) | 37.9% [.30,.46] | 85.0% [.78,.90] | 67.9% | 0.11 | 0.235 | PASS |

Gate: coherence >= 90%, four-class >= 85%, genuine-refusal precision CI lower bound > 0.5.
**Nothing passes.** The best control-passing approach (combined) fails all three checks; the most
accurate approach measured (the judge) fails its control by 3 items out of 48.

### Controls

| approach | unsteered called refusing | unsteered called broken | loops called broken | loops called genuine refusal |
|---|---|---|---|---|
| mechanical | 0.0% | 0.0% | 100% | 0.0% |
| combined | 0.0% | 0.0% | 100% | 0.0% |
| binary judge | 0.0% | **6.2%** | 100% | 0.0% |
| internal | 0.0% | **12.5%** | **93.8%** | 2.1% |

The positive control (loops must be called broken, and must not be called genuine refusals) was
**tightened after its first run**, and the reasoning is recorded in §7.1 rather than banked: as first
written it asked only the second half, and approach 3 passed it while labelling 24 of 48 known loops
`answers`. A control a demonstrably wrong classifier passes is not a control. The change is motivated
independently of any outcome, is symmetric, and removed one of our own results rather than a rival's.

## What the labels say, by condition

Stratum A only - 10 random per condition, so unbiased within each:

| condition | gold labels |
|---|---|
| no-steer | 10 `answers` |
| **dense/all m=1** (the operating point) | **10 `refuses_broken`, 0 `refuses`** |
| dense/all m=2 | 8 `just_broken`, 2 `refuses_broken` |
| static-0.90 m=2 | 6 `just_broken`, 4 `refuses_broken` |
| adaptive_signed-0.90 m=2 | 8 `just_broken`, 2 `refuses_broken` |
| dense/first-4 | 8 `answers`, **2 `refuses`** |
| dense/prompt+1 | 9 `answers`, **1 `refuses`** |
| JOINT-0.90 m=2 | 10 `answers` |

All three clean refusals in the unbiased sample come from the two weakest interventions. On the
enriched sample, all **29** labelled `dense/all m=1` items are `refuses_broken` - zero clean refusals
out of 29, where the combined rule had claimed 15 and the judge 6. Both were over-calling badly.

This is a sharper statement than week 3's "~4% ceiling", and it corroborates it from an independent
direction: refusal is not merely rare, it is **absent where steering is strong** and appears only
where the intervention barely does anything.

## The NLL prediction failed

Pre-registered: a verbatim loop is predictable, so its NLL under the base model should be *lower* than
coherent text - which would have made the literature's one-sided perplexity gate blind to it.
Measured: coherent 0.393, loops 0.803. **Loops are more surprising, not less.** In-Distribution
Steering's criterion is directionally right after all.

The real limitation of perplexity here is duller: 21% of coherent items fall inside the loop range,
and the best achievable threshold reaches 90.6% balanced accuracy where every mechanical feature
reaches 100%. Usable as a second opinion, poor as a primary gate. §4 pre-registered this branch
explicitly, so `classify_internal` was simplified to one-sided by following the rule rather than
revising it.

## Process findings

**A control used as a filter deleted the best instrument.** §7.2 originally skipped any approach that
failed a control, so the judge - the most accurate of the four on both axes - was never scored, and
§8 declared a winner after comparing one candidate against nothing. A negative control is a
**necessary condition on what an instrument's output means, not a measure of its accuracy.** Fixed:
everything is scored, control status travels as a flag, and gate *eligibility* still requires a pass.
The verdict was unchanged either way, which is why the fix was safe to make after the fact.

**Two self-inflicted losses, both now structurally prevented.** A fresh process running only §0-§2
truncated `week3_5_taxonomy.json` and destroyed the §3 and §4 per-item labels. `save_results` now
merges at the top level and writes atomically; the single deliberate truncation point is §0.1, which
rotates the old file to `.prev.json`; and the expensive per-item output has its own files
(`week3_5_judge.json`, `week3_5_internal.json`) written by exactly one cell each. §3 and §4 now
**reload** from those files when the model is not loaded, so a scoring change costs seconds instead of
GPU minutes.

**The sheet guard earned its place twice.** §5 refuses to let a regenerated sheet replace a labelled
one, because the labels are keyed by `sid`. It fired on a real inconsistency (a reload path changed
which sources fed the candidate pool). Its first version asserted and aborted, which left `SHEET`
holding the regenerated ordering - so §7 scored every label against the wrong text and printed a 45%
ceiling and 51% coherence that were pure misalignment. The file on disk is now authoritative
unconditionally, and a mismatch warns rather than raises.

## Caveats that must travel with every number here

- The labels are **agent-produced and human-confirmed**. The confirmation is what makes them
  load-bearing; without it they would be one automatic classifier certifying another.
- **`unsure` was used zero times** across 160 items, on the hardest distinction in the project.
- The **100% self-consistency ceiling is not a self-consistency measure** for a single-context
  labeller that can see both copies of a duplicate.
- No judge-versus-*independent*-human agreement number exists yet.

## What this means for week 4

Week 4's plan was to re-select the operating point by coherence and re-run NB §6, §8 and §9 on a
trustworthy metric. **Both halves are now pointless for refusal.** There is no coherent-and-refusing
operating point to select, and the re-runs would rank methods by an absent quantity. The "empty
region" branch that week-4 planning was asked to pre-register has fired.

Recommended instead:

1. **Bank the negative result.** The measurement finding, plus a hand-confirmed taxonomy showing 100%
   breakage at the operating point and clean refusal only where steering is weakest, plus
   `RELATED_WORK.md`'s finding that two current papers and a July-2026 benchmark score exactly this
   with substring matching. That is a complete contribution and needs no further runs.
2. **Move H1 and H3 to sycophancy.** Extraction is healthy (held-out probe AUC 0.880 at layer 16),
   only the intervention was ever unfinished, and it is scored by A/B probability shift rather than by
   counting refusal-shaped tokens - so it does not inherit the blind spot this project is about.
3. **Keep the judge's two binary questions.** 91.4% on each axis against confirmed labels is a working
   instrument for a narrower job - breakage and answer rates, including in the sycophancy work. Only
   the four-class label and the genuine-refusal class are out of reach.

Still deliberately out of scope: refusal ablation on harmful prompts, and Llama-3.2-1B generalization.
