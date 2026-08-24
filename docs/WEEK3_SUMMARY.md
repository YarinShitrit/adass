# Week 3 summary — Adaptive Sparse Steering (AdaSS)

> Section references written `NB §N` point at the notebook.

*Written 14 August 2026, covering the validation runs of 10–11 August.
What was run and every correction: `WORKLOG.md`. Concepts and code: `ONBOARDING.md`.
Artifacts: `week3_results.json`, `week3_patches.json`, `week3_generations.json`,
`adass_week3_validation.ipynb` (all code cells carry saved outputs), `adass.py`,
`week3_reference_labels.json`.*

---

## Headline finding

**Refusal metrics measure token shape, not behaviour — and this is invisible unless you read
the generations.** Two independent instruments fail the same way for the same reason:

- The **string matcher** used throughout weeks 1–2 has **5.6% precision** against a behavioural
  judge: it fires on any *"I cannot…"* opener regardless of what follows.
- The **graded metric** (`refusal_margin`) built in week 3 *to replace it* tracks the matcher
  (r = 0.78) far better than it tracks behaviour — because it was validated against the matcher,
  and so inherited exactly the blind spot it was built to remove.
- The **behavioural judge** built to adjudicate between them shares the same blind spot on the
  one distinction that matters: reference labelling of its `full_refusal` calls found **7 of 7
  were degenerate loops** (see below). It is reliable for *"did the user get answered"* and
  unreliable for *"was that a refusal or a broken model."*

Three instruments, one shared failure: none of them looks at **repetition structure**, which is
the only thing separating a refusal from a model that has stopped working.

Both instruments score the log-probability or presence of *refusal-shaped tokens*. The two
texts that maximise them are the two worst behavioural outcomes in the experiment:

1. **Degenerate loops** — *"I cannot and I cannot do this. I cannot and I cannot do this…"* —
   the most refusal-shaped text that exists. The two highest margins in the confirmation table
   (2.847, 2.208) are ~98% degenerate output.
2. **Apology-then-answer** — *"I cannot help with that. … However, I can help you brainstorm
   some ideas!"* — scored as a refusal by both instruments; the judge reads it as 97.9%
   answered.

The behavioural bottom line: **the highest genuine refusal rate anywhere in the experiment is
~4%** (dense steering, multiplier 1 — 2/48, Wilson95 [1.2%, 14.0%]). The figure first recorded
there was 14.6%, taken from the judge's `full_refusal` count; a full census of that condition's
33 non-comply generations found every one of those 7 to be a degenerate loop, and 2 genuine
refusals hiding in the class the judge had called degenerate. On benign prompts with this vector,
steering does not reliably induce refusal — it either breaks the model or briefly perturbs the
opening.

The AdaSS hypotheses (H1–H3) are the case study that exposed this. Per the framing decision of
14 Aug, this methodological result is now the project's primary contribution; the sparsity
results below are provisional pending a behaviourally-anchored metric.

### Why "refused" versus "broken" is load-bearing

From the outside both look identical — the user got no answer — so it is worth stating why
conflating them invalidates the comparisons rather than merely mislabelling them.

**The effect and damage axes collapse into one.** The project trades effect against damage; if
what is scored as effect *is* damage, the Pareto frontiers plot damage against damage. Dense
steering at the two multipliers inside the supposed validity domain:

| multiplier | margin ("effect") | genuine refusal | degenerate |
|---|---|---|---|
| 1.0 | +1.620 | 4.2% (census) | 64.6% |
| 2.0 | **+2.847** | ~0–2% | **~98%** |

Turn the steering up and the effect score nearly **doubles** while real refusals fall to ~zero and
breakage becomes near-total. The metric is not noisy; it points the wrong way.

**They are opposite claims about the mechanism.** A refusal means we activated a capability the
model has — the publishable claim. Degeneration means the residual stream was pushed off the
manifold the model can process at all, which is already-known and says nothing about steering.

**Every comparison gains a way to win for the wrong reason.** H1 and H3 rank by effect at matched
KL, so a method can win by destroying the model more efficiently per unit KL. "Adaptive beats
static by 0.2 nats" is equally consistent with "adaptive breaks it at lower KL cost."

**Nothing in the experiment measures coherence.** KL is large whether the model refused *or* fell
apart, so effect and damage both rise under degeneration and neither rises *differently*. That is
why breakage stayed invisible for three weeks rather than appearing as an outlier. The negative
control ("the unsteered model must never refuse") was blind to it by construction, because the
unsteered model does not degenerate either — future controls need a **positive** case: a
known-degenerate output the grader must not call a refusal.

**It should have changed the experimental design, not just the write-up.** At the chosen operating
point roughly two-thirds of dense outputs are already degenerate. The right response is to lower
the multiplier or move layers until the model refuses *and* still writes English — the only regime
where "can this be made surgical?" is a meaningful question. Instead every method was ranked on
the manner of its breaking. Re-selecting the operating point is now a prerequisite for the NB §6/§8/§9
re-runs.

---

## What we did

Week 3 was the validity pass over the weeks 1–2 claims.

1. **Consolidated the machinery** into one `adass.py` (three drifted copies of the steering
   hook retired) and gated everything behind replication: vector norm reproduces to 0.016%,
   three week-2 anchor numbers reproduce within CI, and a forced-vs-generate equivalence test
   passes on all eight position modes.
2. **Printed the generations for the first time**, at 128 tokens instead of 48 — which is what
   overturned the week-2 headline.
3. **Built a graded, generation-free effect metric** (`refusal_margin`), a local behavioural
   judge (single-token A/B/C/D letter scoring, passing its negative control), and restored a
   comparable KL measure.
4. **Ran the honest H1 test**: 119-config sweep (4 mask methods × 4 sparsities × multipliers),
   compared as Pareto frontiers at matched KL with bootstrap CIs — removing the
   renormalisation confound (at s=0.90, week 2 was pushing adaptive masks 1.76–1.80× vs
   static's 1.45×).
5. **Ran the never-executed sections**: per-step re-masking (NB §7), the redesigned position
   experiment (NB §8), the first-ever H3 test (NB §9), confirmation generations (NB §10),
   sycophancy triage (NB §11).
6. **Applied three post-hoc patches** after auditing the run: a judge-based validation gate
   (A), CI-gating for H3 (B), and a behavioural re-run of NB §7 (C). **All three changed the
   answer the notebook had printed.**

## Results, by confidence

### Solid

- **Week 2's headline claim is overturned, and the overturn strengthened.** "Steer only the
  first 4 tokens — same refusal, better quality" is actually apology-then-answer: 97.9%
  answered per the judge, 95.8% per the recovery proxy, reproduced at 128 tokens in Patch A.
  Dense steering, by contrast, genuinely stops the model answering (31.2% answered) — mostly
  via degeneration, not clean refusal.
- **The refusal matcher is 5.6% precision** (144 flags, 136 false positives), now with the
  mechanism: it counts degenerate loops and apology openers as refusals.
- **Both metrics measure token shape, not behaviour** — the headline finding above.
- **The judge shares the blind spot on the refusal/degeneration split.** A full census of
  `dense/all m=1`'s 33 non-comply generations found all 7 of its `full_refusal` calls to be
  degenerate loops, and 2 genuine refusals misfiled as degenerate. It is reliable on the
  `ANSWERED` axis (93.3% agreement), which is what that overturn rests on, and only there.
- **Sycophancy extraction is healthy**: held-out probe AUC 0.880 at layer 16. Week 1 filed it
  as a failed result; it is an *intervention* failure, not an extraction one —
  reclassify to "unfinished."

### Provisional — ranked on a metric now known not to track behaviour

- **H1 (adaptive vs static masks):** within the graded metric's validity domain (mult ≤ 2.0)
  and at matched KL, adaptive masks are never worse than static in any CI-decided cell and win
  5 of 12 (7 ties at n=48); gradient attribution wins tight KL budgets, the signed score loose
  ones. Immune to the position-window artifact (all configs steer every position), but the
  ranking axis is the margin, so this stands or falls with the new metric.
- **H3 (the joint method):** NOT SUPPORTED as measured — position-only beats joint at every
  budget, robust to CI gating and domain restriction (Patch B). But every winning position
  config is a first-*k* gate sitting in the two metrics' **shared blind spot**: the margin is a
  start-of-sequence measure and the KL a whole-sequence average, so gating the first few
  tokens covers nearly all of the numerator while diluting the denominator over ~40 unperturbed
  positions (first-8: margin 2.922 > dense-all's 2.847 at 3.4× less KL). Low late-sequence KL
  *is* the reversion to base behaviour — i.e. the recovery artifact. H3 needs re-measuring, not
  re-interpreting.
- **Position structure (NB §8):** the generated-token effect saturates by position 16 and
  position 2 is singular (largest marginal and only non-trivial necessity). Suspect —
  margin-based; "position 2 matters" may mean "position 2 is where a refusal opener commits."

### Closed

- **Per-step adaptive re-masking (NB §7 / Patch C): no difference.** 0/4 comparisons CI-separated
  under the judge. The earlier "per-step higher in 4/4" hint was the matcher's artifact
  (judge: 1/4) and is withdrawn.

## Process findings worth keeping

- **The propagation pattern** — five times, a fix was applied where the problem was noticed
  rather than everywhere it applied: CI-gating (NB §6.5 not §9.2), the graded metric (everywhere
  but NB §7), 128-token generations (NB §2 not §7/§10), NB §10's gate validating against the
  discredited matcher, and NB §4 validating the new metric against the old one. The notebook must be treated
  as one instrument, not a sequence of cells.
- **Analysis rules changed after seeing data are flagged, not banked.** The H1 verdict reversed
  when the corrected rule (CI gating + validity domain) was applied; recorded with the
  defensibility argument (independently motivated, symmetric, changes what the rule sees rather
  than which way it points).
- **Negative controls caught every grader bug** (the unsteered model must never refuse), and
  persistence-after-every-cell prevented a repeat of the 4-hour data loss.

## Next steps

> **Outcome, 21 August 2026: items 1-4 are all superseded.** Week 3.5 built the classifier and
> scored it against 160 hand-confirmed labels. The gate says no-go, and the reason is not the
> instrument: at this operating point every sampled generation is a *broken* refusal and none is a
> clean one, so items 3 and 4 (re-select the operating point, re-run NB §6/§8/§9) have nothing
> admissible to measure. See `WEEK3_5_SUMMARY.md`. The genuine-refusal ceiling below is corroborated
> and sharpened rather than overturned.
>
> **Superseded in part, 21 August 2026.** Items 1 and 2 below became **week 3.5**, which is in
> progress: three independent classifiers for the four outcomes (refused / answered / refused but
> broken / just broken), anchored to a hand-labelled gold set, with a go/no-go gate for week 4. See
> `WEEK3_5_SUMMARY.md`. Item 2 grew from 9 items to a 160-item stratified sheet, because ~4% refusal
> prevalence means a small sample contains almost none of the class the project turns on. Items 3-5
> stand as written.

### As originally written (week 4), in priority order

1. **Build the behaviourally-anchored effect metric** — judge `answered` primary, a
   window-shifted margin (canned continuations scored after ~20 tokens of the model's own
   generation) as the cheap proxy, validated against the judge this time. Must carry a
   **separate coherence axis**: degeneration is orthogonal to effect, not an extreme of it, and
   window-shifting alone does not fix it. Critical path.
2. **Human-verify 9 specific outputs** (was: hand-label ~40). Model reference labels already
   exist in `week3_reference_labels.json`; what needs human eyes is the 7 `dense/all m=1`
   generations the judge called `full_refusal` (`idx` 5, 17, 19, 38, 41, 42, 46) and the 2
   corrections (`idx` 27, 44). One automatic classifier cannot certify another.
3. **Re-select the operating point.** At `mult = 1.0` roughly two-thirds of dense outputs are
   already degenerate, so the sweep ran where the model was mostly broken. Find the strongest
   setting at which output stays coherent, and define the validity domain by coherence rather
   than by where the margin peaks.
4. **Re-run NB §6/§8/§9 on the new metric.** Expensive; must not start before 1–3.
5. **Fold the confirmed numbers into the write-up.** The weeks 1–2 handover document is
   superseded and now lives in `archive/`; `ONBOARDING.md` replaces it.

Deferred by design (unchanged from the plan): refusal ablation on harmful prompts, and
Llama-3.2-1B generalization — premature until the metric is trustworthy.
