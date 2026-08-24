# Related work: how prior work handles output quality

*Written 20 August 2026. Scope: **one question only** - when a steering paper reports that an
intervention worked, does anything in its evaluation rule out that the model was simply broken?
This is the literature check behind the week-4 priority "build a behaviourally-anchored effect
metric with a separate coherence axis" (`WEEK3_SUMMARY.md` §next steps, item 1).*

*Reading method and verification status: §6. Three papers were read in full from arXiv HTML;
AxBench was read in full as a fourth because it turned out to hold the design worth copying.
Papers characterised second-hand rather than read are marked as such and should not be cited
from this note.*

---

## Summary

| Paper | Effect metric | Output-quality metric | Per-generation coherence gate |
|---|---|---|---|
| **What Drives Representation Steering?** (2604.08524) | ASR: LlamaGuard2 on harmful prompts, **refusal substring matching** on harmless | none | **none** |
| **RepBench** (2607.28008v2) | probe AUROC; substring non-refusal on harmful, **substring refusal rate on Alpaca** | none | **none** |
| **In-Distribution Steering** (2510.13285) | SPI, plus GPT-4.1 judging target behaviour on open generation | **perplexity**, aggregate | no; coherence is bounded *a priori* by an in-distribution constraint |
| **AxBench** (2501.17148v2) | LLM-judge concept score, 0-2 | **LLM-judge fluency score, 0-2, separate axis** | **yes, effectively** - harmonic mean, so a zero on fluency zeroes the result |

The pattern: the two papers that measure *refusal* specifically have no output-quality axis at
all. The two that do have one are not refusal-specific, measure quality in aggregate or bound it
in advance, and neither reports a per-generation refused-versus-broken label.

---

## 1. What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

Cheng, Wiegreffe, Manocha (U. Maryland), arXiv:2604.08524, 9 April 2026.
<https://arxiv.org/abs/2604.08524>

**Why it matters here.** This is the 90-99% static sparsification result week 1 partially
contradicts, and its setup is close to ours: Gemma 2 2B Instruct (plus Llama 3.2 3B),
difference-in-means following Arditi et al., 128 train / 32 validation prompts per side, harmful
prompts from AdvBench and friends, harmless from Alpaca. Their best Gemma DIM vector is **layer
15, position -1**; ours is layer 16. Same model, same method, adjacent layer.

**How they measure effect.** Attack Success Rate, the proportion of completions that bypassed
refusal. Two directions, two instruments:

- **bypass** (subtract the vector, harmful prompts, JailbreakBench and StrongReject): judged by
  **LlamaGuard2 8B**, which reads for unsafe content.
- **induce** (add the vector, harmless Alpaca prompts): judged by **refusal substring matching**,
  lower ASR being better.

Their substring list is twelve markers - "I'm sorry", "I am sorry", "I apologize", "As an AI",
"As a language model", "As an assistant", "I cannot", "I can't", "I am unable to", "I'm unable
to", "I am not able to", "I'm not able to". Compare `REFUSAL_MARKERS` in `adass.py`: ten of those
twelve are in our list verbatim, and our list adds five more. **It is materially the same
instrument, on the same model.** That is what makes our 5.6% precision figure a statement about
their evaluator rather than only about ours.

**How they measure quality.** They do not. Keyword search over the full text returns no measurement
of fluency, perplexity, coherence, repetition, or degeneration, and there is no report of reading
generations. Re-checked 21 August with stem searches rather than whole words: the three incidental
hits are "incoherent top tokens" (about an attention head, not output), a figure-storage note, and
one forward-looking line worth quoting because it concedes the axis - the mechanistic insight
"could inform more targeted or fine-grained steering interventions, with the goal of **improving
concept expression without sacrificing generation quality** (Feng et al. 2026)". Named as future
work, cited to someone else, not measured here. The one manual step in the paper filters the *training* data for the learned
vectors, not the steered outputs.

There is a KL term, but it does not do this job: vector selection minimises
`sigmoid(bypass score) - sigmoid(induce score)` subject to `induce score > 0`, `kl score < 0.1`,
and `layer < 0.8L`, where the KL is measured for **directional ablation** on the harmless
validation set. So it constrains the ablation direction, not the addition that induces refusal,
and it is a distributional distance rather than a coherence check - exactly the quantity week 3
showed rises whether the model refuses or falls apart (`ONBOARDING.md` §3.3).

**The contact point, stated carefully.** Their sparsification claim is an ASR average over Alpaca
(substring), JailbreakBench and StrongReject (LlamaGuard2). On the Alpaca half, a degenerate
`"I cannot"` loop contains a refusal marker, therefore counts as "did not bypass refusal",
therefore counts as successful refusal induction. Our census says that on this model at the
strengths where the effect looks strong, most of those outputs are loops.

What that does **not** license: their bypass-direction results are LlamaGuard-judged and untouched
by our finding, and we have not re-run their sparsification, so we cannot say their 90% number is
wrong. The defensible claim is narrower and still worth making - **on the induce direction, the
instrument that carries their claim cannot separate refusal from collapse, and we measured how
badly on this exact model.**

**One convergence worth recording.** Their gradient-based sparsification beats IE-based, random
dropout, and bottom-k, particularly above 80% sparsity. Week 3's H1 independently found gradient
attribution winning the tight KL budgets (`WEEK3_SUMMARY.md`, provisional results). Two different
setups, same winner among mask scorers. Both remain provisional here, but it is a point of
agreement rather than a conflict.

**Where it leaves AdaSS.** Their masks are **static**: one threshold on gradient scores, one mask
per vector. The per-input axis really is untouched by this paper. The proposal's framing of the
gap holds on this axis.

---

## 2. RepBench: A Benchmark for Representation Engineering

arXiv:2607.28008v2, July 2026. <https://arxiv.org/html/2607.28008>

**What it is.** A benchmark and data layer for representation engineering: 94 representation
targets compiled from benchmark sources, 12 model checkpoints, cross-benchmark transfer measured
by leave-one-benchmark-out AUC and retrieval accuracy. The steering part is a **controlled
data-only substitution**: reproduce Arditi et al.'s refusal-direction extraction and Rimsky et
al.'s CAA hallucination steering, changing *only* the direction-extraction data.

**How they measure effect.** For reading, probe AUROC on unseen JailbreakBench and HarmBench. For
intervention, on Llama-3-8B-Instruct with five fixed refusal interventions: substring-based
non-refusal rate on harmful prompts, and **substring-based refusal rate on Alpaca at +1**. Their
Table 12:

| direction | Alpaca refusal at +1 |
|---|---|
| no intervention | .000 |
| original specialised data | **1.000** |
| RepBench data | .870 |

**How they measure quality.** They do not. Zero hits across the full text for fluency,
perplexity, repetition, and gibberish; the four "coherence" hits are a probing-target category
and a cited paper title. Human effort in the paper audits dataset-to-target mappings, not
intervention outputs.

They *do* scope-limit their causal evaluation, and honestly: "These measure whether the direction
mediates refusal under the reproduced protocol, not whether a non-refusal response contains
harmful assistance." Note which gap that closes and which it leaves open. It disclaims the
harmful-content axis. It says nothing about whether a *refusal-scored* response is a refusal or a
collapse.

**Why this is the strongest external support for the project's contribution.** This is a July 2026
benchmark - the field's current attempt at standardised measurement - reporting **100% refusal
induction on harmless prompts** at +1, on an instrument we have measured at 5.6% precision for
that exact task. Different model (Llama-3-8B-Instruct, not Gemma-2-2b), so we cannot claim their
number is inflated; we do not know that it is. The claim is about the instrument, and it is
sharper here than anywhere else in the literature: if the standard evaluator cannot tell refusal
from collapse, then a benchmark built on it cannot either, however careful everything else about
it is.

---

## 3. In-Distribution Steering: Balancing Control and Coherence in Language Model Generation

Vogels, Wong, Choho, Blangero, Bhan (Ekimetrics, Sorbonne), arXiv:2510.13285, 15 October 2025.
<https://arxiv.org/abs/2510.13285>

**This is the paper that should already be in our related work.** It states our problem in its
abstract: fixed steering strength "degrades text plausibility and coherence," and strong
intervention "may push activations outside the natural distribution of the target behaviour,
increasing the risk of collapse and non plausibility during text generation."

**The method (IDS).** Per-input, per-position steering strength, chosen as the *largest* value that
keeps the steered activation inside the target behaviour's activation distribution. Concretely:
build contrastive activation distributions at the last prompt token, PCA them down (30-42% retained
variance works best), measure Mahalanobis distance to the positive-class distribution, set the
threshold at the **95th percentile** of that distance, and solve for the largest alpha satisfying
the constraint - which has a closed form. Layers are restricted to those whose probe F1 clears a
threshold (performance declines above roughly 80%). Six models including **Gemma-2-2B**, seven
datasets, 150-token greedy generations via TransformerLens.

**How they measure effect and quality.** Effect is SPI (Steering Performance Impact: the fraction
of previously unaligned answers corrected, or of aligned answers wrongly changed), with **GPT-4.1**
judging whether open-ended generations exhibit the target behaviour. Quality is **perplexity**.
Collapse is defined *jointly* - "high perplexity and low SPI" - and they diagnose its cause as
larger average steering factors driving activations out of distribution.

**The number we can use.** Their Table 2, open-ended refusal, on Gemma-2-2B:

| method | SPI | perplexity |
|---|---|---|
| CAA, factor 1 | 0.93 | 6.86 |
| IDS | 0.92 | 7.85 |
| CAA, factor 1.5 | 0.69 | 8.15 |
| MERA | 0.01 | **18.10** |

That is an external, published reference band for our exact model: coherent generation sits near
perplexity 7, collapse near 18. `nll_under_base` in `adass.py` already computes this quantity and
is currently used as nothing more than a reported proxy. This gives it a calibration point that
does not come from our own data.

**Two limits, stated so we do not overclaim.** Their refusal open-generation setting is the
**bypass** direction - the worked collapse example is a prompt they are trying to make Gemma-2-2B
answer - not our induce direction, so the band transfers as a coherence reference and not as a
result about inducing refusal. And their coherence measurement is **aggregate perplexity**, so a
single collapsed generation is visible only in a mean; they have no per-generation
refused-versus-broken label, no repetition measure, and no human read of outputs. Their real
protection against collapse is the in-distribution constraint applied *before* generation, not
detection after it.

**Where it leaves AdaSS.** Directly on H2's territory. IDS chooses a per-input, per-position
steering *coefficient*, which is one of the two axes the proposal called empty, and it does so with
a principled criterion rather than a heuristic. The dimension-selection axis survives untouched -
IDS scales a dense vector, it never asks which coordinates to keep. The proposal's related-work
paragraph needs rewriting on the position/strength axis.

---

## 3b. Arditi et al. 2024 - read in full on 21 August, no longer second-hand

Arditi, Obeso, Syed, Duvenaud, Gurnee, Nanda, *Refusal in Language Models Is Mediated by a Single
Direction*, NeurIPS 2024. arXiv:2406.11717. This is our extraction protocol's source and was
previously characterised here from secondary description; that caveat is now lifted.

**They flag our gap explicitly.** On the substring matcher: *"While effective at detecting memorized
refusals, it does not assess whether the completion is coherent or contains harmful content."* And on
their own instruments: *"it is difficult to measure the coherence of a chat model, and we consider
each metric used flawed in various ways. We use multiple varied metrics to give a broad view of
coherence."*

**What their "coherence" means.** §4.3 and Appendix G measure **capability retention**, not
per-generation coherence: MMLU, ARC, GSM8K, TruthfulQA, TinyHellaSwag, WinoGrande via LM Eval Harness,
plus CE loss on The Pile, Alpaca, and on-distribution data. All on the *orthogonalized* (ablation)
model. The per-generation claim is explicitly qualitative: *"qualitatively, we observe that models
maintain their coherence after undergoing weight orthogonalization."*

**The finding that matters most for us, in Appendix I.1.** They compare activation addition against
directional ablation and report that addition *"causes increased loss over harmless data, in
particular compared to directional ablation"*, with the mechanism named: *"on harmless inputs, adding
the negative refusal direction shifts the harmless activations **off distribution**, resulting in
increased perplexity"*, whereas *"directional ablation shifts harmful activations towards harmless
activations, while also not shifting harmless activations too far off distribution."* Table 9, CE loss
on Alpaca relative to baseline:

| model | ablation | activation addition |
|---|---|---|
| Qwen 1.8B | +0.005 | **+0.259** |
| Qwen 72B | +0.028 | **+0.384** |
| Qwen 14B | +0.004 | +0.111 |
| Gemma 2B | +0.011 | +0.089 |

So the primary refusal paper measured that plain dense activation addition damages the model on
off-target inputs, by 3x to 52x more than ablation, and **chose ablation because of it**. Our week-3.5
result is a quantification of a mechanism this paper named, and the field's response to it was to
change intervention type rather than to make addition surgical.

## 3c. Refusal Beyond a Single Direction (2606.13720, June 2026) - the closest work to our headline

*A Preliminary Comparison of Diff-in-Means and INLP.* Five safety-tuned chat models: Gemma 2B-IT,
Qwen 1.8B-Chat, Yi 6B-Chat, Llama-2 7B-Chat, Llama-3 8B-Instruct. Four interventions: DiM directional
ablation, DiM activation addition, INLP nullspace projection, INLP counterfactual flipping.

**This paper independently reached our headline finding, and it is published.** From §5.3:

> *"The most striking pattern is the prevalence of looping completions under ActAdd, consistent with
> the perplexity spikes in Table 1. On harmless prompts many of these **degenerate outputs happen to
> contain a refusal phrase, which means the high ΔRefusal_harmless under ActAdd is partly a
> measurement artefact** of repetitive generation."*

They also find ActAdd *"behaves as a generic refusal injector rather than as a targeted
refusal-induction on the harmful-harmless axis"*, and that baseline harmless-prompt refusals are
*"rare and incoherent"*.

**Their instrument is close to what week 3.5 built.** An LLM judge (Qwen2.5-14B-Instruct, greedy)
emitting `is_looping_or_repetitive`, `request_satisfied ∈ {yes, partial, no, not_applicable}`,
`initial_refusal_then_compliance`, `is_asking_for_more_context`, plus refusal-reason tags and median
per-example perplexity. Their ActAdd is `h ← h ± w_DiM` - coefficient 1.0, unnormalised, every token
position - which is our `dense/all m=1` exactly.

**What this costs us, and what survives.** We are no longer first to observe that refusal injection on
harmless prompts is inflated by degeneration. What is not duplicated:

- They say *partly* a measurement artefact, qualitatively, and still adopt substring matching as their
  headline metric for harmful prompts. We measured it: 5.6% precision, and the *sign* claim - the
  degenerate loop is the **argmax** of every instrument, with measured effect nearly doubling from
  multiplier 1 to 2 while genuine refusals fall to zero.
- They use one LLM judge with a bias audit. We showed a judge purpose-built for this distinction still
  fails it, 0 of 7 - the blind spot survives one level deeper than a looping flag.
- They have no human labels. We have 160 hand-confirmed items with a per-condition breakdown, giving
  0 clean refusals in 29 sampled at the operating point.
- Their scope is intervention *type* (DiM vs INLP). Per-input sparse masking is untouched.

**And it strengthens the refusal-suppression direction.** They report a looping *spike* on a plot
whose y-axis is expanded over 0-20%, at the same coefficient where we measure 100% broken. Same
intervention, different outcome - so a coherent-suppression regime is populated for somebody. See
`PLAN_REFUSAL_SUPPRESSION.md` for the three candidate explanations and why resolving it comes first.

## 4. AxBench, for contrast: the design worth copying

Wu et al., arXiv:2501.17148v2. <https://arxiv.org/abs/2501.17148>

Already cited in the project for why difference-in-means rather than SAEs. It also happens to
contain the evaluation design our week-4 metric should adopt.

Steering outputs get **three separate LLM-judge scores, each 0 to 2**: concept (is the target
concept present), instruct (does it follow the instruction), fluency (is the text fluent). The
overall score is their **harmonic mean**, which "heavily penalises poor performance on any of these
three subscores" - a zero anywhere is a zero overall. That is a coherence *gate* expressed as an
aggregation rule, and it means no amount of concept score can pay for broken text.

Their stated reason for having a fluency axis at all is worth quoting in full, because it is our
failure mode described from the other side:

> "The fluency score is needed as the model can cheat by generating fragmented tokens that relate
> to the concept and the instruction, while being incoherent to humans."

One more relevant observation from their appendix: existing representation-steering work "often
applies repetition or frequency penalties," which AxBench declines to use, on the grounds that it
does not resemble normal use. So part of the field *suppresses* degeneration at decode time rather
than measuring it - which would hide precisely the effect we found. Our `generate()` is greedy
with no repetition penalty, so we are in AxBench's setting and not in the penalised one. Worth
saying explicitly in the write-up, because "your loops are a decoding artifact" is an obvious
reviewer question.

---

## 5. What this does to our claim

**Retract:** any version of "no prior work checks whether steered output is coherent." AxBench has
a dedicated fluency axis with a gating aggregation; IDS is a whole method built around bounding
the intervention to preserve coherence. That framing would not survive a reviewer with a search
engine.

**Keep, sharpened into three separable claims:**

1. **The papers we contradict do not gate on coherence** - 2604.08524 and RepBench report no
   fluency, perplexity, or repetition measure at all, and both score induced refusal on harmless
   prompts by substring matching. Verified by full-text search. **But this can no longer be said of
   the field**: Arditi et al. flag the gap explicitly and 2606.13720 closes it. Revised 21 August;
   the earlier, broader version of this claim would not have survived review.
2. **Where coherence is measured, it is measured in aggregate.** Mean perplexity (IDS) or a mean
   judge subscore (AxBench) can look healthy while individual generations are loops, and neither
   produces the per-generation label that separates "refused" from "broke". No paper we read carries a
   **positive control** - a known-degenerate output the effect metric must not score as an effect.
3. **Our result is a sign claim, and it now has partial priority against it.** 2606.13720 (June
   2026) reports that ActAdd's harmless-prompt refusal injection is *"partly a measurement artefact of
   repetitive generation"* - the same phenomenon, qualitatively, on five models. What remains ours:
   the **argmax** claim (measured effect nearly doubles from multiplier 1 to 2 while genuine refusals
   fall to zero, so the metric does not merely lose signal, it changes sign); that this survives
   replacing the matcher with a graded metric and then with a purpose-built behavioural judge, which
   fails 0 of 7 on the same distinction; and 160 hand-confirmed labels where they have none. State the
   priority explicitly rather than hoping a reviewer misses it.

**A reframing that helps.** The 5.6% precision number stops being an embarrassing property of our
own matcher and becomes a measured validity property of **the evaluator two current papers use for
this task, on one of the models they use it on**. That is a contribution to their measurement
validity rather than a confession about ours.

---

## 6. What to adopt in week 4

In priority order, and all of it borrowed rather than invented:

1. **AxBench's structure.** Separate coherence subscore, harmonic-mean aggregation so a broken
   generation cannot win. Citable, widely adopted, and it gives our gate a precedent instead of an
   argument.
2. **Perplexity as one coherence signal, calibrated against IDS.** `nll_under_base` already computes
   it. IDS's Gemma-2-2B band (roughly 7 coherent, 18 collapsed) is an external anchor, which is
   exactly what our previous validation attempts lacked.
3. **IDS's in-distribution constraint for the operating-point problem** (week-4 item 3). Rather than
   grid-searching multipliers and checking coherence after the fact, bound the intervention so steered
   activations stay within the 95th-percentile Mahalanobis distance of the target distribution. Needs
   only a PCA and a covariance over activations we already collect, and it answers "define the validity
   domain by coherence" with a criterion rather than a threshold we picked.
4. **Keep the mechanical repetition measures as ours.** Longest verbatim repeated span, distinct-n,
   type-token ratio. This is the piece none of the four papers has, it is per-generation rather than
   aggregate, and it is objectively checkable rather than a model's judgement - which matters because
   the thing we are trying to avoid is one classifier certifying another.
5. **Arditi-style capability retention as a secondary check only** (held-out perplexity, MMLU). It
   shows the model as a whole is intact; it cannot see one broken generation, which is the whole
   problem.

**Positioning consequence.** With IDS occupying adaptive per-position strength and 2604.08524
occupying static sparsification, the residual novelty on the method axis is per-input *dimension*
selection alone. That is thinner than the proposal assumed. It reinforces the 14 August framing
decision: lead with the measurement result, keep AdaSS as the case study that exposed it, and cite
IDS as convergent evidence that coherence has to bound steering strength rather than be checked
afterwards.

---

## 7. Verification status

Read in full from arXiv HTML on 20 August 2026: 2604.08524 (v1), 2607.28008 (v2), 2510.13285 (v1),
2501.17148 (v2). Numbers quoted above are from those texts, with table and section locations named
in each entry.

**Absence claims** - "no fluency metric", "no repetition measure" - were checked by keyword search
over each full text for: fluency, perplexity, coherence, repetition, degenerate/degeneration,
gibberish, qualitative, manual, human. That is stronger than a reading impression and still not
proof: a quality check described in words none of those terms cover would be missed. Appendices
were searched but not read line by line.

Read in full on 21 August, second round: 2406.11717 (Arditi et al.) and 2606.13720. Both are now
quoted from source and the earlier second-hand caveat on Arditi is lifted. Note one correction it
forced: Arditi's capability-retention suite is MMLU, ARC, GSM8K, TruthfulQA, TinyHellaSwag and
WinoGrande plus CE loss on The Pile / Alpaca / on-distribution data - the earlier second-hand summary
here said "held-out perplexity plus MMLU/ARC", which understated it.

**Still not read, characterised second-hand and not to be cited from this note:** Rimsky et al.
(CAA), Tan et al., CAST, SADI, CLAS, MERA, Braun et al.

**One correction to an earlier statement in this project's planning discussion:** IDS was described
as measuring "perplexity plus token-level repetition analysis." It does not. Perplexity is its only
text-quality measure; the full text contains no repetition metric. The claim came from an automated
summary of the paper rather than the paper, which is the same class of error this project keeps
finding, one level out - it does not change any recommendation above, but the repetition measures in
item 4 of §6 are ours to build, not IDS's to borrow.
