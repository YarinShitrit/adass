# Project Proposal

## Project Title

**Adaptive Sparse Steering: Input-Conditional Dimension and Token-Position Masking for Activation Steering in LLMs**

> **Revised 20 August 2026 - related-work framing only.** The original is preserved verbatim at
> `archive/adaptive_sparse_steering_plan_2026-08-08_original.md`. Three things changed here: the
> prior-work paragraph in the problem statement, the novelty claim under innovation highlights, and
> the reference list, all following the full-text review of the closest work in `RELATED_WORK.md`.
>
> **What deliberately did not change:** the hypotheses, implementation steps, and methodology below
> are left as originally written. This document remains a record of historical intent, not a current
> plan. For where the project actually stands, read `WEEK3_SUMMARY.md`: as of 14 August 2026 the
> project leads with a measurement result, with the hypotheses below as the case study that exposed
> it.
>
> One observation worth recording rather than editing away: the Metrics section below already
> specified **"LLM-judge fluency/relevance"** as a quality axis. It was never implemented. The
> coherence blind spot that overturned weeks 1-3 was therefore a gap between this proposal and the
> code, not a gap in the proposal.

## Problem Statement

Activation steering — adding a "steering vector" to a model's hidden activations at inference time — is a lightweight alternative to fine-tuning for controlling LLM behaviors such as refusal and sycophancy (Rimsky et al., 2024; Arditi et al., 2024). Standard steering applies a **fixed, dense vector uniformly at every token position after the prompt**, causing two documented problems: (1) **over-steering** that degrades output quality and increases perplexity (Braun et al., 2025), and (2) **input-inconsistent effectiveness** — the same vector over-steers some prompts and under-steers others (Tan et al., NeurIPS 2024).

Prior work addresses each symptom separately, and the two axes this proposal targets are **not equally open**.

*On strength and position.* Adaptive methods modulate how hard to steer per input (CAST, SADI - ICLR 2025; CLAS, 2026). In-Distribution Steering (IDS - Vogels et al., 2025) goes furthest and is the closest prior work to this proposal's second axis: it sets the coefficient per input **and per token position**, choosing the largest value that keeps the steered activation inside the 95th-percentile Mahalanobis distance of the target behaviour's activation distribution, and it is motivated by precisely the failure invoked above - over-steering that collapses text into implausible output. IDS scales continuously rather than gating positions on or off, so "which positions" remains formally open, but the *benefit* claimed for position sparsity below is now pursued by a stronger and better-motivated criterion.

*On dimensions.* Static sparsification reports that refusal vectors survive zeroing 90-99% of dimensions (arXiv:2604.08524), using a single mask shared across all inputs. Two caveats matter for how much that constrains us: the masks are static by construction, and that paper's induce-refusal evaluation is substring matching on harmless prompts - materially the same instrument this project later measured at 5.6% precision on the same model (`RELATED_WORK.md` §1), so the sparsification claim is better read as holding on the bypass direction than on both.

What no prior work does is **select which dimensions to steer on a per-input basis.** We ask:

> *Can input-conditional selection of both a sparse subset of steering-vector dimensions and a targeted subset of token positions match the behavioral effect of full dense steering while reducing output-quality degradation?*

## Detailed Description of Idea and Innovation Highlights

We propose **Adaptive Sparse Steering (AdaSS)**, which decomposes steering into two per-input decisions:

1. **Dimension masking:** Given a base steering vector *v* (difference-in-means / CAA), for each input we score dimensions by their per-input contribution (e.g., |v_i ⊙ Δh_i|, or fast gradient attribution) and retain only the **top-k** — a *different* sparse mask per prompt, unlike the single static mask of prior sparsification work.
2. **Token-position gating:** We steer only positions where intervention is causally effective, scored by a cheap per-position signal (linear probe on the residual stream, or per-position ΔNLL), instead of every generated token.

The final intervention is *sparse in dimensions and sparse in positions*, with both patterns conditioned on the input.

**Innovation highlights:**

- **Where this sits, after a full-text review of the closest work** (`RELATED_WORK.md`). *The dimension axis is open:* SADI adapts *scaling* over a **static** dimension mask; arXiv:2604.08524 sparsifies with a single **shared** mask; MAT-Steer gates per token over a **static** direction. None selects dimensions per input. *The position axis is not open in the way this proposal assumed:* "Steer Like the LLM" learns per-token *coefficients* over a dense direction, and IDS does the same per input under an explicit coherence bound, which is the same target as H2 below. CAST gates only *whether* to steer. The residual novelty is therefore **per-input dimension selection**, with the joint method (H3) a combination of one new axis and one contested one rather than two new axes.
- **What the project has actually contributed, which this framing predates.** Weeks 1-3 found that refusal metrics score token *shape* rather than behaviour: a degenerate repetition loop is the most refusal-shaped text that exists, and it maximises every instrument tried, including a behavioural judge built to escape the problem. Nothing in the reviewed literature measures this. The two refusal-specific papers (arXiv:2604.08524, RepBench) carry no output-quality axis at all, and AxBench's separate fluency subscore exists because a *concept* score can be gamed by fragmented text, not because a refusal metric's argmax is a broken model. See `WEEK3_SUMMARY.md`; the hypotheses below are the case study that exposed it.
- **Informative either way.** We also answer: *does the optimal sparse subset actually differ across inputs?* A positive result yields a better steering method; a negative result (per-input ≈ static) is a meaningful finding about the geometry of behavioral directions, consistent with Arditi et al.'s single-direction hypothesis.

**Hypotheses:**

- **H1:** Per-input sparse masks match dense steering at ≥90% sparsity and beat an equal-sparsity static mask on held-out prompts.
- **H2:** Steering <30% of positions retains most behavioral effect with lower perplexity/KL degradation than all-position steering.
- **H3:** The joint method achieves the best effect-vs-quality Pareto frontier among all ablations.

## Implementation Steps

1. Set up Gemma-2-2B-it with TransformerLens / `steering-vectors` on Colab; extract difference-in-means vectors for **refusal** (AdvBench + harmless pairs, Arditi et al. protocol) and **sycophancy** (CAA A/B dataset).
2. Reproduce baselines: full dense CAA steering and static top-k sparsification; build the evaluation harness (behavioral shift, refusal scoring, perplexity, KL, LLM-judge quality).
3. Implement per-input dimension masking and token-position gating; sweep sparsity level and position budget; sanity-check each mechanism independently.
4. Run the full ablation grid {dense, static-sparse, adaptive-sparse} × {all positions, targeted positions} for both behaviors; plot effect-vs-quality Pareto curves.
5. Generalization check on Llama-3.2-1B-Instruct; analyze mask overlap across inputs and which positions get selected and why.
6. Write the report and presentation.

## Methodology & Datasets, and Models

- **Models:** Gemma-2-2B-it (primary; fits a free Colab T4), Llama-3.2-1B-Instruct (generalization).
- **Base method:** Difference-in-means / CAA vectors at a mid-layer residual stream — the strongest simple baseline per AxBench (ICML 2025). SAEs are deliberately not the core method (AxBench finds them uncompetitive for steering); Gemma Scope may serve as an optional interpretive lens.
- **Datasets:** AdvBench + Alpaca-style harmless instructions (refusal); CAA contrastive A/B datasets (sycophancy); held-out splits for all claims.
- **Baselines:** no steering; full dense vector; static top-k sparse mask; position-uniform steering.
- **Metrics:** A/B probability shift and refusal rate (behavioral effect); perplexity, KL vs. unsteered model, LLM-judge fluency/relevance (quality); % dimensions zeroed and % positions steered (efficiency). Primary result: Pareto curves of effect vs. quality degradation with confidence intervals.
- **Risk mitigation:** if per-input masks don't beat static, we report a rigorous negative result with mask-overlap analysis; if attribution is too slow on a T4, we fall back to magnitude-based top-k; if the joint method underperforms, the token-position study stands as a self-contained contribution.

## Key References

- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024.
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024.
- Tan et al., *Analysing the Generalisation and Reliability of Steering Vectors*, NeurIPS 2024.
- Lee et al., *Conditional Activation Steering (CAST)*, ICLR 2025.
- Wang et al., *Semantics-Adaptive Dynamic Intervention (SADI)*, ICLR 2025.
- Vogels et al., *In-Distribution Steering: Balancing Control and Coherence in Language Model Generation*, arXiv:2510.13285 - per-input, per-position steering strength bounded to stay in-distribution; the closest prior work to H2, and the source of an external coherence reference band on Gemma-2-2B.
- *What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal*, arXiv:2604.08524 - the static 90-99% sparsification result; its induce-refusal evaluation is substring matching, see `RELATED_WORK.md` §1.
- *RepBench: A Benchmark for Representation Engineering*, arXiv:2607.28008 - current standardised measurement for this field; scores induced refusal on harmless prompts by substring matching, with no output-quality axis.
- Wu et al., *AxBench*, ICML 2025 - why difference-in-means rather than SAEs, and the evaluation design to copy: separate concept, instruction, and **fluency** subscores combined by harmonic mean, so broken text cannot win.
- Hsu et al., *Contextual Linear Activation Steering (CLAS)*, arXiv:2604.24693.
