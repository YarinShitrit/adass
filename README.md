# AdaSS — Adaptive Sparse Steering

Research code investigating whether **activation steering** can be made surgical. We extract a
"refusal" direction from Gemma-2-2b-it by difference-in-means, add it back into the residual stream
at inference time to make the model refuse harmless requests, and ask whether the intervention can
be narrowed — to fewer *dimensions* of the vector, or fewer *token positions* — without losing the
effect.

Final project, NLP course, Reichman University.

---

## Where the project stands

> **Start with [`docs/HANDOVER.md`](docs/HANDOVER.md).** Three pages, current state, and the traps.

The intended contribution was a better effect-versus-damage tradeoff. What the work produced instead
is a **measurement result**, and as of 23 August it has a sharper form than "our metric was bad":

> **The standard evaluation instrument for induced refusal is saturated across the entire useful
> operating range, so it silently corrupts the layer- and strength-selection step that every
> steering paper runs before it measures anything.**

The substring matcher sits at exactly **100% for layers 10–18** while the true clean-refusal rate
underneath it runs 93.8% → 97.9% → 47.9% → 12.5%. Week 1 chose the operating point by that metric's
argmax, and landed on layer 16 — the one layer in the range where steering destroys the model rather
than steering it. The metric failure and the operating-point failure are one bug, one level apart.

![The selection metric is flat where the behaviour changes](figures/fig_matcher_saturation.png)

**Still open:** `‖v‖` grows 6.2× with depth and the multiplier scales the raw vector, so depth and
strength are confounded in every layer comparison the project has made. `notebooks/05` §6 is the
pre-registered test that separates them. Both outcomes leave the finding above intact.

---

## Quick start

### Local

```bash
git clone <your-repo-url> adass && cd adass
python -m pip install -e ".[notebook]"
huggingface-cli login          # needs the two gated licences accepted, see below
python -c "import adass; print(adass.paths.describe()); print(adass.pick_device(), adass.pick_dtype(adass.pick_device()))"
```

Expect `cuda torch.bfloat16` or `mps torch.bfloat16`. **If it prints `float16`, stop** — Gemma-2
emits broken text in fp16, and that failure looks exactly like the degeneration this project studies.

Then open `notebooks/05_week4_layers.ipynb`. Its bootstrap cell finds the repo root by walking up
from wherever the kernel started, so the working directory does not matter.

### Colab

Open `notebooks/05_week4_layers.ipynb` and run the bootstrap cell. It detects Colab, clones the repo,
installs the package, and prompts for a HuggingFace login. Two things to set once:

1. In the bootstrap cell, set `GITHUB_REPO = "your-username/adass"`.
2. In Colab **Secrets** (the key icon in the sidebar), add `GH_TOKEN` — a fine-grained GitHub PAT
   with read access to the repo. The token is never written into the notebook, so nothing
   credential-shaped can be committed by accident. Without it the cell falls back to mounting Drive.

Then, before running the GPU sections:

```python
%env ADASS_LOAD_MODEL=1
```

**Do not set `HF_HUB_OFFLINE=1` on Colab.** That advice in the docs assumes weights already cached;
on a fresh runtime nothing is, and `from_pretrained` fails outright.

### Prerequisites

Two **gated** HuggingFace repos, and it is easy to accept the first and be confused by the second:

- the model — <https://huggingface.co/google/gemma-2-2b-it>
- the harmful prompts — <https://huggingface.co/datasets/walledai/AdvBench>

(`tatsu-lab/alpaca`, the harmless prompts, is open.) ~6 GB for the model, ~8–10 GB RAM or VRAM at
the default batch sizes. Runs on CUDA, Apple Silicon (MPS), or CPU.

> `transformers` is pinned **`>=4.56,<5` on purpose.** Notebook 03 §1 is a replication gate against
> week-2 numbers, and changing the modelling library's major version underneath a replication test
> confounds "did my refactor break something" with "did the library change something". Validate that
> gate before relaxing the pin.

---

## Layout

```
adass/
├── adass/                  the package — import adass
│   ├── core.py             model loading, the Steer3 hook, splits, metrics, masks, graders
│   └── paths.py            repo-relative path resolution, so notebooks are CWD-independent
├── notebooks/
│   ├── 01_week1_baselines.ipynb     ┐
│   ├── 02_week2_adaptive.ipynb      │ ARCHIVAL — the historical record.
│   ├── 03_week3_validation.ipynb    │ Read, do not re-run.
│   ├── 04_week3_5_taxonomy.ipynb    ┘
│   └── 05_week4_layers.ipynb        LIVE — runs locally or on Colab unchanged
├── data/
│   ├── gold/               the 160 hand labels + the blind sheet. IRREPLACEABLE, and only
│   │                       meaningful as a pair — labels are keyed by sid
│   ├── generations/        the 384 generations everything is scored against
│   ├── vectors/            refusal_dirs.pt — one row per layer, INDEXED [layer + 1]
│   └── results/            every run's output, merged and rotated by adass.save_results
├── figures/
├── config/                 the operating point, loaded rather than hard-coded
└── docs/                   HANDOVER, ONBOARDING, WORKLOG, RELATED_WORK, the plans
```

Paths resolve from the repo, not the working directory: `adass.artifact("week3_generations.json")`
finds it wherever it lives, and `adass.save_results(obj, "week4_layers.json")` writes to
`data/results/`. Override the root with the `ADASS_ROOT` environment variable.

---

## Documents, in reading order

| # | File | What it is |
|---|---|---|
| 1 | [`docs/HANDOVER.md`](docs/HANDOVER.md) | Current state in ~3 pages: the saturated-selection finding, the layer reversal, the open confound, which instrument to trust for what, and the traps. **The entry point.** |
| 2 | [`docs/WEEK3_5_SUMMARY.md`](docs/WEEK3_5_SUMMARY.md) | The four-class classifier and the hand labels. **Its conclusion is superseded**; its methods and controls are not. |
| 3 | [`docs/WEEK3_SUMMARY.md`](docs/WEEK3_SUMMARY.md) | How the project got here — the measurement result and what it overturned. |
| 4 | [`docs/ONBOARDING.md`](docs/ONBOARDING.md) | The long guide: the residual stream, position semantics, every metric, the code, the glossary. Written for someone new to interpretability. |
| 5 | [`docs/WORKLOG.md`](docs/WORKLOG.md) | Every correction — what we believed, what was wrong, how it was caught. Roughly half are corrections to our own analysis. |
| 6 | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Six papers read in full, including the **priority claim against arXiv:2606.13720**. |
| 7 | [`docs/PLAN_REFUSAL_SUPPRESSION.md`](docs/PLAN_REFUSAL_SUPPRESSION.md) | The main line from here. `PLAN_SYCOPHANCY.md` is the alternative venue. |

`NB §6` means section 6 of a notebook; a bare `§6` inside a document means section 6 of that document.

---

## What to pick up first

1. **Run `notebooks/05` §3–§7 on a GPU.** §2 is already run. §3–§5 give the 22 August reversal
   reproducible code for the first time — `steps123_results.json` was produced in a session that was
   never saved, and no cell anywhere writes it. §6 settles the strength/depth confound under a rule
   written above the code. §7 leaves a normalised strength axis behind. About an hour on a T4.
2. **~40 blind hand labels at a coherent layer** (12 is the natural choice). No GPU. All 160 existing
   gold labels come from layer-16 conditions, so both instruments are currently used
   out-of-distribution wherever the coherent regime is measured. Until this runs, no layer-10-to-14
   number is hand-confirmed.
3. **H1 and H3 at a coherent layer**, per `docs/PLAN_REFUSAL_SUPPRESSION.md`. The masking machinery
   is behaviour- and layer-agnostic and needs no changes. **Compare at matched relative strength**,
   not matched multiplier — matched raw multiplier is what made week 2's H1 test unfair.

---

## Conventions worth knowing before you change anything

Each of these exists because something went wrong without it. `docs/WORKLOG.md` has the incidents.

- **`adass/core.py` is the source of the module.** It used to be generated by a `%%writefile` cell in
  notebook 04, and the two silently drifted. That cell is now inert and kept for provenance.
- **`save_results` merges** at the top level, so a partial re-run cannot delete a section it did not
  compute. The one deliberate truncation point is each notebook's §0.1, which rotates to
  `.prev.json`. This is not redundancy to simplify away — a partial re-run once wiped hours of GPU
  output off disk.
- **`make_splits(seed=0)` with the default `train_n=128`.** `train_n=160` shifts `harmless_test` by
  32 items, so generations get paired with the wrong prompts. That invalidated a whole run once.
- **Decision rules are written in a markdown cell above the code**, so a result that contradicts the
  hypothesis is recorded rather than reinterpreted.
- **Every instrument needs a negative *and* a positive control.** "The unsteered model must never
  refuse" is blind to a grader that confuses refusal with degeneration — which is what happened.
- **Read the generations.** Every reversal in this project came from somebody finally printing the
  text.
