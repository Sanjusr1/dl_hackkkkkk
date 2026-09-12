# DACL — Disaster-Adaptive Continual Learning

A benchmark and reference implementation for **continually** learning to recognise
natural disasters from aerial imagery: absorbing each new event without losing
what was learned from previous ones.

## Motivation

When a major disaster occurs, aerial and satellite imagery becomes available
fast. After the flash flood that struck Nepal's Rasuwa District on 26 August
2026, the International Charter "Space and Major Disasters" was activated at the
request of Nepal's Department of Hydrology and Meteorology, with UNITAR as
managing agency, and Copernicus EMS was activated alongside it — roughly 4,689
buildings fell inside the flood-affected zone and 41 bridges were washed away.

A model that has to be retrained from scratch on the full historical archive
every time this happens is operationally useless: retraining is slow, the
archive may not be redistributable, and storage is bounded. A model that is
simply fine-tuned on the new event suffers **catastrophic forgetting** — in our
runs, naive fine-tuning collapses to exactly 1/T accuracy, retaining only the
most recent disaster type.

This repo measures how well different continual-learning strategies close that
gap, under two protocols that a deployed damage-assessment system actually
faces.

## Protocols

| Protocol | Task = | Label space | Question it answers |
|---|---|---|---|
| **Class-incremental (CIL)** | new disaster *types* | grows | Can the model learn a disaster it has never seen without losing the others? |
| **Domain-incremental (DIL)** | new *event* / sensor / geography | fixed | Can the model absorb a new acquisition under a known taxonomy? |

No task identity is given at test time. Every evaluation is over the full label
space seen so far — masking logits to the current task would report
"task-incremental" accuracy, which is much higher and not something a deployed
system can do.

## Methods

All nine share an identical stream, augmentation and optimiser, so the only
thing that differs between rows of a results table is the strategy.

| Name | Family | Notes |
|---|---|---|
| `finetune` | lower bound | its gap to `joint` *is* forgetting |
| `joint` | upper bound | retrains on all data seen so far |
| `frozen` | control | frozen pretrained features, head only — quantifies how much comes from ImageNet init rather than from CL |
| `ewc` | regularisation | diagonal Fisher, online variant by default |
| `lwf` | regularisation | KD from the frozen previous model |
| `er` | rehearsal | reservoir buffer + replay CE |
| `er_ace` | rehearsal | asymmetric CE; blocks the negative gradient new data applies to old classes |
| `derpp` | rehearsal | replays stored *logits* as well as labels |
| `icarl` | rehearsal | herding exemplars + nearest-mean-of-exemplars classification |

`ewc` and `lwf` are rehearsal-free, which matters when imagery cannot be
retained — Charter-supplied products often come with redistribution
restrictions.

## Metrics

From the accuracy matrix `R[i, j]` = accuracy on task *j* after training task *i*:

- **ACC** — final average accuracy
- **AvgInc** — average incremental accuracy (useful *throughout* the stream, not just at the end)
- **BWT** — backward transfer; negative means forgetting
- **Forget** — average max drop from each task's best-ever accuracy
- **LA** — learning accuracy (diagonal mean): separates "forgot the past" from "too rigid to learn the new disaster". Average accuracy alone conflates these.
- **FWT** — forward transfer vs. an untrained model
- **Macro-F1** — disaster classes are badly imbalanced; a model that silently stops predicting a rare class keeps a respectable accuracy

## Install

```bash
pip install -r requirements.txt
```

## Quickstart — no download required

A procedural synthetic corpus (6 disaster signatures over a shared terrain
prior) ships with the repo so the whole pipeline runs on a CPU in minutes:

```bash
PYTHONPATH=src python -m dacl.run --config configs/synthetic_cil.yaml --method finetune
PYTHONPATH=src python -m dacl.run --config configs/synthetic_cil.yaml --method derpp
```

Use it to verify a new method before spending GPU hours. It is a correctness
sandbox, **not** a source of reportable results.

## One-command run

For a complete demo run that tests the code, runs the benchmark, generates
plots, and builds the frontend dashboard:

```bash
python run_project.py --synthetic
```

On Kaggle, attach both datasets first:

- this project dataset
- `banasmitajena/aiderv2-dataset`

Then run one command from the project folder:

```bash
bash run.sh --fast --seeds 0
```

For the final GPU sweep:

```bash
bash run.sh --full --seeds 0 1 2
```

Outputs:

- `runs/benchmark/summary.json` — result table data
- `figures/` — accuracy matrices, retention curves, comparison plots
- `frontend/index.html` — presentation dashboard

## Real data — AIDERv2

16,723 aerial images, 4 classes (earthquake/collapsed building, flood, fire,
normal), official split 13,399 / 1,670 / 1,654.

```bash
pip install kaggle                     # needs ~/.kaggle/kaggle.json
kaggle datasets download -d banasmitajena/aiderv2-dataset
unzip aiderv2-dataset.zip -d raw_aiderv2

python scripts/prepare_aiderv2.py --src raw_aiderv2 --dst data/aiderv2 --resize 256
```

The official split is preserved; re-splitting a corpus that ships its own makes
results incomparable with published numbers on it. Validation is folded into
train (we do no per-task model selection).

Then run the primary experiment:

```bash
python scripts/run_benchmark.py --config configs/aiderv2_cil.yaml \
    --methods finetune frozen ewc lwf er er_ace derpp icarl joint \
    --seeds 0 1 2 3 4
python scripts/plot_results.py --runs runs/benchmark/* --out figures/
```

### Experimental design note

With 4 classes the stream is **3 tasks of sizes [2, 1, 1]**. The first task gets
two classes deliberately: a single-class task makes cross-entropy degenerate.

`shuffle_classes: true` ties the class **order** to the run seed, so sweeping
seeds sweeps orders. With only 3 tasks the class order is the dominant source of
variance — single-seed results here are close to meaningless. Report mean ± std
over at least 5 seeds.

AIDERv2 carries no event or sensor metadata, so a real domain-incremental stream
is not available from it. `configs/aiderv2_dil_shift.yaml` builds one by
*partitioning* (never duplicating) each class across three **simulated**
acquisition conditions — clear nadir, dusk haze, low-res oblique. Report it as
simulated photometric/resolution shift. For real per-event streams, use xBD
(`configs/xbd_damage_dil.yaml`).

## Results on the synthetic sandbox

Mechanism check only — 3 tasks, 2 classes each, smallcnn, single seed:

| method | ACC | BWT | forgetting |
|---|---|---|---|
| finetune | 33.3% | −100% | 100% |
| ewc | 33.3% | −100% | 100% |
| lwf | 33.3% | −100% | 100% |
| er | 86.7% | −14.2% | 14.2% |
| derpp | 98.3% | −2.5% | 2.5% |
| icarl | 100% | 0% | 0% |
| joint *(upper bound)* | 98.3% | −0.8% | 0.8% |

Fine-tuning keeps a perfect diagonal and collapses to exactly 1/T. EWC and LwF
failing to beat it is the **expected** class-incremental result, not a bug: both
penalties are live (LwF's KD term sits around 0.23), but neither addresses
task-recency bias in the growing head. This is a finding worth writing up, not a
problem to debug away.

## Configuration

One YAML plus a seed fully describes a run. Configs compose via `_base_`:

```bash
PYTHONPATH=src python -m dacl.run --config configs/aiderv2_cil.yaml \
    --method derpp --seed 3 \
    --set method.params.buffer_size=1000 train.epochs=8
```

Two defaults worth knowing:

- **Cosine head** (`model.head: cosine`) — with a plain linear head, the newest
  classes get systematically larger logit norms simply because they are the only
  ones receiving gradient. Normalising features and weights removes that degree
  of freedom. For linear heads in CIL, weight alignment is applied at each task
  boundary instead.
- **Replay stores un-augmented views** — buffering an already-augmented crop
  bakes one random augmentation into memory forever and measurably hurts replay.

## Layout

```
configs/           experiment configs (compose via _base_)
scripts/
  prepare_aiderv2.py   Kaggle layout -> expected layout, optional resize/domains
  run_benchmark.py     methods x seeds sweep -> comparison table
  plot_results.py      accuracy matrices, retention curves, task confusion
src/dacl/
  config.py        typed config + YAML + CLI overrides
  data/            synthetic generator, dataset scanning, scenario builder
  models/network.py  backbones + growing linear/cosine heads
  methods/         base, buffer, baselines, regularization, replay
  engine/          trainer, evaluator, metrics
tests/             59 tests, run in <5s
```

## Tests

```bash
python -m pytest -q
```

## Adding a method

Subclass `ContinualMethod`, implement `observe()` (and optionally `begin_task`,
`end_task`, `predict`), register it in `methods/__init__.py`. The
parametrised integration test picks it up automatically.

## References

- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017 (EWC)
- Li & Hoiem, *Learning without Forgetting*, ECCV 2016
- Rebuffi et al., *iCaRL: Incremental Classifier and Representation Learning*, CVPR 2017
- Buzzega et al., *Dark Experience for General Continual Learning*, NeurIPS 2020 (DER++)
- Caccia et al., *New Insights on Reducing Abrupt Representation Change*, ICLR 2022 (ER-ACE)
- Zhao et al., *Maintaining Discrimination and Fairness in Class Incremental Learning*, CVPR 2020 (weight alignment)
- Lopez-Paz & Ranzato, *Gradient Episodic Memory*, NeurIPS 2017 (BWT/FWT)
- Shianios et al., *A Benchmark and Investigation of Deep-Learning-Based Techniques for Detecting Natural Disasters in Aerial Images*, CAIP 2023 (AIDERv2)
