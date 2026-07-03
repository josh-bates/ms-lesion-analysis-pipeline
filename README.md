# MS Lesion Analysis Pipeline

An end-to-end, **honest prototype** pipeline for multiple sclerosis (MS) lesion
analysis on brain MRI, built as a learning / demonstration project for a
computational-neuroimaging research role. It runs the whole journey a lab actually
does — **quality control → FSL preprocessing → lesion metrics → a deep-learning
lesion segmenter → models relating imaging to clinical disability → a Gradio GUI** —
and every module is heavily commented to *teach* each step and each design choice.

> **Scope & honesty.** This is a prototype on a **tiny** public cohort (30 MS
> patients). The deep-learning Dice and the EDSS models are *illustrative of method*,
> not clinical-grade. Every stage reports its numbers plainly, with baselines and
> caveats. Small sample size is stated wherever it matters.

---

## Data

[open_ms_data](https://github.com/muschellij2/open_ms_data), **cross-sectional**
cohort: 30 MS patients, each with `FLAIR`, `T1W`, `T2W` and an expert `consensus_gt`
lesion mask, provided in four spatial variants (`raw`, `coregistered`,
`coregistered_resampled`, `MNI`). We default to **`coregistered`** so every modality
and the lesion mask already sit on the FLAIR grid (masks overlay the images voxel-
for-voxel). The repo also ships a demographics table (age, sex, MS type, EDSS), which
Stage 1 extracts to `outputs/demographics.csv`.

The dataset is **not committed** (see `.gitignore`); clone it into `./open_ms_data`:
```bash
git clone https://github.com/muschellij2/open_ms_data
```

---

## Quick start

```bash
./setup.sh                          # venv + deps + FSL check (see notes below)
source .venv/bin/activate
python -m tests.smoke_test          # fast synthetic end-to-end check (seconds)
python run_pipeline.py --help       # run the real pipeline, stage by stage
python app.py                       # launch the GUI
```

**Environment notes (learned the hard way on this WSL box):**
- **FSL** is often configured only in *login* shells (`~/.profile`), so a Python
  process from an IDE/pytest/Gradio may not see `bet`/`flirt`/`fast`. `src/fsl_utils.py`
  auto-detects `FSLDIR` and repairs `PATH` for the running process.
- If `python -m venv` fails (`ensurepip` missing, no sudo), `setup.sh` falls back to
  the `virtualenv` package automatically.
- **PyTorch** CPU wheels live on a separate index; `setup.sh` installs torch from
  there first, then the rest from PyPI.

---

## The stages, in plain language

### Stage 1 — Quality control & curation (`src/qc.py`)
Open every subject with `nibabel`; confirm all modalities + the mask exist; record
shape, voxel size, orientation, and intensity range; flag inconsistencies; and parse
the demographics/EDSS table. **Garbage in, garbage out** — one transposed image or a
mask that doesn't line up with its FLAIR would silently poison every later number.
→ `outputs/qc_report.csv`, `outputs/demographics.csv`.
*Real finding it caught:* the cohort has 2 different voxel grids, and the masks carry
a benign **sub-voxel (~0.4 mm) header origin offset** vs FLAIR — distinguished from a
true grid mismatch so we don't false-alarm on all 30 subjects.

### Stage 2 — Preprocessing with FSL (`src/preprocess.py`)
Per subject, from raw FLAIR + T1: **`bet`** (skull-strip) → **`flirt`** (rigid T1→FLAIR)
→ **`fast`** (segment CSF/GM/WM). We downsample to **2 mm isotropic** first so the whole
cohort processes in minutes instead of hours (volumes in mm³ are ~resolution-
independent). **QC:** Dice between our `bet` mask and the dataset's provided brain mask
(patient01 ≈ **0.96**). → `derivatives/<subject>/`, `outputs/preprocess_qc.csv`.

### Stage 3 — Imaging metrics (`src/metrics.py`)
From the expert mask (+ Stage 2's WM map): lesion **volume** (mm³), lesion **count**
(26-connectivity components), **lesion load** (lesion ÷ white-matter volume), and
**mean FLAIR intensity** in lesions (raw and z-scored). → `outputs/imaging_metrics.csv`.
*Sanity check that validates the whole chain:* lesions read **+1.1 to +1.6 SD brighter
than brain on FLAIR** — the hyperintensity that makes FLAIR the MS modality.

### Stage 4 — U-Net lesion segmentation (`src/segment.py`)
A small **2D U-Net** (MONAI/PyTorch) on axial FLAIR slices, learning from the consensus
masks. **Split by SUBJECT** (not slice) so no patient appears in both train and test —
the key to an honest Dice. z-score normalised; Dice reported on **held-out subjects**;
model saved to `models/`. CPU-friendly, GPU-capable.

### Stage 5 — Imaging → disability (`src/predict_disability.py`)
Two simple, cross-validated scikit-learn models: **(a)** classify low vs high lesion
burden, **(b)** estimate **EDSS** from imaging features. Subjects with missing EDSS are
dropped (27 remain). Every model is compared to a trivial baseline (majority class /
mean predictor). We *expect* EDSS regression to be weak (see the paradox below) and
report that honestly rather than cherry-pick.

### Stage 6 — Gradio GUI (`app.py`)
Upload one FLAIR `.nii.gz` → show the middle axial slice → skull-strip → overlay the
U-Net predicted lesions → display metrics + predicted burden class + estimated EDSS.
Works on a **single scan without the full dataset**, loading only the saved models.

---

## Why these neuroimaging choices? (the "teach me" bit)

**Why FLAIR for MS lesions?** FLAIR (FLuid-Attenuated Inversion Recovery) suppresses
the bright signal of cerebrospinal fluid while keeping lesions bright. MS plaques are
**hyperintense** on FLAIR and sit next to the ventricles, where on a plain T2 they'd be
lost in bright CSF. So FLAIR gives the best lesion-to-background contrast — which is why
it's our segmentation input and why our metrics confirmed lesions are ~1–2 SD brighter
than surrounding brain.

**Why skull-strip (brain extraction)?** The skull, scalp, eyes and neck are bright and
irrelevant. Removing them means tissue segmentation, intensity normalisation and the
U-Net all see *brain only* — non-brain tissue would corrupt FAST's intensity model and
distort any whole-image normalisation.

**Why rigid (6-DOF) registration for T1→FLAIR?** It's the same head in one scanning
session, just repositioned — only rotations + translations should change. A 12-DOF
affine would also scale/shear the brain, hiding real anatomy. Same subject, same day →
rigid.

**Why z-score normalisation (over brain voxels)?** MRI intensity is in *arbitrary
units* — identical tissue can read 200 on one scanner and 800 on another. Z-scoring to
mean 0 / SD 1 lets the model learn lesion **shape and contrast**, not a scanner's
brightness. We compute the statistics over brain voxels so the big black air background
doesn't dominate. (Alternatives: min-max is outlier-sensitive; histogram matching is
heavier — z-score is the transparent standard.)

**What does Dice measure?** `Dice = 2·|A ∩ B| / (|A| + |B|)`, from 0 (no overlap) to 1
(identical). It's the standard overlap score for segmentation, robust to the fact that
masks are mostly background. We use it twice: brain-mask agreement (Stage 2) and lesion
segmentation quality (Stage 4).

**What is EDSS?** The **Expanded Disability Status Scale**, the standard clinical score
of MS disability: 0 (normal) to 10, in half-point steps, driven by a neurological exam.
It's **ordinal**, and crucially it reflects spinal-cord, brainstem, visual and cognitive
function — not just brain-lesion load.

**The clinico-radiological paradox.** Brain-lesion load on conventional MRI correlates
only *modestly* with EDSS, because EDSS is heavily influenced by spinal-cord and
cognitive factors that brain FLAIR lesions don't capture. So a **weak EDSS regression is
the honest, expected result**, not a bug — and reporting it truthfully (with baselines
and CV) is the point of Stage 5.

---

## How each stage maps to the job requirements

| Job requirement | Where it's demonstrated |
|---|---|
| **Automated MRI preprocessing pipelines** | `src/preprocess.py` — scripted `bet`→`flirt`→`fast`, idempotent, batch over the cohort, every command logged |
| **Lesion segmentation on structural MRI** | `src/segment.py` (2D U-Net on FLAIR) and `src/metrics.py` (quantifying the expert masks) |
| **Industry-standard neuroimaging toolkit (FSL)** | `src/fsl_utils.py` wraps FSL `bet` / `flirt` / `fast` / `fslmaths` / `fslstats`, with auto-detection and full command logging |
| **Large-dataset curation & quality control** | `src/qc.py` — inventory, geometry/orientation checks, mask–image correspondence, demographics parsing → `qc_report.csv` |
| **Python pipeline automation + Git version control** | `run_pipeline.py` orchestrator, modular `src/`, `tests/smoke_test.py`, this repo under Git |
| **Deep learning (PyTorch/MONAI) for medical images** | `src/segment.py` — MONAI U-Net, DiceLoss, subject-level split, held-out Dice, saved model |
| **Building & validating models relating imaging to clinical disability** | `src/predict_disability.py` — burden classifier + EDSS regressor, cross-validated, baseline-compared, caveated |

---

## Project layout

```
ms_project/
├── run_pipeline.py        # orchestrates all stages on the real dataset
├── app.py                 # Stage 6: Gradio GUI (single-scan demo)
├── requirements.txt
├── setup.sh
├── src/
│   ├── config.py          # paths, subject discovery, FSL-aware config
│   ├── fsl_utils.py        # logged subprocess wrappers for bet/flirt/fast/...
│   ├── image_utils.py      # NIfTI I/O, z-score, resampling, synthetic phantom
│   ├── qc.py               # Stage 1
│   ├── preprocess.py       # Stage 2
│   ├── metrics.py          # Stage 3
│   ├── segment.py          # Stage 4 (MONAI/PyTorch U-Net)
│   └── predict_disability.py  # Stage 5
├── tests/
│   └── smoke_test.py       # synthetic tiny-NIfTI end-to-end test (every stage)
├── derivatives/  outputs/  models/  logs/     # all generated (git-ignored)
```

---

## The smoke test

The real dataset is multi-GB and FSL/U-Net stages take minutes; that's a poor feedback
loop. `tests/smoke_test.py` generates tiny **synthetic brain phantoms** — same file
names, same physics (bright FLAIR lesions in white matter) — and runs *every stage and
the GUI* on them in seconds. It's how we validate the plumbing before (and independently
of) the slow real runs. Each stage has a guarded check that reports PASS/SKIP, so the
suite stays green as the pipeline is built up.

```bash
python -m tests.smoke_test
```

---

## Limitations (stated plainly)

- **n = 30** (27 with EDSS). All learned metrics are illustrative, not clinical.
- **2 mm downsampling** for FSL trades small-lesion precision for cohort-scale speed.
- **2D** U-Net ignores through-plane context (a 3D U-Net would likely do better on a GPU).
- Tissue segmentation runs `fast` on a FLAIR-derived brain mask at 2 mm — adequate for a
  WM-volume denominator, not a validated tissue-volumetry pipeline.
- EDSS is only weakly predictable from brain-lesion imaging (clinico-radiological paradox).
