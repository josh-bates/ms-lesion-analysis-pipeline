# MS Lesion Analysis Pipeline

A small, honest **prototype** neuroimaging pipeline for multiple sclerosis (MS)
lesion analysis, built as a learning / demonstration project. It runs end to end:
quality control → FSL preprocessing → lesion metrics → a deep-learning lesion
segmenter → simple models relating imaging to clinical disability → a Gradio GUI.

> **Status:** built in stages. This README grows as each stage lands. The pipeline
> is designed so the fast **synthetic smoke test** validates every stage (and the
> GUI) before the multi-GB dataset download and slow FSL runs finish.

---

## Data

[open_ms_data](https://github.com/muschellij2/open_ms_data), **cross-sectional**
cohort: 30 MS patients, each with `FLAIR`, `T1W`, `T2W` and an expert
`consensus_gt` lesion mask, in four spatial variants (`raw`, `coregistered`,
`coregistered_resampled`, `MNI`). We default to **coregistered** so every modality
and the lesion mask already sit on the FLAIR grid. The README of that repo also
ships a demographics table (age, sex, MS type, EDSS) which we extract to
`outputs/demographics.csv`.

The dataset is **not** committed here (see `.gitignore`); clone it into
`./open_ms_data`.

---

## Quick start

```bash
./setup.sh                          # make .venv, install deps, check for FSL
source .venv/bin/activate
python -m tests.smoke_test          # fast synthetic end-to-end check (seconds)
python run_pipeline.py --help       # run the real pipeline, stage by stage
```

---

## Project layout

```
ms_project/
├── run_pipeline.py        # orchestrates all stages on the real dataset
├── app.py                 # Gradio GUI (single-scan demo)          [Stage 6]
├── requirements.txt
├── setup.sh
├── src/
│   ├── config.py          # all paths, subject discovery, FSL-aware config
│   ├── fsl_utils.py       # logged subprocess wrappers for bet/flirt/fast/...
│   ├── image_utils.py     # NIfTI I/O, z-score normalisation, synthetic phantom
│   ├── qc.py              # Stage 1: inventory + QC + demographics
│   ├── preprocess.py      # Stage 2: bet / flirt / fast
│   ├── metrics.py         # Stage 3: lesion volume/count/load/intensity
│   ├── segment.py         # Stage 4: 2D U-Net (MONAI/PyTorch)
│   └── predict_disability.py  # Stage 5: burden classifier + EDSS regressor
├── tests/
│   └── smoke_test.py      # synthetic tiny-NIfTI end-to-end test
├── derivatives/           # per-subject processed images   (generated)
├── outputs/               # CSV reports + figures           (generated)
├── models/                # trained model weights           (generated)
└── logs/                  # run + FSL command logs          (generated)
```

---

## The stages, in plain language

*(Detailed neuroimaging rationale and the job-requirements mapping table are added
to this section as each stage is implemented.)*

- **Stage 1 – QC.** Open every scan, confirm all modalities + the mask exist, and
  record shape / voxel size / orientation / intensity range. Flag anything
  inconsistent. Parse demographics + EDSS.
- **Stage 2 – Preprocess.** Skull-strip (FSL `bet`), rigidly register T1→FLAIR
  (`flirt`), segment tissue into grey/white/CSF (`fast`).
- **Stage 3 – Metrics.** From the expert mask: lesion volume (mm³), lesion count,
  lesion load (fraction of white matter), mean FLAIR intensity in lesions.
- **Stage 4 – Segmentation.** Train a 2D U-Net on axial FLAIR slices to predict
  lesions; evaluate with Dice on held-out **subjects**.
- **Stage 5 – Disability.** Relate imaging metrics to clinical disability: classify
  low vs high lesion burden and estimate EDSS.
- **Stage 6 – GUI.** Upload one FLAIR scan, skull-strip it, overlay the predicted
  lesion mask, and show metrics + predicted burden class + estimated EDSS.

---

## A note on honesty

This is a prototype on a **tiny** dataset (30 subjects). Deep-learning Dice on 30
subjects and EDSS regression on ~26 (after dropping missing EDSS) are illustrative,
not clinical. Every stage reports its numbers plainly and states this limitation.
