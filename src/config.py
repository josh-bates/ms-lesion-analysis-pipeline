"""
config.py
=========
Central configuration for the whole MS-lesion pipeline.

WHY a single config module?
    Every other module (qc, preprocess, metrics, segment, ...) needs to agree on
    *where the data lives*, *where outputs go*, and *how to find FSL*. Putting that
    in one place means there is exactly one thing to change when a path moves, and
    the smoke test can point the whole pipeline at a throw-away synthetic dataset
    just by overriding a couple of values.

Design choice:
    We expose plain module-level constants (PROJECT_ROOT, DATA_ROOT, ...) *and*
    small helper functions. The alternative was a big config class or a YAML file.
    For a teaching prototype, module-level constants are the most readable: you can
    `from src import config` and immediately see every path. Anything that a caller
    might legitimately want to override (data root, which image "variant" to use)
    is also readable from an environment variable so the smoke test and the GUI can
    redirect the pipeline without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this src/ package. We compute it from
# this file's location so the project keeps working no matter what the current
# working directory is when a script is launched.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# The open_ms_data repository is cloned inside the project. We point at the
# cross-sectional cohort (one scan session per patient), which is what the task
# asks for. DATA_ROOT can be overridden with the MS_DATA_ROOT env var so the
# smoke test can substitute a tiny synthetic dataset.
DATA_ROOT: Path = Path(
    os.environ.get("MS_DATA_ROOT", PROJECT_ROOT / "open_ms_data" / "cross_sectional")
)

# open_ms_data ships every subject in four spatial "variants":
#   raw                    - original scanner space, each modality its own grid
#   coregistered           - T1/T2 rigidly aligned to FLAIR (native FLAIR grid)
#   coregistered_resampled - coregistered + resampled to an isotropic grid
#   MNI                    - warped into standard MNI152 template space
# For analysis we default to "coregistered": the lesion mask (consensus_gt) and
# all modalities already sit on the FLAIR grid, so masks and images line up
# voxel-for-voxel without us doing anything. (Alternative: MNI space, which makes
# subjects comparable to each other but resamples/blurs the data and needs a
# non-linear registration we don't want to depend on for a prototype.)
DATA_VARIANT: str = os.environ.get("MS_DATA_VARIANT", "coregistered")

# Target isotropic voxel size (mm) for the FSL preprocessing stage. The raw scans
# are ~0.47mm in-plane; running bet/flirt/fast at that resolution takes minutes per
# subject. We downsample to 2mm so the whole 30-subject cohort processes in minutes
# (see image_utils.resample_to_iso for the full rationale/trade-off). Set to 0 or
# None to disable downsampling and use native resolution.
PREPROCESS_ISO_MM: float = float(os.environ.get("MS_PREPROCESS_ISO_MM", "2.0"))

# Where everything we generate goes. Kept separate from the input data so the
# raw dataset stays pristine (a basic reproducibility / data-hygiene habit).
DERIVATIVES_DIR: Path = PROJECT_ROOT / "derivatives"   # per-subject processed images
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"           # CSVs, figures, reports
MODELS_DIR: Path = PROJECT_ROOT / "models"             # trained model weights
LOGS_DIR: Path = PROJECT_ROOT / "logs"                 # command + run logs

# Canonical output file names (referenced from several modules, so name them once).
QC_REPORT_CSV: Path = OUTPUTS_DIR / "qc_report.csv"
DEMOGRAPHICS_CSV: Path = OUTPUTS_DIR / "demographics.csv"
IMAGING_METRICS_CSV: Path = OUTPUTS_DIR / "imaging_metrics.csv"
UNET_MODEL_PATH: Path = MODELS_DIR / "unet_lesion_2d.pt"

# Standard file names inside each subject folder in open_ms_data.
MODALITY_FILES: dict[str, str] = {
    "FLAIR": "FLAIR.nii.gz",
    "T1W": "T1W.nii.gz",
    "T2W": "T2W.nii.gz",
}
LESION_MASK_FILE: str = "consensus_gt.nii.gz"      # expert consensus lesion mask
BRAINMASK_FILE: str = "brainmask.nii.gz"           # provided brain mask (coreg variant)


def ensure_dirs() -> None:
    """Create all output directories if they don't already exist.

    Called at the start of every stage so a fresh checkout 'just works'. Using
    exist_ok=True keeps this idempotent (safe to call repeatedly).
    """
    for d in (DERIVATIVES_DIR, OUTPUTS_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def variant_dir(variant: str | None = None) -> Path:
    """Return the directory holding all subjects for a given spatial variant."""
    return DATA_ROOT / (variant or DATA_VARIANT)


def list_subjects(variant: str | None = None) -> list[str]:
    """Return sorted subject IDs (e.g. ['patient01', ...]) present on disk.

    We discover subjects from the filesystem rather than hard-coding patient01..30
    so the exact same code runs over the synthetic smoke-test dataset (which might
    have only 4 fake subjects).
    """
    vdir = variant_dir(variant)
    if not vdir.is_dir():
        return []
    subs = [p.name for p in vdir.iterdir() if p.is_dir() and p.name.startswith("patient")]
    return sorted(subs)


def subject_dir(subject: str, variant: str | None = None) -> Path:
    """Path to one subject's folder within a variant."""
    return variant_dir(variant) / subject


def subject_files(subject: str, variant: str | None = None) -> dict[str, Path]:
    """Return a dict of the standard files for a subject.

    Keys: 'FLAIR', 'T1W', 'T2W', 'lesion_mask', 'brainmask'. Values are Paths that
    may or may not exist; the QC stage is responsible for checking existence and
    reporting what's missing (rather than crashing here).
    """
    sdir = subject_dir(subject, variant)
    files = {mod: sdir / fname for mod, fname in MODALITY_FILES.items()}
    files["lesion_mask"] = sdir / LESION_MASK_FILE
    files["brainmask"] = sdir / BRAINMASK_FILE
    return files


def subject_derivatives_dir(subject: str) -> Path:
    """Where processed images for one subject are written."""
    return DERIVATIVES_DIR / subject
