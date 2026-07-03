"""
metrics.py  -  Stage 3: quantitative imaging metrics
====================================================
Turn each subject's expert lesion mask (+ tissue segmentation from Stage 2) into a
small set of NUMBERS that describe their MS lesion burden. These numbers are what
Stage 5 relates to clinical disability (EDSS), and what the GUI reports for an
uploaded scan.

Per subject we compute:

  * lesion_volume_mm3   - total lesion volume in cubic millimetres. Volume (not
                          voxel count) so scans at different resolutions are
                          comparable: a voxel COUNT depends on voxel size, a volume
                          in mm^3 does not.
  * lesion_count        - number of DISTINCT lesions = connected components of the
                          mask. Two subjects can share a lesion volume but differ in
                          whether it's one big plaque or many small ones - clinically
                          different pictures.
  * lesion_load_wm_frac - lesion volume as a fraction of white-matter volume. This
                          normalises burden by how much white matter a person has
                          (bigger brains have more WM), and MS lesions live in white
                          matter, so WM is the natural denominator.
  * mean_flair_lesion   - mean FLAIR intensity inside the lesions, both raw and
                          z-scored over the brain. Raw MRI intensity is in arbitrary
                          units (not comparable across scans), so the z-scored value
                          ("how many standard deviations above typical brain tissue")
                          is the comparable one.

Design: this stage does NOT need FSL and is fast. It needs Stage 2's white-matter
map ONLY for lesion_load; if a subject hasn't been preprocessed yet we still emit
every other metric and leave lesion_load as NaN (never crash on partial data).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage

from . import config, image_utils

log = logging.getLogger("metrics")


# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------

def count_lesions(mask: np.ndarray, *, min_voxels: int = 0) -> int:
    """Count connected components (distinct lesions) in a binary mask.

    Connectivity choice: we use FULL 3D connectivity (26-neighbourhood: faces +
    edges + corners) via a 3x3x3 structuring element. Alternatives are 6- (faces
    only) or 18- (faces+edges) connectivity, which would split a diagonally-touching
    lesion into two. 26-connectivity is the common, permissive default for lesion
    counting; the trade-off is that two lesions touching only at a corner are counted
    as one. `min_voxels` can drop tiny specks, but the expert consensus mask is clean
    so we default to 0 (no filtering).
    """
    structure = np.ones((3, 3, 3), dtype=int)  # 26-connectivity
    labelled, n = ndimage.label(mask.astype(bool), structure=structure)
    if min_voxels > 0 and n > 0:
        sizes = ndimage.sum(np.ones_like(labelled), labelled, index=range(1, n + 1))
        n = int((sizes >= min_voxels).sum())
    return int(n)


def _wm_volume_mm3(subject: str) -> float | None:
    """White-matter volume (mm^3) from Stage 2's FAST output, or None if not run.

    FAST's pve_2 is the white-matter partial-volume map (WM is brightest on T1, so
    it's the highest-intensity class). Summing the partial-volume fractions and
    multiplying by the voxel volume gives a sub-voxel-accurate WM volume.
    """
    wm_path = config.subject_derivatives_dir(subject) / "fast_pve_2.nii.gz"
    if not wm_path.exists():
        return None
    img = image_utils.load_nifti(wm_path)
    vox = image_utils.voxel_volume_mm3(img)
    return float(np.asarray(img.dataobj).sum() * vox)


def _native_brain_mask(subject: str, reference_img):
    """Return a brain mask on the native FLAIR grid (for z-scoring lesion intensity).

    We reuse Stage 2's bet brain mask (computed at 2mm) and resample it up onto the
    native FLAIR grid. If preprocessing hasn't run, fall back to None and let the
    z-score routine build a crude threshold mask.
    """
    bet_mask = config.subject_derivatives_dir(subject) / "flair_brain_mask.nii.gz"
    if not bet_mask.exists():
        return None
    resampled = image_utils.resample_like(
        image_utils.load_nifti(bet_mask), reference_img, interpolation="nearest"
    )
    return np.asarray(resampled.dataobj).astype(bool)


# ---------------------------------------------------------------------------
# Per-subject metrics
# ---------------------------------------------------------------------------

def metrics_subject(subject: str) -> dict:
    """Compute all imaging metrics for one subject and return a CSV row dict."""
    files = config.subject_files(subject)
    row: dict = {"subject": subject}

    # Lesion mask + FLAIR are both native (coregistered variant) -> same grid, so
    # intensity sampling and volume are exact and need no resampling.
    mask_img = image_utils.load_nifti(files["lesion_mask"])
    mask = np.asarray(mask_img.dataobj) > 0
    vox_vol = image_utils.voxel_volume_mm3(mask_img)

    n_lesion_vox = int(mask.sum())
    row["lesion_voxels"] = n_lesion_vox
    row["lesion_volume_mm3"] = round(n_lesion_vox * vox_vol, 1)
    row["lesion_count"] = count_lesions(mask)

    # --- lesion load as a fraction of white matter --------------------------
    wm_vol = _wm_volume_mm3(subject)
    if wm_vol and wm_vol > 0:
        row["wm_volume_mm3"] = round(wm_vol, 1)
        row["lesion_load_wm_frac"] = round(row["lesion_volume_mm3"] / wm_vol, 5)
    else:
        row["wm_volume_mm3"] = np.nan
        row["lesion_load_wm_frac"] = np.nan  # Stage 2 not run for this subject yet

    # --- mean FLAIR intensity inside lesions --------------------------------
    flair_img = image_utils.load_nifti(files["FLAIR"])
    flair = np.asarray(flair_img.dataobj, dtype=np.float32)
    if n_lesion_vox > 0:
        row["mean_flair_lesion_raw"] = round(float(flair[mask].mean()), 2)
        # Z-score the FLAIR over brain voxels, then average inside lesions. This
        # "SDs above typical brain tissue" is comparable across subjects/scanners,
        # unlike the raw value.
        brain = _native_brain_mask(subject, flair_img)
        z = image_utils.zscore_normalise(flair, brain)
        row["mean_flair_lesion_z"] = round(float(z[mask].mean()), 3)
    else:
        row["mean_flair_lesion_raw"] = np.nan
        row["mean_flair_lesion_z"] = np.nan

    return row


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_metrics(subjects: list[str] | None = None, *, write: bool = True) -> pd.DataFrame:
    """Compute imaging metrics for all subjects -> outputs/imaging_metrics.csv.

    Returns the DataFrame. Kept imaging-only (no EDSS merge) so this file is a clean
    'what the images say'; Stage 5 joins it to demographics/EDSS.
    """
    config.ensure_dirs()
    if subjects is None:
        subjects = config.list_subjects()
    if not subjects:
        log.warning("no subjects found")
        return pd.DataFrame()

    log.info("computing imaging metrics for %d subjects", len(subjects))
    rows = []
    for sub in subjects:
        try:
            rows.append(metrics_subject(sub))
        except Exception as e:  # noqa: BLE001
            log.error("metrics failed for %s: %s", sub, e)
            rows.append({"subject": sub, "error": str(e)})
    df = pd.DataFrame(rows)

    if write:
        df.to_csv(config.IMAGING_METRICS_CSV, index=False)
        log.info("wrote %s", config.IMAGING_METRICS_CSV)

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "-" * 60)
    print(f"IMAGING METRICS SUMMARY  ({len(df)} subjects)")
    print("-" * 60)
    n_load = int(df["lesion_load_wm_frac"].notna().sum()) if "lesion_load_wm_frac" in df else 0
    print(f"lesion_load computed for {n_load}/{len(df)} "
          f"(rest await Stage 2 preprocessing)")
    for col, label, scale in [
        ("lesion_volume_mm3", "lesion volume (cm3)", 1 / 1000),
        ("lesion_count", "lesion count", 1),
        ("lesion_load_wm_frac", "lesion load (% WM)", 100),
        ("mean_flair_lesion_z", "mean FLAIR z in lesion", 1),
    ]:
        if col in df and df[col].notna().any():
            s = df[col].dropna() * scale
            print(f"  {label:26s} median={s.median():7.2f}  range[{s.min():.2f}, {s.max():.2f}]")
    print("-" * 60)
