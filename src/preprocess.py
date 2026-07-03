"""
preprocess.py  -  Stage 2: FSL preprocessing pipeline
=====================================================
Turn raw scanner images into clean, aligned, brain-only images with tissue labels,
using the industry-standard FSL toolkit. This is the "automated MRI preprocessing
pipeline" step, and it is where we lean on FSL (bet/flirt/fast) rather than any
hand-rolled algorithm.

Per subject, starting from RAW FLAIR and RAW T1:

  0. downsample FLAIR to an isotropic grid (default 2mm; config.PREPROCESS_ISO_MM)
             -> a ~45x smaller volume so bet/flirt/fast finish in seconds not
             minutes, letting us process all 30 subjects. Everything below runs on
             this grid. See image_utils.resample_to_iso for the fidelity trade-off.

  1. bet   (Brain Extraction Tool) on FLAIR
             -> a brain-extracted FLAIR + a binary brain mask, in FLAIR space.
             We skull-strip because everything after this (tissue segmentation,
             intensity normalisation, lesion metrics) should see brain only - the
             skull, scalp, eyes and neck are bright, irrelevant, and would wreck
             both FAST's intensity model and any z-scoring.

  2. flirt (linear registration) of raw T1 -> raw FLAIR, rigid (6 DOF)
             -> the T1 resampled onto the FLAIR voxel grid. Raw T1 and raw FLAIR
             are the SAME head in one session but on totally different grids
             (we verified raw T1 is e.g. 408x512x152 vs FLAIR 192x512x512), so we
             must align them before we can combine information across modalities.
             Rigid (rotations+translations only) is correct for same-subject,
             same-day scans - the anatomy's size/shape must not change.

  3. Apply the FLAIR brain mask to the registered T1 -> brain-only T1.

  4. fast  (tissue segmentation) on the brain-only T1
             -> partial-volume maps for CSF / grey matter / white matter. We run
             FAST on T1 (not FLAIR) because T1 has the best grey/white contrast.
             White-matter volume from here feeds the "lesion load" metric in Stage 3.

  5. QC:  Dice overlap between OUR bet brain mask and the dataset's PROVIDED
             brainmask.nii.gz. Dice in [0,1]; ~0.95+ means our skull-strip agrees
             with the reference. This is a cheap, quantitative check that the most
             error-prone step (brain extraction) didn't misbehave.

All outputs go to derivatives/<subject>/. FSL is slow on these 512x512 volumes
(minutes per subject), so run_preprocess() processes a small SUBSET by default and
skips subjects already done (idempotent); pass limit=None to process everyone.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, fsl_utils, image_utils

log = logging.getLogger("preprocess")

# Which spatial variant to read raw inputs from. The task says "starting from raw
# FLAIR and T1", so we default to the 'raw' variant. If it's absent (e.g. the
# synthetic smoke dataset only builds one variant), we fall back to the configured
# DATA_VARIANT so the exact same code still runs in tests.
RAW_VARIANT = "raw"

# Default number of subjects to preprocess when called with no explicit list.
# Keeps the demo/pipeline fast; override with limit=None for the full cohort.
DEFAULT_LIMIT = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_inputs(subject: str) -> tuple:
    """Return (flair_path, t1_path) for a subject, preferring the raw variant.

    Falls back to the configured variant if a 'raw' folder doesn't exist so the
    synthetic smoke test (which builds only one variant) exercises this code too.
    """
    raw = config.variant_dir(RAW_VARIANT)
    variant = RAW_VARIANT if raw.is_dir() else config.DATA_VARIANT
    files = config.subject_files(subject, variant=variant)
    return files["FLAIR"], files["T1W"]


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

        Dice = 2 * |A ∩ B| / (|A| + |B|)

    Ranges 0 (no overlap) to 1 (identical). Dice is the standard overlap metric in
    medical image segmentation because it rewards agreement while being robust to
    the huge class imbalance typical of masks (most voxels are background). We use
    it here for brain-mask agreement, and again in Stage 4 for lesion segmentation.
    (Alternative: Jaccard/IoU = |A∩B|/|A∪B|; Dice is just a monotonic re-scaling and
    is the field convention, so we report Dice.)
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # two empty masks are trivially identical
    return float(2.0 * np.logical_and(a, b).sum() / denom)


# ---------------------------------------------------------------------------
# Per-subject preprocessing
# ---------------------------------------------------------------------------

def preprocess_subject(subject: str, *, overwrite: bool = False) -> dict:
    """Run bet -> flirt -> fast for one subject; return a dict of QC facts.

    Idempotent: if the key outputs already exist and overwrite is False, we skip
    the expensive FSL calls and just (re)compute the Dice QC from the existing mask.
    """
    if not fsl_utils.fsl_available():
        raise RuntimeError("FSL not available - cannot run preprocessing")

    out_dir = config.subject_derivatives_dir(subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    flair_raw, t1_raw = _raw_inputs(subject)

    # Output paths (named so a human browsing derivatives/ knows what each is).
    flair_ds = out_dir / "flair_2mm.nii.gz"               # downsampled FLAIR (bet/flirt ref)
    flair_brain = out_dir / "flair_brain.nii.gz"
    flair_mask = fsl_utils.brain_mask_path(flair_brain)   # flair_brain_mask.nii.gz
    t1_to_flair = out_dir / "t1_to_flair.nii.gz"
    t1_to_flair_mat = out_dir / "t1_to_flair.mat"
    t1_brain = out_dir / "t1_brain.nii.gz"
    fast_base = out_dir / "fast"                          # fast_pve_0/1/2.nii.gz

    done = flair_mask.exists() and t1_brain.exists() and (out_dir / "fast_pve_2.nii.gz").exists()
    if done and not overwrite:
        log.info("[%s] outputs exist - skipping FSL (use overwrite=True to redo)", subject)
    else:
        # --- 0) downsample FLAIR to an isotropic grid (speed; see config) ----
        # We resample the FLAIR ONCE and use it as the common reference grid for
        # everything below. flirt then resamples T1 straight onto this grid, so the
        # whole subject lives in one tidy low-res space.
        iso = config.PREPROCESS_ISO_MM
        if iso and iso > 0:
            log.info("[%s] downsample FLAIR to %.1fmm isotropic", subject, iso)
            ds = image_utils.resample_to_iso(image_utils.load_nifti(flair_raw), iso,
                                             interpolation="continuous")
            image_utils.save_like(ds, np.asarray(ds.dataobj), flair_ds)
            flair_ref = flair_ds
        else:
            flair_ref = flair_raw  # native resolution (slow path)

        # --- 1) brain-extract the FLAIR -------------------------------------
        log.info("[%s] bet (brain extraction) on FLAIR", subject)
        fsl_utils.bet(flair_ref, flair_brain, frac=0.4, robust=True)

        # --- 2) rigid-register raw T1 onto the (downsampled) FLAIR ----------
        # Using flair_ref as the flirt reference means the registered T1 comes out
        # on the SAME 2mm grid - registration and resampling in one step.
        log.info("[%s] flirt (rigid T1 -> FLAIR, %.1fmm grid)", subject, iso or 0)
        fsl_utils.flirt_rigid(t1_raw, flair_ref, t1_to_flair, t1_to_flair_mat)

        # --- 3) apply the FLAIR brain mask to the registered T1 -------------
        log.info("[%s] apply brain mask to registered T1", subject)
        fsl_utils.fslmaths_mask(t1_to_flair, flair_mask, t1_brain)

        # --- 4) tissue segmentation on the brain-only T1 --------------------
        log.info("[%s] fast (CSF/GM/WM tissue segmentation)", subject)
        fsl_utils.fast_segment(t1_brain, fast_base, n_classes=3)

    # --- 5) QC: Dice vs the provided brain mask -----------------------------
    row: dict = {"subject": subject, "status": "ok"}
    my_mask_img = image_utils.load_nifti(flair_mask)
    my_mask = np.asarray(my_mask_img.dataobj).astype(bool)
    row["brain_voxels"] = int(my_mask.sum())

    provided = config.subject_files(subject)["brainmask"]  # native-res, coregistered variant
    if provided.exists():
        # Our mask is on the 2mm grid; the provided mask is native ~0.47mm. Resample
        # the provided mask ONTO our grid (nearest-neighbour, never interpolate
        # labels) so the two masks are voxel-comparable, then Dice them.
        ref_img = image_utils.resample_like(
            image_utils.load_nifti(provided), my_mask_img, interpolation="nearest"
        )
        ref = np.asarray(ref_img.dataobj).astype(bool)
        row["dice_vs_provided"] = round(dice(my_mask, ref), 4)
    else:
        row["dice_vs_provided"] = np.nan
        row["status"] = "no-provided-brainmask"

    # Record tissue volumes (mm^3) - white-matter volume is reused by Stage 3.
    fast_wm = out_dir / "fast_pve_2.nii.gz"  # pve_2 = WM (brightest tissue on T1)
    if fast_wm.exists():
        wm_img = image_utils.load_nifti(fast_wm)
        vox_vol = image_utils.voxel_volume_mm3(wm_img)
        # pve maps are partial-volume fractions in [0,1]; summing * voxel volume
        # gives a soft (sub-voxel-accurate) tissue volume.
        row["wm_volume_mm3"] = round(float(np.asarray(wm_img.dataobj).sum() * vox_vol), 1)

    log.info("[%s] done: brain_voxels=%d dice_vs_provided=%s",
             subject, row["brain_voxels"], row.get("dice_vs_provided"))
    return row


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_preprocess(subjects: list[str] | None = None,
                   *, limit: int | None = DEFAULT_LIMIT,
                   overwrite: bool = False,
                   write: bool = True) -> pd.DataFrame:
    """Preprocess subjects and write outputs/preprocess_qc.csv.

    Parameters
    ----------
    subjects  : explicit list; if None, use the first `limit` discovered subjects.
    limit     : cap on how many subjects to process when `subjects` is None. FSL is
                slow, so we default to a subset for demos. Pass None to do all.
    overwrite : re-run FSL even if outputs already exist.
    """
    config.ensure_dirs()
    if not fsl_utils.fsl_available():
        log.warning("FSL not available - skipping preprocessing stage entirely.")
        return pd.DataFrame()

    if subjects is None:
        subjects = config.list_subjects()
        if limit is not None:
            subjects = subjects[:limit]

    log.info("preprocessing %d subject(s): %s", len(subjects), subjects)
    print(f"\nPreprocessing {len(subjects)} subject(s) with FSL "
          f"(bet -> flirt -> fast). This is the slow stage.")
    if limit is not None and len(config.list_subjects()) > len(subjects):
        print(f"  (subset for speed; run with limit=None to process all "
              f"{len(config.list_subjects())} subjects)")

    rows = []
    for i, sub in enumerate(subjects, 1):
        print(f"  [{i}/{len(subjects)}] {sub} ...", flush=True)
        try:
            rows.append(preprocess_subject(sub, overwrite=overwrite))
        except Exception as e:  # noqa: BLE001 - one bad subject must not abort the batch
            log.error("preprocess failed for %s: %s", sub, e)
            rows.append({"subject": sub, "status": f"error:{e}"})

    df = pd.DataFrame(rows)
    if write and not df.empty:
        out = config.OUTPUTS_DIR / "preprocess_qc.csv"
        df.to_csv(out, index=False)
        log.info("wrote %s", out)
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n" + "-" * 60)
    print(f"PREPROCESS SUMMARY  ({len(df)} subjects)")
    print("-" * 60)
    if "dice_vs_provided" in df:
        d = df["dice_vs_provided"].dropna()
        if len(d):
            print(f"Dice vs provided brainmask: mean={d.mean():.3f} "
                  f"min={d.min():.3f} max={d.max():.3f}")
            # A low Dice is worth calling out explicitly.
            poor = df[df["dice_vs_provided"] < 0.9]
            if len(poor):
                print("  subjects with Dice < 0.90 (inspect BET):",
                      list(poor["subject"]))
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("non-ok status:")
        for _, r in bad.iterrows():
            print(f"  {r['subject']}: {r['status']}")
    print("-" * 60)
