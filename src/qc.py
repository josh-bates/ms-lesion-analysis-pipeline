"""
qc.py  -  Stage 1: Quality Control + demographics curation
==========================================================
Before doing ANY analysis, a neuroimaging team inventories the dataset and checks
it for problems. Garbage in -> garbage out: a single subject with a transposed
image, a missing modality, or a lesion mask that doesn't line up with its FLAIR
can silently poison every downstream metric and model. This stage is the
"large-dataset curation and quality control" step.

What it does
------------
1. For every subject, load each modality + the lesion mask with nibabel and record:
     - image shape (voxel grid dimensions)
     - voxel size in mm (from the header 'zooms' / pixdim)
     - orientation code (e.g. 'LPS' - how voxel axes map to Left/Right,
       Posterior/Anterior, Superior/Inferior)
     - intensity range (min/max) and, for the mask, the number of lesion voxels
2. Flag inconsistencies:
     - a missing modality or mask,
     - FLAIR and mask that DON'T share the same grid (shape/affine) - which would
       mean lesion labels don't correspond to FLAIR voxels,
     - a mask that isn't binary, or is empty.
3. Write outputs/qc_report.csv (one row per subject).
4. Parse the demographics/EDSS table -> outputs/demographics.csv, merge it onto
   the QC table, and print a human-readable summary.

Everything is defensive: a bad subject is FLAGGED, not fatal, so one broken scan
never aborts QC over the other 29.
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import pandas as pd

from . import config
from . import image_utils

log = logging.getLogger("qc")


# ---------------------------------------------------------------------------
# Per-image inspection
# ---------------------------------------------------------------------------

def _inspect_image(path) -> dict:
    """Return a dict of QC facts about one NIfTI image (without loading full data twice).

    We read the header for cheap facts (shape, zooms, orientation) and the data
    array once for intensity range. For a mask the 'max' tells us the label values.
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return {
        "shape": tuple(int(s) for s in img.shape),
        "zooms_mm": tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "intensity_min": float(np.nanmin(data)),
        "intensity_max": float(np.nanmax(data)),
        "affine": img.affine,
    }


def compare_grids(a_affine, a_shape, b_affine, b_shape) -> tuple[str, float]:
    """Classify how two image grids relate. Returns (category, origin_offset_mm).

    Categories, from best to worst:
      'match'         - same shape AND same affine (identical grid). Ideal.
      'origin-offset' - same shape and same 3x3 (rotation+voxel scaling) but the
                        world ORIGIN differs by LESS than one voxel. Voxels still
                        correspond index-for-index, so array-based analysis (which
                        is what we do) is safe; this is a benign header quirk worth
                        noting, not a real misalignment.
      'mismatch'      - different shape, different voxel directions/scaling, or a
                        translation of a voxel or more. Here a lesion labelled at
                        (i,j,k) would NOT map to FLAIR voxel (i,j,k): serious.

    WHY split it out? open_ms_data's coregistered masks sit on the FLAIR grid but
    carry a ~0.4 mm origin offset in the header. Treating that as a fatal mismatch
    would wrongly condemn ~all subjects; ignoring geometry entirely would miss a
    genuinely transposed image. This function draws the line at "do voxels still
    correspond?" - the thing that actually matters for our voxel-wise metrics.
    """
    a_affine, b_affine = np.asarray(a_affine), np.asarray(b_affine)
    if tuple(a_shape) != tuple(b_shape):
        return "mismatch", float("nan")
    # Compare the 3x3 (rotation + voxel scaling). If these differ, axes/resolution
    # differ and voxels do not correspond.
    if not np.allclose(a_affine[:3, :3], b_affine[:3, :3], atol=1e-3):
        return "mismatch", float("nan")
    # Same shape + same 3x3 -> only the origin (translation) can differ.
    offset = float(np.linalg.norm(a_affine[:3, 3] - b_affine[:3, 3]))
    voxel_diag = float(np.linalg.norm(np.diag(a_affine[:3, :3])))  # ~one-voxel length
    if offset < 1e-3:
        return "match", offset
    if offset < voxel_diag:  # sub-voxel: indices still line up
        return "origin-offset", offset
    return "mismatch", offset


# ---------------------------------------------------------------------------
# Per-subject QC
# ---------------------------------------------------------------------------

def qc_subject(subject: str) -> dict:
    """Run QC for one subject and return a flat dict (one CSV row).

    'flags' accumulates human-readable problem strings; an empty flags field means
    the subject passed. We key most numeric facts off FLAIR because FLAIR is our
    primary modality (see README: MS lesions are brightest / most conspicuous on
    FLAIR).
    """
    files = config.subject_files(subject)
    row: dict = {"subject": subject}
    flags: list[str] = []   # serious problems -> WARN/FAIL
    notes: list[str] = []   # benign-but-noteworthy observations -> still PASS

    # 1) existence of every expected file --------------------------------------
    present = {}
    for key in ("FLAIR", "T1W", "T2W", "lesion_mask"):
        present[key] = files[key].exists()
        if not present[key]:
            flags.append(f"missing:{key}")
    row["has_FLAIR"] = present["FLAIR"]
    row["has_T1W"] = present["T1W"]
    row["has_T2W"] = present["T2W"]
    row["has_mask"] = present["lesion_mask"]

    # If FLAIR itself is missing we can't characterise the subject further.
    if not present["FLAIR"]:
        row["flags"] = ";".join(flags) or "ok"
        row["notes"] = ""
        row["status"] = "FAIL"
        return row

    # 2) inspect FLAIR (the reference grid) ------------------------------------
    flair = _inspect_image(files["FLAIR"])
    row["shape"] = "x".join(str(s) for s in flair["shape"])
    row["voxel_mm"] = "x".join(str(z) for z in flair["zooms_mm"])
    row["orientation"] = flair["orientation"]
    row["flair_min"] = round(flair["intensity_min"], 1)
    row["flair_max"] = round(flair["intensity_max"], 1)
    # Voxel volume is handy for the metrics stage and a sanity check here.
    row["voxel_volume_mm3"] = round(float(np.prod(flair["zooms_mm"])), 4)

    # 3) other modalities must share FLAIR's grid (they should, in coregistered) --
    for key in ("T1W", "T2W"):
        if present[key]:
            info = _inspect_image(files[key])
            cat, off = compare_grids(flair["affine"], flair["shape"], info["affine"], info["shape"])
            # In 'raw' space modalities legitimately have different grids; only
            # judge alignment when we expect it (coregistered/resampled/MNI).
            if config.DATA_VARIANT != "raw":
                if cat == "mismatch":
                    flags.append(f"grid-mismatch:{key}")
                elif cat == "origin-offset":
                    notes.append(f"{key}-origin-offset:{off:.2f}mm")

    # 4) lesion mask checks -----------------------------------------------------
    if present["lesion_mask"]:
        mimg = nib.load(str(files["lesion_mask"]))
        mdata = np.asarray(mimg.dataobj)
        uniq = np.unique(mdata)
        n_lesion_vox = int((mdata > 0).sum())
        row["mask_voxels"] = n_lesion_vox
        row["mask_volume_mm3"] = round(n_lesion_vox * row["voxel_volume_mm3"], 1)
        # A consensus lesion mask should be binary {0,1}.
        if not np.all(np.isin(uniq, [0, 1])):
            flags.append(f"mask-not-binary:{list(uniq[:5])}")
        if n_lesion_vox == 0:
            flags.append("mask-empty")
        # The mask must correspond to FLAIR voxels. A true grid mismatch is serious;
        # a sub-voxel origin offset (open_ms_data's ~0.4mm header quirk) is benign
        # because array indices still line up - recorded as a note, not a flag.
        cat, off = compare_grids(flair["affine"], flair["shape"], mimg.affine, mimg.shape)
        if cat == "mismatch":
            flags.append("mask-grid-mismatch")
        elif cat == "origin-offset":
            notes.append(f"mask-origin-offset:{off:.2f}mm")

    # 5) provided brain mask (optional) - just note its presence for Stage 2 Dice.
    row["has_brainmask"] = files["brainmask"].exists()

    row["flags"] = ";".join(flags) if flags else "ok"
    row["notes"] = ";".join(notes) if notes else ""
    row["status"] = "PASS" if not flags else "WARN"
    return row


# ---------------------------------------------------------------------------
# Demographics parsing
# ---------------------------------------------------------------------------

def build_demographics() -> pd.DataFrame:
    """Parse the open_ms_data demographics table into a tidy demographics.csv.

    open_ms_data ships this as patient_info.csv alongside the cross-sectional data
    (it is the same table rendered in the repo README). Columns:
        patient_id, age, sex, ms_type, edss, criteria
    Missing values are the literal string 'N/A' (3 rows have no ms_type/EDSS). We:
      * map patient_id (1..30) -> our 'patientNN' subject IDs so it merges onto QC,
      * coerce 'N/A' -> NaN for age/edss so pandas treats them numerically,
      * keep EDSS as a float (it is an ordinal 0-10 clinical score, half steps).
    """
    src_csv = config.DATA_ROOT / "patient_info.csv"
    if not src_csv.exists():
        log.warning("demographics source not found: %s", src_csv)
        return pd.DataFrame()

    df = pd.read_csv(src_csv, na_values=["N/A", "NA", ""])
    # Normalise column names defensively (dataset uses lower-case already).
    df.columns = [c.strip().lower() for c in df.columns]
    # Build the subject ID that matches folder names: patient01 .. patient30.
    df["subject"] = df["patient_id"].apply(lambda i: f"patient{int(i):02d}")
    # Ensure numeric types where appropriate.
    for col in ("age", "edss"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Reorder with subject first.
    cols = ["subject", "patient_id", "age", "sex", "ms_type", "edss", "criteria"]
    df = df[[c for c in cols if c in df.columns]]

    config.ensure_dirs()
    df.to_csv(config.DEMOGRAPHICS_CSV, index=False)
    log.info("wrote demographics: %s (%d rows)", config.DEMOGRAPHICS_CSV, len(df))
    return df


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_qc(write: bool = True) -> pd.DataFrame:
    """Run QC over all subjects, merge demographics, write qc_report.csv, print summary.

    Returns the merged DataFrame so callers (and the smoke test) can inspect it.
    """
    config.ensure_dirs()
    subjects = config.list_subjects()
    if not subjects:
        log.warning("no subjects found under %s", config.variant_dir())
        return pd.DataFrame()

    log.info("running QC on %d subjects", len(subjects))
    rows = []
    for sub in subjects:
        try:
            rows.append(qc_subject(sub))
        except Exception as e:  # noqa: BLE001 - QC must never abort on one bad subject
            log.error("QC failed hard for %s: %s", sub, e)
            rows.append({"subject": sub, "status": "ERROR", "flags": f"exception:{e}"})
    qc_df = pd.DataFrame(rows)

    # Merge demographics (left join keeps every subject even if demographics missing).
    demo = build_demographics()
    if not demo.empty:
        merged = qc_df.merge(demo, on="subject", how="left")
    else:
        merged = qc_df

    if write:
        merged.to_csv(config.QC_REPORT_CSV, index=False)
        log.info("wrote QC report: %s", config.QC_REPORT_CSV)

    _print_summary(merged)
    return merged


def _print_summary(df: pd.DataFrame) -> None:
    """Print a compact, screen-shareable QC summary."""
    print("\n" + "-" * 60)
    print(f"QC SUMMARY  ({len(df)} subjects)")
    print("-" * 60)
    if "status" in df:
        counts = df["status"].value_counts().to_dict()
        print("status:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    # Show any subjects with flags so problems are visible at a glance.
    if "flags" in df:
        flagged = df[df["flags"].fillna("ok") != "ok"]
        if len(flagged):
            print(f"\nflagged subjects ({len(flagged)}):")
            for _, r in flagged.iterrows():
                print(f"  {r['subject']}: {r['flags']}")
        else:
            print("no subjects flagged - all consistent")
    # Benign notes (e.g. the known sub-voxel origin offset) shown separately so a
    # reviewer sees they were detected and consciously judged harmless.
    if "notes" in df:
        noted = df[df["notes"].fillna("") != ""]
        if len(noted):
            example = noted["notes"].iloc[0]
            print(f"notes: {len(noted)} subjects carry benign header notes "
                  f"(e.g. {noted['subject'].iloc[0]}: {example})")
    # Consistency of geometry across the cohort (common shape/voxel/orientation?).
    for col in ("shape", "voxel_mm", "orientation"):
        if col in df:
            vals = df[col].dropna().unique()
            tag = "uniform" if len(vals) == 1 else f"{len(vals)} distinct"
            print(f"{col}: {tag}  {list(vals)[:4]}{' ...' if len(vals) > 4 else ''}")
    # Demographics snapshot.
    if "edss" in df:
        n_edss = int(df["edss"].notna().sum())
        print(f"\ndemographics: EDSS present for {n_edss}/{len(df)} "
              f"(missing {len(df) - n_edss})")
        if "ms_type" in df:
            print("ms_type:", df["ms_type"].value_counts(dropna=False).to_dict())
    print("-" * 60)
