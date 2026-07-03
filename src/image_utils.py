"""
image_utils.py
==============
Small, dependency-light helpers for reading/writing NIfTI images and the two
image operations that show up all over the pipeline:

  * z-score normalisation over brain voxels, and
  * a synthetic "brain phantom" generator used by the smoke test.

Everything here uses nibabel (the standard Python library for neuroimaging file
formats) and numpy. Keeping these helpers in one place means every stage loads
and normalises images the *same* way - a common source of silent bugs is two
modules normalising slightly differently.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI image object (keeps header + affine, does not read data yet)."""
    return nib.load(str(path))


def load_data(path: Path) -> np.ndarray:
    """Load just the voxel array as float32.

    We force float32: the raw dtype might be int16, and doing z-scoring or Dice
    on integers would truncate. float32 (not float64) keeps memory reasonable for
    whole-brain volumes.
    """
    return np.asarray(load_nifti(path).dataobj, dtype=np.float32)


def save_like(reference: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> Path:
    """Save `data` as a NIfTI that borrows the affine + header of `reference`.

    WHY reuse the affine? The affine maps voxel indices -> millimetre world
    coordinates. If we invented our own, a derived mask would no longer line up
    with the original scan in any viewer. Reusing the reference affine guarantees
    the output overlays perfectly on its input.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data), affine=reference.affine, header=reference.header)
    nib.save(img, str(out_path))
    return out_path


def resample_to_iso(img: nib.Nifti1Image, mm: float, *, interpolation: str = "continuous") -> nib.Nifti1Image:
    """Resample an image to isotropic `mm` voxels, keeping its world position.

    WHY resample (downsample) at all?
        These scans are ~0.47 mm in-plane - beautiful, but ~45x more voxels than a
        2 mm grid. FSL's bet/flirt/fast scale with voxel count, so native resolution
        makes each subject take minutes. For a proof-of-concept whose outputs are
        physical VOLUMES (lesion load, tissue volumes in mm^3), 2 mm is plenty: a
        volume in mm^3 is resolution-independent to first order. We trade a little
        precision on tiny lesions (stated as a limitation) for a ~45x speed-up that
        lets us process all 30 subjects instead of a handful.

    HOW: we build a target affine that keeps the original orientation/direction
    cosines but rescales each axis to `mm`, then let nilearn compute the new shape
    and origin so the brain stays in the same world location (masks still overlay).

    interpolation:
        'continuous' for intensity images (FLAIR/T1), 'nearest' for label masks
        (never interpolate integer labels - that invents fractional classes).
    """
    from nilearn.image import resample_img  # imported lazily so core stages don't need nilearn

    affine = np.asarray(img.affine, dtype=np.float64)
    R = affine[:3, :3]
    zooms = np.linalg.norm(R, axis=0)           # current voxel sizes from column norms
    zooms[zooms == 0] = 1.0
    target_R = R / zooms * float(mm)            # same directions, rescaled to `mm`
    return resample_img(
        img,
        target_affine=target_R,
        interpolation=interpolation,
        copy_header=True,
        force_resample=True,
    )


def resample_like(source: nib.Nifti1Image, reference: nib.Nifti1Image, *, interpolation: str = "nearest") -> nib.Nifti1Image:
    """Resample `source` onto `reference`'s grid (e.g. put a native mask on the 2mm grid).

    Used to bring the dataset's native-resolution provided brain mask onto our
    downsampled FLAIR grid so we can Dice-compare them voxel-for-voxel.
    """
    from nilearn.image import resample_to_img

    return resample_to_img(
        source, reference, interpolation=interpolation, copy_header=True, force_resample=True
    )


def voxel_volume_mm3(img: nib.Nifti1Image) -> float:
    """Volume of a single voxel in mm^3 = product of the three voxel dimensions.

    Read from the header's zooms (pixdim). Needed to convert a voxel COUNT into a
    physical VOLUME - two scans with different resolution can have the same lesion
    volume but very different voxel counts.
    """
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def zscore_normalise(volume: np.ndarray, brain_mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalise: (x - mean) / std, computed over BRAIN voxels only.

    WHY z-score, and why only over brain voxels?
        MRI intensities are in arbitrary units - the same tissue can read 200 on
        one scanner and 800 on another. A learning model must not have to learn
        each scanner's scale, so we standardise every volume to mean 0, std 1.
        We compute the statistics over brain voxels only because the large black
        air background would drag the mean down and shrink the std, wasting the
        model's dynamic range on empty space.
        (Alternatives: min-max scaling is sensitive to a single bright outlier
        voxel; histogram matching is more powerful but heavier and harder to
        explain - z-scoring is the standard, transparent default for MS lesion
        work.)

    If no mask is given we build a crude one (voxels above the volume mean), which
    is enough for the synthetic phantom and as a fallback.
    """
    volume = volume.astype(np.float32)
    if brain_mask is None:
        brain_mask = volume > volume.mean()
    brain_mask = brain_mask.astype(bool)
    if brain_mask.sum() == 0:
        return np.zeros_like(volume)
    vals = volume[brain_mask]
    mean, std = float(vals.mean()), float(vals.std())
    if std == 0:
        return np.zeros_like(volume)
    out = (volume - mean) / std
    # Zero out non-brain so downstream code never sees normalised background noise.
    out[~brain_mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Synthetic phantom (used by tests/smoke_test.py)
# ---------------------------------------------------------------------------

def _ellipsoid(shape, centre, radii) -> np.ndarray:
    """Boolean mask of an ellipsoid - the building block for a fake 'brain'."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = centre
    rz, ry, rx = radii
    d = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return d <= 1.0


def make_synthetic_subject(
    out_dir: Path,
    *,
    shape=(48, 64, 64),
    voxel_size=(3.0, 2.0, 2.0),
    n_lesions: int = 3,
    seed: int = 0,
    with_brainmask: bool = True,
) -> dict[str, Path]:
    """Write a tiny but *structurally realistic* fake subject to `out_dir`.

    It produces FLAIR/T1W/T2W volumes, a consensus lesion mask and (optionally) a
    brain mask, using the SAME file names as open_ms_data so every real stage runs
    against it unchanged. This is what lets us validate the whole pipeline and the
    GUI in seconds, before the multi-GB download and slow FSL runs finish.

    The phantom deliberately encodes the physics the pipeline relies on:
      * an ellipsoidal brain sitting inside empty 'air' (so skull-strip has a job),
      * WM brighter than GM on T1 (so FAST-like ordering is meaningful), and
      * lesions that are HYPER-intense on FLAIR (bright) - exactly the appearance
        MS lesions have, which is why FLAIR is the modality we segment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # World geometry: a diagonal affine encoding the voxel sizes in mm.
    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0]).astype(np.float32)

    centre = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    brain = _ellipsoid(shape, centre, (shape[0] * 0.38, shape[1] * 0.40, shape[2] * 0.40))
    # Inner ellipsoid = white matter; the shell between it and the brain edge = grey matter.
    wm = _ellipsoid(shape, centre, (shape[0] * 0.26, shape[1] * 0.28, shape[2] * 0.28))
    gm = brain & ~wm

    # Base intensities (arbitrary units, chosen to mimic real contrast ordering).
    flair = np.zeros(shape, np.float32)
    t1 = np.zeros(shape, np.float32)
    t2 = np.zeros(shape, np.float32)
    # T1: WM brightest, GM mid, CSF/air dark.
    t1[gm] = 400.0
    t1[wm] = 650.0
    # FLAIR/T2: WM/GM mid; we'll add bright lesions on FLAIR below.
    flair[brain] = 300.0
    t2[brain] = 350.0

    # Scatter lesions inside white matter and make them FLAIR-hyperintense.
    lesion = np.zeros(shape, bool)
    wm_idx = np.argwhere(wm)
    for _ in range(n_lesions):
        cz, cy, cx = wm_idx[rng.integers(len(wm_idx))]
        r = rng.integers(2, 4)
        blob = _ellipsoid(shape, (cz, cy, cx), (r, r, r)) & wm
        lesion |= blob
    flair[lesion] = 900.0   # hyperintense on FLAIR - the signature of MS lesions
    t2[lesion] = 800.0

    # Add mild Gaussian noise so normalisation/segmentation aren't trivial.
    for arr in (flair, t1, t2):
        arr += rng.normal(0, 15, size=shape).astype(np.float32)
        np.clip(arr, 0, None, out=arr)

    def _save(arr, name):
        p = out_dir / name
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        return p

    paths = {
        "FLAIR": _save(flair, "FLAIR.nii.gz"),
        "T1W": _save(t1, "T1W.nii.gz"),
        "T2W": _save(t2, "T2W.nii.gz"),
        "lesion_mask": _save(lesion.astype(np.uint8), "consensus_gt.nii.gz"),
    }
    if with_brainmask:
        paths["brainmask"] = _save(brain.astype(np.uint8), "brainmask.nii.gz")
    return paths


def make_synthetic_dataset(root: Path, n_subjects: int = 4) -> Path:
    """Create a full mini-cohort under `root/<variant>/patientNN/...`.

    Layout mirrors open_ms_data/cross_sectional so pointing MS_DATA_ROOT at `root`
    makes the real pipeline treat this as the dataset. Subjects are given a range
    of lesion counts so downstream metrics/models see some variation.
    """
    from . import config

    root = Path(root)
    vdir = root / config.DATA_VARIANT
    for i in range(1, n_subjects + 1):
        sub = f"patient{i:02d}"
        make_synthetic_subject(
            vdir / sub,
            n_lesions=1 + (i % 4),   # 1..4 lesions, gives between-subject variation
            seed=i,
        )
    return root
