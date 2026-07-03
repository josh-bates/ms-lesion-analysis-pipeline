"""
segment.py  -  Stage 4: 2D U-Net lesion segmentation (MONAI / PyTorch)
=====================================================================
Train a small convolutional neural network to segment MS lesions on axial FLAIR
slices, learning from the expert consensus masks. This is the "deep learning for
medical image analysis" step.

The essential ideas, and the choices we make (with alternatives noted):

WHY a U-Net?
    U-Net is the default architecture for biomedical image segmentation: an encoder
    that compresses the image to "what is here" and a decoder that expands back to
    "where exactly", with skip connections so fine boundaries survive. We use MONAI's
    implementation (MONAI = medical-imaging library built on PyTorch) rather than
    writing one by hand, for the same reason we use FSL: use the validated tool.

WHY 2D (per-slice) not 3D?
    A 3D U-Net sees whole-brain context and usually wins, but it is far heavier to
    train and needs a GPU to be practical. For a CPU-friendly prototype we segment
    each axial slice independently. Trade-off: the model can't use through-plane
    context, so it may miss lesions that are only obvious across slices. Stated as a
    limitation; the code is structured so swapping in a 3D UNet is a small change.

WHY split by SUBJECT, not by slice?
    THE most important methodological point here. Adjacent slices from one patient
    are highly correlated. If slices from the same subject appear in both train and
    test, the model can "memorise" that subject and the test Dice is inflated - a
    classic data-leakage bug. We therefore assign each SUBJECT wholly to train,
    validation, or test, so the test set is subjects the model has never seen.

WHY z-score normalisation on brain voxels?
    (See image_utils.zscore_normalise.) MRI intensities are arbitrary units; we
    standardise each volume to mean 0 / std 1 over brain voxels so the network
    learns lesion SHAPE/CONTRAST, not a scanner's brightness.

WHAT is Dice, and why report it on held-out subjects?
    Dice = 2*|pred ∩ truth| / (|pred| + |truth|), in [0,1]; it's the standard overlap
    score for segmentation. We report it on the held-out TEST subjects because that
    estimates how well the model generalises to new patients - the only number that
    matters. On 30 subjects this is illustrative, not clinical (stated as a caveat).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import nibabel as nib

from . import config, image_utils

log = logging.getLogger("segment")

# --- working resolution / slice size --------------------------------------
# We train at 2mm isotropic and resize every axial slice to a fixed square so the
# network sees a consistent input (QC found the cohort has 2 different native
# shapes). 128x128 is small enough for CPU, big enough to keep lesions visible.
SEG_ISO_MM = 2.0
SLICE_SIZE = 128


@dataclass
class SegConfig:
    """Everything needed to reproduce training / run inference. Saved with the model
    so the GUI normalises an uploaded scan exactly as training did."""
    iso_mm: float = SEG_ISO_MM
    slice_size: int = SLICE_SIZE
    channels: tuple = (16, 32, 64, 128)  # U-Net feature widths per level (small=CPU)
    max_epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-3
    neg_per_pos: float = 1.0    # how many lesion-free slices to keep per lesion slice
    seed: int = 42


# ---------------------------------------------------------------------------
# Data preparation: volumes -> normalised, fixed-size axial slices
# ---------------------------------------------------------------------------

def _canonical_iso(img: nib.Nifti1Image, mm: float, *, interpolation: str) -> nib.Nifti1Image:
    """Reorient to canonical RAS then resample to `mm` isotropic.

    Reorienting to RAS first means "axial" is ALWAYS the last axis, regardless of a
    scan's native orientation (our data is LPS). That makes slice extraction below
    orientation-proof - a subtle but important robustness point.
    """
    can = nib.as_closest_canonical(img)
    return image_utils.resample_to_iso(can, mm, interpolation=interpolation)


def _resize_slice(arr2d: np.ndarray, size: int, *, order: int) -> np.ndarray:
    """Resize a 2D slice to size x size. order=1 bilinear for images, 0 nearest for masks."""
    from scipy.ndimage import zoom
    zy = size / arr2d.shape[0]
    zx = size / arr2d.shape[1]
    return zoom(arr2d, (zy, zx), order=order)


def load_subject_slices(subject: str, cfg: SegConfig, *, rng=None):
    """Return lists (images, masks) of fixed-size normalised axial slices for a subject.

    We keep every slice that contains lesion (the positives) plus a random sample of
    lesion-free brain slices (negatives), balanced by cfg.neg_per_pos. WHY balance?
    Most axial slices contain no lesion; training on all of them drowns the signal
    and the network learns to predict "all background". Subsampling negatives is a
    simple, transparent fix. (Alternative: keep all slices with a class-weighted or
    focal loss - heavier; balancing is the clearer teaching choice.)
    """
    rng = rng or np.random.default_rng(cfg.seed)
    files = config.subject_files(subject)

    flair_img = _canonical_iso(image_utils.load_nifti(files["FLAIR"]), cfg.iso_mm, interpolation="continuous")
    mask_img = _canonical_iso(image_utils.load_nifti(files["lesion_mask"]), cfg.iso_mm, interpolation="nearest")
    flair = np.asarray(flair_img.dataobj, dtype=np.float32)
    mask = (np.asarray(mask_img.dataobj) > 0).astype(np.float32)

    # z-score over a brain mask (prefer Stage 2's bet mask; else threshold fallback).
    brain = flair > np.percentile(flair[flair > 0], 20) if (flair > 0).any() else None
    flair = image_utils.zscore_normalise(flair, brain)

    # Axial slices are along the last (S) axis after canonical reorientation.
    n_slices = flair.shape[2]
    has_lesion = [k for k in range(n_slices) if mask[:, :, k].any()]
    # brain-containing but lesion-free slices, as negative candidates
    lesion_free = [k for k in range(n_slices)
                   if k not in has_lesion and (flair[:, :, k] != 0).any()]
    n_neg = min(len(lesion_free), int(round(len(has_lesion) * cfg.neg_per_pos)))
    negatives = list(rng.choice(lesion_free, size=n_neg, replace=False)) if n_neg else []

    images, masks = [], []
    for k in has_lesion + negatives:
        img2d = _resize_slice(flair[:, :, k], cfg.slice_size, order=1)
        msk2d = _resize_slice(mask[:, :, k], cfg.slice_size, order=0)
        images.append(img2d.astype(np.float32))
        masks.append((msk2d > 0.5).astype(np.float32))
    return images, masks


# ---------------------------------------------------------------------------
# Subject-level train/val/test split (leakage-free)
# ---------------------------------------------------------------------------

def split_subjects(subjects: list[str], cfg: SegConfig):
    """Deterministically split SUBJECTS into train/val/test (~70/15/15).

    With tiny N we still guarantee at least one subject in val and test so the run is
    meaningful. Seeded so the split is reproducible across runs.
    """
    rng = np.random.default_rng(cfg.seed)
    subs = list(subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    test = subs[:n_test]
    val = subs[n_test:n_test + n_val]
    train = subs[n_test + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Torch dataset / model (imported lazily so non-DL stages don't need torch)
# ---------------------------------------------------------------------------

def _build_dataset(subjects, cfg, rng):
    import torch

    all_imgs, all_masks = [], []
    for sub in subjects:
        imgs, masks = load_subject_slices(sub, cfg, rng=rng)
        all_imgs.extend(imgs)
        all_masks.extend(masks)
    if not all_imgs:
        return None
    # Shape -> (N, 1, H, W): a channel dim of 1 (single FLAIR modality).
    X = torch.from_numpy(np.stack(all_imgs)[:, None, :, :]).float()
    Y = torch.from_numpy(np.stack(all_masks)[:, None, :, :]).float()
    return torch.utils.data.TensorDataset(X, Y)


def build_model(cfg: SegConfig):
    """Create a small 2D U-Net (MONAI). out_channels=1 + sigmoid => binary mask."""
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=cfg.channels,
        strides=(2,) * (len(cfg.channels) - 1),
        num_res_units=2,
    )


def _dice_from_logits(logits, target, threshold=0.5, eps=1e-6):
    """Dice on a batch, thresholding sigmoid(logits). Used for validation/test."""
    import torch

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return dice  # per-sample


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_unet(subjects: list[str] | None = None, *, max_epochs: int | None = None,
               smoke: bool = False, write: bool = True, cfg: SegConfig | None = None):
    """Train the 2D U-Net and evaluate Dice on held-out subjects.

    smoke=True shrinks everything (1 epoch, tiny model) for the synthetic test.
    Returns a dict of results (per-split Dice, model path).
    """
    import torch
    from monai.losses import DiceLoss

    cfg = cfg or SegConfig()
    if smoke:
        cfg.max_epochs = 1
        cfg.channels = (8, 16)
        cfg.batch_size = 4
    if max_epochs is not None:
        cfg.max_epochs = max_epochs

    config.ensure_dirs()
    subjects = subjects or config.list_subjects()
    if len(subjects) < 3:
        raise RuntimeError(f"need >=3 subjects to split train/val/test, got {len(subjects)}")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)

    train_s, val_s, test_s = split_subjects(subjects, cfg)
    log.info("subject split: train=%s val=%s test=%s", train_s, val_s, test_s)
    print(f"Split by SUBJECT (no leakage): {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")

    train_ds = _build_dataset(train_s, cfg, rng)
    val_ds = _build_dataset(val_s, cfg, rng)
    if train_ds is None:
        raise RuntimeError("no training slices produced (no lesions found?)")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = build_model(cfg).to(device)
    # DiceLoss (sigmoid=True) optimises overlap directly - the right objective for
    # imbalanced segmentation (a plain pixel cross-entropy would favour predicting
    # all-background). We could add CE (DiceCELoss); Dice alone is the clear default.
    loss_fn = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training {cfg.max_epochs} epoch(s) on {len(train_ds)} slices "
          f"(device={device.type})...")
    for epoch in range(cfg.max_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_ds)
        val_dice = _evaluate(model, val_ds, device) if val_ds is not None else float("nan")
        print(f"  epoch {epoch + 1}/{cfg.max_epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.3f}")

    # --- final evaluation on held-out TEST subjects, per subject ------------
    per_subject = {}
    for sub in test_s:
        ds = _build_dataset([sub], cfg, rng)
        per_subject[sub] = _evaluate(model, ds, device) if ds is not None else float("nan")
    test_dice = float(np.nanmean(list(per_subject.values()))) if per_subject else float("nan")

    print("\n" + "-" * 60)
    print("SEGMENTATION (U-Net) RESULTS")
    print("-" * 60)
    print(f"held-out TEST subjects: {test_s}")
    for s, d in per_subject.items():
        print(f"  {s}: Dice={d:.3f}")
    print(f"mean test Dice = {test_dice:.3f}")
    if not smoke:
        print("NOTE: 30-subject prototype - Dice is illustrative, not clinical-grade.")
    print("-" * 60)

    result = {"test_dice": test_dice, "per_subject": per_subject,
              "train": train_s, "val": val_s, "test": test_s}

    if write:
        _save_model(model, cfg, result)
    return result


def _evaluate(model, dataset, device) -> float:
    """Mean Dice over a dataset (all its slices), in eval mode."""
    import torch

    if dataset is None or len(dataset) == 0:
        return float("nan")
    model.eval()
    dl = torch.utils.data.DataLoader(dataset, batch_size=16)
    dices = []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            dices.append(_dice_from_logits(model(xb), yb).cpu().numpy())
    return float(np.concatenate(dices).mean())


# ---------------------------------------------------------------------------
# Persistence + inference (reused by the GUI)
# ---------------------------------------------------------------------------

def _save_model(model, cfg: SegConfig, result: dict) -> Path:
    import torch

    path = config.UNET_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg),
                "test_dice": result.get("test_dice")}, path)
    log.info("saved model -> %s", path)
    print(f"saved model -> {path}")
    return path


def load_model(path: Path | None = None):
    """Load a trained model + its SegConfig for inference."""
    import torch

    path = path or config.UNET_MODEL_PATH
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SegConfig(**ckpt["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt.get("test_dice")


def predict_volume(flair_path: Path, model=None, cfg: SegConfig | None = None):
    """Predict a lesion mask for a whole FLAIR volume; return a NIfTI on the model grid.

    Used by the GUI. Steps mirror training exactly: canonical+2mm resample, z-score,
    slice, resize to SLICE_SIZE, predict each slice, resize the prediction back, and
    stack into a volume. Reproducing the training preprocessing at inference time is
    essential - a model only works on inputs shaped like what it trained on.
    """
    import torch

    if model is None:
        model, cfg, _ = load_model()
    cfg = cfg or SegConfig()

    flair_img = _canonical_iso(image_utils.load_nifti(flair_path), cfg.iso_mm, interpolation="continuous")
    flair = np.asarray(flair_img.dataobj, dtype=np.float32)
    brain = flair > np.percentile(flair[flair > 0], 20) if (flair > 0).any() else None
    norm = image_utils.zscore_normalise(flair, brain)

    H, W, n = flair.shape
    pred_vol = np.zeros_like(flair, dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for k in range(n):
            sl = norm[:, :, k]
            if not (sl != 0).any():
                continue  # skip empty slices
            resized = _resize_slice(sl, cfg.slice_size, order=1)
            x = torch.from_numpy(resized[None, None]).float()
            prob = torch.sigmoid(model(x))[0, 0].numpy()
            pred = (prob > 0.5).astype(np.float32)
            # resize prediction back to this slice's native (H, W); nearest for labels
            back = _resize_to(pred, (H, W))
            pred_vol[:, :, k] = (back > 0.5).astype(np.uint8)

    return nib.Nifti1Image(pred_vol, affine=flair_img.affine), flair_img


def _resize_to(arr2d, hw):
    from scipy.ndimage import zoom
    zy = hw[0] / arr2d.shape[0]
    zx = hw[1] / arr2d.shape[1]
    return zoom(arr2d, (zy, zx), order=0)
