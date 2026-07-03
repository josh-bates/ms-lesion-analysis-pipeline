"""
app.py  -  Stage 6: Gradio GUI for a single-scan MS lesion demo
==============================================================
A small web UI you can screen-share. Upload ONE FLAIR .nii.gz and it:

  1. shows the middle axial slice,
  2. runs brain extraction (FSL bet),
  3. overlays the U-Net predicted lesion mask,
  4. reports quantitative metrics (lesion volume, count, mean FLAIR z), and
  5. shows the predicted lesion-burden class and estimated EDSS.

Design choices:
  * Everything the GUI needs at runtime is a saved artefact from earlier stages
    (the U-Net weights, the disability models). The GUI itself trains nothing, so it
    is fast and works on a single uploaded scan WITHOUT the full dataset present -
    exactly the "demo on one scan" requirement.
  * The heavy analysis lives in analyze_scan(), a plain function with no Gradio in
    its signature, so it is unit/smoke-testable without launching a server.
  * gradio is imported lazily (inside build_demo/main) so importing this module - or
    calling analyze_scan from the smoke test - does not require gradio to be present.
  * Every step degrades gracefully: no FSL -> threshold-based brain mask with a note;
    no trained U-Net -> skip the overlay with a note. The UI never hard-crashes on a
    missing artefact, which matters when demoing on an unfamiliar machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: render figures to arrays, no display needed
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src import config, image_utils, fsl_utils, metrics as metrics_mod
from src import segment
from src import predict_disability


# ---------------------------------------------------------------------------
# Brain extraction for an uploaded scan
# ---------------------------------------------------------------------------

def _brain_mask_for(flair_img: nib.Nifti1Image, workdir: Path) -> tuple[np.ndarray, str]:
    """Return (brain_mask, note). Uses FSL bet if available, else a threshold fallback.

    We operate on the already-downsampled canonical FLAIR that the U-Net used, so the
    mask lines up with the displayed slice and the prediction.
    """
    data = np.asarray(flair_img.dataobj, dtype=np.float32)
    if fsl_utils.fsl_available():
        tmp_in = workdir / "flair_for_bet.nii.gz"
        nib.save(flair_img, str(tmp_in))
        out = workdir / "brain.nii.gz"
        try:
            fsl_utils.bet(tmp_in, out, frac=0.4, robust=True)
            mask = np.asarray(image_utils.load_nifti(fsl_utils.brain_mask_path(out)).dataobj) > 0
            return mask.astype(bool), "FSL bet"
        except Exception as e:  # noqa: BLE001
            return _threshold_brain(data), f"threshold fallback (bet failed: {e})"
    return _threshold_brain(data), "threshold fallback (FSL not available)"


def _threshold_brain(data: np.ndarray) -> np.ndarray:
    """Crude brain mask: everything above the 20th percentile of positive intensity."""
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros_like(data, dtype=bool)
    return data > np.percentile(pos, 20)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_overlay(slice_img: np.ndarray, mask_slice: np.ndarray | None,
                    brain_slice: np.ndarray | None, title: str) -> np.ndarray:
    """Render a grayscale slice with optional red lesion overlay to an RGB array."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    # .T and origin='lower' give the conventional radiological upright view.
    ax.imshow(slice_img.T, cmap="gray", origin="lower")
    if brain_slice is not None:
        ax.contour(brain_slice.T, levels=[0.5], colors="deepskyblue", linewidths=0.6)
    if mask_slice is not None and mask_slice.any():
        overlay = np.ma.masked_where(~mask_slice.T.astype(bool), mask_slice.T)
        ax.imshow(overlay, cmap="autumn", alpha=0.6, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


# ---------------------------------------------------------------------------
# The core analysis (server-independent, testable)
# ---------------------------------------------------------------------------

def analyze_scan(flair_path: str) -> dict:
    """Analyse one uploaded FLAIR volume end to end.

    Returns a dict with: 'slice_rgb', 'overlay_rgb' (numpy images), 'report' (markdown
    string), and 'metrics' (the computed feature dict). Safe to call from tests.
    """
    flair_path = str(flair_path)
    notes: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)

        # --- U-Net prediction (also gives us the canonical 2mm FLAIR to display) ---
        model_available = config.UNET_MODEL_PATH.exists()
        if model_available:
            model, seg_cfg, test_dice = segment.load_model()
            pred_img, flair_img = segment.predict_volume(flair_path, model=model, cfg=seg_cfg)
            pred = np.asarray(pred_img.dataobj).astype(bool)
        else:
            # No trained model: still show the scan, but no lesion overlay/metrics.
            flair_img = segment._canonical_iso(
                image_utils.load_nifti(flair_path), segment.SEG_ISO_MM, interpolation="continuous")
            pred = np.zeros(flair_img.shape, dtype=bool)
            test_dice = None
            notes.append("No trained U-Net found - run Stage 4 (segment) to enable prediction.")

        flair = np.asarray(flair_img.dataobj, dtype=np.float32)

        # --- brain extraction ---
        brain, bet_note = _brain_mask_for(flair_img, workdir)
        notes.append(f"Brain extraction: {bet_note}.")

        # --- metrics from the PREDICTED mask ---
        vox_vol = image_utils.voxel_volume_mm3(flair_img)
        lesion_vox = int(pred.sum())
        z = image_utils.zscore_normalise(flair, brain)
        metrics = {
            "lesion_volume_mm3": round(lesion_vox * vox_vol, 1),
            "lesion_count": metrics_mod.count_lesions(pred) if lesion_vox else 0,
            "mean_flair_lesion_z": round(float(z[pred].mean()), 3) if lesion_vox else float("nan"),
        }
        brain_vol_cm3 = round(int(brain.sum()) * vox_vol / 1000, 1)

        # --- clinical estimates from the disability models ---
        payload = predict_disability.load_disability_models()
        clinical = predict_disability.predict_from_metrics(metrics, payload)
        if payload is None:
            notes.append("No disability models found - run Stage 5 to enable EDSS/burden.")

        # --- images: middle axial slice + overlay ---
        mid = flair.shape[2] // 2
        # pick the slice with the most predicted lesion if any, else the middle
        if lesion_vox:
            per_slice = pred.reshape(-1, pred.shape[2]).sum(axis=0)
            mid = int(np.argmax(per_slice))
        slice_rgb = _render_overlay(flair[:, :, mid], None, brain[:, :, mid],
                                    f"FLAIR (axial slice {mid})")
        overlay_rgb = _render_overlay(flair[:, :, mid], pred[:, :, mid], brain[:, :, mid],
                                      "Predicted lesions (red)")

    report = _format_report(metrics, brain_vol_cm3, clinical, test_dice, notes)
    return {"slice_rgb": slice_rgb, "overlay_rgb": overlay_rgb,
            "report": report, "metrics": metrics}


def _format_report(metrics, brain_vol_cm3, clinical, test_dice, notes) -> str:
    lv = metrics["lesion_volume_mm3"]
    lines = [
        "### Quantitative results",
        "",
        f"- **Lesion volume:** {lv:,.0f} mm³ ({lv/1000:.2f} cm³)",
        f"- **Lesion count:** {metrics['lesion_count']}",
        f"- **Mean FLAIR z in lesions:** {metrics['mean_flair_lesion_z']}"
        "  *(SDs above typical brain tissue)*",
        f"- **Brain volume (extracted):** {brain_vol_cm3:.0f} cm³",
        "",
        "### Clinical estimates (from imaging)",
        f"- **Lesion-burden class:** {clinical['burden_class'] or 'n/a'}",
        f"- **Estimated EDSS:** {clinical['estimated_edss'] if clinical['estimated_edss'] is not None else 'n/a'}",
    ]
    if test_dice is not None:
        lines += ["", f"*(U-Net held-out test Dice ≈ {test_dice:.3f}.)*"]
    lines += ["", "---",
              "*Prototype on 30 subjects - estimates are illustrative, not diagnostic.*"]
    if notes:
        lines += ["", "**Notes:** " + " ".join(notes)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI (built lazily)
# ---------------------------------------------------------------------------

def _gradio_fn(file_obj):
    """Adapter: Gradio hands us an uploaded file; return the three UI outputs."""
    if file_obj is None:
        return None, None, "Please upload a FLAIR .nii.gz file."
    res = analyze_scan(file_obj.name if hasattr(file_obj, "name") else file_obj)
    return res["slice_rgb"], res["overlay_rgb"], res["report"]


def build_demo():
    import gradio as gr

    with gr.Blocks(title="MS Lesion Analysis (prototype)") as demo:
        gr.Markdown(
            "# MS Lesion Analysis — prototype\n"
            "Upload a **FLAIR** `.nii.gz`. The app skull-strips it, runs a 2D U-Net to "
            "segment lesions, and reports imaging metrics plus a predicted lesion-burden "
            "class and estimated EDSS.\n\n"
            "*Research/education prototype trained on 30 subjects — not for clinical use.*"
        )
        with gr.Row():
            inp = gr.File(label="FLAIR volume (.nii.gz)", file_types=[".gz", ".nii"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            out_slice = gr.Image(label="Middle axial FLAIR (brain outline in blue)")
            out_overlay = gr.Image(label="Predicted lesions (red overlay)")
        out_report = gr.Markdown()
        btn.click(_gradio_fn, inputs=inp, outputs=[out_slice, out_overlay, out_report])
    return demo


def main():
    demo = build_demo()
    # share=False keeps it local; set server_name='0.0.0.0' to expose on a LAN.
    demo.launch()


if __name__ == "__main__":
    main()
