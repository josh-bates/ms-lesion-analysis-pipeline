"""
tests/smoke_test.py
===================
A *fast* end-to-end smoke test that exercises every stage of the pipeline on
tiny SYNTHETIC NIfTI volumes - no real download, no multi-minute FSL runs, no GPU.

WHY a smoke test like this?
    The real dataset is several GB and FSL/U-Net stages take minutes each. That is
    a terrible feedback loop while writing code. Instead we generate a handful of
    small fake brains (see image_utils.make_synthetic_dataset) that carry the same
    file names and the same *physics* (bright FLAIR lesions, WM brighter than GM),
    then run each stage against them. If the plumbing is broken - a wrong path, a
    shape bug, a mislabelled column - this catches it in seconds.

    "Smoke test" = just switch it on and check nothing catches fire. It asserts that
    each stage RUNS and produces the right *kind* of output; it is deliberately NOT
    a correctness/accuracy test (a 48-voxel phantom can't tell you your Dice is good).

HOW IT GROWS
    The pipeline is built in stages. Each check below is guarded so that stages not
    yet implemented are reported as SKIP rather than failing the run. As each module
    lands, its check activates automatically.

Run it with:   .venv/bin/python -m tests.smoke_test
Exit code 0 = all implemented stages passed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Make 'src' importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Check:
    """Tiny test harness: records PASS / FAIL / SKIP and prints a readable log."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, name: str) -> None:
        self.passed += 1
        print(f"  [PASS] {name}")

    def skip(self, name: str, why: str) -> None:
        self.skipped += 1
        print(f"  [SKIP] {name}  ({why})")

    def fail(self, name: str, err: Exception) -> None:
        self.failed += 1
        print(f"  [FAIL] {name}")
        print("         " + "".join(traceback.format_exception_only(type(err), err)).strip())

    def summary(self) -> int:
        print("\n" + "=" * 60)
        print(f"SMOKE TEST: {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        print("=" * 60)
        return 1 if self.failed else 0


def main() -> int:
    chk = Check()
    print("=" * 60)
    print("MS LESION PIPELINE - SYNTHETIC SMOKE TEST")
    print("=" * 60)

    # ---- Build a throw-away synthetic dataset and redirect the pipeline at it ----
    # We create everything under one temp directory so nothing pollutes the project.
    tmp = Path(tempfile.mkdtemp(prefix="ms_smoke_"))
    os.environ["MS_DATA_ROOT"] = str(tmp / "data" / "cross_sectional")

    # Import AFTER setting MS_DATA_ROOT so config picks it up.
    from src import config, image_utils

    # Point derivatives/outputs/models at the temp dir too, so a smoke run never
    # overwrites real results.
    config.DERIVATIVES_DIR = tmp / "derivatives"
    config.OUTPUTS_DIR = tmp / "outputs"
    config.MODELS_DIR = tmp / "models"
    config.LOGS_DIR = tmp / "logs"
    config.QC_REPORT_CSV = config.OUTPUTS_DIR / "qc_report.csv"
    config.DEMOGRAPHICS_CSV = config.OUTPUTS_DIR / "demographics.csv"
    config.IMAGING_METRICS_CSV = config.OUTPUTS_DIR / "imaging_metrics.csv"
    config.UNET_MODEL_PATH = config.MODELS_DIR / "unet_lesion_2d.pt"
    config.ensure_dirs()

    print(f"\nScratch dir: {tmp}")

    # -------------------------------------------------------------------------
    # Stage 0: scaffolding - synthetic data generation + shared utilities
    # -------------------------------------------------------------------------
    print("\n[Stage 0] scaffold: synthetic data + utilities")
    n_subjects = 4
    try:
        image_utils.make_synthetic_dataset(tmp / "data" / "cross_sectional", n_subjects=n_subjects)
        subs = config.list_subjects()
        assert len(subs) == n_subjects, f"expected {n_subjects} subjects, found {subs}"
        chk.ok(f"generated {n_subjects} synthetic subjects: {subs}")
    except Exception as e:  # noqa: BLE001
        chk.fail("synthetic dataset generation", e)
        return chk.summary()  # nothing else can run without data

    try:
        files = config.subject_files(subs[0])
        for key in ("FLAIR", "T1W", "T2W", "lesion_mask", "brainmask"):
            assert files[key].exists(), f"missing {key} for {subs[0]}"
        chk.ok("expected files present for a subject")
    except Exception as e:  # noqa: BLE001
        chk.fail("subject file layout", e)

    try:
        import numpy as np
        vol = image_utils.load_data(files["FLAIR"])
        mask = image_utils.load_data(files["brainmask"]).astype(bool)
        z = image_utils.zscore_normalise(vol, mask)
        inside = z[mask]
        assert abs(inside.mean()) < 1e-3, f"z-score mean not ~0: {inside.mean()}"
        assert abs(inside.std() - 1.0) < 1e-2, f"z-score std not ~1: {inside.std()}"
        assert np.all(z[~mask] == 0), "background should be zeroed after z-score"
        chk.ok("z-score normalisation gives mean~0/std~1 over brain")
    except Exception as e:  # noqa: BLE001
        chk.fail("z-score normalisation", e)

    # Report FSL availability (informational - not a failure if absent).
    try:
        from src import fsl_utils
        if fsl_utils.fsl_available():
            chk.ok("FSL detected on PATH (bet/flirt/fast available)")
        else:
            chk.skip("FSL availability", "FSL not found - FSL stages will be skipped")
    except Exception as e:  # noqa: BLE001
        chk.fail("FSL detection", e)

    # -------------------------------------------------------------------------
    # Stage 1: QC  (activates once src/qc.py exists)
    # -------------------------------------------------------------------------
    print("\n[Stage 1] QC")
    _run_stage(chk, "qc", lambda: _smoke_qc(config))

    # -------------------------------------------------------------------------
    # Stage 2: preprocessing (FSL) - only if FSL is present
    # -------------------------------------------------------------------------
    print("\n[Stage 2] preprocess")
    _run_stage(chk, "preprocess", lambda: _smoke_preprocess(config))

    # -------------------------------------------------------------------------
    # Stage 3: imaging metrics
    # -------------------------------------------------------------------------
    print("\n[Stage 3] metrics")
    _run_stage(chk, "metrics", lambda: _smoke_metrics(config))

    # -------------------------------------------------------------------------
    # Stage 4: U-Net segmentation (tiny, 1 epoch)
    # -------------------------------------------------------------------------
    print("\n[Stage 4] segment (U-Net)")
    _run_stage(chk, "segment", lambda: _smoke_segment(config))

    # -------------------------------------------------------------------------
    # Stage 5: disability models
    # -------------------------------------------------------------------------
    print("\n[Stage 5] predict_disability")
    _run_stage(chk, "predict_disability", lambda: _smoke_disability(config))

    # -------------------------------------------------------------------------
    # Stage 6: GUI import / core function (no server launch)
    # -------------------------------------------------------------------------
    print("\n[Stage 6] app (GUI)")
    _run_stage(chk, "app", lambda: _smoke_app(config))

    return chk.summary()


# ---------------------------------------------------------------------------
# Per-stage smoke bodies. Each raises NotImplementedError until its module lands;
# _run_stage turns that into a SKIP so the overall run stays green as we build up.
# ---------------------------------------------------------------------------

def _run_stage(chk: "Check", name: str, body) -> None:
    try:
        body()
    except _NotReady as nr:
        chk.skip(name, str(nr))
    except Exception as e:  # noqa: BLE001
        chk.fail(name, e)
    else:
        chk.ok(f"{name} stage ran on synthetic data")


class _NotReady(Exception):
    """Raised by a stage body when its module is not implemented yet."""


def _import_or_skip(modname: str):
    try:
        __import__(modname)
    except ModuleNotFoundError as e:
        # Only treat *this* module being missing as 'not ready'; re-raise real
        # missing-dependency errors so we don't hide a broken import.
        if modname.split(".")[-1] in str(e):
            raise _NotReady(f"{modname} not implemented yet")
        raise
    import importlib
    return importlib.import_module(modname)


def _smoke_qc(config) -> None:
    qc = _import_or_skip("src.qc")
    df = qc.run_qc(write=True)
    assert len(df) == len(config.list_subjects()), "QC row count != subject count"
    assert config.QC_REPORT_CSV.exists(), "qc_report.csv not written"


def _smoke_preprocess(config) -> None:
    from src import fsl_utils
    if not fsl_utils.fsl_available():
        raise _NotReady("FSL not available")
    pre = _import_or_skip("src.preprocess")
    subs = config.list_subjects()
    pre.preprocess_subject(subs[0])


def _smoke_metrics(config) -> None:
    metrics = _import_or_skip("src.metrics")
    df = metrics.run_metrics(write=True)
    assert len(df) >= 1, "no metrics rows produced"


def _smoke_segment(config) -> None:
    seg = _import_or_skip("src.segment")
    # A 1-epoch run on the phantom just proves the training loop + I/O work.
    seg.train_unet(max_epochs=1, smoke=True)


def _smoke_disability(config) -> None:
    dis = _import_or_skip("src.predict_disability")
    dis.run_disability_models(smoke=True)


def _smoke_app(config) -> None:
    app = _import_or_skip("app")
    # Just prove the analysis function runs on one uploaded file - do NOT launch
    # the web server inside the test.
    subs = config.list_subjects()
    flair = config.subject_files(subs[0])["FLAIR"]
    app.analyze_scan(str(flair))


if __name__ == "__main__":
    raise SystemExit(main())
