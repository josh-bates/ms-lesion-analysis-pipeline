"""
fsl_utils.py
============
Thin, *logged* wrappers around the FSL command-line tools we use:
    bet      - Brain Extraction Tool  (skull-stripping)
    flirt    - linear (rigid/affine) registration
    fast     - tissue segmentation into GM / WM / CSF
    fslmaths - image arithmetic (apply masks, threshold, ...)
    fslstats - report simple statistics from an image

WHY call FSL through subprocess instead of a Python re-implementation?
    FSL is the *industry-standard* neuroimaging toolkit. A research team wants the
    exact, citable, validated algorithm - not a hand-rolled skull-strip. So we shell
    out to the real tools and treat Python as the orchestrator. The cost is that we
    must find FSL on PATH and capture its output; that is exactly what this module
    handles, once, for everybody.

Every command we run is logged (to stdout and to logs/fsl_commands.log) so the run
is fully auditable - you can copy any logged line into a terminal and reproduce it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import config

log = logging.getLogger("fsl")


# ---------------------------------------------------------------------------
# Finding FSL
# ---------------------------------------------------------------------------
# On this machine FSL is set up in ~/.profile, which only runs for *login* shells.
# A Python process launched by an IDE / Gradio / pytest may NOT have FSLDIR set or
# the FSL bin dir on PATH. So we locate FSL ourselves and, if needed, repair the
# environment for the current process. This is the kind of "works on my machine"
# gap that bites real pipelines, so we handle it explicitly and loudly.

def _candidate_fsl_dirs() -> list[Path]:
    """Common install locations to probe for FSLDIR."""
    cands = []
    if os.environ.get("FSLDIR"):
        cands.append(Path(os.environ["FSLDIR"]))
    cands += [
        Path.home() / "fsl",
        Path("/usr/local/fsl"),
        Path("/usr/share/fsl"),
        Path("/opt/fsl"),
    ]
    return cands


def find_fsl() -> Path | None:
    """Return a valid FSLDIR (a dir whose share/fsl/bin/bet exists), or None."""
    for d in _candidate_fsl_dirs():
        if d and (d / "share" / "fsl" / "bin" / "bet").exists():
            return d
        if d and (d / "bin" / "bet").exists():  # older FSL layout
            return d
    return None


def setup_fsl_env() -> bool:
    """Ensure FSLDIR is set and the FSL bin dir is on PATH for THIS process.

    Returns True if FSL is available afterwards, False otherwise. Safe to call
    many times. We do NOT raise on failure: several stages (QC, metrics, the
    U-Net) do not need FSL, and we would rather run those than abort the whole
    pipeline just because FSL is missing.
    """
    # Already usable?
    if shutil.which("bet"):
        return True

    fsldir = find_fsl()
    if fsldir is None:
        log.warning("FSL not found in any known location; FSL stages will be skipped.")
        return False

    # Repair the environment the same way ${FSLDIR}/etc/fslconf/fsl.sh would.
    os.environ["FSLDIR"] = str(fsldir)
    os.environ.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")  # make outputs .nii.gz
    for bindir in (fsldir / "share" / "fsl" / "bin", fsldir / "bin"):
        if bindir.is_dir():
            os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    ok = shutil.which("bet") is not None
    if ok:
        log.info("FSL configured: FSLDIR=%s", fsldir)
    return ok


def fsl_available() -> bool:
    """Convenience predicate used by callers to decide whether to run FSL stages."""
    return setup_fsl_env()


# ---------------------------------------------------------------------------
# Running commands with logging
# ---------------------------------------------------------------------------
_LOG_FILE_HANDLER_ADDED = False


def _ensure_command_log() -> None:
    """Attach a file handler that records every FSL command to logs/fsl_commands.log."""
    global _LOG_FILE_HANDLER_ADDED
    if _LOG_FILE_HANDLER_ADDED:
        return
    config.ensure_dirs()
    fh = logging.FileHandler(config.LOGS_DIR / "fsl_commands.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logging.getLogger("fsl").addHandler(fh)
    _LOG_FILE_HANDLER_ADDED = True


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run one external command, logging the exact command line and its duration.

    Parameters
    ----------
    cmd   : the command as a list of strings (never a single shell string - that
            avoids shell-injection surprises and quoting bugs).
    check : if True, raise RuntimeError on a non-zero exit code.

    We log BEFORE running (so a hang is attributable to a specific command) and the
    elapsed time AFTER (useful when you're wondering why the pipeline is slow).
    """
    _ensure_command_log()
    setup_fsl_env()  # make sure FSL is on PATH even if caller forgot
    printable = " ".join(str(c) for c in cmd)
    log.info("RUN: %s", printable)
    start = time.time()
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    dt = time.time() - start
    log.info("   -> exit=%d  time=%.1fs", proc.returncode, dt)
    if proc.stdout.strip():
        log.debug("   stdout: %s", proc.stdout.strip())
    if proc.returncode != 0:
        log.error("   stderr: %s", proc.stderr.strip())
        if check:
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}): {printable}\n{proc.stderr.strip()}"
            )
    return proc


# ---------------------------------------------------------------------------
# High-level FSL operations
# ---------------------------------------------------------------------------
# Each function returns the Path(s) it produced so callers can chain steps.

def bet(in_img: Path, out_brain: Path, *, frac: float = 0.4, robust: bool = True) -> Path:
    """Skull-strip `in_img`, writing brain-extracted image + a binary brain mask.

    frac (-f): fractional intensity threshold. Lower => keeps MORE brain (larger
    mask). We default to 0.4 rather than FSL's 0.5 because FLAIR brains are often
    slightly over-stripped at 0.5; 0.4 is a common, conservative choice you would
    then eyeball. `-R` (robust) re-estimates the brain centre iteratively and is
    more reliable on whole-head images at the cost of some runtime.

    FSL writes the mask as <out_brain>_mask.nii.gz by passing -m.
    """
    out_brain = Path(out_brain)
    cmd = ["bet", str(in_img), str(out_brain), "-m", "-f", str(frac)]
    if robust:
        cmd.append("-R")
    run_cmd(cmd)
    return out_brain


def brain_mask_path(bet_out: Path) -> Path:
    """Given the 'out_brain' base bet() was called with, return the mask path.

    bet appends '_mask' before the .nii.gz suffix, e.g.
    T1_brain.nii.gz -> T1_brain_mask.nii.gz.
    """
    bet_out = Path(bet_out)
    name = bet_out.name
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return bet_out.with_name(name[: -len(suf)] + "_mask" + suf)
    return bet_out.with_name(bet_out.stem + "_mask.nii.gz")


def flirt_rigid(moving: Path, reference: Path, out_img: Path, out_mat: Path) -> Path:
    """Rigidly register `moving` to `reference` (6 degrees of freedom).

    We use dof=6 (rigid: rotations + translations only) because T1 and FLAIR are
    the SAME patient in one session - only head position differs, so shape must be
    preserved. (Alternatives: dof=12 affine would also let the image scale/shear,
    which is wrong for within-subject same-day scans and could hide real anatomy.)
    `-cost corratio` (correlation ratio) is FLIRT's default and works well across
    different MRI contrasts (T1 vs FLAIR have different tissue intensities).
    """
    cmd = [
        "flirt", "-in", str(moving), "-ref", str(reference),
        "-out", str(out_img), "-omat", str(out_mat),
        "-dof", "6", "-cost", "corratio",
    ]
    run_cmd(cmd)
    return Path(out_img)


def fast_segment(brain_img: Path, out_basename: Path, *, n_classes: int = 3) -> Path:
    """Segment a *brain-extracted* image into tissue classes with FSL FAST.

    FAST expects the skull already removed (it models 3 tissue intensities, not
    skull/scalp), which is why preprocess runs bet first. With n_classes=3 FAST
    produces partial-volume maps pve_0/1/2 that correspond to CSF, GM, WM ordered
    by mean intensity on a T1 (CSF darkest, WM brightest). Output files are named
    <out_basename>_pve_0.nii.gz etc.
    """
    cmd = [
        "fast", "-t", "1",            # -t 1 => input is a T1-weighted image
        "-n", str(n_classes),         # number of tissue classes
        "-g",                          # also write binary segmentation per class
        "-o", str(out_basename),      # output basename
        str(brain_img),
    ]
    run_cmd(cmd)
    return Path(out_basename)


def fslmaths_mask(in_img: Path, mask: Path, out_img: Path) -> Path:
    """Multiply `in_img` by a binary `mask` (i.e. apply a brain mask)."""
    run_cmd(["fslmaths", str(in_img), "-mas", str(mask), str(out_img)])
    return Path(out_img)


def fslstats_volume(mask: Path) -> float:
    """Return the volume (mm^3) of the non-zero region of a mask via fslstats -V.

    fslstats -V prints '<n_voxels> <volume_mm3>'; we take the second number.
    Provided as a cross-check against our own numpy-based volume in metrics.py -
    two independent tools agreeing is a cheap correctness guarantee.
    """
    proc = run_cmd(["fslstats", str(mask), "-V"])
    parts = proc.stdout.split()
    return float(parts[1]) if len(parts) >= 2 else 0.0
