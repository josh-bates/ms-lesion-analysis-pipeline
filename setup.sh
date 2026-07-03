#!/usr/bin/env bash
# =============================================================================
# setup.sh - one-shot environment setup for the MS lesion pipeline.
#
# What it does:
#   1. creates a Python virtual environment in ./.venv
#   2. installs everything in requirements.txt
#   3. checks that FSL is discoverable and prints a friendly report
#
# Usage:   ./setup.sh          (from the project root)
# Then:    source .venv/bin/activate
#
# Design note: we keep this a plain bash script (not a Makefile / poetry / conda)
# because the target environment is "Ubuntu on WSL with system Python + FSL", and
# a readable shell script is the least surprising thing on that platform.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
VENV=.venv

echo "==> 1/3  Creating virtual environment in ${VENV}"
if [ ! -d "${VENV}" ]; then
    # Prefer the stdlib venv. On some Debian/WSL setups python3-venv/ensurepip is
    # missing; fall back to the 'virtualenv' package (installed to the user site),
    # which bundles its own pip and needs no root.
    if ${PY} -m venv "${VENV}" 2>/dev/null; then
        echo "    created with python -m venv"
    else
        echo "    python -m venv unavailable (missing ensurepip); using virtualenv"
        ${PY} -m pip install --user --break-system-packages -q virtualenv
        ${PY} -m virtualenv "${VENV}"
    fi
else
    echo "    ${VENV} already exists - reusing"
fi

echo "==> 2/3  Installing Python dependencies (this can take a few minutes)"
"${VENV}/bin/python" -m pip install --upgrade -q pip
# torch's CPU wheels are on a dedicated index; install it there FIRST, then the
# rest from PyPI (monai etc. are not on the torch index). On a CUDA machine you can
# drop the --index-url line to get a GPU-enabled torch build.
"${VENV}/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"${VENV}/bin/pip" install -r requirements.txt

echo "==> 3/3  Checking for FSL"
# FSL is usually configured only in login shells (~/.profile), so a plain script
# may not see it. Probe the common locations the pipeline itself probes.
FSL_FOUND=""
for d in "${FSLDIR:-}" "$HOME/fsl" /usr/local/fsl /usr/share/fsl /opt/fsl; do
    if [ -n "$d" ] && { [ -x "$d/share/fsl/bin/bet" ] || [ -x "$d/bin/bet" ]; }; then
        FSL_FOUND="$d"; break
    fi
done
if [ -n "$FSL_FOUND" ]; then
    echo "    FSL found at: ${FSL_FOUND}"
else
    echo "    WARNING: FSL not found. QC / metrics / U-Net / GUI still work,"
    echo "             but the preprocessing stage (bet/flirt/fast) will be skipped."
    echo "             Install FSL from https://fsl.fmrib.ox.ac.uk and re-run."
fi

echo ""
echo "Done. Next steps:"
echo "    source ${VENV}/bin/activate"
echo "    python -m tests.smoke_test        # fast synthetic end-to-end check"
echo "    python run_pipeline.py --help     # run the real pipeline"
