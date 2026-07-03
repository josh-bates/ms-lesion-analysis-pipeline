#!/usr/bin/env python3
"""
run_pipeline.py
===============
One orchestrator to run the whole MS lesion analysis end to end on the REAL
open_ms_data dataset, stage by stage:

    1. qc        - inventory + quality control, build demographics.csv
    2. preprocess- FSL brain extraction, registration, tissue segmentation
    3. metrics   - lesion volume / count / load / intensity per subject
    4. segment   - train + evaluate the 2D U-Net lesion segmenter
    5. disability- classify lesion burden + estimate EDSS from imaging metrics

Design choice: a thin CLI (argparse) that just calls each module's top-level
function in order. All the real logic lives in the src/ modules so it is unit-
testable and reusable by the GUI; this file only sequences them and prints
progress. You can run a single stage with --stage or --only.

    python run_pipeline.py                 # run every stage in order
    python run_pipeline.py --only qc       # run just QC
    python run_pipeline.py --from metrics  # run metrics onward
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Ordered list of pipeline stages. Each entry: (name, callable-returning-None).
# Callables are imported lazily inside _stage_fn so that a missing heavy dependency
# (e.g. torch) never blocks unrelated stages.
STAGE_ORDER = ["qc", "preprocess", "metrics", "segment", "disability"]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _stage_fn(name: str):
    """Lazily import and return the entry-point callable for a stage."""
    if name == "qc":
        from src import qc
        return qc.run_qc
    if name == "preprocess":
        from src import preprocess
        return preprocess.run_preprocess
    if name == "metrics":
        from src import metrics
        return metrics.run_metrics
    if name == "segment":
        from src import segment
        return segment.train_unet
    if name == "disability":
        from src import predict_disability
        return predict_disability.run_disability_models
    raise ValueError(f"unknown stage: {name}")


def _selected_stages(args) -> list[str]:
    if args.only:
        return [args.only]
    stages = STAGE_ORDER
    if args.from_stage:
        i = STAGE_ORDER.index(args.from_stage)
        stages = STAGE_ORDER[i:]
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=STAGE_ORDER, help="run only this stage")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_ORDER, help="run from this stage onward")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    from src import config
    config.ensure_dirs()

    stages = _selected_stages(args)
    print("=" * 64)
    print("MS LESION ANALYSIS PIPELINE")
    print(f"data root : {config.DATA_ROOT}  (variant: {config.DATA_VARIANT})")
    print(f"subjects  : {len(config.list_subjects())} found")
    print(f"stages    : {stages}")
    print("=" * 64)

    for name in stages:
        print(f"\n----- STAGE: {name} -----")
        start = time.time()
        try:
            fn = _stage_fn(name)
        except ImportError as e:
            print(f"[skip] {name}: module not implemented yet ({e})")
            continue
        fn()
        print(f"----- {name} done in {time.time() - start:.1f}s -----")

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
