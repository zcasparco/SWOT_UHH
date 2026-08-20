#!/usr/bin/env python3
"""
Combine the per-file <stem>_alias.pkl outputs from run_aliased_batch.py into
a single all_segments dict, the same shape produced by the notebook:

    all_segments = {"left": [...], "right": [...]}

Run after the sbatch job (or array) has finished:

    python combine_aliased_results.py --output-dir /path/to/results/swot_alias \
        --out /path/to/results/all_segments.pkl

Reads the manifest.csv written by run_aliased_batch.py so it only touches
files that actually succeeded.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Directory containing the per-file *_alias.pkl outputs and manifest.csv.")
    p.add_argument("--manifest", type=Path, default=None,
                    help="Defaults to <output-dir>/manifest.csv.")
    p.add_argument("--out", type=Path, required=True,
                    help="Path to write the combined all_segments pickle.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    manifest_path = args.manifest or (args.output_dir / "manifest.csv")

    all_segments = {"left": [], "right": []}

    n_files = 0
    n_missing = 0

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["error"]:
                continue

            pkl_path = Path(row["output"])
            if not pkl_path.exists():
                n_missing += 1
                continue

            with open(pkl_path, "rb") as pf:
                payload = pickle.load(pf)

            for seg in payload["segments"]:
                all_segments[seg.swath].append(seg)

            n_files += 1

    print(f"Combined {n_files} files ({n_missing} listed as OK but missing on disk).")
    print(f'left : {len(all_segments["left"])} segments')
    print(f'right: {len(all_segments["right"])} segments')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(all_segments, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
