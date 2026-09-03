#!/usr/bin/env python3
"""Print SAPS dataset descriptors from scraper JSONL output."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


DATASET_CLASSES = {
    "jacobi": "JacobiDataset",
    "cg": "CGDataset",
    "jacobi_cg": "PreconditionedCGDataset",
    "block_jacobi_cg": "PreconditionedCGDataset",
    "lsqr": "LSQRDataset",
    "gmres": "GMRESDataset",
}


def _paths(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return [Path(path) for path in sorted(glob.glob("results-*.jsonl"))]


def _descriptor(solver: str, record: dict[str, Any]) -> str:
    name = json.dumps(record["matrix_name"])
    nnz = int(record["nnz"])
    if solver in {"jacobi_cg", "block_jacobi_cg"}:
        return f'PreconditionedCGDataset({name}, "unknown"),'
    return f"{DATASET_CLASSES[solver]}({name}, nnz={nnz}),"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print SAPS dataset descriptors for converged scraper results."
    )
    parser.add_argument("jsonl", nargs="*", help="JSONL files. Defaults to results-*.jsonl.")
    args = parser.parse_args()

    descriptors: dict[str, list[tuple[str, str]]] = {
        solver: [] for solver in DATASET_CLASSES
    }
    seen: set[tuple[str, str]] = set()

    for path in _paths(args.jsonl):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = f"{record['matrix_group']}/{record['matrix_name']}"
                for solver, result in record.get("results", {}).items():
                    key = (solver, record["matrix_name"])
                    if key in seen or result.get("converged") is not True:
                        continue
                    seen.add(key)
                    descriptors[solver].append((label, _descriptor(solver, record)))

    for solver, items in descriptors.items():
        print(f"# {solver} ({len(items)})")
        for label, descriptor in sorted(items):
            print(f"{descriptor}  # {label}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
