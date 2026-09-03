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

MAX_SUITESPARSE_RHS_PER_MATRIX = 100


def _paths(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return [Path(path) for path in sorted(glob.glob("results-*.jsonl"))]


def _descriptor(solver: str, record: dict[str, Any]) -> str:
    source_name = record.get(
        "source_name",
        f"{record['matrix_group']}/{record['matrix_name']}",
    )
    name = json.dumps(source_name)
    nnz = int(record["nnz"])
    rhs_index = record.get("rhs_index")
    rhs_arg = (
        f", rhs_index={int(rhs_index)}"
        if record.get("rhs_kind") == "suitesparse" and rhs_index is not None
        else ""
    )
    if solver in {"jacobi_cg", "block_jacobi_cg"}:
        return f'PreconditionedCGDataset({name}, "unknown"{rhs_arg})'
    return f"{DATASET_CLASSES[solver]}({name}, nnz={nnz}{rhs_arg})"


def _passes_rhs_filter(record: dict[str, Any]) -> bool:
    if record.get("rhs_kind") != "suitesparse":
        return True
    return int(record["rhs_index"]) < MAX_SUITESPARSE_RHS_PER_MATRIX


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print SAPS dataset descriptors for converged scraper results."
    )
    parser.add_argument("jsonl", nargs="*", help="JSONL files. Defaults to results-*.jsonl.")
    args = parser.parse_args()

    descriptors: dict[str, list[tuple[str, str]]] = {
        solver: [] for solver in DATASET_CLASSES
    }
    seen: set[tuple[str, str, Any]] = set()

    for path in _paths(args.jsonl):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not _passes_rhs_filter(record):
                    continue
                for solver, result in record.get("results", {}).items():
                    source_name = record.get(
                        "source_name",
                        f"{record['matrix_group']}/{record['matrix_name']}",
                    )
                    key = (
                        solver,
                        source_name,
                        record.get("rhs_index"),
                    )
                    if key in seen or result.get("converged") is not True:
                        continue
                    seen.add(key)
                    label = source_name
                    if record.get("rhs_kind") == "suitesparse":
                        label = f"{label} rhs{record.get('rhs_index')}"
                    descriptors[solver].append((label, _descriptor(solver, record)))

    for solver, items in descriptors.items():
        print(f"{solver}_datasets = [")
        for _label, descriptor in sorted(items):
            print(f"    {descriptor},")
        print("]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
