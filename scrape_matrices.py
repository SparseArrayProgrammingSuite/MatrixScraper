#!/usr/bin/env python3
# ruff: noqa: E402
"""Probe SuiteSparse matrices by running SAPS solver benchmarks on them.

This script intentionally calls the SAPS benchmark implementations instead of
estimating spectral radii or condition numbers. Each output line is a JSON
record for one matrix with per-solver convergence results.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sps

import ssgetpy

SAPS_REPO_NAME = "SparseAutoschedulingBenchmark"
SCRIPT_DIR = Path(__file__).resolve().parent
SAPS_REPO_DIR = Path(
    os.environ.get("SAPS_REPO_DIR", SCRIPT_DIR.parent / SAPS_REPO_NAME)
)
SAPS_SRC_DIR = SAPS_REPO_DIR / "src"
SAPS_SCIPY_FRAMEWORK = SAPS_REPO_DIR / "frameworks" / "saps_scipy.py"


if (SAPS_SRC_DIR / "saps").exists():
    sys.path.insert(0, str(SAPS_SRC_DIR))

import saps.benchmarks.GMRES as saps_gmres
import saps.benchmarks.cg as saps_cg
import saps.benchmarks.jacobi as saps_jacobi
import saps.benchmarks.lsqr as saps_lsqr
import saps.benchmarks.preconditioned_cg as saps_preconditioned_cg
import saps.downloaders.suitesparse as saps_suitesparse_downloader
from binsparse.conversions import from_scipy
from saps.framework import load_framework
from saps.benchmarks.GMRES import GMRESBenchmark, GMRESDataset, GMRESGenerator
from saps.benchmarks.cg import CGBenchmark, CGDataset, CGGenerator
from saps.benchmarks.jacobi import JacobiBenchmark, JacobiDataset, JacobiGenerator
from saps.benchmarks.lsqr import LSQRBenchmark, LSQRDataset, LSQRGenerator
from saps.benchmarks.preconditioned_cg import (
    BlockJacobiCGGenerator,
    JacobiCGGenerator,
    JacobiPreconditionedCGBenchmark,
    PreconditionedCGBenchmark,
    PreconditionedCGDataset,
)


SOLVER_NAMES = (
    "jacobi",
    "cg",
    "jacobi_cg",
    "block_jacobi_cg",
    "lsqr",
    "gmres",
)

LSQR_MATRIX_KINDS = frozenset({"least squares problem"})

ACCEPTED_MATRIX_KINDS = frozenset(
    {
        "tomography problem",
        "thermal problem",
        "theoretical/quantum chemistry problem sequence",
        "theoretical/quantum chemistry problem",
        "subsequent theoretical/quantum chemistry problem",
        "subsequent structural problem",
        "subsequent semiconductor device problem",
        "subsequent power network problem",
        "subsequent optimization problem",
        "subsequent computational fluid dynamics problem",
        "subsequent circuit simulation problem",
        "subsequent 2d/3d problem",
        "structural problem sequence",
        "structural problem",
        "semiconductor process problem",
        "semiconductor device problem sequence",
        "semiconductor device problem",
        "robotics problem",
        "random 2d/3d problem",
        "power network problem sequence",
        "power network problem",
        "optimization problem sequence",
        "optimization problem",
        "optimal control problem",
        "model reduction problem",
        "materials problem",
        "frequency domain circuit simulation problem",
        "electromagnetics problem",
        "eigenvalue/model reduction problem",
        "economic problem",
        "data analytics problem",
        "computer vision problem",
        "computer graphics/vision problem",
        "computational fluid dynamics problem sequence",
        "computational fluid dynamics problem",
        "computational fluid dynamics",
        "computational chemistry problem",
        "circuit simulation problem sequence",
        "circuit simulation problem",
        "chemical process simulation problem sequence",
        "chemical process simulation problem",
        "chemical oceanography problem",
        "acoustics problem",
        "2d/3d problem sequence",
        "2d/3d problem",
    }
)


def _matrix_kind(matrix: Any) -> str:
    return str(getattr(matrix, "kind", "")).strip().lower()


def _load_scipy_framework() -> Any:
    if not SAPS_SCIPY_FRAMEWORK.exists():
        raise FileNotFoundError(
            "Could not find the SAPS SciPy framework. Set SAPS_REPO_DIR to the "
            f"{SAPS_REPO_NAME} checkout."
        )
    return load_framework(SAPS_SCIPY_FRAMEWORK)


@dataclass
class SolverSpec:
    benchmark: Any
    generator: Any
    dataset_cls: type[Any]
    square_required: bool = True
    residual_kind: str = "linear"
    tolerance: float = 1e-6
    max_iters: int = 100
    accepted_kinds: frozenset[str] = ACCEPTED_MATRIX_KINDS
    include_nnz: bool = True
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    restart_limit: int | None = None

    def accepts_matrix_kind(self, matrix_kind: str) -> bool:
        return matrix_kind in self.accepted_kinds

    def make_dataset(self, matrix_name: str, nnz: int) -> Any:
        kwargs = dict(self.dataset_kwargs)
        if self.include_nnz:
            kwargs["nnz"] = nnz
        return self.dataset_cls(matrix_name, **kwargs)

    def benchmark_meta(self, A: sps.spmatrix) -> dict[str, Any]:
        meta = {
            "tolerance": self.tolerance,
            "max_iters": self.max_iters,
        }
        if self.restart_limit is not None:
            meta["restart"] = min(self.restart_limit, A.shape[0])
        return meta

    def result_settings(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "max_iters": self.max_iters,
        }

    def skip_reason(self, matrix_kind: str, rows: int, cols: int) -> str | None:
        if not self.accepts_matrix_kind(matrix_kind):
            return f"{self.benchmark.name} does not accept matrix kind {matrix_kind!r}"
        if self.square_required and rows != cols:
            return f"{self.benchmark.name} requires a square matrix"
        return None


def _as_real_sparse_matrix(matrix: Any) -> sps.spmatrix:
    if not sps.issparse(matrix):
        raise TypeError("Expected SAPS to provide a SciPy sparse matrix")
    A = matrix
    if np.iscomplexobj(A.data):
        raise ValueError("SAPS solver benchmarks expect real-valued matrices")
    A = A.astype(np.float64).tocoo(copy=False)
    A.sum_duplicates()
    return A.tocsr()


@contextlib.contextmanager
def _saps_suitesparse_context(
    suitesparse_data_dir: Path,
    seed: int,
) -> Iterator[None]:
    original_fetches = {
        saps_cg: saps_cg.fetch_suitesparse_linear_system,
        saps_gmres: saps_gmres.fetch_suitesparse_linear_system,
        saps_jacobi: saps_jacobi.fetch_suitesparse_linear_system,
        saps_lsqr: saps_lsqr.fetch_suitesparse_linear_system,
        saps_preconditioned_cg: saps_preconditioned_cg.fetch_suitesparse_linear_system,
    }

    def _fetch_suitesparse_linear_system(source_name: str):
        A, b, meta = saps_suitesparse_downloader.load_suitesparse_matrix(
            source_name,
            data_dir=suitesparse_data_dir,
        )
        has_real_rhs = bool(meta["has_b_file"])
        if not has_real_rhs:
            b = saps_suitesparse_downloader.random_rhs_for_matrix(
                A.tocoo(),
                seed=seed,
            )
        return from_scipy(A), b, has_real_rhs

    for module in original_fetches:
        module.fetch_suitesparse_linear_system = _fetch_suitesparse_linear_system

    try:
        yield
    finally:
        for module, fetch in original_fetches.items():
            module.fetch_suitesparse_linear_system = fetch


SOLVERS = {
    "jacobi": SolverSpec(
        benchmark=JacobiBenchmark(),
        generator=JacobiGenerator(),
        dataset_cls=JacobiDataset,
        max_iters=1000,
    ),
    "cg": SolverSpec(
        benchmark=CGBenchmark(),
        generator=CGGenerator(),
        dataset_cls=CGDataset,
    ),
    "jacobi_cg": SolverSpec(
        benchmark=JacobiPreconditionedCGBenchmark(),
        generator=JacobiCGGenerator(),
        dataset_cls=PreconditionedCGDataset,
        include_nnz=False,
        dataset_kwargs={"condition_number": "unknown"},
    ),
    "block_jacobi_cg": SolverSpec(
        benchmark=PreconditionedCGBenchmark(),
        generator=BlockJacobiCGGenerator(),
        dataset_cls=PreconditionedCGDataset,
        include_nnz=False,
        dataset_kwargs={"condition_number": "unknown"},
    ),
    "lsqr": SolverSpec(
        benchmark=LSQRBenchmark(),
        generator=LSQRGenerator(),
        dataset_cls=LSQRDataset,
        square_required=False,
        residual_kind="least_squares",
        accepted_kinds=LSQR_MATRIX_KINDS,
    ),
    "gmres": SolverSpec(
        benchmark=GMRESBenchmark(),
        generator=GMRESGenerator(),
        dataset_cls=GMRESDataset,
        restart_limit=50,
    ),
}


def _safe_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _linear_residual(
    A: sps.spmatrix,
    b: np.ndarray,
    x: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    residual = b - A @ x
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    relative_residual = residual_norm / max(b_norm, 1e-300)
    return {
        "residual_norm": _safe_float(residual_norm),
        "relative_residual": _safe_float(relative_residual),
        "converged": bool(relative_residual < tolerance),
    }


def _least_squares_residual(
    A: sps.spmatrix,
    b: np.ndarray,
    x: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    residual = b - A @ x
    gradient = A.T @ residual
    residual_norm = np.linalg.norm(residual)
    gradient_norm = np.linalg.norm(gradient)
    b_norm = np.linalg.norm(b)
    atb_norm = np.linalg.norm(A.T @ b)
    relative_residual = residual_norm / max(b_norm, 1e-300)
    relative_gradient = gradient_norm / max(atb_norm, 1e-300)
    return {
        "residual_norm": _safe_float(residual_norm),
        "relative_residual": _safe_float(relative_residual),
        "gradient_norm": _safe_float(gradient_norm),
        "relative_gradient": _safe_float(relative_gradient),
        "converged": bool(relative_residual < tolerance),
    }


def _run_solver(
    solver_name: str,
    matrix_name: str,
    matrix_kind: str,
    rows: int,
    cols: int,
    nnz: int,
    xp: Any,
) -> dict[str, Any]:
    spec = SOLVERS[solver_name]
    skip_reason = spec.skip_reason(matrix_kind, rows, cols)
    if skip_reason is not None:
        return {
            "status": "skipped",
            "converged": False,
            "reason": skip_reason,
            **spec.result_settings(),
        }

    start = time.perf_counter()
    try:
        dataset = spec.make_dataset(matrix_name, nnz)
        problem = spec.generator.generate(dataset)
        data = []
        for value in problem.inputs:
            data.append(xp.from_binsparse(value))
        meta = dict(problem.meta)
        run_A = _as_real_sparse_matrix(data[0])
        run_b = np.asarray(data[1], dtype=np.float64).reshape(-1)
        meta = {**meta, **spec.benchmark_meta(run_A)}
        output = spec.benchmark.benchmark(xp, data, meta)
        elapsed = time.perf_counter() - start
        x = np.asarray(output[0], dtype=np.float64).reshape(-1)
        residual = (
            _least_squares_residual(run_A, run_b, x, spec.tolerance)
            if spec.residual_kind == "least_squares"
            else _linear_residual(run_A, run_b, x, spec.tolerance)
        )
        return {
            "status": "ok",
            "seconds": _safe_float(elapsed),
            **spec.result_settings(),
            **residual,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return {
            "status": "error",
            "converged": False,
            "seconds": _safe_float(elapsed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **spec.result_settings(),
        }


def _done_matrix_names(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    names = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                names.add(json.loads(line)["matrix_name"])
            except (KeyError, json.JSONDecodeError):
                continue
    return names


def _chunk_items(items: list[Any], chunk_count: int, chunk_index: int) -> list[Any]:
    chunk = []
    for item_index, item in enumerate(items):
        if item_index % chunk_count == chunk_index:
            chunk.append(item)
    return chunk


def _search_matrices(
    args: argparse.Namespace,
    solvers: Iterable[str],
) -> list[Any]:
    specs = [SOLVERS[solver_name] for solver_name in solvers]
    matrices = list(ssgetpy.search())
    accepted_matrices = []
    for matrix in matrices:
        matrix_kind = _matrix_kind(matrix)
        for spec in specs:
            if spec.accepts_matrix_kind(matrix_kind):
                accepted_matrices.append(matrix)
                break
    matrices = accepted_matrices
    matrices.sort(key=lambda matrix: (matrix.group, matrix.name))
    if args.shuffle:
        random.Random(args.seed).shuffle(matrices)
    return _chunk_items(matrices, args.chunk_count, args.chunk_index)


def _matrix_record(
    matrix: Any,
    solvers: Iterable[str],
    xp: Any,
) -> dict[str, Any]:
    rows = int(matrix.rows)
    cols = int(matrix.cols)
    matrix_kind = _matrix_kind(matrix)
    nnz = int(matrix.nnz)

    results = {}
    for solver_name in solvers:
        results[solver_name] = _run_solver(
            solver_name,
            matrix.name,
            matrix_kind,
            rows,
            cols,
            nnz,
            xp,
        )

    return {
        "matrix_name": matrix.name,
        "matrix_group": matrix.group,
        "matrix_kind": matrix_kind,
        "shape": [rows, cols],
        "n": cols,
        "nnz": nnz,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download SuiteSparse matrices and call SAPS solver benchmarks to "
            "record whether they converge."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scrape_matrices.jsonl"),
        help=(
            "JSONL output path. Existing matrix names are skipped unless --force "
            "is set."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/suitesparse"),
        help="Directory used by ssgetpy for downloaded matrices.",
    )
    parser.add_argument(
        "--solver",
        action="append",
        choices=SOLVER_NAMES,
        help="Solver to run. Repeat to run multiple solvers. Defaults to all solvers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()

    if args.chunk_count < 1:
        parser.error("--chunk-count must be at least 1")
    if args.chunk_index < 0 or args.chunk_index >= args.chunk_count:
        parser.error("--chunk-index must be between 0 and --chunk-count - 1")

    solvers = args.solver if args.solver is not None else list(SOLVER_NAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    completed = set() if args.force else _done_matrix_names(args.output)
    matrices = _search_matrices(args, solvers)

    xp = _load_scipy_framework()
    with _saps_suitesparse_context(args.data_dir, args.seed), args.output.open(
        "a",
        encoding="utf-8",
    ) as output:
        for matrix in matrices:
            if matrix.name in completed:
                print(f"Skipping {matrix.name}; already present in {args.output}")
                continue

            print(f"Checking {matrix.group}/{matrix.name}", flush=True)
            try:
                record = _matrix_record(matrix, solvers, xp)
            except Exception as exc:  # noqa: BLE001
                record = {
                    "matrix_name": matrix.name,
                    "matrix_group": matrix.group,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
