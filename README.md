# Matrix Scraper

Tooling for probing SuiteSparse matrices against the solver implementations in
the SAPS benchmark repository. `saps` is installed from GitHub through Poetry.
The scraper uses SAPS' SciPy framework from that Poetry-installed `saps`
checkout.
Every searched SuiteSparse entry gets a JSONL record. Per-solver results record
whether the solver ran, skipped the matrix, errored, converged, or failed to
converge.
The scraper asks `ssgetpy` for the full SuiteSparse index by default. Use
Slurm chunking only to divide that full index across jobs.

Install:

```bash
poetry install
```

Run locally:

```bash
poetry run python scrape_matrices.py
```

Solver convergence uses `rel_tol=1e-6`. The `max_iter` value is `100`,
except Jacobi uses `1000`.

Run on Slurm:

```bash
sbatch --array=0-31 scrape-matrices.slurm
```

Pass extra scraper arguments through `SAPS_SCRAPE_ARGS`, for example:

```bash
SAPS_SCRAPE_ARGS="--solver cg --solver gmres" \
  sbatch --array=0-31 scrape-matrices.slurm
```

Each job writes JSONL to `results-<chunk>.jsonl` in the directory where you ran
`sbatch`. Use `SAPS_SCRAPE_OUTPUT_DIR` to override that location.
