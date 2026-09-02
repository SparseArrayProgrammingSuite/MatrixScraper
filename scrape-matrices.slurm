#!/usr/bin/env bash
#SBATCH --job-name=saps-scrape-matrices
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=scrape-matrices-%A_%a.log

set -euo pipefail

scraper_directory="${MATRIX_SCRAPER_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$scraper_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  scraper_directory="$script_directory"
fi

export PATH="$HOME/.local/bin:$PATH"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  task_min="${SLURM_ARRAY_TASK_MIN:-0}"
  chunk_index=$((SLURM_ARRAY_TASK_ID - task_min))
  chunk_count="${SAPS_SCRAPE_CHUNK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
else
  chunk_index="${SAPS_SCRAPE_CHUNK_INDEX:-0}"
  chunk_count="${SAPS_SCRAPE_CHUNK_COUNT:-1}"
fi

output_dir="${SAPS_SCRAPE_OUTPUT_DIR:-$scraper_directory/.saps/outputs/matrix-scraper}"
data_dir="${SAPS_SCRAPE_DATA_DIR:-$scraper_directory/data/suitesparse}"
output_file="$output_dir/results-${chunk_index}.jsonl"

mkdir -p "$output_dir"

scrape_args=(
  --output "$output_file"
  --data-dir "$data_dir"
  --chunk-count "$chunk_count"
  --chunk-index "$chunk_index"
)

if [[ -n "${SAPS_SCRAPE_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$SAPS_SCRAPE_ARGS"
  scrape_args+=("${extra_args[@]}")
fi

cd "$scraper_directory"

poetry run python scrape_matrices.py "${scrape_args[@]}"
