#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_dir> <test_csv> [extra_args...]"
  echo "Example: $0 /tmp/baseline_run /tmp/dataset.csv --threshold_method youden"
  echo "Example: $0 /tmp/prototype_run /tmp/dataset.csv --single-stage"
  exit 1
fi

run_dir="$1"
test_csv="$2"
shift 2
extra_args=("$@")

if [[ ! -d "$run_dir" ]]; then
  echo "Run directory not found: $run_dir"
  exit 1
fi

config_path="$run_dir/config.yml"
if [[ ! -f "$config_path" ]]; then
  echo "Run config not found: $config_path"
  exit 1
fi

if [[ ! -f "$test_csv" ]]; then
  echo "Test CSV not found: $test_csv"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  python_cmd=(uv run python)
else
  python_cmd=(python)
fi

model_type="$(
  "${python_cmd[@]}" -c '
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
model_type = config.get("model", {}).get("type")
if not model_type:
    raise SystemExit("Could not determine model.type from config.yml")
print(model_type)
' "$config_path"
)"

if [[ "$model_type" == "simple_predictor" ]]; then
  if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
    cmd=(uv run effector-bincls evaluate-baseline)
  else
    cmd=(python -m effector_bincls.evaluation.baseline)
  fi
  echo "Detected baseline run from $config_path"
elif [[ "$model_type" == "simple" ]]; then
  if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
    cmd=(uv run effector-bincls evaluate-prototype)
  else
    cmd=(python -m effector_bincls.evaluation.prototype)
  fi

  user_requested_single_stage=0
  for arg in "${extra_args[@]}"; do
    if [[ "$arg" == "--single-stage" ]]; then
      user_requested_single_stage=1
      break
    fi
  done

  if [[ $user_requested_single_stage -eq 0 ]]; then
    shopt -s nullglob
    two_stage_checkpoints=("$run_dir"/fold_*/finetuning/checkpoint.pt)
    single_stage_checkpoints=("$run_dir"/fold_*/checkpoint.pt)
    shopt -u nullglob

    if (( ${#two_stage_checkpoints[@]} > 0 )); then
      echo "Detected two-stage prototype run from checkpoint layout"
    elif (( ${#single_stage_checkpoints[@]} > 0 )); then
      echo "Detected single-stage prototype run from checkpoint layout"
      extra_args=(--single-stage "${extra_args[@]}")
    else
      echo "No prototype checkpoints found under $run_dir"
      exit 1
    fi
  else
    echo "Detected prototype run from $config_path"
    echo "Using caller-supplied --single-stage override"
  fi
else
  echo "Unsupported model.type in $config_path: $model_type"
  exit 1
fi

"${cmd[@]}" \
  --run_dir "$run_dir" \
  --test_csv "$test_csv" \
  "${extra_args[@]}"
