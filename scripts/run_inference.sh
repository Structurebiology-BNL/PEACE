#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <embedding_dir> <model_dir> [threshold] [extra_args...]"
  echo "Example: $0 /tmp/packed_embeddings /tmp/prototype_run 0.5 --single-stage"
  exit 1
fi

embedding_dir="$1"
model_dir="$2"
threshold="0.5"
extra_args=()

if [[ $# -ge 3 ]]; then
  if [[ "$3" == --* ]]; then
    extra_args=("${@:3}")
  else
    threshold="$3"
    if [[ $# -gt 3 ]]; then
      extra_args=("${@:4}")
    fi
  fi
fi

if [[ ! -d "$embedding_dir" ]]; then
  echo "Packed embedding dataset directory not found: $embedding_dir"
  exit 1
fi

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir"
  exit 1
fi

cmd=(python -m effector_bincls.inference.prototype)
if command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  cmd=(uv run effector-bincls infer-prototype)
fi

"${cmd[@]}" \
  --embedding_dir "$embedding_dir" \
  --model_dir "$model_dir" \
  --output_file "$model_dir/predictions.csv" \
  --threshold "$threshold" \
  "${extra_args[@]}"
