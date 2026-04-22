#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_VIDEO="${1:-videoset.mp4}"
OUTPUT_ROOT="${2:-parallel_benchmarks}"
MODEL_PATH="${MODEL_PATH:-models/yolo11s.onnx}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MPI_WORLD_SIZES="${MPI_WORLD_SIZES:-2 3 4}"
SCHEDULERS="${SCHEDULERS:-dynamic static-contiguous static-round-robin}"
LD_PREFIX="${LD_PREFIX:-/usr/lib/wsl/lib}"

mkdir -p "$OUTPUT_ROOT"

printf "world_size\tscheduler\ttotal_wall_ms\tdistributed_processing_ms\taverage_detection_ms\taverage_processing_ms\taverage_people_per_frame\toutput_dir\n"

for world_size in $MPI_WORLD_SIZES; do
  for scheduler in $SCHEDULERS; do
    run_name="np${world_size}_$(echo "$scheduler" | tr '-' '_')"
    run_output_dir="${OUTPUT_ROOT}/${run_name}"

    LD_LIBRARY_PATH="${LD_PREFIX}:${LD_LIBRARY_PATH:-}" \
      mpirun -np "$world_size" ./build/rescue_video_analyzer \
        --input "$INPUT_VIDEO" \
        --output-dir "$run_output_dir" \
        --batch-size "$BATCH_SIZE" \
        --resize-width 640 \
        --model "$MODEL_PATH" \
        --scheduler "$scheduler" \
        --no-annotated-video \
        >/tmp/"${run_name}".log 2>&1

    python3 - "$run_output_dir/summary.json" "$world_size" "$scheduler" "$run_output_dir" <<'PY'
import json
import sys

summary_path, world_size, scheduler, output_dir = sys.argv[1:5]
with open(summary_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

print(
    f"{world_size}\t{scheduler}\t"
    f"{data['total_wall_ms']:.3f}\t"
    f"{data['distributed_processing_ms']:.3f}\t"
    f"{data['average_detection_ms']:.3f}\t"
    f"{data['average_processing_ms']:.3f}\t"
    f"{data['average_people_per_frame']:.3f}\t"
    f"{output_dir}"
)
PY
  done
done
