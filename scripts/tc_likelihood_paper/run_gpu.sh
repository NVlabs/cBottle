#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# GPU runs behind the figures, in dependency order:
#   CHECKPOINT=/path/to/checkpoints OUT=/path/to/output ./run_gpu.sh \
#     samples fig2 fig3 tier2 tier1 cost classify tier4
# LAUNCH runs the multi-sample stages on many GPUs, e.g. "torchrun --nproc-per-node 8".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CHECKPOINT:?set CHECKPOINT to the cBottle checkpoint root}"
: "${OUT:?set OUT to the output root}"
LAUNCH="${LAUNCH:-python3}"
export MPI4PY_RC_INITIALIZE=0 HDF5_USE_FILE_LOCKING=FALSE
ALL_SEPTEMBERS=(--start-time 1979-09-01T00:00:00 --end-time 2018-09-30T00:00:00)
FIG3_PERIOD=(--start-time 2004-09-01T00:00:00 --end-time 2009-09-26T18:00:00)

ODDS="$HERE/../inference_odds_ratio.py"
odds() { $LAUNCH "$ODDS" --checkpoint "$CHECKPOINT" "$@"; }
odds_one_gpu() { python3 "$ODDS" --checkpoint "$CHECKPOINT" "$@"; }
clf() { python3 "$HERE/tc_classifier.py" "$1" --checkpoint "$CHECKPOINT" "${@:2}"; }

stage_samples() {
  odds "$OUT/guidance_0.0" --guidance-scale-prefactor 0.0 "${ALL_SEPTEMBERS[@]}"
  odds "$OUT/guidance_0.1" --guidance-scale-prefactor 0.1 "${ALL_SEPTEMBERS[@]}" \
    --forward-guidance --run-backward-sampling
}

stage_fig2() {
  local spec name on off
  for spec in 15_20:15:20 15_200:15:200 full:0:200; do
    IFS=: read -r name on off <<<"$spec"
    odds_one_gpu "$OUT/fig2_windows/$name" --guidance-scale-prefactor 1.0 \
      --start-time 2018-09-29T12:00:00 --end-time 2018-09-29T12:00:00 \
      --guidance-on "$on" --guidance-off "$off" --extra-steps 0 \
      --forward-guidance --compute-forward-divergences --run-backward-sampling \
      --save-divergence-traces
  done
}

stage_fig3() {
  local scale
  for scale in 0.05 0.2 0.5 1.0; do
    odds "$OUT/fig3_sweep/guidance_$scale" --guidance-scale-prefactor "$scale" \
      "${FIG3_PERIOD[@]}" --forward-guidance --run-backward-sampling
  done
}

stage_tier2() {
  local spec name scale
  for spec in full:0.1 full_g0.01:0.01; do
    IFS=: read -r name scale <<<"$spec"
    odds "$OUT/tier2_windows/$name" --guidance-scale-prefactor "$scale" \
      "${FIG3_PERIOD[@]}" --forward-guidance --run-backward-sampling \
      --guidance-on 0 --guidance-off 200 --extra-steps-window 15,20
  done
}

stage_tier1() {
  local n e k
  for n in 18 36; do
    for e in 0 5 25; do
      for k in 1 3 6; do
        odds "$OUT/tier1_cost_sweep/s${n}_e${e}_k${k}" \
          --guidance-scale-prefactor 0.1 --x0-dir "$OUT/guidance_0.1" \
          --start-time 2004-09-01T00:00:00 --end-time 2008-09-05T18:00:00 \
          --max-samples 500 --run-backward-sampling \
          --num-steps "$n" --extra-steps "$e" --divergence-samples "$k"
      done
    done
  done
}

stage_cost() {
  odds_one_gpu "$OUT/tier1_cost_calibration/guided" --guidance-scale-prefactor 1.0 \
    --start-time 2004-09-01T00:00:00 --end-time 2004-09-30T18:00:00 \
    --max-samples 5 --forward-guidance --skip-write-nc
}

stage_classify() {
  clf samples --maps-dir "$OUT/control_tcs" "$OUT/guidance_0.0"
  clf samples "$OUT/guidance_0.1" "$OUT"/fig3_sweep/guidance_* "$OUT"/tier2_windows/*
  clf era5 --maps-dir "$OUT/era5_tcs" "$OUT/guidance_0.0" "$OUT/guidance_0.1"
}

stage_tier4() {
  mkdir -p "$OUT/tier4_classifier"
  python3 "$HERE/eval_classifier_vs_sigma.py" --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT/tier4_classifier"
}

for stage in "$@"; do
  "stage_$stage"
done
