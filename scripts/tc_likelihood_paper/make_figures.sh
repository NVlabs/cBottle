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
# Figures from the output root of run_gpu.sh or an archive of pack_data.py:
#   OUT=/path/to/output [FIGS=$OUT/figures] ./make_figures.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT:?set OUT to the output root of run_gpu.sh or pack_data.py}"
FIGS="${FIGS:-$OUT/figures}"
DERIVED="$OUT/derived"
MAPS=(--era5-maps "$OUT/era5_tcs" --control-maps "$OUT/control_tcs")
run() { python3 "$HERE/$1" "${@:2}"; }

mkdir -p "$FIGS" "$DERIVED"
[[ -e "$DERIVED/era5_hpx3_metrics.csv" ]] ||
  run compute_physical_metrics.py --output-dir "$DERIVED" \
    "$OUT/guidance_0.0" "$OUT/guidance_0.1"
run importance_sampling.py --ema-dir "$OUT" --metrics-dir "$DERIVED" \
  --output-dir "$DERIVED"

run plot_guidance_windows.py --base-dir "$OUT/fig2_windows" \
  --output "$FIGS/guidance_windows"
run plot_log_weights.py --output "$FIGS/log_weights" \
  --run 0.05="$OUT/fig3_sweep/guidance_0.05" \
  --run 0.1="$OUT/guidance_0.1" \
  --run 0.2="$OUT/fig3_sweep/guidance_0.2" \
  --run 0.5="$OUT/fig3_sweep/guidance_0.5" \
  --run 1.0="$OUT/fig3_sweep/guidance_1.0"
run plot_is_comparison.py "${MAPS[@]}" --csv-dir "$DERIVED/fig4" \
  --output "$FIGS/is_comparison_classifier"
run plot_sampler_settings.py --sweep-dir "$OUT/tier1_cost_sweep" \
  --guided-dir "$OUT/guidance_0.1" --cost-dir "$OUT/tier1_cost_calibration/guided" \
  --output "$FIGS/sampler_settings"
run guidance_window_table.py --ema-dir "$OUT" \
  --output "$FIGS/guidance_windows_table.csv"
run plot_is_comparison.py "${MAPS[@]}" --csv-dir "$DERIVED/wind_hpx3" \
  --xlabel "HPX3 max 10 m wind threshold (m/s)" --output "$FIGS/is_comparison_wind"
run plot_classifier_vs_intensity.py --metrics-dir "$DERIVED" \
  --output "$FIGS/classifier_vs_intensity"
run plot_classifier_vs_sigma.py --input-dir "$OUT/tier4_classifier" \
  --output "$FIGS/classifier_vs_sigma"
