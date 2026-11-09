# TC likelihood paper

Scripts behind the inferences and figures of *Towards accurate extreme event
likelihoods from diffusion model climate emulators* (Geosci. Model Dev.,
egusphere-2026-2610): importance-sampled frequencies of tropical cyclones (TCs)
near Miami in September, from classifier-guided cBottle samples.

The figures can be redrawn on a CPU from the archived outputs, or everything
can be rerun on GPUs with the public EMA checkpoints.

| Figure | Script | `run_gpu.sh` stages |
|---|---|---|
| Fig. 2, guidance windows | `plot_guidance_windows.py` | `fig2` |
| Fig. 3, log weights | `plot_log_weights.py` | `samples fig3` |
| Fig. 4, TC exceedance frequencies and maps | `importance_sampling.py`, `plot_is_comparison.py` | `samples classify` |
| Guidance-window table | `guidance_window_table.py` | `samples tier2 classify` |
| Fig. A1, sampler settings and cost | `plot_sampler_settings.py` | `samples tier1 cost classify` |
| Fig. A2, wind-speed exceedance frequencies | `compute_physical_metrics.py`, `importance_sampling.py`, `plot_is_comparison.py` | `samples` |
| Fig. A3, classifier versus intensity | `plot_classifier_vs_intensity.py` | `samples classify` |
| Fig. A4, classifier skill versus noise level | `eval_classifier_vs_sigma.py`, `plot_classifier_vs_sigma.py` | `tier4` |

## Redrawing the figures from the archived outputs

The archive on [Zenodo](https://doi.org/10.5281/zenodo.24089312) unpacks to
`tc_likelihood_paper_data/`, about 5 MB in the layout of a `run_gpu.sh` output
directory. With cBottle installed:

```bash
OUT=/path/to/tc_likelihood_paper_data ./make_figures.sh
```

Figures go to `$OUT/figures`, intermediate tables to `$OUT/derived`. On
`run_gpu.sh` output, `make_figures.sh` first extracts the wind and pressure
metrics of the samples and ERA5 into `$OUT/derived`, unless they are there.

## Rerunning the inferences

**Checkpoints.** Download the EMA weights from the
[nvidia/cbottle](https://huggingface.co/nvidia/cbottle) Hugging Face repository
and arrange them as `cbottle.inference.load("cbottle-3d-moe-tc")` expects:

```bash
HF=/path/to/download ROOT=/path/to/checkpoints
mkdir -p $ROOT/cBottle-3d $ROOT/cBottle-3d-tc
for s in 000512000 002048000 009856000; do
  ln -s $HF/cBottle-3d-ema/ema-checkpoint-$s.checkpoint \
    $ROOT/cBottle-3d/training-state-$s.checkpoint
done
ln -s $HF/cBottle-3d-tc-ema/ema-checkpoint-002176000.checkpoint \
  $ROOT/cBottle-3d-tc/training-state-002176000.checkpoint
```

**Data.** The datasets are located through the environment variables of
`cbottle.config.environment`:

* `AMIP_MID_MONTH_SST`: AMIP SST forcing (`scripts/download_amip_sst.py`), all
  sampling stages;
* `V6_ERA5_ZARR`, `V6_ERA5_ZARR_PROFILE`: ERA5 on HEALPix level 6
  (`scripts/era5_data_prep`), for the ERA5 classifier probabilities and metrics
  and `tier4`;
* `IBTRACS_DATA_PATH`: IBTrACS track labels, for `tier4`.

**Runs.** In dependency order:

```bash
export CHECKPOINT=$ROOT OUT=/path/to/output
LAUNCH="torchrun --nproc-per-node 8" ./run_gpu.sh samples fig2 fig3 tier2 tier1 cost classify tier4
./make_figures.sh
python pack_data.py --out $OUT --dest /path/to/archive
```

`LAUNCH` starts the multi-sample stages on many GPUs; `fig2` and `cost` always
run on one GPU. `tier1` rescores the guided samples of `samples`.

## Notes

* **Seeds.** Each GPU seeds with `--seed` plus its rank and takes a contiguous
  block of timestamps, so samples are identical to the paper's only with the
  same number of GPUs: 128 for the control run, 256 for the guided run, 64 for
  the `fig3`, `tier2` and `tier1` runs. With the same seeds, samples and
  likelihood terms reproduce exactly; classifier probabilities and the guidance
  divergence of full-trajectory runs agree to within 1 % (GPU kernels).
* **Full-trajectory guidance at γ = 0.01** (`tier2`) was chosen so that the
  mean classifier probability of its samples matches that of the [15, 20]
  window at γ = 0.1 (about 0.25). The calibration runs are not included.
* **Heun.** The sampler-settings sweep in the archive includes
  `s36_e25_k3_heun`, made with an unreleased sampler variant;
  `inference_odds_ratio.py` integrates with Euler steps only. The plot draws the
  point when the run is present.
* **ERA5 classifier probabilities** in the archive (`era5_tc_probability`
  columns and `era5_tcs`) were computed with the non-EMA classifier weights, as
  in the paper; `run_gpu.sh classify` uses the EMA classifier. The archived ERA5
  map averages the Septembers of 1995-2018.
* **Bootstrap intervals** of Fig. A1 depend on the order of the samples and can
  differ slightly from the paper.
* **Environment.** cBottle's `fastgen` dependency, used only for distillation,
  requires torch 2.10. With an older torch, install cBottle with `--no-deps` and
  add `numpy`, `zarr>=3`, `xarray`, `cartopy`, `netCDF4`, `h5netcdf`, `h5py`,
  `python-dotenv`, `zict` and `s3fs`.

## Archive contents

`pack_data.py` writes, in the layout of the output directory:

* `*/likelihoods.csv`: the per-sample likelihood terms and classifier
  probabilities of every run, merged over ranks;
* `control_tcs/`, `era5_tcs/`: time-mean HPX3 TC probability maps (Fig. 4b, c);
* `fig2_windows/`: the 10 m zonal wind of the three Fig. 2 samples and their
  divergence traces;
* `tier4_classifier/`: classifier skill and reliability versus noise level;
* `derived/`: maximum 10 m wind and minimum sea-level pressure in the Miami
  HPX3 pixel for the samples and ERA5.

The ERA5-derived values contain modified Copernicus Climate Change Service
information (2026).
