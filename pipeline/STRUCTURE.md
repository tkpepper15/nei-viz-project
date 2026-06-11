# Directory Structure

Annotated map of every file in the repository. Update this when files are added or removed.

Data, model weights, figures, and logs are excluded from version control (see `.gitignore`).

---

## Root

```
pipeline/
├── 01_generate_dataset_identifiable.py generate training data (uniform in identifiable space)
├── 02_train_transformer.py             train FisherAwareTransformer; writes to models/fisher_vN/
├── train.sh                            train the FisherAwareTransformer (frozen-encoder fine-tuning)
├── backend_api.py                      Flask API exposing MDN + GPF to the frontend
├── start_api.sh                        launch the Flask API (port 5003)
│
├── benchmark_full_pipeline.py          end-to-end temporal benchmark via live API
├── calibrate_temperature.py            post-hoc temperature scaling for MDN uncertainty
├── diagnose_model.py                   MDN component collapse, gradient, and dead-neuron checks
├── mechanistic_admissibility.py        admissibility test: is a mechanistic claim supported by data?
├── visualize_raw_params.py             raw-parameter error visualization (Ra, Rb, Ca, Cb)
├── visualize_temporal_comparison.py    ECM vs GPF vs FFBS tracking comparison plots
│
├── README.md                           project overview, quick start, key results
├── STRUCTURE.md                        this file
├── requirements.txt                    pip dependencies
├── .gitignore                          excludes data/, models/, logs/, figures/, results/
```

---

## eval/ — Evaluation Suite

All experiments write JSON to `results/eval/`. Import shared utilities from `eval/shared.py`.

```
eval/
├── shared.py             shared: data loaders, metrics, CR bounds, model loader
├── E1_identifiability.py CR bound distribution; Fisher effective rank; easy/hard strata
├── E2_static.py          MDN vs ECM vs CR bound on static spectra; calibration; stratified MAE
├── E3_temporal.py        GPF vs FFBS vs ECM-per-step; MAE curves by pathology
├── E4_degeneracy.py      symmetric MAE gain; beta tracking; PTG fork detection timing
├── run_all.py            orchestrator: runs E1-E4, forwards common flags, reports timing
└── report.py             reads result JSONs; prints ASCII tables; --latex for paper fragments
```

---

## src/ — Library Code

```
src/
├── models/
│   ├── fisher_transformer.py   FisherAwareTransformer: encoder + MDN head (K components, full cov)
│   └── __init__.py
│
├── physics/
│   ├── eis_fisher.py           Fisher information matrix; CR bounds in identifiable space;
│   │                           compute_impedance; compute_cr_bounds_identifiable
│   └── __init__.py
│
├── pipeline/
│   ├── gpf.py                  GaussianParticleFilter: 128-particle IEKF update + FFBS sweep;
│   │                           ffbs_backward_sweep; BOUNDS_LOW/MID/HIGH
│   ├── ptg.py                  build_ptg: HDBSCAN clustering -> PTGResult with ForkEvent list
│   └── __init__.py             public API: GaussianParticleFilter, ffbs_backward_sweep, build_ptg
│
├── baselines/
│   └── deterministic_fit.py    classical ECM: L-BFGS-B with random restarts (DeterministicPhysicsFit)
│
├── data/
│   ├── eis_dataset_v2.py       PyTorch Dataset for static train/val/test CSVs
│   └── randles_circuit_simulator.py  impedance simulation; used by data generation scripts
│
└── evaluation/
    ├── symmetric_metrics.py    optimal Ra/Rb, Ca/Cb assignment MAE (symmetric_raw_mae)
    └── __init__.py
```

---

## simulation/ — Temporal Dataset Generation

```
simulation/
├── generate_temporal_dataset.py  writes temporal_dataset.h5 (100k traj x 200 steps x 100 freq)
├── pathology_models.py           per-pathology drift models: healthy, barrier_breakdown, etc.
├── trajectory_generator.py       stochastic trajectory simulator with pathology transitions
└── __init__.py
```

---

## scripts/ — Development and Figure Scripts

These are not part of the core pipeline. They produce figures and intermediate analysis.

```
scripts/
├── analysis/
│   ├── generate_dataset.py                 generate the training dataset (mixed_distribution_v2)
│   ├── estimate_process_noise.py           fit GPF process noise covariance from real data
│   ├── test_parameter_assignment.py        symmetric Ra/Rb/Ca/Cb assignment accuracy by regime
│   └── truth_init_comparison.py            compare cold-start vs truth-initialized GPF
│
└── figures/
    ├── fig_convergence_preview.py          GPF convergence preview (synthetic, runs in seconds)
    ├── fig_ffbs_variance_reduction.py      FFBS variance reduction vs per-step ECM
    ├── fig_snr_by_parameter.py             signal-to-noise ratio per identifiable parameter
    ├── fig_accuracy_by_parameter.py        MAE per parameter: MDN vs ECM vs CR bound
    ├── fig_gpf_sweep_progression.py        GPF posterior narrowing across frequency sweep
    ├── fig_cramer_rao_bounds.py            Cramer-Rao lower bound analysis and efficiency ratios
    ├── fig_precision_breakdown.py          spectrum information vs extracted precision by parameter
    └── plot_nyquist_comparison.py          Nyquist plot comparison across methods
```

---

## scripts/validation/ — Multi-Circuit Generalization

```
scripts/validation/
└── multi_circuit_temporal_analysis.py   test temporal smoothness on multiple circuit topologies
                                          (simple RC, Randles, RCRC); validates generalizability
```

---

## docs/ — Background Documentation

```
docs/
├── PROBLEM_OUTLINE.md      detailed formulation of the identifiability problem
├── DEGENERACY.md           Ra/Rb/Ca/Cb degeneracy: math, implications, resolution
├── IDENTIFIABILITY_TABLE.md per-parameter identifiability ranking with sensitivity analysis
└── KNOWLEDGE_GAP.md        known limitations and open questions
```

---

## data/ — Datasets (not tracked in git)

```
data/
├── mixed_distribution_v2/          primary static dataset (train/val/test CSVs + metadata.json)
│                                   100 frequencies, log-uniform sampling, N=50k/5k/5k
├── identifiable_uniform_v1/        alternative dataset sampled in identifiable space
├── temporal_v1/                    temporal dataset (HDF5, 100k traj x 200 steps x 100 freq)
│                                   keys: params_log10, derived_log10, impedance_real/imag,
│                                         pathology (bytes), frequencies, time_minutes
├── temporal_test/                  small temporal test set for quick sanity checks
└── process_noise.npz               fitted process noise covariance (from estimate_process_noise.py)
```

---

## models/ — Trained Checkpoints (not tracked in git)

```
models/
└── fisher_v10/             current best model
    ├── best_model.pt       val_mae=0.2166; config includes use_drt=False
    ├── checkpoint_*.pt     intermediate checkpoints
    ├── final_model.pt
    └── training_log.csv
```

Earlier versions (v7, v8, v9) are stored locally but not used by the eval suite.

---

## results/ — Evaluation Outputs (not tracked in git)

```
results/
└── eval/
    ├── E1_identifiability.json
    ├── E2_static.json
    ├── E3_temporal.json
    └── E4_degeneracy.json
```

---

## logs/ — Runtime Logs (not tracked in git)

Per-run CSV files written by the GPF and training scripts. Format:
- `kalman_<timestamp>.csv`  — per-step GPF diagnostics
- `smc_steps_<timestamp>.csv` — particle filter step statistics
- `train_v2.log`, `training_output.log` — training stdout captures
