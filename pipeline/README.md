# EIS Parameter Extraction — ML Pipeline

Inference pipeline for extracting circuit parameters from electrochemical impedance spectra (EIS) of retinal pigment epithelium (RPE) cells.

The core problem is a non-identifiable inverse problem: the five raw circuit parameters [Ra, Rb, Ca, Cb, Rsh] cannot all be recovered from a single impedance spectrum. The pipeline works in a provably identifiable coordinate system and provides calibrated uncertainty at every stage.

---

## The Problem

The RPE equivalent circuit is a modified Randles model:

```
     ----[Rsh]----+----------+------
                  |          |
              [Ra]|      [Rb]|
              [Ca]|      [Cb]|
                  |          |
                  +----------+
        Rsh (shunt)   apical + basolateral RC branches
```

**Structural degeneracy:** Ra and Rb (and Ca and Cb) are interchangeable — swapping the apical and basolateral labels produces an identical spectrum. The Fisher information matrix has effective rank 3, not 5.

**Identifiable coordinates:** The five quantities that *are* recoverable are:

| Parameter   | Formula                              | Interpretation              |
|-------------|--------------------------------------|-----------------------------|
| `tau_big`   | max(Ra·Ca, Rb·Cb)                    | dominant time constant      |
| `tau_small` | min(Ra·Ca, Rb·Cb)                    | secondary time constant     |
| `TER`       | Rsh·(Ra+Rb) / (Rsh+Ra+Rb)           | transepithelial resistance  |
| `TEC`       | Ca·Cb / (Ca+Cb)                      | transepithelial capacitance |
| `Rsh`       | Rsh                                  | shunt/paracellular path     |

All evaluation and model outputs use log10 of these quantities.

---

## Inference Architecture

```
Single spectrum Z(omega)
        |
        v
  FisherAwareTransformer
  - MDN over identifiable space (K=3 components, full covariance)
  - Trained on mixed_distribution_v2 (log-uniform sampling)
  - val MAE: tau_big 0.504, tau_small 0.275, TER 0.015, TEC 0.034, Rsh 0.256
        |
        v  (single-spectrum result: identifiable params + uncertainty)


Sequential spectra Z_1, Z_2, ..., Z_T
        |
        v
  Gaussian Particle Filter (GPF)
  - 128 particles in log10 raw parameter space
  - IEKF update (2 iterations) per step using full spectral likelihood
  - Biological prior on Ra > Rb, Ca > Cb (apical dominant)
  - Process model: Brownian motion with pathology-specific drift
        |
        v
  FFBS Backward Sweep (offline)
  - Forward-Filter Backward-Smoother (Briers-Doucet-Maskell)
  - Uses stored particle history; refines all timesteps using future data
        |
        v
  Posterior Trajectory Graph (PTG)
  - HDBSCAN clustering of smoothed particles -> directed trajectory graph
  - Viterbi decoding for MAP mode sequence
  - Fork events = mechanistic bifurcations (e.g. Rsh-decline vs Rb-increase)
```

---

## Quick Start

```bash
# install dependencies
pip install -r requirements.txt

# generate training data
python scripts/analysis/generate_dataset.py
# or for identifiable-space sampling
python 01_generate_dataset_identifiable.py

# train the transformer
bash train.sh

# run the full evaluation suite
python -m eval.run_all

# print paper-ready tables from results
python -m eval.report

# start the Flask API (for frontend integration)
bash start_api.sh
```

---

## Evaluation Suite

All evaluation lives under `eval/` and writes JSON to `results/eval/`.

| Experiment | File                        | What it measures                                               |
|------------|-----------------------------|----------------------------------------------------------------|
| E1         | `eval/E1_identifiability.py` | CR bound distribution, Fisher effective rank, easy/hard strata |
| E2         | `eval/E2_static.py`          | MDN vs ECM vs CR bound on static spectra; calibration          |
| E3         | `eval/E3_temporal.py`        | GPF vs FFBS vs ECM-per-step; convergence curves by pathology   |
| E4         | `eval/E4_degeneracy.py`      | Symmetric metrics; beta tracking; PTG fork detection timing    |

```bash
# full suite
python -m eval.run_all

# fast development pass (skips slow ECM, small n)
python -m eval.run_all --no-ecm --n 200 --n-traj 5

# E1 and E2 in parallel
python -m eval.run_all --parallel

# single experiment
python -m eval.E2_static --n 500

# read results
python -m eval.report
python -m eval.report --latex    # LaTeX table fragments
```

---

## Key Results (n=1000, log10 median MAE)

Static inference:

| Parameter   | CR bound | MDN MAE | ECM MAE |
|-------------|----------|---------|---------|
| tau_big     | ~0.08    | 0.504   | ~0.95   |
| tau_small   | ~0.05    | 0.275   | ~0.60   |
| TER         | ~0.008   | 0.015   | ~0.04   |
| TEC         | ~0.010   | 0.034   | ~0.08   |
| Rsh         | ~0.013   | 0.256   | ~0.45   |

TER and TEC approach the Cramer-Rao bound. Rsh is limited by spectral insensitivity in the healthy regime (Rsh >> TER means the spectrum carries little Rsh information).

---

## File Reference

See `STRUCTURE.md` for the annotated directory tree.

---

## Requirements

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib h5py tqdm joblib flask
```

Python 3.10+. No GPU required; training on CPU takes ~30 min for 100 epochs.
