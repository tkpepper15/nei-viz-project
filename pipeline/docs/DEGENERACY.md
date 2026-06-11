# Parameter Degeneracy in the RPE Circuit Model

## What is Parameter Degeneracy?

Parameter degeneracy occurs when multiple parameter combinations produce identical impedance spectra.
For the RCRC Randles circuit, three structural degeneracies exist.

---

### 1. Time-Constant Degeneracy

**Constraint:** `Ra * Ca = tau_a = constant`

For a fixed time constant, infinitely many (Ra, Ca) pairs satisfy the constraint:

```
Ra = 100 Ohm,  Ca = 100 uF  ->  tau_a = 0.01 s
Ra = 1000 Ohm, Ca = 10 uF   ->  tau_a = 0.01 s
Ra = 10000 Ohm, Ca = 1 uF   ->  tau_a = 0.01 s
```

All produce identical impedance spectra. The degenerate manifold is a hyperbola
in (Ra, Ca) space: `Ca = tau_a / Ra`.

The same applies to (Rb, Cb) via `tau_b = Rb * Cb`.

**Resolution:** Work in identifiable coordinates tau_big = max(tau_a, tau_b),
tau_small = min(tau_a, tau_b). The individual R and C values are not recoverable
from spectral data alone.

---

### 2. TER Degeneracy (Parallel Resistance Manifold)

**Constraint:** `TER = Rsh * (Ra + Rb) / (Rsh + Ra + Rb) = constant`

For fixed TER and Rsh, multiple (Ra, Rb) pairs produce the same DC resistance.
Example with TER = 857 Ohm, Rsh = 2000 Ohm (-> Ra + Rb = 1500 Ohm):

```
(Ra = 1000, Rb = 500)  valid
(Ra = 750,  Rb = 750)  valid
(Ra = 1200, Rb = 300)  valid
```

**Resolution:** TER and Rsh are both recoverable. Ra and Rb individually are not.

---

### 3. Permutation Symmetry

**Constraint:** (Ra, Ca) <-> (Rb, Cb) swap produces identical spectrum.

The circuit is symmetric between apical and basolateral membranes. Any labeling
that preserves tau_a and tau_b gives the same impedance.

**Resolution in this pipeline:** The biological prior enforces Ra > Rb and Ca > Cb
(apical membrane dominates). This breaks the permutation symmetry for inference
but does not change what is observable from the spectrum.

For evaluation, the symmetric MAE metric (`src/evaluation/symmetric_metrics.py`)
finds the optimal assignment before computing error, so performance metrics are
not penalized for the arbitrary apical/basolateral labeling.

---

## Fisher Information and Effective Rank

The Fisher information matrix F(theta) = J^T J / sigma^2 quantifies how much
information the spectrum carries about each parameter. For the RCRC circuit:

- **Effective rank of F: 3** (not 5). Two null directions correspond to the
  tau_a and tau_b manifolds described above.
- **Identifiable subspace:** [tau_big, tau_small, TER, TEC, Rsh].
- **Cramer-Rao bounds** in this subspace are finite and parameter-dependent.
  Rsh has the largest CR bound (spectral insensitivity at Rsh >> TER), while
  TER and TEC have the smallest (high spectral sensitivity).

See `eval/E1_identifiability.py` for the empirical rank distribution and
CR bound distribution over the test set.

---

## Model Comparison Summary

| Aspect               | ECM (L-BFGS-B) | FisherAwareTransformer (MDN)   |
|----------------------|----------------|-------------------------------|
| Manifold awareness   | Partial (optimizer picks one point arbitrarily) | Yes (works in identifiable space) |
| Uncertainty output   | None           | Per-parameter posterior std   |
| Handles permutation  | No (arbitrary labeling) | Yes (identifiable coords)   |
| CR efficiency (TER)  | ~0.2           | ~0.5 (approaches bound)      |
| CR efficiency (Rsh)  | ~0.03          | ~0.05 (both far from bound)  |

---

## Key Insights

1. **Ra and Ca individually are not identifiable** from spectral data. tau_a = Ra * Ca is.

2. **Rsh is weakly identifiable** in the healthy regime where Rsh >> TER. The
   sensitivity of the impedance spectrum to Rsh is approximately TER/Rsh, which
   is ~8% when Rsh = 10 kOhm and TER = 800 Ohm. This is the dominant limitation
   on Rsh tracking, not model capacity.

3. **TER and TEC are the most recoverable quantities**. They approach the
   Cramer-Rao lower bound in static inference.

4. **Sequential tracking (GPF) helps Rsh** because temporal continuity constrains
   the drift, partially compensating for low spectral sensitivity.

---

## Scientific Interpretation

Traditional EIS analysis often assumes all parameters are identifiable. This is
incorrect for the RCRC circuit. The pipeline is designed around this fact:

- Training and inference operate in the identifiable subspace.
- The transformer MDN outputs a distribution in identifiable coordinates.
- The GPF particle state is in raw coordinates but the ordering constraint
  (Ra > Rb, Ca > Cb) eliminates the permutation degeneracy.
- Evaluation uses symmetric metrics to avoid penalizing harmless label swaps.

---

## References

- Srinivasan et al. — RPE equivalent circuit topology (parallel formulation)
- Bishop (1994) — Mixture Density Networks for multimodal posteriors
- Transtrum et al. (2015) — Sloppy model manifolds and parameter identifiability
- Ciucci et al. (2019) — Identifiability analysis for EIS
- Briers, Doucet & Maskell (2010) — Forward-Filter Backward-Smoother
