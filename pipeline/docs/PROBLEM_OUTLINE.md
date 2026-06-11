# ECM Parameter Extraction: Problem Outline

## Overview

We are trying to extract 5 circuit parameters (Ra, Rb, Ca, Cb, Rsh) from electrochemical impedance spectroscopy (EIS) data. This is an inverse problem with fundamental identifiability challenges.

---

## 1. FUNDAMENTAL IDENTIFIABILITY PROBLEMS

### 1.1 Parameter Degeneracy (Solution Manifold)

**Problem:** Multiple parameter combinations produce identical impedance spectra.

**Affected parameters:**
- Ra and Rb: Only their sum (Ra + Rb) is constrained by TER
- Ca and Cb: Only their series combination (TEC) is constrained
- Individual R and C values trade off via tau = R * C

**Mathematical structure:**
```
Given: TER, TEC, tau_a, tau_b, Rsh
Constraint: Ra + Rb = TER * Rsh / (Rsh - TER)
Constraint: Ca * Cb / (Ca + Cb) = TEC
Constraint: Ra * Ca = tau_a, Rb * Cb = tau_b

Free parameter: alpha = Ra / (Ra + Rb) in (0, 1)
```

**Impact:** Any alpha value satisfying TEC constraint is equally valid from impedance alone.

---

### 1.2 Rsh-Rsum Confounding

**Problem:** Rsh and Rsum = Ra + Rb appear together in the parallel formula.

**Formula:** `TER = Rsh * Rsum / (Rsh + Rsum)`

**Sensitivity analysis:**
```
Rsum = TER * Rsh / (Rsh - TER)

When Rsh/TER = 1.02:  Rsum = 50 * TER
When Rsh/TER = 1.10:  Rsum = 10 * TER
When Rsh/TER = 2.00:  Rsum = 2 * TER
When Rsh/TER = 10.0:  Rsum = 1.11 * TER
```

**Impact:** Small errors in Rsh estimation cause large errors in Rsum when Rsh ≈ TER.

---

### 1.3 Time Constant Separation

**Problem:** Two RC branches produce overlapping features in the spectrum.

**When tau_a ≈ tau_b:** Single broad arc, cannot separate individual time constants.

**When tau_a >> tau_b (or vice versa):** Two distinct arcs, but:
- Still cannot determine which is "apical" vs "basolateral"
- Assignment is arbitrary (symmetry)

**Current mitigation:** Training data enforces tau_ratio ≥ 2 for separation.

---

### 1.4 Frequency Range Limitations

**Problem:** Finite frequency range limits observable information.

**DC limit (f → 0):**
- TER is exact IF lowest frequency captures the DC plateau
- If lowest f is too high, TER is underestimated

**HF limit (f → ∞):**
- TEC extraction assumes Z → 1/(jω*TEC)
- If highest f is too low, TEC extraction fails
- Large TEC (high capacitance) requires lower frequencies to see capacitive behavior

**Impact:** TEC extraction failed on 50% of test samples (0.55 decades error vs 0.003 target).

---

## 2. DATA DISTRIBUTION PROBLEMS

### 2.1 Rsh/TER Ratio Distribution

**Problem:** 60% of training samples have Rsh/TER < 1.5.

**Why it matters:**
- When Rsh ≈ TER, the inverse problem is ill-conditioned
- Small impedance differences correspond to large parameter differences
- MDN learns this distribution; deterministic methods don't

**Data statistics:**
```
Rsh/TER ratio:
  min: 1.00
  max: 477.08
  median: 1.24
  60% below 1.5
  70% below 2.0
```

---

### 2.2 Parameter Independence

**Problem:** Parameters are generated independently (uncorrelated).

**Correlation matrix shows:**
- Ra, Rb, Ca, Cb, Rsh all have ~0 correlation with each other
- Only definitional correlations exist (tau_a with Ra and Ca)

**Why it matters:**
- No exploitable structure in parameter space
- MDN cannot learn "if Ra is high, then Rb is usually low"
- Each parameter must be inferred independently from spectrum

---

### 2.3 Extreme Parameter Ratios

**Problem:** Some samples have extreme asymmetry.

**Examples from validation:**
- Ra/Rb ratios from 1:184 to 184:1
- tau_ratio from 2 to 200,000+

**Impact:**
- MDN can fail catastrophically on edge cases (Sample 5 with tau = 10^60)
- Numerical instability in extreme regimes

---

### 2.4 Noise in Training Data

**Problem:** 94% of samples have negative Z_real values at some frequencies.

**Cause:** Simulated measurement noise added during generation.

**Impact:**
- Negative Z_real is physically impossible for RC circuits
- Can confuse both MDN and deterministic extraction
- Extreme noise cases cause MDN failures

---

## 3. ALGORITHM PROBLEMS

### 3.1 Deterministic Extraction: TEC Failure

**Problem:** HF limit extraction for TEC fails on many samples.

**Root cause:**
- Assumes Z_imag → -1/(ω*TEC) at high frequency
- Only valid when ω >> 1/tau for both branches
- Large TEC means lower characteristic frequency, may not reach asymptote

**Current performance:** 0.55 decades (vs MDN 0.003)

---

### 3.2 Deterministic Extraction: Rsh Estimation

**Problem:** Cannot reliably estimate Rsh from spectrum shape.

**Attempted methods:**
- Arc width analysis
- Nyquist plot geometry
- Phase derivative analysis

**All fail because:** Rsh affects spectrum subtly when Rsh ≈ TER + Rsum.

**Current performance:** 0.49 decades (vs MDN 0.249)

---

### 3.3 Deterministic Extraction: Time Constant Estimation

**Problem:** Cannot reliably extract tau_a, tau_b from spectrum.

**Challenges:**
- Overlapping RC contributions
- Characteristic frequency peaks are broad
- Phase analysis is noise-sensitive

**Current performance:** 1.7-1.8 decades (vs MDN 0.69)

---

### 3.4 Optimization Landscape

**Problem:** Impedance fitting has many local minima.

**Observation:** Different (tau_a, tau_b, Rsh) combinations give similar impedance fit.

**Impact:**
- Global optimization (differential_evolution) finds good impedance fit
- But may find wrong parameter combination
- No way to distinguish without prior information

---

### 3.5 Grid Search Limitations

**Problem:** Grid search is slow and may miss optimal solutions.

**Current approach:**
- 30 Rsh values × 25 tau_a × 25 tau_b × 19 alpha = 356,250 evaluations
- Each evaluation requires impedance computation (100 frequencies)
- Still misses solutions between grid points

---

## 4. MDN MODEL PROBLEMS

### 4.1 Edge Case Failures

**Problem:** MDN produces catastrophically wrong predictions on some samples.

**Example (Sample 5):**
- True tau_a = 2.8e-4 s
- Predicted tau_a = 5.5e+60 s (64 decades error!)
- TER also wrong by 3 decades

**Cause:** Extreme Ra/Rb asymmetry (1:184) + high noise.

---

### 4.2 TEC Performance Inconsistency

**Problem:** MDN achieves 0.003 decades on TEC but deterministic gets 0.55.

**Implication:** MDN is doing something beyond simple HF extraction.

**Hypothesis:**
- MDN learns TEC from full spectrum shape, not just HF limit
- Uses learned priors about typical TEC values
- May be inferring TEC from tau information

---

### 4.3 Time Constant Accuracy Ceiling

**Problem:** MDN achieves ~0.69 decades on tau (about 5x error factor).

**This is NOT "solved":**
- 0.69 decades means predictions are within 5x of truth on average
- For tau_a = 0.001 s, MDN predicts 0.0002 - 0.005 s range
- Significant uncertainty remains

**Question:** Is this a fundamental limit or can it be improved?

---

### 4.4 Uncertainty Calibration

**Problem:** MDN predicts uncertainty but calibration is unclear.

**Observations:**
- Mean predicted sigma ~0.28 decades for tau
- But actual MAE is 0.69 decades
- Predicted uncertainty may be underestimated

---

## 5. THEORETICAL QUESTIONS

### 5.1 What is the Identifiability Limit?

**Unknown:** What is the theoretical minimum error achievable for each parameter?

**Factors:**
- Noise level in data
- Frequency range
- True parameter values (some regimes harder than others)

**Needed:** Fisher information analysis or Cramer-Rao bounds.

---

### 5.2 Optimal Parameterization

**Question:** Are (Ra, Rb, Ca, Cb, Rsh) the right parameters to predict?

**Alternative parameterizations:**
- (TER, TEC, tau_a, tau_b, alpha) - separates identifiable from degenerate
- (TER, TEC, tau_ratio, tau_geometric_mean, Rsh/TER)
- Direct observable space

---

### 5.3 Information Content of Spectrum

**Question:** How much information does the spectrum contain about each parameter?

**Observable information:**
- DC limit → TER (exact)
- HF limit → TEC (if captured)
- Arc peaks → tau_a, tau_b (approximate)
- Arc shape → Rsh/Rsum ratio (weak)

**Missing information:**
- Alpha (Ra/Rb split) - not in spectrum
- Absolute scale of R vs C (only products matter)

---

## 6. SUMMARY: KEY PROBLEMS TO SOLVE

### High Priority (Blocking Progress)

1. **Rsh/Rsum separation** - Need reliable Rsh estimation when Rsh ≈ TER
2. **TEC extraction** - Need method that works across all TEC values
3. **Time constant extraction** - Need to match MDN's 0.69 decades

### Medium Priority (Improvement Opportunities)

4. **Edge case handling** - Prevent catastrophic MDN failures
5. **Optimal parameterization** - Work in identifiable parameter space
6. **Uncertainty quantification** - Reliable confidence intervals

### Research Questions

7. **Fundamental limits** - What's theoretically achievable?
8. **Prior incorporation** - How to inject distribution knowledge into deterministic methods?
9. **Manifold characterization** - How to properly represent the degenerate solution space?

---

## 7. PROPOSED SOLUTION DIRECTIONS

### Direction A: Better Deterministic Extraction

- Reformulate in terms of identifiable parameters (TER, TEC, tau_ratio, etc.)
- Use Rsh/TER ratio instead of absolute Rsh
- Develop TEC extraction that doesn't require HF asymptote

### Direction B: Hybrid Deterministic + Learned

- Extract TER exactly (physics)
- Use MDN only for ambiguous parameters (tau, Rsh)
- Combine physics constraints with learned priors

### Direction C: Reparameterized MDN

- Train MDN to predict (TER, TEC, tau_a, tau_b, Rsh/TER) instead of raw parameters
- Separate identifiable from degenerate
- Output manifold bounds for Ra, Rb, Ca, Cb

### Direction D: Improved Training Data

- Generate data with Rsh/TER uniformly distributed (not peaked at 1.2)
- Ensure frequency range captures TEC asymptote for all samples
- Add more samples in challenging regimes (extreme ratios)

### Direction E: Theoretical Analysis

- Compute Fisher information for parameter identifiability
- Derive Cramer-Rao bounds for each parameter
- Prove what's fundamentally achievable vs what requires priors
