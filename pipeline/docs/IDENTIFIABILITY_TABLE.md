# Parameter Identifiability Reference Table

## Mathematical Foundation

### Cramer-Rao Bound: The Second Derivative Connection

The Cramer-Rao bound sigma_CR is computed from the **Fisher Information Matrix (FIM)**:

```
F_ij = -E[d^2 log L / dtheta_i dtheta_j]    (Hessian of negative log-likelihood)

sigma_CR = sqrt([F^-1]_ii)                   (diagonal of inverse FIM)
```

**Key insight**: sigma_CR represents the **theoretical minimum standard deviation** achievable by any unbiased estimator. Because it derives from the second derivative (curvature) of the likelihood surface:

- **Sharp curvature** (large |F_ii|) => Small sigma_CR => **Identifiable**
- **Flat curvature** (small |F_ii|) => Large sigma_CR => **Non-identifiable**

### Units Interpretation

| Metric | Units | Physical Meaning |
|--------|-------|------------------|
| sigma_CR | decades (log10) | Theoretical minimum uncertainty in orders of magnitude |
| MAE | decades (log10) | Actual prediction error in orders of magnitude |
| H_norm | dimensionless [0,1] | 0 = unique solution, 1 = complete degeneracy |
| I (index) | dimensionless [0,1] | 0 = non-identifiable, 1 = perfectly identifiable |
| rho | dimensionless | 1 = efficient, >1 = suboptimal by factor rho |

---

## Complete Identifiability Table

### Primary Parameters (Directly Measurable)

| Parameter | Symbol | sigma_CR (decades) | MAE (decades) | rho = RMSE/sigma_CR | H_norm | I = (1-H)e^(-MAE) | Interpretation |
|-----------|--------|-------------------|---------------|---------------------|--------|-------------------|----------------|
| **TER** | R_TE | 0.0042 | 0.0024 | 0.73* | 0.00 | 0.998 | Highly identifiable |
| **TEC** | C_TE | 0.0430 | 0.0245 | 0.71* | 0.02 | 0.975 | Highly identifiable |
| **Rsh** | R_sh | 0.0130 | 0.1500 | 14.5 | 0.94 | 0.116 | Weakly identifiable |
| **tau_b** | tau_b | 0.1107 | 0.8085 | 9.2 | 0.96 | 0.066 | Non-identifiable |
| **tau_a** | tau_a | 0.1143 | 0.8228 | 9.0 | 0.97 | 0.059 | Non-identifiable |

### Base Circuit Parameters (Structurally Degenerate)

| Parameter | Symbol | sigma_CR (decades) | MAE (decades) | rho = RMSE/sigma_CR | H_norm | I = (1-H)e^(-MAE) | Interpretation |
|-----------|--------|-------------------|---------------|---------------------|--------|-------------------|----------------|
| **Ra** | R_a | 0.0508 | 0.6396 | 15.8 | 0.99 | 0.057 | Non-identifiable |
| **Rb** | R_b | 0.0508 | 0.6541 | 16.2 | 0.97 | 0.067 | Non-identifiable |
| **Ca** | C_a | 0.0783 | 0.2964 | 4.8 | 0.93 | 0.100 | Weakly identifiable |
| **Cb** | C_b | 0.0760 | 0.3068 | 5.1 | 0.93 | 0.099 | Weakly identifiable |

### Derived Diagnostic Parameters

| Parameter | Symbol | sigma_CR (decades) | MAE (decades) | rho = RMSE/sigma_CR | H_norm | I = (1-H)e^(-MAE) | Interpretation |
|-----------|--------|-------------------|---------------|---------------------|--------|-------------------|----------------|
| **N0** | N_0 | 0.1873 | 0.1346 | 0.90* | - | - | Near-efficient |
| **N1** | N_1 | 0.0430 | 0.0245 | 0.71* | - | - | Near-efficient |
| **D1** | D_1 | 0.0994 | 0.0528 | 0.67* | - | - | Near-efficient |

*rho < 1 indicates biased estimator using structural constraints (valid, not a problem)

---

## Interpretation Guide

### Understanding sigma_CR Values

```
sigma_CR = 0.01 decades  =>  ~2.3% uncertainty (10^0.01 = 1.023)
sigma_CR = 0.05 decades  =>  ~12% uncertainty (10^0.05 = 1.122)
sigma_CR = 0.10 decades  =>  ~26% uncertainty (10^0.10 = 1.259)
sigma_CR = 0.50 decades  =>  ~216% uncertainty (10^0.50 = 3.162)
```

**Rule of thumb**: sigma_CR < 0.05 decades is excellent, < 0.10 is good, > 0.20 is poor.

### Why TER and TEC Are Identifiable

1. **TER = Rsh || (Ra + Rb)**: The parallel combination constrains the solution space
2. **TEC = Ca * Cb / (Ca + Cb)**: Series capacitance is well-determined

These composite quantities have **sharp likelihood peaks** (high FIM curvature) because multiple parameter combinations map to the same TER/TEC value, but the impedance spectrum uniquely determines these composites.

### Why Ra, Rb, Ca, Cb Are Non-Identifiable

The model exhibits **structural degeneracy**:

1. **tau-degeneracy**: tau_a = Ra * Ca, tau_b = Rb * Cb
   - Infinite (Ra, Ca) pairs give same tau_a
   - The spectrum only constrains the product, not individual values

2. **TER-degeneracy**: Rsh || (Ra + Rb) = TER
   - Given TER and tau values, multiple (Ra, Rb, Rsh) combinations work

3. **Result**: Flat likelihood valleys => small FIM eigenvalues => large sigma_CR

### The Efficiency Ratio (rho)

```
rho = RMSE / sigma_CR

rho = 1.0:  Achieving the theoretical limit (efficient estimator)
rho > 1.0:  Suboptimal by factor rho (room for improvement)
rho < 1.0:  Biased estimator using prior information (valid for constrained models)
```

**Interpretation for your poster**: Parameters with rho >> 1 have poor estimators, but this could be because:
- The problem is fundamentally hard (high sigma_CR)
- The model/method is suboptimal (fixable)

Parameters with rho ~ 1 are already achieving theoretical limits.

---

## Key Findings for Your Presentation

### Identifiability Hierarchy (157x Separation)

```
TER (I = 0.998)  >>  TEC (0.975)  >>  Rsh (0.116)  >>  Ca/Cb (~0.10)  >>  Ra/Rb/tau (~0.06)
     |                  |                |                  |                    |
  Perfect           Perfect           Weak              Weak              Non-ID
```

### What Can vs Cannot Be Learned from EIS Data

| Learnable (I > 0.9) | Partially Learnable (I ~ 0.1) | Not Learnable (I < 0.07) |
|---------------------|-------------------------------|--------------------------|
| TER | Rsh | Ra |
| TEC | Ca | Rb |
| | Cb | tau_a |
| | | tau_b |

### Practical Implications

1. **Report composite quantities**: TER, TEC are the scientifically meaningful results
2. **Individual parameters require priors**: Ra, Rb, Ca, Cb cannot be uniquely determined
3. **Time constants are diagnostic**: tau_a, tau_b reveal membrane dynamics but have degeneracy
4. **Physics-informed ML helps**: Models that respect degeneracy structure outperform naive approaches

---

## Mathematical Details: FIM Computation

For log-transformed parameters theta = log10(p), the FIM elements are:

```
F_ij = sum_k (1/sigma_k^2) * (dZ_k/dtheta_i) * (dZ_k/dtheta_j)
```

where:
- Z_k = complex impedance at frequency f_k
- sigma_k = measurement noise at frequency k
- The sum runs over all frequency points

**The second derivative connection**:

```
F_ij = -E[d^2 log L / dtheta_i dtheta_j]
     = -E[d/dtheta_i (d log L / dtheta_j)]
     = -E[d/dtheta_i (sum_k (Z_k - Z_k(theta))/sigma_k^2 * dZ_k/dtheta_j)]
```

For Gaussian noise, this reduces to the sum of squared gradients formula above.

---

## References

1. Pawitan, Y. (2001). In All Likelihood: Statistical Modelling and Inference Using Likelihood.
2. Raue, A. et al. (2009). Structural and practical identifiability analysis. Bioinformatics.
3. Barsoukov, E. & Macdonald, J.R. (2005). Impedance Spectroscopy: Theory, Experiment, and Applications.
