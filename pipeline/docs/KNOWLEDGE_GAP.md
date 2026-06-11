# Knowledge Gap Visualization Guide

## What This Shows

These visualizations demonstrate the **knowledge gap** - what CAN vs CANNOT be learned from impedance spectroscopy alone. This is the gap that transformers help excavate by learning which parameters are identifiable and which are fundamentally degenerate.

## The Four Key Visualizations

### 1. Identifiability Hierarchy (`identifiability_hierarchy.png`)

**What it shows:**
- **Left panel**: Base parameters (Ra, Rb, Ca, Cb, Rsh) ranked by relative uncertainty (σ/μ)
- **Right panel**: Derived quantities (TER, τa, τb, TEC) - these SHOULD be identifiable

**Color coding:**
- 🟢 **Green** (σ/μ < 0.5): IDENTIFIABLE - the transformer predicts these tightly
- 🟠 **Orange** (0.5 ≤ σ/μ < 1.0): WEAKLY IDENTIFIABLE - moderate uncertainty
- 🔴 **Red** (σ/μ ≥ 1.0): NON-IDENTIFIABLE - wide uncertainty (degenerate!)

**What to look for:**
- Individual resistances/capacitances (Ra, Ca, Rb, Cb) should be RED (non-identifiable)
- Derived quantities (τa, τb, TER) should be GREEN (identifiable)
- This shows the **knowledge gap**: we CAN identify τ but NOT individual R and C!

### 2. Manifold Uncertainty Alignment (`manifold_uncertainty_alignment.png`)

**What it shows:**
- **Left panel**: Uncertainty ellipses in (Ra, Ca) space aligned with τa=constant manifold
- **Right panel**: τa predictions from different points on the manifold

**The key insight:**
When we generate spectra from different (Ra, Ca) pairs all satisfying Ra×Ca = τa:
- The transformer predicts **different Ra and Ca values** (red dots scattered)
- BUT predicts the **same τa** every time! (right panel shows tight clustering)
- The blue uncertainty ellipses are ELONGATED along the black manifold curve

**This is the visual proof of the knowledge gap:**
- Individual Ra, Ca: CANNOT be recovered (scatter along manifold)
- Product τa = Ra×Ca: CAN be recovered (tight predictions)

### 3. Fisher vs Learned Identifiability (`fisher_vs_learned.png`)

**What it shows:**
Compares **theoretical identifiability** (Fisher Information) to **learned identifiability** (transformer uncertainty).

**Left panel: Fisher Information singular values**
- Large singular values (σ > 1000) = stiff directions = identifiable
- Small singular values (σ < 1) = sloppy directions = non-identifiable

**Right panel: Transformer inverse relative uncertainty**
- High values (1/σ > 10) = tight predictions = identifiable
- Low values (1/σ < 2) = wide predictions = non-identifiable

**What to look for:**
- The **rank ordering should match** between panels
- Parameters with large Fisher singular values should have small transformer uncertainty
- This proves the transformer **discovered the identifiability structure** without being told!

### 4. Knowledge Gap Summary (`knowledge_gap_summary.png`)

**What it shows:**
Comprehensive 4-panel view of the complete knowledge gap.

**Top 3 panels:**
- **Left (Green)**: Identifiable parameters
- **Middle (Orange)**: Weakly identifiable parameters
- **Right (Red)**: Non-identifiable parameters

**Bottom panel:**
- Derived quantities (TER, τa, τb, TEC) with color-coded identifiability

**What to look for:**
- Most base parameters (Ra, Ca, Rb, Cb) should be in the RED zone
- Derived quantities should be in the GREEN zone
- This shows that **combinations are identifiable even when individual parameters are not**

## The Knowledge Gap: What It Means

### What CAN Be Learned from Impedance Spectra

✅ **Time constants** (τa = Ra×Ca, τb = Rb×Cb)
- Physics: These control frequency-dependent behavior
- Observable: Phase peaks, semicircle features

✅ **Transepithelial Resistance** (TER = parallel combination)
- Physics: DC limit of impedance
- Observable: Low-frequency plateau

✅ **Shunt resistance** (Rsh)
- Physics: Paracellular pathway
- Observable: Relatively independent

❓ **Transepithelial Capacitance** (TEC = series combination)
- Physics: High-frequency behavior
- Observable: If spectrum extends to high enough frequencies

### What CANNOT Be Learned (The Gap!)

❌ **Individual resistances** (Ra, Rb separately)
- Degenerate: Multiple (Ra, Rb) pairs give same TER
- Manifold: Ra + Rb ≈ constant (for fixed TER, Rsh)

❌ **Individual capacitances** (Ca, Cb separately)
- Degenerate: Multiple (Ca, Cb) pairs give same TEC
- Manifold: Ca || Cb ≈ constant

❌ **Unique (R, C) pairs** for each membrane
- Degenerate: Infinite (R, C) pairs satisfy R×C = τ
- Manifold: Hyperbola C = τ/R

## How Transformers Excavate This Gap

### What Classical Methods Do Wrong

**Deterministic fitting (L-BFGS-B):**
- Must pick ONE point on the degenerate manifold
- Unstable: Small noise → different solution
- No uncertainty quantification

**Naive ML models:**
- Make confident predictions about non-identifiable parameters
- Hallucinate: Predict specific Ra, Ca without respecting Ra×Ca = τ constraint
- Spurious correlations not grounded in physics

### What Physics-Informed Transformers Do Right

✅ **Learn the manifold structure**
- Uncertainty spreads along τ-degeneracy (elongated ellipses)
- Strong negative correlations: Ra ↑ ⇒ Ca ↓ (maintaining τ)

✅ **Distinguish identifiable from non-identifiable**
- Tight predictions for τa, τb, TER (green bars)
- Wide predictions for Ra, Ca, Rb, Cb (red bars)

✅ **Respect physics constraints**
- Ra×Ca ≈ τa (product preserved even when individual values vary)
- Ra+Rb ≈ constant (TER constraint satisfied)

✅ **Quantify uncertainty correctly**
- Wide uncertainty = "I don't know" for degenerate parameters
- Tight uncertainty = "I know" for identifiable quantities

## Scientific Implications

### For EIS Analysis

1. **Stop reporting non-identifiable parameters as point estimates**
   - Ra, Ca individually are meaningless
   - Report τa and uncertainty range for (Ra, Ca)

2. **Acknowledge degeneracy in publications**
   - "Multiple (Ra, Ca) pairs consistent with data"
   - Report confidence intervals or posterior distributions

3. **Use derived quantities**
   - TER (transepithelial resistance) is identifiable
   - τa, τb (time constants) are identifiable
   - Ra, Ca separately are NOT

### For Machine Learning

1. **Physics losses are essential**
   - Without them, models hallucinate confidently
   - Spectrum reconstruction loss alone is insufficient

2. **Uncertainty quantification reveals identifiability**
   - Wide uncertainty = learned non-identifiability
   - Correlation structure = learned degeneracy manifolds

3. **MDNs capture multimodal posteriors**
   - Mixture components can represent multiple degenerate solutions
   - Better than single Gaussian (averages modes)

### For Future Experiments

1. **Break degeneracy with additional measurements**
   - Direct capacitance measurements
   - Voltage-dependent impedance
   - Paired optical/electrical readouts

2. **Constrain parameter space with biology**
   - Known membrane capacitance ranges
   - Expected resistance ratios
   - Physiological constraints

3. **Accept fundamental limits**
   - Some parameters cannot be uniquely determined
   - Report identifiable quantities instead
   - Acknowledge the knowledge gap

## Running the Analysis

### Basic Usage

```bash
cd pipeline

python visualize_knowledge_gap.py \
  --model models/phase1_marginal_20251225_180118_log_enc_queries_mdn2_perm_inv_ent0p010_freq_wt
```

### Output

Four PNG files in `knowledge_gap_analysis/`:
1. `identifiability_hierarchy.png` - Which parameters are identifiable
2. `manifold_uncertainty_alignment.png` - Uncertainty along degenerate manifolds
3. `fisher_vs_learned.png` - Theory vs learned identifiability
4. `knowledge_gap_summary.png` - Complete overview

## Interpreting Your Results

### Good Results (Physics-Informed Model)

✅ **Identifiability hierarchy**:
- τb: Green (most identifiable - dominates impedance)
- Rsh: Green or Orange (relatively independent)
- TER: Green (low-frequency limit)
- τa: Orange (secondary time constant)
- Ra, Ca, Rb, Cb: Red (degenerate)

✅ **Manifold alignment**:
- Blue ellipses elongated parallel to black curve
- τa predictions cluster tightly (low scatter on right panel)
- Mean error in τa < 10%

✅ **Fisher vs learned**:
- Both panels show same rank ordering
- Correlation coefficient > 0.8

### Bad Results (No Physics Constraints)

❌ **All parameters show similar uncertainty**
- No distinction between identifiable and non-identifiable
- Model doesn't know what it doesn't know

❌ **Ellipses circular or random**
- No alignment with theoretical manifolds
- Model ignores physics constraints

❌ **Fisher and learned disagree**
- Transformer learned wrong structure
- Need stronger physics losses

## Next Steps

### Enhanced Visualizations

1. **3D manifold projection** - Show intersection of τa and TER manifolds
2. **Noise robustness** - Test identifiability under measurement noise
3. **Frequency dependence** - Which frequency bands inform which parameters

### Model Improvements

1. **Phase 2 training** - Joint consistency optimization
2. **Constrained sampling** - Project posterior onto physics manifolds
3. **Active learning** - Sample from high-uncertainty regions

### Experimental Validation

1. **Synthetic validation** - Generate spectra with known ground truth
2. **Noise injection** - Test robustness to realistic measurement errors
3. **Real RPE data** - Apply to experimental impedance measurements

---

**Generated**: 2026-01-04
**Tool**: `visualize_knowledge_gap.py`
**Key Message**: The transformer doesn't hallucinate - it learns which parameters are fundamentally unknowable!
