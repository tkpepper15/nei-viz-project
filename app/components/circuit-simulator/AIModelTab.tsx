"use client";

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { CircuitParameters } from './types/parameters';
import { ModelSnapshot } from './types';
import { ML_API_BASE } from '../../config/api';
import { PILL_BASE, DIVIDER, LABEL_BTN_ACTIVE, LABEL_BTN_INACTIVE, NODE_HEADER, NODE_HEADER_FOCUSED, ICON, NODE_CHROME, NODE_DELETE_BTN, TYPE } from './ui/tokens';
import { SpiderAxisConfig } from './visualizations/SpiderPlot3D';

// ---- Types ----

interface ActiveModelInfo {
  model: string;
  epoch: number | null;
  val_mae: number | null;
  param_mae: Record<string, number> | null;
  rbpf_version?: string;
  rbpf_bounds?: { low: number[]; high: number[]; params: string[] };
}

interface AvailableModel {
  name: string;
  val_mae: number | null;
  epoch: number | null;
  param_space: string | null;
  mae_source: string | null;
  active: boolean;
}

type ParameterKey = 'R1' | 'C1' | 'R2' | 'TER' | 'TEC' | 'Ra' | 'Rb' | 'Ca' | 'Cb';
const PARAM_KEYS: ParameterKey[] = ['R1', 'C1', 'R2', 'TER', 'TEC', 'Ra', 'Rb', 'Ca', 'Cb'];



interface DerivedSeries { mean: number[]; q25: number[]; q75: number[]; }

interface GroundTruth {
  Ra: number; Rb: number; Ca: number; Cb: number; Rsh: number;
  R1: number; C1: number; R2: number; TER: number; TEC: number;
}

interface Timepoint {
  time_min: number;
  Z_real: number[];
  Z_imag: number[];
  frequencies: number[];
  ground_truth: GroundTruth;
}

interface RealSample {
  name: string;
  donor: string;
  interval_min: number;
  atp_lo: number;
  atp_hi: number;
  timepoints: Timepoint[];
}

// Analytics types

// Per-timepoint: 15 stratified MC draws in display units for spaghetti lines
type RawSamplesMap = Record<ParameterKey, number[]>;

// Single MDN proposal mean in display units (all 9 params)
type ProposalDisplay = Record<ParameterKey, number>;

// Transformer analytics extracted from a single forward pass
interface AnalyticsSnapshot {
  proposals: ProposalDisplay[];   // K=3 component means in display units
  weights: number[];              // K mixture weights
  attention: number[][][];        // layers × n_heads × n_freq (mean over query dim)
  pooling: number[] | null;       // n_freq pooling weights
}

interface ParticleScatterFrame {
  Ca: number[];        // µF/cm²
  Cb: number[];
  Ra: number[];        // kΩ·cm²
  Rb: number[];
  hyp: number[];       // hypothesis index 0-3 per particle
  time_min?: number[]; // absolute time tag per particle (for cross-timepoint joint density)
}

interface _SmcDiag {
  ess_frac: number;
  p_Ca_gt_Cb: number;
  p_Ra_lt_Rb: number;
  hyp_weights: number[];  // K=4 hypothesis weights (sum to 1)
}

// Kalman smoother result from the done event (constant-velocity augmented model)
interface KalmanResult {
  time_min: number[];
  param_names: string[];          // ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']
  // Position estimates
  mu_filtered: number[][];        // (T, 5) causal estimate mean (log10)
  std_filtered: number[][];       // (T, 5) causal estimate std
  mu_smoothed: number[][];        // (T, 5) non-causal smoothed mean (log10)
  std_smoothed: number[][];       // (T, 5) non-causal smoothed std
  // Velocity estimates: dθ/dt in log10/min
  mu_vel_filtered?: number[][];   // (T, 5) causal velocity estimate
  std_vel_filtered?: number[][];  // (T, 5) causal velocity std
  mu_vel_smoothed?: number[][];   // (T, 5) smoothed velocity
  std_vel_smoothed?: number[][];  // (T, 5) smoothed velocity std
  // Dense temporal grid: 0.5-min resolution + 30-min extrapolation (identifiable space)
  t_grid?: number[];              // (N,) query times in minutes
  mu_grid?: number[][];           // (N, 5) predicted position (log10) — identifiable
  std_grid?: number[][];          // (N, 5) prediction std
  // Raw [Ra, Rb, Ca, Cb, Rsh] space — unimodal when canonical_mode is active
  raw_param_names?: string[];
  mu_smoothed_raw?: number[][];
  std_smoothed_raw?: number[][];
  mu_vel_smoothed_raw?: number[][];
  mu_grid_raw?: number[][];       // (N, 5) predicted position (log10) — raw space
  std_grid_raw?: number[][];      // (N, 5) prediction std — raw space
  // Per-param linear trend over the full experiment (from Kalman smoothed positions)
  trends?: Array<{
    slope: number;       // log10/min, linear regression on smoothed positions
    net_change: number;  // log10 displacement first→last
    mean_vel: number;    // mean instantaneous velocity (log10/min)
    direction: 'rising' | 'falling' | 'stable';
  }>;
  // Posterior geometry: eigendecomposition of measurement covariance at each timepoint
  geometry?: {
    eigen_vals: number[][];      // (T, 5) descending eigenvalues of cov_meas
    eigen_vecs: number[][][];    // (T, 5, 5) eigenvectors as rows (descending)
    identifiability: number[];   // (T,) λ_min/λ_max — 0=degenerate, 1=isotropic
    innov_proj: number[][];      // (T, 5) Kalman innovation projected onto eigenvectors
  };
}

// Change-point detected on Kalman innovations
interface ChangePoint {
  timepoint: number;
  time_min: number | null;
  mahal: number;
  dominant_param: string;
  direction: string;
  delta_log: number[];
  interpretation: string;
}

interface AdmissibilityParamScore {
  O: number; A: number; T: number; C: number; product: number;
}
interface AdmissibilityClaimLanguage {
  tier: 'supported' | 'hedged' | 'weak' | 'refused';
  text: string;
}
interface BiophysicalMode {
  pc: number;
  name: string;
  label: string;
  eigenvalue: number;
  loadings: Record<string, number>;
}
interface Admissibility {
  scores: Record<string, AdmissibilityParamScore>;
  claim_language: Record<string, AdmissibilityClaimLanguage>;
  biophysical_modes: BiophysicalMode[];
  mechanistic_inconsistency: boolean;
}

interface TimeSeriesResult {
  time_min: number[];
  predictions: Record<ParameterKey, DerivedSeries>;
  ecm: Record<ParameterKey, number[]> | null;
  attribution: {
    d_Ra: number[]; d_Rb: number[]; d_Ca: number[];
    d_Cb: number[]; d_Rsh: number[]; d_RaRb: number[];
  };
  groundTruth: Record<ParameterKey, number[]> | null;
  atpLo: number | null;
  atpHi: number | null;
  timing: { dl_ms: number[]; ecm_ms: number[] };
  pRaIsR1: number[];
  pCaIsC1: number[];
  rawSamples: RawSamplesMap[];
  posteriorStd: number[][];
  analyticsPerTimepoint: (AnalyticsSnapshot | null)[];
  analyticsLast: AnalyticsSnapshot | null;
  dlResnorm: number[];
  ecmResnorm: number[];
  degHR: number[];
  degHC: number[];
  particleScatter: ParticleScatterFrame[];
  smcEss: number[];
  smcPCaGtCb: number[];
  smcPRaLtRb: number[];
  hypWeights: number[][];  // (n_timepoints, K=4) hypothesis weights per timepoint
  smoothedTrajectories: SmoothedTrajectory[];
  clusterPaths: SmoothedTrajectory[];
  ecmColdPath: EcmColdPoint[];
  mdnPredictions: Record<ParameterKey, DerivedSeries>;
  mdnEntropy: number[];  // per-timepoint mean log10 std across identifiable params
  // Kalman smoother and change-point detection (populated at done event)
  kalman: KalmanResult | null;
  changepoints: ChangePoint[];
  admissibility: Admissibility | null;
  // Per-step relabeling fraction (canonical mode only)
  relabelingFrac: (number | null)[];
  // FFBS posterior summary + mechanism analysis (populated at done event)
  posteriorSummary: Record<string, PosteriorParamSummary> | null;
  mechanism: Mechanism | null;
  // Which pipeline stages were active when this result was computed
  activePipeline?: { useTransformer: boolean; useRBPF: boolean; useKalman: boolean; useEcm: boolean };
}

// ---- Cold ECM baseline path (from done event) ----
interface EcmColdPoint {
  time_min: number;
  TER: number; TEC: number; tau_big: number; tau_small: number;
  Rsh: number; Ra: number; Rb: number; Ca: number; Cb: number;
}

// ---- Smoothed trajectory types (from done event) ----
// R1/C1/R2 are derived: R1=min(Ra,Rb) transcellular, C1=min(Ca,Cb) membrane, R2=Rsh paracellular
type TrajectoryParam = 'TER' | 'TEC' | 'tau_big' | 'tau_small' | 'Rsh' | 'Ra' | 'Rb' | 'Ca' | 'Cb' | 'R1' | 'C1' | 'R2';

// Grouped: identifiable/clinical first, raw circuit elements second
const TRAJ_PARAMS_IDENTIFIABLE: TrajectoryParam[] = ['TER', 'TEC', 'R1', 'C1', 'R2', 'tau_big', 'tau_small'];
const TRAJ_PARAMS_RAW: TrajectoryParam[]           = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'];
const _TRAJ_PARAMS: TrajectoryParam[] = [...TRAJ_PARAMS_IDENTIFIABLE, ...TRAJ_PARAMS_RAW]; // kept for reference

const TRAJ_PARAM_LABELS: Record<TrajectoryParam, string> = {
  TER:       'TER (kΩ)',
  TEC:       'TEC (µF)',
  tau_big:   'τ_big (s)',
  tau_small: 'τ_small (s)',
  Rsh:       'Rsh (kΩ)',
  Ra:        'Ra (kΩ)',
  Rb:        'Rb (kΩ)',
  Ca:        'Ca (µF)',
  Cb:        'Cb (µF)',
  R1:        'R1 transcellular (kΩ)',
  C1:        'C1 membrane (µF)',
  R2:        'R2 paracellular (kΩ)',
};

interface SmoothedPathPoint {
  t: number;
  time_min: number;
  TER: number;       TER_q05: number;       TER_q25: number;       TER_q75: number;       TER_q95: number;
  TEC: number;       TEC_q05: number;       TEC_q25: number;       TEC_q75: number;       TEC_q95: number;
  tau_big: number;   tau_big_q05: number;   tau_big_q25: number;   tau_big_q75: number;   tau_big_q95: number;
  tau_small: number; tau_small_q05: number; tau_small_q25: number; tau_small_q75: number; tau_small_q95: number;
  Rsh: number;       Rsh_q05: number;       Rsh_q25: number;       Rsh_q75: number;       Rsh_q95: number;
  Ra: number;        Ra_q05: number;        Ra_q25: number;        Ra_q75: number;        Ra_q95: number;
  Rb: number;        Rb_q05: number;        Rb_q25: number;        Rb_q75: number;        Rb_q95: number;
  Ca: number;        Ca_q05: number;        Ca_q25: number;        Ca_q75: number;        Ca_q95: number;
  Cb: number;        Cb_q05: number;        Cb_q25: number;        Cb_q75: number;        Cb_q95: number;
}

interface SmoothedTrajectory {
  rank: number;
  probability: number;
  hypothesis: number;
  label: string;
  pCaGtCb?: number;   // fraction of particles where Ca > Cb (cluster paths only)
  pRaLtRb?: number;   // fraction of particles where Ra < Rb (cluster paths only)
  path: SmoothedPathPoint[];
}

function trajDisplayValue(param: TrajectoryParam, raw: number): number {
  if (param === 'TER' || param === 'Rsh' || param === 'Ra' || param === 'Rb' || param === 'R1' || param === 'R2') return raw / 1000;  // Ω → kΩ
  if (param === 'TEC' || param === 'Ca' || param === 'Cb' || param === 'C1') return raw * 1e6;   // F → µF
  return raw;                                                                                      // tau: seconds as-is
}

// Extract raw value from SmoothedPathPoint — handles derived R1/C1/R2.
function extractSmoothedValue(param: TrajectoryParam, pt: SmoothedPathPoint): number {
  if (param === 'R1') return Math.min(pt.Ra, pt.Rb);
  if (param === 'C1') return Math.min(pt.Ca, pt.Cb);
  if (param === 'R2') return pt.Rsh;
  return (pt as unknown as Record<string, number>)[param] ?? 0;
}
function extractSmoothedQ05(param: TrajectoryParam, pt: SmoothedPathPoint): number {
  if (param === 'R1') return Math.min(pt.Ra_q05, pt.Rb_q05);
  if (param === 'C1') return Math.min(pt.Ca_q05, pt.Cb_q05);
  if (param === 'R2') return pt.Rsh_q05;
  return (pt as unknown as Record<string, number>)[`${param}_q05`] ?? 0;
}
function extractSmoothedQ25(param: TrajectoryParam, pt: SmoothedPathPoint): number {
  if (param === 'R1') return Math.min(pt.Ra_q25, pt.Rb_q25);
  if (param === 'C1') return Math.min(pt.Ca_q25, pt.Cb_q25);
  if (param === 'R2') return pt.Rsh_q25;
  return (pt as unknown as Record<string, number>)[`${param}_q25`] ?? 0;
}
function extractSmoothedQ75(param: TrajectoryParam, pt: SmoothedPathPoint): number {
  if (param === 'R1') return Math.min(pt.Ra_q75, pt.Rb_q75);
  if (param === 'C1') return Math.min(pt.Ca_q75, pt.Cb_q75);
  if (param === 'R2') return pt.Rsh_q75;
  return (pt as unknown as Record<string, number>)[`${param}_q75`] ?? 0;
}
function extractSmoothedQ95(param: TrajectoryParam, pt: SmoothedPathPoint): number {
  if (param === 'R1') return Math.min(pt.Ra_q95, pt.Rb_q95);
  if (param === 'C1') return Math.min(pt.Ca_q95, pt.Cb_q95);
  if (param === 'R2') return pt.Rsh_q95;
  return (pt as unknown as Record<string, number>)[`${param}_q95`] ?? 0;
}

// Extract raw value from EcmColdPoint for any TrajectoryParam.
function extractEcmValue(param: TrajectoryParam, pt: EcmColdPoint): number {
  if (param === 'R1') return Math.min(pt.Ra, pt.Rb);
  if (param === 'C1') return Math.min(pt.Ca, pt.Cb);
  if (param === 'R2') return pt.Rsh;
  return (pt as unknown as Record<string, number>)[param] ?? 0;
}


// ---- Posterior / mechanism types ----

interface PosteriorParamSummary {
  median: number;
  q05: number;
  q95: number;
  identifiability: number;  // 0-1, higher = more constrained posterior
}

interface MechanismHypothesis {
  name: string;
  label: string;
  probability: number;
}

interface EvidenceWindow {
  t_start: number;
  t_end: number;
  label: string;
  contribution: number;
}

interface Mechanism {
  hypotheses: MechanismHypothesis[];
  evidence_time: EvidenceWindow[];
  established: string[];
  ambiguous: string[];
}

interface ObservabilityIdentEntry {
  score: number;
  status: 'strong' | 'moderate' | 'ambiguous';
  note: string;
  structurally_ambiguous?: boolean;
}

interface ObservabilityResult {
  snr_score: number;
  freq_coverage_score: number;
  drift_score: number;
  drift_label: string;
  identifiability: Record<string, ObservabilityIdentEntry>;
  event_windows: Array<{ t_start: number; t_end: number; confidence: number }>;
  n_sweeps: number;
  freq_range: [number, number];
}

// ---- Node Graph Types ----

type NodeType =
  | 'source-data'           // Load real or synthetic data
  | 'process-ml'            // Inference: constraints + model config
  | 'viz-results'           // Results panel: posteriors, mechanism, confidence
  | 'viz-trajectory'        // Trajectory line chart
  | 'viz-diagnostics'       // Filter health: ESS + identifiability + Kalman dynamics
  | 'source-observability'  // Data quality + frequency coverage + event detection
  | 'viz-mechanism'         // Hypothesis posteriors + evidence attribution
  | 'viz-spiderplot';       // 3D spider plot tornado — circuit parameters over time

type PortKind = 'impedance-series' | 'ml-result';
// DataClass mirrors the node category that produces the data — used for connection validation.
// 'source': any raw data output (impedance series, model grid) from source-* nodes
// 'process': parameter estimates from process-* nodes (ml-result)
type DataClass = 'source' | 'process';
type TrajView  = 'trajectory' | 'error' | 'derivative' | 'uncertainty' | 'table';

interface PortDef {
  id: string;
  side: 'in' | 'out';
  kind: PortKind;
  dataClass: DataClass;
  accepts?: DataClass[]; // input ports only — overrides exact dataClass match when set
  label: string;
  yFrac: number; // vertical position as fraction of node height, 0=top 1=bottom
}

function portsCompatible(fromPort: PortDef, toPort: PortDef): boolean {
  if (toPort.accepts) return toPort.accepts.includes(fromPort.dataClass);
  return fromPort.dataClass === toPort.dataClass;
}

interface NodeDef {
  label: string;
  category: 'source' | 'process' | 'viz' | 'other';
  ports: PortDef[];
  defaultWidth: number;
  defaultHeight: number;
  portColor: string;
}

// Design tokens — categorical role colors, desaturated ~25% from material palette.
// Process is intentionally the most present (pipeline engine), Source most muted.
const C_SOURCE  = '#6469a0'; // most muted — data origin
const C_PROCESS = '#c4a040'; // most present — transformation engine
const C_VIZ     = '#3d7a60'; // calm — output/read

const NODE_DEFS: Record<NodeType, NodeDef> = {
  'source-data': {
    label: 'Data', category: 'source', portColor: C_SOURCE,
    defaultWidth: 300, defaultHeight: 320,
    ports: [
      { id: 'out',     side: 'out', kind: 'impedance-series', dataClass: 'source', label: 'To Inference', yFrac: 0.38 },
      { id: 'out-viz', side: 'out', kind: 'impedance-series', dataClass: 'source', label: 'To Visualizer', yFrac: 0.62 },
    ],
  },
  'process-ml': {
    label: 'Inference', category: 'process', portColor: C_PROCESS,
    defaultWidth: 340, defaultHeight: 620,
    ports: [
      { id: 'in',     side: 'in',  kind: 'impedance-series', dataClass: 'source',  label: 'Z(ω, t)',    yFrac: 0.5 },
      { id: 'out-ml', side: 'out', kind: 'ml-result',        dataClass: 'process', label: 'θ(t), Σ(t)', yFrac: 0.5 },
    ],
  },
  'viz-results': {
    label: 'Results', category: 'viz', portColor: C_VIZ,
    defaultWidth: 280, defaultHeight: 620,
    ports: [{ id: 'in', side: 'in', kind: 'ml-result', dataClass: 'process', label: 'ML Result', yFrac: 0.5 }],
  },
  'viz-trajectory': {
    label: 'Trajectory', category: 'viz', portColor: C_VIZ,
    defaultWidth: 700, defaultHeight: 400,
    ports: [
      { id: 'in-1', side: 'in', kind: 'ml-result', dataClass: 'process', label: 'Source A', yFrac: 0.35 },
      { id: 'in-2', side: 'in', kind: 'ml-result', dataClass: 'process', label: 'Source B', yFrac: 0.65 },
    ],
  },
  'viz-diagnostics': {
    label: 'Filter Health', category: 'viz', portColor: C_VIZ,
    defaultWidth: 500, defaultHeight: 480,
    ports: [{ id: 'in', side: 'in', kind: 'ml-result', dataClass: 'process', label: 'ML Result', yFrac: 0.5 }],
  },
  'source-observability': {
    label: 'Observability', category: 'source', portColor: C_SOURCE,
    defaultWidth: 300, defaultHeight: 400,
    ports: [
      { id: 'in',  side: 'in',  kind: 'impedance-series', dataClass: 'source',  label: 'Z(ω, t)',    yFrac: 0.5 },
      { id: 'out', side: 'out', kind: 'impedance-series', dataClass: 'source',  label: 'Passthrough', yFrac: 0.5 },
    ],
  },
  'viz-mechanism': {
    label: 'Mechanism', category: 'viz', portColor: C_VIZ,
    defaultWidth: 360, defaultHeight: 340,
    ports: [{ id: 'in', side: 'in', kind: 'ml-result', dataClass: 'process', label: 'ML Result', yFrac: 0.5 }],
  },
  'viz-spiderplot': {
    label: 'Spider 3D', category: 'viz', portColor: C_VIZ,
    defaultWidth: 560, defaultHeight: 640,
    ports: [{ id: 'in', side: 'in', kind: 'impedance-series', dataClass: 'source', label: 'Impedance Data', yFrac: 0.5 }],
  },
};

// Per-node config (discriminated union)

interface ParamConstraint {
  enabled: boolean;
  min: number;
  max: number;
  ddtMax: number;   // max |Δlog10(param)| per measurement step
  monotone: 'up' | 'down' | 'free';
}

const DEFAULT_CONSTRAINTS: Record<string, ParamConstraint> = {
  Ra:  { enabled: true, min: 10,   max: 50000,   ddtMax: 0.10, monotone: 'free' },
  Rb:  { enabled: true, min: 10,   max: 50000,   ddtMax: 0.10, monotone: 'free' },
  Ca:  { enabled: true, min: 1e-9, max: 1e-4,    ddtMax: 0.05, monotone: 'free' },
  Cb:  { enabled: true, min: 1e-9, max: 1e-4,    ddtMax: 0.05, monotone: 'free' },
  Rsh: { enabled: true, min: 100,  max: 1000000, ddtMax: 0.05, monotone: 'free' },
};

function fmtPy(v: number): string {
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  const s = v.toExponential();
  const [m, eStr] = s.split('e');
  const e = parseInt(eStr);
  const mf = parseFloat(m);
  return Math.abs(mf - Math.round(mf)) < 0.0001 ? `${Math.round(mf)}e${e}` : v.toExponential(1);
}

function constraintsToCode(c: Record<string, ParamConstraint>, cfg: { useTransformer: boolean; useRBPF: boolean; useKalman: boolean; useEcm: boolean }): string {
  const rows = Object.entries(c).map(([name, p]) => {
    const pad = name.length < 3 ? ' '.repeat(3 - name.length) : '';
    const comment = p.enabled ? '' : '  # disabled';
    return `    '${name}':${pad} Param(min=${fmtPy(p.min)}, max=${fmtPy(p.max)}, ddt_max=${p.ddtMax.toFixed(2)}, direction='${p.monotone}'),${comment}`;
  }).join('\n');
  return `from circuit_filter import Param, FilterConfig

# Per-parameter biological constraints
# ddt_max: max |Δlog10(param)| per measurement step
# direction: 'free' | 'up' | 'down'
constraints = {
${rows}
}

# Symmetry: enforce Ra > Rb (apical > basolateral convention)
canonical = 'Ra_gt_Rb'

config = FilterConfig(
    use_transformer = ${cfg.useTransformer ? 'True' : 'False'},
    use_rbpf        = ${cfg.useRBPF ? 'True' : 'False'},
    use_kalman      = ${cfg.useKalman ? 'True' : 'False'},
    use_ecm         = ${cfg.useEcm ? 'True' : 'False'},
)
`;
}

function codeToConstraints(code: string, prev: { useTransformer: boolean; useRBPF: boolean; useKalman: boolean; useEcm: boolean; constraints: Record<string, ParamConstraint> }): { constraints: Record<string, ParamConstraint>; useTransformer: boolean; useRBPF: boolean; useKalman: boolean; useEcm: boolean } {
  const constraints: Record<string, ParamConstraint> = {};
  for (const [name, def] of Object.entries(DEFAULT_CONSTRAINTS)) {
    const re = new RegExp(`'${name}'\\s*:\\s*Param\\(([^)]+)\\)`);
    const m = code.match(re);
    if (!m) { constraints[name] = { ...def }; continue; }
    const args = m[1];
    const getN = (key: string, fallback: number) => { const r = args.match(new RegExp(`${key}=([0-9e.+\\-]+)`)); return r ? parseFloat(r[1]) : fallback; };
    const dirM = args.match(/direction='(free|up|down)'/);
    const disabledLine = new RegExp(`'${name}'[^\\n]*#\\s*disabled`).test(code);
    constraints[name] = {
      enabled: !disabledLine,
      min: getN('min', def.min),
      max: getN('max', def.max),
      ddtMax: getN('ddt_max', def.ddtMax),
      monotone: (dirM?.[1] as 'free' | 'up' | 'down') ?? def.monotone,
    };
  }
  const boolFlag = (key: string, fallback: boolean) => { const m = code.match(new RegExp(`${key}\\s*=\\s*(True|False)`)); return m ? m[1] === 'True' : fallback; };
  return {
    constraints,
    useTransformer: boolFlag('use_transformer', prev.useTransformer),
    useRBPF:        boolFlag('use_rbpf',        prev.useRBPF),
    useKalman:      boolFlag('use_kalman',       prev.useKalman),
    useEcm:         boolFlag('use_ecm',          prev.useEcm),
  };
}

interface SourceDataCfg         { kind: 'source-data';           selection: string; syntheticTrajIdx: number | null; }
interface ProcessMLCfg {
  kind: 'process-ml';
  useTransformer: boolean;
  useRBPF: boolean;
  useKalman: boolean;
  useEcm: boolean;
  modelOverride?: string;
  constraints: Record<string, ParamConstraint>;
  editMode: 'visual' | 'code';
  modelCode: string;
}
interface VizResultsCfg         { kind: 'viz-results'; }
interface VizTrajectoryCfg      { kind: 'viz-trajectory';        param: TrajectoryParam; view: TrajView; }
interface VizDiagnosticsCfg     { kind: 'viz-diagnostics';       paramIdx: number; }
interface SourceObservabilityCfg{ kind: 'source-observability';  _placeholder: true; }
interface VizMechanismCfg       { kind: 'viz-mechanism';         _placeholder: true; }
interface TimeAnnotation { id: string; label: string; tStart: number; tEnd: number; color: string; isHidden?: boolean; }
interface VizSpiderplotCfg { kind: 'viz-spiderplot'; showVertGrid: boolean; annotations: TimeAnnotation[]; timeUnit: 'h' | 'm'; }

type NodeConfig =
  | SourceDataCfg
  | ProcessMLCfg
  | VizResultsCfg
  | VizTrajectoryCfg
  | VizDiagnosticsCfg
  | SourceObservabilityCfg
  | VizMechanismCfg
  | VizSpiderplotCfg;

// Output data that flows through edges
type ImpedanceSeries = { Z_real: number[]; Z_imag: number[]; frequencies: number[]; time_min: number };
type NodeOutput =
  | { kind: 'impedance-series'; sequences: ImpedanceSeries[]; groundTruth: Record<ParameterKey, number[]> | null; atpLo: number | null; atpHi: number | null; label: string; idGt: { names: string[]; series: number[][]; dtMinutes: number | null } | null }
  | { kind: 'ml-result'; result: TimeSeriesResult }
  | { kind: 'observability'; result: ObservabilityResult };

interface GraphNode {
  id: string; type: NodeType;
  x: number; y: number; width: number; height: number;
  config: NodeConfig;
  output: NodeOutput | null;
  status: 'idle' | 'running' | 'done' | 'error';
  stale?: boolean;
  enabled?: boolean;
  errorMsg: string | null;
  progress: { done: number; total: number } | null;
}

interface GraphEdge {
  id: string;
  fromNodeId: string; fromPortId: string;
  toNodeId: string;   toPortId: string;
}

function portPos(n: GraphNode, portId: string): { x: number; y: number } {
  const port = NODE_DEFS[n.type].ports.find(p => p.id === portId);
  const yFrac = port?.yFrac ?? 0.5;
  const x = port?.side === 'out' ? n.x + n.width : n.x;
  return { x, y: n.y + n.height * yFrac };
}

function edgePath(
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  _fromNode: GraphNode | undefined,
  _toNode: GraphNode | undefined,
  _allNodes: GraphNode[]
): string {
  const dx = p2.x - p1.x;
  const dist = Math.hypot(dx, p2.y - p1.y);

  if (dx >= 0) {
    const h = Math.max(50, dist * 0.4);
    return `M${p1.x},${p1.y} C${p1.x + h},${p1.y} ${p2.x - h},${p2.y} ${p2.x},${p2.y}`;
  }

  // Backward: exit right from source, arc through vertical midpoint, enter from left at target
  const escape = Math.max(70, Math.abs(dx) * 0.3 + 40);
  const midY = (p1.y + p2.y) / 2;
  const ex1 = p1.x + escape;
  const ex2 = p2.x - escape;
  const midX = (ex1 + ex2) / 2;
  return [
    `M${p1.x},${p1.y}`,
    `C${ex1},${p1.y} ${ex1},${midY} ${midX},${midY}`,
    `C${ex2},${midY} ${ex2},${p2.y} ${p2.x},${p2.y}`,
  ].join(' ');
}

// Preview path while dragging a new connection — follows cursor with a smooth curve
function pendingEdgePath(p1: { x: number; y: number }, p2: { x: number; y: number }): string {
  const dx = p2.x - p1.x;
  const dist = Math.hypot(dx, p2.y - p1.y);
  const h = dx >= 0
    ? Math.max(60, dist * 0.42)
    : Math.max(80, dist * 0.45 + Math.abs(dx) * 0.3);
  return `M${p1.x},${p1.y} C${p1.x + h},${p1.y} ${p2.x - h},${p2.y} ${p2.x},${p2.y}`;
}

function makeDefaultConfig(type: NodeType, _refParams: CircuitParameters): NodeConfig {
  switch (type) {
    case 'source-data': return { kind: 'source-data', selection: 'real:0', syntheticTrajIdx: null };
    case 'process-ml': {
      const base = { useTransformer: true, useRBPF: true, useKalman: true, useEcm: true };
      return { kind: 'process-ml', ...base, modelOverride: undefined, constraints: { ...DEFAULT_CONSTRAINTS }, editMode: 'visual', modelCode: constraintsToCode(DEFAULT_CONSTRAINTS, base) };
    }
    case 'viz-results':           return { kind: 'viz-results' };
    case 'viz-trajectory':        return { kind: 'viz-trajectory', param: 'TER', view: 'trajectory' };
    case 'viz-diagnostics':       return { kind: 'viz-diagnostics', paramIdx: 0 };
    case 'source-observability':  return { kind: 'source-observability', _placeholder: true };
    case 'viz-mechanism':         return { kind: 'viz-mechanism', _placeholder: true };
    case 'viz-spiderplot':        return { kind: 'viz-spiderplot', showVertGrid: false, annotations: [], timeUnit: 'h' };
  }
}


interface ZoomActions {
  zoomIn: () => void;
  zoomOut: () => void;
  zoomReset: () => void;
}

interface AIModelTabProps {
  groundTruthParams: CircuitParameters;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  onZoomChange?: (scale: number) => void;
  zoomActionsRef?: React.MutableRefObject<ZoomActions | null>;
}

// ---- Pure helpers ----




function emptyResult(): TimeSeriesResult {
  return {
    time_min: [],
    predictions: {
      R1: { mean: [], q25: [], q75: [] },
      C1: { mean: [], q25: [], q75: [] },
      R2: { mean: [], q25: [], q75: [] },
      TER: { mean: [], q25: [], q75: [] },
      TEC: { mean: [], q25: [], q75: [] },
      Ra: { mean: [], q25: [], q75: [] },
      Rb: { mean: [], q25: [], q75: [] },
      Ca: { mean: [], q25: [], q75: [] },
      Cb: { mean: [], q25: [], q75: [] },
    },
    ecm: { R1: [], C1: [], R2: [], TER: [], TEC: [], Ra: [], Rb: [], Ca: [], Cb: [] },
    attribution: { d_Ra: [], d_Rb: [], d_Ca: [], d_Cb: [], d_Rsh: [], d_RaRb: [] },
    groundTruth: null,
    atpLo: null,
    atpHi: null,
    timing: { dl_ms: [], ecm_ms: [] },
    pRaIsR1: [],
    pCaIsC1: [],
    rawSamples: [],
    posteriorStd: [],
    analyticsPerTimepoint: [],
    analyticsLast: null,
    dlResnorm: [],
    ecmResnorm: [],
    degHR: [],
    degHC: [],
    particleScatter: [],
    smcEss: [],
    smcPCaGtCb: [],
    smcPRaLtRb: [],
    hypWeights: [],
    smoothedTrajectories: [],
    clusterPaths: [],
    ecmColdPath: [],
    mdnPredictions: { R1: { mean: [], q25: [], q75: [] }, C1: { mean: [], q25: [], q75: [] },
      R2: { mean: [], q25: [], q75: [] }, TER: { mean: [], q25: [], q75: [] },
      TEC: { mean: [], q25: [], q75: [] }, Ra: { mean: [], q25: [], q75: [] },
      Rb: { mean: [], q25: [], q75: [] }, Ca: { mean: [], q25: [], q75: [] },
      Cb: { mean: [], q25: [], q75: [] } },
    mdnEntropy: [],
    kalman: null,
    changepoints: [],
    admissibility: null,
    relabelingFrac: [],
    posteriorSummary: null,
    mechanism: null,
  };
}


// ---- Main component ----

const GRAPH_STORAGE_KEY = 'ai-model-graph-v2';

function loadGraph(): { nodes: GraphNode[]; edges: GraphEdge[]; nextId: number } | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(GRAPH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { nodes: GraphNode[]; edges: unknown[]; nextId: number };
    const nodes = parsed.nodes.map(n => {
      // Migrate old node types to source-data
      const rawType = (n as unknown as { type: string }).type;
      const isSrcLegacy = rawType === 'eval-rbpf' || rawType === 'source-real' || rawType === 'source-synthetic';
      const migratedType: NodeType = isSrcLegacy ? 'source-data' : n.type as NodeType;
      const rawKind = (n.config as unknown as { kind: string }).kind;
      const isKindLegacy = rawKind === 'eval-rbpf' || rawKind === 'source-real' || rawKind === 'source-synthetic';
      const rawCfg = n.config as unknown as Record<string, unknown>;
      const migratedConfig: NodeConfig = isKindLegacy
        ? {
            kind: 'source-data' as const,
            selection: rawKind === 'source-real'
              ? `real:${(rawCfg.selectedSampleIdx as number) ?? 0}`
              : `synthetic:${(rawCfg.pathology as string) ?? 'random'}`,
            syntheticTrajIdx: (rawCfg.trajIdx as number | null) ?? null,
          } as SourceDataCfg
        // also migrate intermediate format that had mode/selectedSampleIdx/pathology/trajIdx
        : rawKind === 'source-data' && 'mode' in rawCfg
          ? {
              kind: 'source-data' as const,
              selection: rawCfg.mode === 'real'
                ? `real:${(rawCfg.selectedSampleIdx as number) ?? 0}`
                : `synthetic:${(rawCfg.pathology as string) ?? 'random'}`,
              syntheticTrajIdx: (rawCfg.trajIdx as number | null) ?? null,
            } as SourceDataCfg
          : n.config;
      return { ...n, type: migratedType, config: migratedConfig, output: null, status: 'idle' as const, errorMsg: null, progress: null, stale: false, enabled: (n as GraphNode).enabled ?? true };
    });
    // Migrate edges: add default portIds if missing (old format)
    const edges: GraphEdge[] = (parsed.edges as Array<{ id: string; fromNodeId: string; fromPortId?: string; toNodeId: string; toPortId?: string }>)
      .map(e => ({ id: e.id, fromNodeId: e.fromNodeId, fromPortId: e.fromPortId ?? 'out', toNodeId: e.toNodeId, toPortId: e.toPortId ?? 'in-1' }));
    return { nodes, edges, nextId: parsed.nextId };
  } catch {
    return null;
  }
}

function saveGraph(nodes: GraphNode[], edges: GraphEdge[], nextId: number) {
  if (typeof window === 'undefined') return;
  try {
    const stripped = nodes.map(n => ({ ...n, output: null, status: 'idle' as const, errorMsg: null, progress: null, stale: false }));
    localStorage.setItem(GRAPH_STORAGE_KEY, JSON.stringify({ nodes: stripped, edges, nextId }));
  } catch { /* storage full — ignore */ }
}

// ---- Compare layouts — source-data node groups (synthetic mode) ----


export const AIModelTab: React.FC<AIModelTabProps> = ({ groundTruthParams, minFreq: _minFreq, maxFreq: _maxFreq, numPoints: _numPoints, onZoomChange, zoomActionsRef }) => {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [nextId, setNextId] = useState<number>(0);
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [showEdges, setShowEdges] = useState(true);
  const [pendingDeleteNodeId, setPendingDeleteNodeId] = useState<string | null>(null);
  const [realSamples, setRealSamples] = useState<RealSample[]>([]);
  const realSamplesRef = useRef<RealSample[]>([]);   // always-current mirror — avoids stale closure in runNode
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);
  const [sampleLoadError, setSampleLoadError] = useState<string | null>(null);
  const hasLoadedRef = useRef(false);
  const dragRef = useRef<{ nodeId: string; sx: number; sy: number; ox: number; oy: number } | null>(null);
  const dragPosRef = useRef<{ nodeId: string; x: number; y: number } | null>(null);
  const rafRef = useRef<number>(0);
  const resizeRef = useRef<{ nodeId: string; sx: number; sy: number; ow: number; oh: number } | null>(null);
  const connectRef = useRef<{ fromNodeId: string; fromPortId: string } | null>(null);
  const [pendingMouse, setPendingMouse] = useState<{ x: number; y: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [vp, setVp] = useState({ x: 80, y: 80, scale: 1 });
  const vpRef = useRef({ x: 80, y: 80, scale: 1 });
  const [isSpaceDown, setIsSpaceDown] = useState(false);
  const isSpaceRef = useRef(false);
  const panRef = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);
  const abortRefs = useRef<Map<string, AbortController>>(new Map());
  const edgesRef = useRef(edges);
  const nodesRef = useRef(nodes);
  const hydratedRef = useRef(false);
  useEffect(() => { edgesRef.current = edges; }, [edges]);
  useEffect(() => { vpRef.current = vp; }, [vp]);

  // Expose zoom state and actions to parent
  useEffect(() => { onZoomChange?.(vp.scale); }, [vp.scale, onZoomChange]);
  if (zoomActionsRef) {
    zoomActionsRef.current = {
      zoomIn:    () => zoomTo(1.25),
      zoomOut:   () => zoomTo(1 / 1.25),
      zoomReset: () => { const newVp = { x: 80, y: 80, scale: 1 }; vpRef.current = newVp; setVp(newVp); },
    };
  }
  // Keep nodesRef in sync — updated synchronously inside setNodes so downstream
  // auto-run logic always sees the latest node outputs without waiting for a render.
  const updateNode = useCallback((id: string, updates: Partial<GraphNode>) =>
    setNodes(prev => {
      const next = prev.map(n => n.id === id ? { ...n, ...updates } : n);
      nodesRef.current = next;
      return next;
    }), []);

  const removeNode = useCallback((id: string) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setEdges(prev => prev.filter(e => e.fromNodeId !== id && e.toNodeId !== id));
    setFocusedNodeId(prev => prev === id ? null : prev);
  }, []);

  const removeEdgesToPort   = useCallback((toNodeId: string,   toPortId: string)       => setEdges(prev => prev.filter(e => !(e.toNodeId   === toNodeId   && e.toPortId   === toPortId))), []);
  const removeEdgesFromPort = useCallback((fromNodeId: string, fromPortId: string)     => setEdges(prev => prev.filter(e => !(e.fromNodeId === fromNodeId && e.fromPortId === fromPortId))), []);

  const addNode = useCallback((type: NodeType) => {
    const id = `n${nextId}`;
    const def = NODE_DEFS[type];
    // Place at viewport center so the node appears where the user is looking
    const el = scrollRef.current;
    const { x: vpX, y: vpY, scale } = vpRef.current;
    const centerX = el ? (el.clientWidth  / 2 - vpX) / scale - def.defaultWidth  / 2 : 80;
    const centerY = el ? (el.clientHeight / 2 - vpY) / scale - def.defaultHeight / 2 : 80;
    // Stagger slightly when multiple nodes are added in a row
    const off = (nextId * 24) % 120;
    setNodes(prev => [...prev, {
      id, type, x: centerX + off, y: centerY + off,
      width: def.defaultWidth, height: def.defaultHeight,
      config: makeDefaultConfig(type, groundTruthParams),
      output: null, status: 'idle', stale: false, enabled: true, errorMsg: null, progress: null,
    }]);
    setFocusedNodeId(id);
    setNextId(n => n + 1);
    setShowAddMenu(false);
  }, [nextId, groundTruthParams]);

  const addEdge = useCallback((fromNodeId: string, fromPortId: string, toNodeId: string, toPortId: string) => {
    const id = `e${fromNodeId}.${fromPortId}>${toNodeId}.${toPortId}`;
    // Only remove existing edge to the same target port — allows multiple ports per node
    setEdges(prev => [...prev.filter(e => !(e.toNodeId === toNodeId && e.toPortId === toPortId)), { id, fromNodeId, fromPortId, toNodeId, toPortId }]);
  }, []);

  // Returns the output from the first connected in-port, or a specific port by ID.
  // Uses refs so cascaded auto-runs see fresh upstream outputs without waiting for render.
  const getNodeInput = useCallback((nodeId: string, portId?: string): NodeOutput | null => {
    const node = nodesRef.current.find(n => n.id === nodeId);
    if (!node) return null;
    const targetPortId = portId ?? NODE_DEFS[node.type].ports.find(p => p.side === 'in')?.id;
    if (!targetPortId) return null;
    const edge = edgesRef.current.find(e => e.toNodeId === nodeId && e.toPortId === targetPortId);
    return edge ? (nodesRef.current.find(n => n.id === edge.fromNodeId)?.output ?? null) : null;
  }, []);


  const toCanvas = useCallback((cx: number, cy: number) => {
    const el = scrollRef.current;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    const { x, y, scale } = vpRef.current;
    return { x: (cx - rect.left - x) / scale, y: (cy - rect.top - y) / scale };
  }, []);

  // Persist graph layout/config (not runtime output) to localStorage.
  // Declared before the hydrate effect so it fires first on mount while
  // hydratedRef is still false — preventing the empty initial state from
  // overwriting stored data before hydration completes.
  useEffect(() => {
    if (!hydratedRef.current) return;
    saveGraph(nodes, edges, nextId);
  }, [nodes, edges, nextId]);

  // Hydrate from localStorage after mount. Runs after the persist effect on
  // the initial render (effects fire in declaration order), so the persist
  // guard above is still false when this fires — safe to load and set state.
  useEffect(() => {
    const saved = loadGraph();
    if (saved) {
      setNodes(saved.nodes);
      setEdges(saved.edges);
      setNextId(saved.nextId);
    }
    hydratedRef.current = true;
  }, []);

  // Auto-run process nodes when they become stale (upstream completed).
  // Viz nodes re-render automatically via props; only process nodes need explicit re-run.
  useEffect(() => {
    const stale = nodes.find(n =>
      n.stale &&
      n.enabled !== false &&
      n.status !== 'running' &&
      n.status !== 'error' &&
      (n.config.kind === 'process-ml' || n.config.kind === 'source-observability')
    );
    if (stale) runNode(stale.id);
  // runNode is intentionally excluded — it changes on every nodes update which
  // would cause unnecessary re-evaluations. The ref-based getNodeInput inside
  // runNode always sees fresh state regardless.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes]);

  // Eager-load real samples on mount so the dropdown is populated immediately
  useEffect(() => {
    if (hasLoadedRef.current) return;
    setIsLoadingSamples(true);
    setSampleLoadError(null);
    fetch(`${ML_API_BASE}/real_sample_data`)
      .then(r => { if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`); return r.json(); })
      .then(data => {
        const s = data.samples as RealSample[];
        realSamplesRef.current = s;   // sync ref before state — runNode reads ref
        setRealSamples(s);
        hasLoadedRef.current = true;
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Unknown error';
        setSampleLoadError(msg);
      })
      .finally(() => setIsLoadingSamples(false));
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (panRef.current) {
        const { sx, sy, ox, oy } = panRef.current;
        const newVp = { ...vpRef.current, x: ox + e.clientX - sx, y: oy + e.clientY - sy };
        vpRef.current = newVp;
        setVp(newVp);
        return;
      }
      if (dragRef.current) {
        const { nodeId, sx, sy, ox, oy } = dragRef.current;
        const s = vpRef.current.scale;
        dragPosRef.current = {
          nodeId,
          x: ox + (e.clientX - sx) / s,
          y: oy + (e.clientY - sy) / s,
        };
        if (!rafRef.current) {
          rafRef.current = requestAnimationFrame(() => {
            rafRef.current = 0;
            const pos = dragPosRef.current;
            if (pos) setNodes(prev => prev.map(n => n.id === pos.nodeId ? { ...n, x: pos.x, y: pos.y } : n));
          });
        }
      }
      if (resizeRef.current) {
        const { nodeId, sx, sy, ow, oh } = resizeRef.current;
        const s = vpRef.current.scale;
        setNodes(prev => prev.map(n => n.id === nodeId
          ? { ...n, width: Math.max(220, ow + (e.clientX - sx) / s), height: Math.max(150, oh + (e.clientY - sy) / s) } : n));
      }
      if (connectRef.current) setPendingMouse(toCanvas(e.clientX, e.clientY));
    };
    const onUp = () => {
      panRef.current = null;
      if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; }
      const pos = dragPosRef.current;
      if (pos) setNodes(prev => prev.map(n => n.id === pos.nodeId ? { ...n, x: pos.x, y: pos.y } : n));
      dragPosRef.current = null;
      dragRef.current = null;
      resizeRef.current = null;
      document.body.style.cursor = '';
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  }, [toCanvas]);

  // Wheel: pinch-to-zoom (ctrlKey) + trackpad pan
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    // Returns true when the event originates inside a scrollable node body that
    // still has room to scroll in the gesture direction. In that case we let the
    // browser handle the scroll naturally instead of panning the canvas.
    const targetIsScrollable = (target: EventTarget | null): boolean => {
      let node = target as HTMLElement | null;
      while (node && node !== el) {
        const style = window.getComputedStyle(node);
        const oy = style.overflowY;
        if ((oy === 'auto' || oy === 'scroll') && node.scrollHeight > node.clientHeight) {
          return true;
        }
        node = node.parentElement;
      }
      return false;
    };

    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) {
        // Pinch-to-zoom always takes over — prevent page zoom
        e.preventDefault();
        const rect = el.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        // 0.996 ≈ 3× more sensitive than the previous 0.999
        const factor = Math.pow(0.996, e.deltaY);
        setVp(prev => {
          const newScale = Math.max(0.08, Math.min(4, prev.scale * factor));
          const ratio = newScale / prev.scale;
          const newVp = { x: cx - (cx - prev.x) * ratio, y: cy - (cy - prev.y) * ratio, scale: newScale };
          vpRef.current = newVp;
          return newVp;
        });
      } else {
        // Two-finger scroll: if the pointer is over a scrollable node body let
        // the browser scroll that element; otherwise pan the canvas.
        if (targetIsScrollable(e.target)) return;
        e.preventDefault();
        setVp(prev => {
          const newVp = { ...prev, x: prev.x - e.deltaX, y: prev.y - e.deltaY };
          vpRef.current = newVp;
          return newVp;
        });
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Space key: hold space + drag to pan (Figma-style)
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) {
        isSpaceRef.current = true;
        setIsSpaceDown(true);
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        isSpaceRef.current = false;
        setIsSpaceDown(false);
        panRef.current = null;
      }
    };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); };
  }, []);

  const zoomTo = useCallback((factor: number, pivotX?: number, pivotY?: number) => {
    const el = scrollRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const cx = pivotX ?? rect.width / 2;
    const cy = pivotY ?? rect.height / 2;
    setVp(prev => {
      const newScale = Math.max(0.08, Math.min(4, prev.scale * factor));
      const ratio = newScale / prev.scale;
      const newVp = { x: cx - (cx - prev.x) * ratio, y: cy - (cy - prev.y) * ratio, scale: newScale };
      vpRef.current = newVp;
      return newVp;
    });
  }, []);

  const propagateStale = useCallback((completedNodeId: string) => {
    setNodes(prev => {
      const currentEdges = edgesRef.current;
      const visited = new Set<string>();
      const queue = [completedNodeId];
      while (queue.length > 0) {
        const cur = queue.shift()!;
        currentEdges.filter(e => e.fromNodeId === cur).forEach(e => {
          if (!visited.has(e.toNodeId)) { visited.add(e.toNodeId); queue.push(e.toNodeId); }
        });
      }
      return prev.map(n => visited.has(n.id) && n.status === 'done' ? { ...n, stale: true } : n);
    });
  }, []);

  const runNode = useCallback(async (nodeId: string) => {
    const n = nodes.find(x => x.id === nodeId);
    if (!n || n.enabled === false) return;
    updateNode(nodeId, { status: 'running', errorMsg: null, progress: null, stale: false });

    try {
      if (n.config.kind === 'source-data') {
        const cfg = n.config;
        const colonIdx = cfg.selection.indexOf(':');
        const selType = cfg.selection.slice(0, colonIdx);
        const selVal  = cfg.selection.slice(colonIdx + 1);

        if (selType === 'real') {
          let samples = realSamplesRef.current;
          if (!samples.length) {
            const res = await fetch(`${ML_API_BASE}/real_sample_data`);
            if (!res.ok) throw new Error(`API ${res.status}`);
            const data = await res.json();
            samples = data.samples as RealSample[];
            realSamplesRef.current = samples;
            setRealSamples(samples);
            hasLoadedRef.current = true;
          }
          const sample = samples[parseInt(selVal, 10) || 0];
          if (!sample) throw new Error('Sample not found');
          const sequences = sample.timepoints.map(tp => ({ Z_real: tp.Z_real, Z_imag: tp.Z_imag, frequencies: tp.frequencies, time_min: tp.time_min }));
          const gt: Record<ParameterKey, number[]> = { R1: [], C1: [], R2: [], TER: [], TEC: [], Ra: [], Rb: [], Ca: [], Cb: [] };
          for (const tp of sample.timepoints) {
            for (const k of (['R1','C1','R2','TER','TEC'] as ParameterKey[])) gt[k].push(tp.ground_truth[k]);
            gt.Ra.push(tp.ground_truth.Ra/1000); gt.Rb.push(tp.ground_truth.Rb/1000);
            gt.Ca.push(tp.ground_truth.Ca*1e6);  gt.Cb.push(tp.ground_truth.Cb*1e6);
          }
          const realDt = sample.timepoints.length > 1 ? sample.timepoints[1].time_min - sample.timepoints[0].time_min : sample.interval_min;
          updateNode(nodeId, { status: 'done', output: { kind: 'impedance-series', sequences, groundTruth: gt, atpLo: sample.atp_lo, atpHi: sample.atp_hi, label: `${sample.name} · ${sample.donor} · ${sample.timepoints.length}pt`, idGt: { names: ['TER','TEC','R2'], series: sample.timepoints.map(tp => [tp.ground_truth.TER, tp.ground_truth.TEC, tp.ground_truth.R2 ?? 0]), dtMinutes: realDt } } });
          propagateStale(nodeId);
          return;
        }

        if (selType === 'synthetic') {
          const pathology = selVal === 'random' ? null : selVal;
          const res = await fetch(`${ML_API_BASE}/synthetic_trajectory`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pathology, traj_idx: cfg.syntheticTrajIdx }),
          });
          if (!res.ok) throw new Error(`API ${res.status}`);
          const data = await res.json() as { error?: string; pathology: string; traj_idx: number; time_min: number[]; dt_minutes?: number; frequencies: number[]; Z_real: number[][]; Z_imag: number[][]; id_gt: number[][]; id_names: string[]; raw_gt?: number[][] };
          if (data.error) throw new Error(data.error);
          const sequences: ImpedanceSeries[] = data.Z_real.map((zr, i) => ({
            Z_real: zr, Z_imag: data.Z_imag[i], frequencies: data.frequencies, time_min: data.time_min[i],
          }));
          // Convert idGt + raw_gt → groundTruth in display units (kΩ, µF)
          const synGt: Record<ParameterKey, number[]> = { R1: [], C1: [], R2: [], TER: [], TEC: [], Ra: [], Rb: [], Ca: [], Cb: [] };
          const terIdx = data.id_names.indexOf('TER');
          const tecIdx = data.id_names.indexOf('TEC');
          const rshIdx = data.id_names.indexOf('Rsh');
          if (terIdx >= 0) synGt.TER = data.id_gt.map(row => Math.pow(10, row[terIdx]) / 1000);
          if (tecIdx >= 0) synGt.TEC = data.id_gt.map(row => Math.pow(10, row[tecIdx]) * 1e6);
          if (rshIdx >= 0) synGt.R2  = data.id_gt.map(row => Math.pow(10, row[rshIdx]) / 1000);
          if (data.raw_gt) {
            // raw_gt: log10[Ra, Rb, Ca, Cb, Rsh] in Ω/F units
            synGt.Ra = data.raw_gt.map(r => Math.pow(10, r[0]) / 1000);
            synGt.Rb = data.raw_gt.map(r => Math.pow(10, r[1]) / 1000);
            synGt.Ca = data.raw_gt.map(r => Math.pow(10, r[2]) * 1e6);
            synGt.Cb = data.raw_gt.map(r => Math.pow(10, r[3]) * 1e6);
          }
          updateNode(nodeId, {
            status: 'done',
            config: { ...cfg, syntheticTrajIdx: data.traj_idx },
            output: {
              kind: 'impedance-series',
              sequences,
              groundTruth: synGt,
              atpLo: null,
              atpHi: null,
              label: `${data.pathology.replace(/_/g, ' ')} · #${data.traj_idx} · ${sequences.length}pt`,
              idGt: { names: data.id_names, series: data.id_gt, dtMinutes: data.dt_minutes ?? null },
            },
          });
          propagateStale(nodeId);
          return;
        }
      }

      if (n.config.kind === 'source-observability') {
        const inputData = getNodeInput(nodeId);
        if (!inputData || inputData.kind !== 'impedance-series') throw new Error('Connect impedance data first');
        // Passthrough: expose the same impedance data on the out port
        updateNode(nodeId, { status: 'done', output: inputData });
        propagateStale(nodeId);
        return;
      }

      if (n.config.kind === 'process-ml') {
        const inputData = getNodeInput(nodeId);
        if (!inputData || inputData.kind !== 'impedance-series') throw new Error('Connect impedance data first');
        const prev2 = abortRefs.current.get(nodeId); if (prev2) prev2.abort();
        const ctrl = new AbortController(); abortRefs.current.set(nodeId, ctrl);
        const cfg2 = n.config;
        const acc = emptyResult(); acc.groundTruth = inputData.groundTruth; acc.atpLo = inputData.atpLo; acc.atpHi = inputData.atpHi;
        const snap = (): TimeSeriesResult => ({
          time_min: [...acc.time_min],
          predictions: Object.fromEntries(PARAM_KEYS.map(k => [k, { mean: [...acc.predictions[k].mean], q25: [...acc.predictions[k].q25], q75: [...acc.predictions[k].q75] }])) as Record<ParameterKey, DerivedSeries>,
          ecm: acc.ecm ? Object.fromEntries(PARAM_KEYS.map(k => [k, [...acc.ecm![k]]])) as Record<ParameterKey, number[]> : null,
          attribution: { d_Ra: [...acc.attribution.d_Ra], d_Rb: [...acc.attribution.d_Rb], d_Ca: [...acc.attribution.d_Ca], d_Cb: [...acc.attribution.d_Cb], d_Rsh: [...acc.attribution.d_Rsh], d_RaRb: [...acc.attribution.d_RaRb] },
          groundTruth: acc.groundTruth, atpLo: acc.atpLo, atpHi: acc.atpHi,
          timing: { dl_ms: [...acc.timing.dl_ms], ecm_ms: [...acc.timing.ecm_ms] },
          pRaIsR1: [...acc.pRaIsR1], pCaIsC1: [...acc.pCaIsC1], rawSamples: [...acc.rawSamples], posteriorStd: [...acc.posteriorStd],
          analyticsPerTimepoint: [...acc.analyticsPerTimepoint], analyticsLast: acc.analyticsLast,
          dlResnorm: [...acc.dlResnorm], ecmResnorm: [...acc.ecmResnorm], degHR: [...acc.degHR], degHC: [...acc.degHC],
          particleScatter: [...acc.particleScatter], smcEss: [...acc.smcEss], smcPCaGtCb: [...acc.smcPCaGtCb], smcPRaLtRb: [...acc.smcPRaLtRb], hypWeights: [...acc.hypWeights],
          smoothedTrajectories: [...acc.smoothedTrajectories], clusterPaths: [...acc.clusterPaths], ecmColdPath: [...acc.ecmColdPath],
          mdnPredictions: Object.fromEntries(PARAM_KEYS.map(k => [k, { mean: [...acc.mdnPredictions[k].mean], q25: [...acc.mdnPredictions[k].q25], q75: [...acc.mdnPredictions[k].q75] }])) as Record<ParameterKey, DerivedSeries>,
          mdnEntropy: [...acc.mdnEntropy], kalman: acc.kalman, changepoints: [...acc.changepoints], admissibility: acc.admissibility, relabelingFrac: [...acc.relabelingFrac],
          posteriorSummary: acc.posteriorSummary, mechanism: acc.mechanism,
          activePipeline: { useTransformer: cfg2.useTransformer, useRBPF: cfg2.useRBPF, useKalman: cfg2.useKalman, useEcm: cfg2.useEcm },
        });
        const response = await fetch(`${ML_API_BASE}/mc_temporal_analysis_stream`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify((() => {
            const paramOrder = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh'];
            const active = cfg2.constraints ? paramOrder.filter(p => cfg2.constraints[p]?.enabled) : [];
            const rbpfBounds = active.length > 0 ? { params: active, low: active.map(p => cfg2.constraints[p].min), high: active.map(p => cfg2.constraints[p].max) } : undefined;
            const ddtConstraints = active.length > 0 ? { params: active, max: active.map(p => cfg2.constraints[p].ddtMax), direction: active.map(p => cfg2.constraints[p].monotone) } : undefined;
            return { sequences: inputData.sequences, n_samples: 2000, n_display: 50, include_ecm: cfg2.useEcm, use_sequential: cfg2.useRBPF !== false, canonical_mode: 'Ra_gt_Rb', sample_id: `n${nodeId}_${Date.now()}`, model_override: cfg2.modelOverride || undefined, rbpf_bounds: rbpfBounds, ddt_constraints: ddtConstraints };
          })()),
          signal: ctrl.signal,
        });
        if (!response.ok) throw new Error(`API ${response.status}`);
        const reader = response.body!.getReader(); const decoder = new TextDecoder(); let buffer = '';
        outer: while (true) {
          const { done, value } = await reader.read(); if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n'); buffer = parts.pop() ?? '';
          for (const part of parts) {
            const dl = part.split('\n').find(l => l.startsWith('data: ')); if (!dl) continue;
            try {
              const p = JSON.parse(dl.slice(6));
              if (p.done) {
                if (p.smoothed_trajectories) acc.smoothedTrajectories = p.smoothed_trajectories as SmoothedTrajectory[];
                if (p.cluster_paths) acc.clusterPaths = p.cluster_paths as SmoothedTrajectory[];
                if (p.ecm_cold_path) acc.ecmColdPath = p.ecm_cold_path as EcmColdPoint[];
                if (p.kalman && cfg2.useKalman !== false) acc.kalman = p.kalman as KalmanResult;
                if (p.changepoints) acc.changepoints = p.changepoints as ChangePoint[];
                if (p.admissibility) acc.admissibility = p.admissibility as Admissibility;
                if (p.posterior_summary) acc.posteriorSummary = p.posterior_summary as Record<string, PosteriorParamSummary>;
                if (p.mechanism) acc.mechanism = p.mechanism as Mechanism;
                break outer;
              }
              if (p.error) continue;
              acc.time_min.push(p.time_min);
              for (const k of PARAM_KEYS) {
                acc.predictions[k].mean.push(p.predictions[k].mean); acc.predictions[k].q25.push(p.predictions[k].q25); acc.predictions[k].q75.push(p.predictions[k].q75);
                if (p.ecm && acc.ecm) acc.ecm[k].push(p.ecm[k]);
              }
              if (p.timing?.dl_ms != null) acc.timing.dl_ms.push(p.timing.dl_ms); if (p.timing?.ecm_ms != null) acc.timing.ecm_ms.push(p.timing.ecm_ms);
              if (p.dl_resnorm != null) acc.dlResnorm.push(p.dl_resnorm); if (p.ecm?.resnorm != null) acc.ecmResnorm.push(p.ecm.resnorm);
              const diag = p.smc_diag ?? p.rbpf_diag;
              if (diag) { acc.smcEss.push(diag.ess_frac ?? 0); acc.smcPCaGtCb.push(diag.p_Ca_gt_Cb ?? 0); acc.smcPRaLtRb.push(diag.p_Ra_lt_Rb ?? 0); if (diag.hyp_weights) acc.hypWeights.push(diag.hyp_weights); }
              if (p.mdn_entropy != null) acc.mdnEntropy.push(p.mdn_entropy); acc.relabelingFrac.push(p.relabeling_frac ?? null);
              if (p.analytics) { acc.analyticsPerTimepoint.push(p.analytics as AnalyticsSnapshot); acc.analyticsLast = p.analytics as AnalyticsSnapshot; } else acc.analyticsPerTimepoint.push(null);
              if (p.mdn_predictions) { for (const k of PARAM_KEYS) { const mp = (p.mdn_predictions as Record<ParameterKey,{mean:number;q25:number;q75:number}>)[k]; if (mp) { acc.mdnPredictions[k].mean.push(mp.mean); acc.mdnPredictions[k].q25.push(mp.q25); acc.mdnPredictions[k].q75.push(mp.q75); } } }
              updateNode(nodeId, { progress: { done: p.index + 1, total: p.total }, output: { kind: 'ml-result', result: snap() } });
            } catch { /* partial chunk */ }
          }
        }
        if (acc.ecm && acc.ecm.R1.length === 0) acc.ecm = null;
        updateNode(nodeId, { status: 'done', output: { kind: 'ml-result', result: snap() }, progress: null });
        propagateStale(nodeId);
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') { updateNode(nodeId, { status: 'idle' }); return; }
      updateNode(nodeId, { status: 'error', errorMsg: err instanceof Error ? err.message : 'Error', progress: null });
    }
  }, [nodes, getNodeInput, updateNode, propagateStale]);

  const stopNode = useCallback((nodeId: string) => {
    abortRefs.current.get(nodeId)?.abort();
    updateNode(nodeId, { status: 'idle', progress: null });
  }, [updateNode]);

  const cancelConnect = useCallback(() => { connectRef.current = null; setPendingMouse(null); }, []);

  return (
    <div className="h-full flex flex-col bg-[#0d0d0f] relative" onKeyDown={e => e.key === 'Escape' && cancelConnect()} tabIndex={-1}>

      {/* Toolbar — bottom center */}
      {(() => {
        const staleProcessNodes = nodes.filter(n => n.stale && n.type === 'process-ml');
        return staleProcessNodes.length > 0 ? (
          <button
            onClick={() => staleProcessNodes.forEach(n => runNode(n.id))}
            className="absolute bottom-16 left-1/2 -translate-x-1/2 z-30 px-3 py-1.5 text-[10px] font-mono rounded-full border border-[#c4a040]/40 bg-[#c4a040]/10 text-[#c4a040] hover:bg-[#c4a040]/20 transition-colors duration-100 flex items-center gap-1.5"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[#c4a040] animate-pulse"/>
            Recompute {staleProcessNodes.length} stale node{staleProcessNodes.length > 1 ? 's' : ''}
          </button>
        ) : null;
      })()}
      <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 z-30 ${PILL_BASE}`}>
        <button
          onClick={() => setShowEdges(v => !v)}
          className={showEdges ? LABEL_BTN_ACTIVE : LABEL_BTN_INACTIVE}
          title={showEdges ? 'Hide connections' : 'Show connections'}
        >
          <svg className={ICON.md} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {showEdges
              ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/>
              : <><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3l18 18"/></>
            }
          </svg>
          Wires
        </button>

        <div className={DIVIDER} />

        <div className="relative">
          <button
            onClick={() => setShowAddMenu(v => !v)}
            onBlur={() => setTimeout(() => setShowAddMenu(false), 150)}
            className={LABEL_BTN_ACTIVE}
          >
            <svg className={ICON.md} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4"/></svg>
            Add node
          </button>
          {showAddMenu && (
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 bg-[#0d0d0f] border border-[#1e1e24] rounded-lg overflow-hidden min-w-56 shadow-xl shadow-black/60">
              {([
                ['Sources',   ['source-data', 'source-observability']],
                ['Process',   ['process-ml']],
                ['Visualize', ['viz-trajectory', 'viz-diagnostics', 'viz-mechanism', 'viz-spiderplot']],
              ] as [string, NodeType[]][]).map(([group, types]) => (
                <React.Fragment key={group}>
                  <div className="px-3 py-1.5 text-[9px] text-[#454549] uppercase tracking-widest border-b border-[#1e1e24]">{group}</div>
                  {types.map(t => (
                    <button key={t} onMouseDown={() => addNode(t)}
                      className="w-full text-left px-3 py-2 text-xs text-[#7a7a82] hover:bg-[#131316] hover:text-[#dddde2] transition-colors duration-100 flex items-center gap-2.5">
                      <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: NODE_DEFS[t].portColor }}/>
                      {NODE_DEFS[t].label}
                    </button>
                  ))}
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-hidden"
        style={{
          backgroundImage: 'radial-gradient(circle, #1e1e24 1px, transparent 1px)',
          backgroundSize: `${24 * vp.scale}px ${24 * vp.scale}px`,
          backgroundPosition: `${vp.x % (24 * vp.scale)}px ${vp.y % (24 * vp.scale)}px`,
          cursor: isSpaceDown ? (panRef.current ? 'grabbing' : 'grab') : 'default',
        }}
        onMouseUp={cancelConnect}
        onMouseDown={e => {
          if (isSpaceRef.current && e.button === 0) {
            panRef.current = { sx: e.clientX, sy: e.clientY, ox: vpRef.current.x, oy: vpRef.current.y };
            e.preventDefault();
          }
        }}
        onClick={() => { if (!panRef.current) setFocusedNodeId(null); }}
      >
        <div style={{ position: 'absolute', left: 0, top: 0, transform: `translate(${vp.x}px, ${vp.y}px) scale(${vp.scale})`, transformOrigin: '0 0', width: 0, height: 0 }}>
          {/* SVG edge overlay — overflow visible so edges render beyond bounds */}
          <svg style={{ position: 'absolute', left: 0, top: 0, width: 8000, height: 6000, pointerEvents: 'none', overflow: 'visible', zIndex: 2 }}>
            <defs>
              {([
                [C_SOURCE,  'source'],
                [C_PROCESS, 'process'],
                [C_VIZ,     'viz'],
              ] as [string, string][]).map(([color, name]) => (
                <marker key={name} id={`arrow-${name}`} markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
                  <path d="M0,0.5 L6,3 L0,5.5 Z" fill={color} opacity="0.5"/>
                </marker>
              ))}
            </defs>
            {showEdges && edges.map(edge => {
              const from = nodes.find(n => n.id === edge.fromNodeId);
              const to   = nodes.find(n => n.id === edge.toNodeId);
              if (!from || !to) return null;
              const p1 = portPos(from, edge.fromPortId);
              const p2 = portPos(to, edge.toPortId);
              const cat = NODE_DEFS[from.type].category;
              const markerName = cat === 'viz' ? 'viz' : cat;
              // Edges connected to the focused node surface slightly on the emphasis system
              const isActive = focusedNodeId === from.id || focusedNodeId === to.id;
              return (
                <path key={edge.id}
                  d={edgePath(p1, p2, from, to, nodes)}
                  fill="none"
                  stroke={NODE_DEFS[from.type].portColor}
                  strokeWidth={isActive ? 1.2 : 1}
                  strokeOpacity={isActive ? 0.6 : 0.3}
                  markerEnd={`url(#arrow-${markerName})`}
                  style={{ transition: 'stroke-opacity 0.15s, stroke-width 0.15s' }}
                />
              );
            })}
            {connectRef.current && pendingMouse && (() => {
              const from = nodes.find(n => n.id === connectRef.current!.fromNodeId);
              if (!from) return null;
              const p1 = portPos(from, connectRef.current!.fromPortId);
              return (
                <path
                  d={pendingEdgePath(p1, pendingMouse)}
                  fill="none" stroke="#2a2a33" strokeWidth={1} strokeDasharray="4 3"
                />
              );
            })()}
          </svg>

          {nodes.map(node => {
            const def = NODE_DEFS[node.type];
            const isFocused = focusedNodeId === node.id;
            const inputData  = getNodeInput(node.id);
            const inputData2 = getNodeInput(node.id, 'in-2');
            const isConnecting = !!connectRef.current;
            const isDisabled = node.enabled === false;

            return (
              <div key={node.id}
                style={{ position: 'absolute', left: node.x, top: node.y, width: node.width, height: node.height, zIndex: isFocused ? 15 : 5, boxShadow: isFocused ? 'inset 0 1px 0 rgba(255,255,255,0.04)' : 'none', transition: 'box-shadow 0.15s' }}
                className={`flex flex-col rounded-lg border transition-colors duration-150 ${node.stale ? 'border-[#c4a040]/40' : isFocused ? 'border-[#2a2a33]' : 'border-[#1e1e24]'} bg-[#131316] ${isDisabled ? 'opacity-35' : ''}`}
                onClick={e => { e.stopPropagation(); setFocusedNodeId(node.id); }}
              >
                {/* Named port circles — left side = inputs, right side = outputs */}
                {def.ports.map(port => {
                  const isInput = port.side === 'in';
                  const hasEdge = isInput
                    ? edges.some(e => e.toNodeId === node.id && e.toPortId === port.id)
                    : edges.some(e => e.fromNodeId === node.id && e.fromPortId === port.id);
                  const isCompatible = isConnecting && connectRef.current ? (() => {
                    const fromNode = nodes.find(n => n.id === connectRef.current!.fromNodeId);
                    const fromPort = fromNode ? NODE_DEFS[fromNode.type].ports.find(p => p.id === connectRef.current!.fromPortId) : null;
                    return fromPort ? portsCompatible(fromPort, port) : false;
                  })() : false;
                  const filled = isInput ? (isCompatible || hasEdge) : hasEdge;
                  const portStyle: React.CSSProperties = {
                    position: 'absolute',
                    ...(isInput ? { left: -4 } : { right: -4 }),
                    top: `${port.yFrac * 100}%`,
                    transform: 'translateY(-50%)',
                    width: 8, height: 8, borderRadius: '50%', zIndex: 25,
                    border: `1.5px solid ${def.portColor}`,
                    background: filled ? def.portColor + 'b3' : '#0d0d0f',
                    opacity: isInput ? (isCompatible ? 1 : hasEdge ? 0.75 : 0.4) : (hasEdge ? 0.75 : 0.28),
                    cursor: isInput ? (hasEdge && !isConnecting ? 'pointer' : 'crosshair') : 'crosshair',
                    transition: 'background 0.1s, opacity 0.1s',
                  };
                  if (isInput) {
                    return (
                      <div key={port.id} style={portStyle}
                        title={`${port.label}${hasEdge ? ' · click to disconnect' : ' · drop here'}`}
                        onMouseUp={e => {
                          e.stopPropagation();
                          if (connectRef.current) {
                            const fromNode = nodes.find(n => n.id === connectRef.current!.fromNodeId);
                            const fromPort = fromNode ? NODE_DEFS[fromNode.type].ports.find(p => p.id === connectRef.current!.fromPortId) : null;
                            if (fromPort && portsCompatible(fromPort, port)) addEdge(connectRef.current.fromNodeId, connectRef.current.fromPortId, node.id, port.id);
                            connectRef.current = null; setPendingMouse(null);
                          }
                        }}
                        onClick={e => { e.stopPropagation(); if (!isConnecting && hasEdge) removeEdgesToPort(node.id, port.id); }}
                      />
                    );
                  } else {
                    return (
                      <div key={port.id} style={portStyle}
                        title={`${port.label}${hasEdge ? ' · drag to rewire · click to disconnect' : ' · drag to connect'}`}
                        onMouseDown={e => {
                          e.stopPropagation(); e.preventDefault();
                          connectRef.current = { fromNodeId: node.id, fromPortId: port.id };
                          setPendingMouse(portPos(node, port.id));
                        }}
                        onClick={e => { e.stopPropagation(); if (!pendingMouse && hasEdge) removeEdgesFromPort(node.id, port.id); }}
                      />
                    );
                  }
                })}

                {/* Header */}
                <div
                  className={isFocused ? NODE_HEADER_FOCUSED : NODE_HEADER}
                  style={{ borderLeft: `2px solid ${def.portColor}` }}
                  onMouseDown={e => {
                    setFocusedNodeId(node.id);
                    dragRef.current = { nodeId: node.id, sx: e.clientX, sy: e.clientY, ox: node.x, oy: node.y };
                    document.body.style.cursor = 'grabbing';
                    e.preventDefault();
                  }}
                >
                  <span className={`${TYPE.nodeLabel} flex-1 min-w-0 truncate select-none`}>{def.label}</span>
                  {node.status === 'running' && node.progress && (
                    <span className={TYPE.nodeProgress}>
                      {node.progress.done}/{node.progress.total}
                    </span>
                  )}
                  <ActiveToggle enabled={node.enabled} onToggle={e => { e.stopPropagation(); updateNode(node.id, { enabled: node.enabled === false }); }}/>
                  <button onClick={e => { e.stopPropagation(); setPendingDeleteNodeId(node.id); }} onMouseDown={e => e.stopPropagation()}
                    className={NODE_DELETE_BTN}
                    title="Delete node">
                    <svg className={ICON.md} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                    </svg>
                  </button>
                </div>

                {node.status === 'running' && (
                  <div className="h-px bg-[#1e1e24] flex-shrink-0">
                    <div className="h-full bg-[#c4a040]/50 transition-all" style={{ width: node.progress ? `${(node.progress.done/node.progress.total)*100}%` : '25%' }}/>
                  </div>
                )}

                {/* Body */}
                <div className={`flex-1 min-h-0 overflow-hidden bg-[#0d0d0f] rounded-b-lg ${isDisabled ? 'pointer-events-none' : ''}`}>
                  <NodeBody node={node} inputData={inputData} inputData2={inputData2} realSamples={realSamples}
                    isLoadingSamples={isLoadingSamples} sampleLoadError={sampleLoadError}
                    groundTruthParams={groundTruthParams}
                    onRun={() => runNode(node.id)} onStop={() => stopNode(node.id)}
                    onConfigChange={cfg => updateNode(node.id, { config: cfg })}
                    onSizeChange={(w, h) => updateNode(node.id, { width: w, height: h })}
                  />
                </div>

                {/* Resize handle */}
                <div style={{ position: 'absolute', bottom: 0, right: 0, width: 18, height: 18, cursor: 'se-resize', zIndex: 26 }}
                  onMouseDown={e => { resizeRef.current = { nodeId: node.id, sx: e.clientX, sy: e.clientY, ow: node.width, oh: node.height }; e.preventDefault(); e.stopPropagation(); }}>
                  <svg width="8" height="8" viewBox="0 0 8 8" className="absolute bottom-1.5 right-1.5 text-[#252529] hover:text-[#454549] transition-colors duration-100">
                    <path d="M7 1L1 7M7 4L4 7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                  </svg>
                </div>
              </div>
            );
          })}

        </div>
      </div>

      {pendingDeleteNodeId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" onMouseDown={e => e.stopPropagation()}>
          <div className="absolute inset-0 bg-black/60" onClick={() => setPendingDeleteNodeId(null)}/>
          <div className="relative bg-[#16161a] border border-[#2a2a33] rounded-lg shadow-xl p-5 w-72 flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <svg className={`${ICON.xl} text-[#c45a5a] flex-shrink-0`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
              </svg>
              <span className="text-[#dddde2] text-sm font-medium">Delete node?</span>
            </div>
            <p className="text-[#7a7a82] text-xs leading-relaxed">This will remove the node and all its connections. This action cannot be undone.</p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setPendingDeleteNodeId(null)}
                className="px-3 py-1.5 text-xs text-[#7a7a82] hover:text-[#dddde2] bg-[#1e1e24] hover:bg-[#2a2a33] rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => { removeNode(pendingDeleteNodeId); setPendingDeleteNodeId(null); }}
                className="px-3 py-1.5 text-xs text-white bg-[#c45a5a] hover:bg-[#d46a6a] rounded transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---- NodeBody ----

interface NodeBodyProps {
  node: GraphNode;
  inputData: NodeOutput | null;
  inputData2: NodeOutput | null;
  realSamples: RealSample[];
  isLoadingSamples: boolean;
  sampleLoadError: string | null;
  groundTruthParams: CircuitParameters;
  onRun: () => void;
  onStop: () => void;
  onConfigChange: (cfg: NodeConfig) => void;
  onSizeChange: (w: number, h: number) => void;
}

function NodeBody(p: NodeBodyProps) {
  switch (p.node.type) {
    case 'source-data':          return <SourceDataBody          {...p}/>;
    case 'process-ml':           return <ProcessMLBody           {...p}/>;
    case 'viz-results':          return <ResultsBody             {...p}/>;
    case 'viz-trajectory':       return <VizTrajectoryBody       {...p}/>;
    case 'viz-diagnostics':      return <VizDiagnosticsBody      {...p}/>;
    case 'source-observability': return <ObservabilityBody       {...p}/>;
    case 'viz-mechanism':        return <MechanismBody           {...p}/>;
    case 'viz-spiderplot':       return <VizSpiderplotBody       {...p}/>;
  }
}

function ActiveToggle({ enabled, onToggle }: { enabled?: boolean; onToggle: (e: React.MouseEvent) => void }) {
  const isOn = enabled !== false;
  return (
    <button onClick={onToggle} onMouseDown={e => e.stopPropagation()}
      className="focus:outline-none flex-shrink-0"
      title={isOn ? 'Disable node' : 'Enable node'}>
      <div className={`w-7 h-3.5 rounded-full transition-colors duration-150 relative ${isOn ? 'bg-[#3d7a60]' : 'bg-[#1e1e24]'}`}>
        <div className={`absolute top-0.5 w-2.5 h-2.5 rounded-full bg-[#dddde2] transition-all duration-150 ${isOn ? 'left-[14px]' : 'left-0.5'}`}/>
      </div>
    </button>
  );
}

function RunBtn({ status, onRun, onStop, label = 'Run', disabled, className = '' }: { status: GraphNode['status']; onRun: () => void; onStop: () => void; label?: string; disabled?: boolean; className?: string }) {
  return status === 'running'
    ? <button onClick={onStop} className={`px-2.5 py-1 text-[10px] rounded border border-[#7a3a3a]/60 text-[#7a3a3a] hover:text-[#a05050] transition-colors duration-100 focus:outline-none ${className}`}>Stop</button>
    : <button onClick={onRun} disabled={disabled} className={`px-2.5 py-1 text-[10px] rounded transition-colors duration-100 focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed ${status === 'done' ? 'border border-[#1e1e24] text-[#7a7a82] hover:border-[#2a2a33] hover:text-[#dddde2]' : 'border border-[#2a2a33] bg-[#131316] text-[#dddde2] hover:border-[#3a3a46] hover:text-white'} ${className}`}>{status === 'done' ? 'Rerun' : label}</button>;
}

// ---- Synthetic pathology options ----
const SYNTHETIC_PATHOLOGIES = [
  { value: 'random',             label: 'Random' },
  { value: 'healthy',            label: 'Healthy' },
  { value: 'maturation',         label: 'Maturation' },
  { value: 'barrier_breakdown',  label: 'Barrier Breakdown' },
  { value: 'apical_injury',      label: 'Apical Injury' },
  { value: 'basolateral_injury', label: 'Basolateral Injury' },
  { value: 'oxidative_stress',   label: 'Oxidative Stress' },
  { value: 'recovery',           label: 'Recovery' },
  { value: 'mixed_pathology',    label: 'Mixed Pathology' },
  { value: 'unknown',            label: 'Unknown' },
];

const DATA_PARAM_COLORS: Record<string, string> = {
  tau_big: '#6469a0', tau_small: '#9a6a9a', TER: '#c4a040', TEC: '#40a0a0', Rsh: '#5a9a60',
  R1: '#6469a0', R2: '#5a9a60', C1: '#9a6a9a',
};
const DATA_PARAM_LABELS: Record<string, string> = {
  tau_big: 'τ_big', tau_small: 'τ_sm', TER: 'TER', TEC: 'TEC', Rsh: 'Rsh',
  R1: 'R1', R2: 'R2', C1: 'C1',
};


function DataPreview({ output }: { output: Extract<NodeOutput, { kind: 'impedance-series' }> }) {
  const T = output.sequences.length;
  if (T === 0) return null;

  const freqs = output.sequences[0]?.frequencies ?? [];
  const fLo = freqs.length ? Math.min(...freqs) : null;
  const fHi = freqs.length ? Math.max(...freqs) : null;
  const fmtHz = (f: number) => f >= 1000 ? `${(f/1000).toFixed(0)}k` : f < 1 ? f.toFixed(3) : f.toFixed(0);
  const is3PEIS = output.atpHi != null && output.atpHi > (output.atpLo ?? 0);
  const dtMin = output.idGt?.dtMinutes ?? (T > 1 ? output.sequences[1].time_min - output.sequences[0].time_min : null);
  const dtLabel = dtMin != null ? (dtMin < 1 ? `${(dtMin * 60).toFixed(0)}s` : `${dtMin.toFixed(1)}min`) : null;

  // Nyquist thumbnail: first, middle, last
  const picks = T <= 3 ? Array.from({ length: T }, (_, i) => i) : [0, Math.floor(T / 2), T - 1];
  const NW = 90; const NH = 60;
  const allZr: number[] = []; const allZi: number[] = [];
  for (const i of picks) {
    allZr.push(...output.sequences[i].Z_real);
    allZi.push(...output.sequences[i].Z_imag.map(v => -v));
  }
  const nMg = 4;
  const minZr = Math.min(...allZr); const maxZr = Math.max(...allZr);
  const minZi = Math.min(...allZi); const maxZi = Math.max(...allZi);
  const rnZr = maxZr - minZr || 1; const rnZi = maxZi - minZi || 1;
  const nxX = (v: number) => nMg + ((v - minZr) / rnZr) * (NW - 2 * nMg);
  const nxY = (v: number) => NH - nMg - ((v - minZi) / rnZi) * (NH - 2 * nMg);
  const arcColors = ['#6469a0', '#9a6a9a', '#4ade80'];
  const arcs = picks.map((p, ci) => ({
    pts: output.sequences[p].Z_real.map((zr, fi) => `${nxX(zr).toFixed(1)},${nxY(-output.sequences[p].Z_imag[fi]).toFixed(1)}`).join(' '),
    color: arcColors[ci % arcColors.length],
    opacity: ci === picks.length - 1 ? 0.95 : 0.4,
  }));

  // Bode thumbnail: |Z| vs log frequency for first + last timepoint
  const BW = 90; const BH = 60;
  const bMg = 4;
  const bSeqs = picks.length > 0 ? [output.sequences[0], output.sequences[picks[picks.length - 1]]] : [];
  const allMag: number[] = [];
  for (const s of bSeqs) allMag.push(...s.Z_real.map((zr, fi) => Math.sqrt(zr * zr + s.Z_imag[fi] * s.Z_imag[fi])));
  const logFLo = fLo ? Math.log10(fLo) : 0;
  const logFHi = fHi ? Math.log10(fHi) : 5;
  const logFRange = logFHi - logFLo || 1;
  const minMag = Math.min(...allMag.filter(v => v > 0)); const maxMag = Math.max(...allMag);
  const logMagLo = Math.log10(Math.max(minMag, 1e-10)); const logMagHi = Math.log10(Math.max(maxMag, 1e-9));
  const logMagRange = logMagHi - logMagLo || 1;
  const bx = (f: number) => bMg + ((Math.log10(Math.max(f, 1e-10)) - logFLo) / logFRange) * (BW - 2 * bMg);
  const by = (m: number) => BH - bMg - ((Math.log10(Math.max(m, 1e-10)) - logMagLo) / logMagRange) * (BH - 2 * bMg);
  const bodeColors = ['#6469a0', '#4ade80'];
  const bodeCurves = bSeqs.map((s, ci) => ({
    pts: s.frequencies.map((f, fi) => {
      const mag = Math.sqrt(s.Z_real[fi] ** 2 + s.Z_imag[fi] ** 2);
      return `${bx(f).toFixed(1)},${by(mag).toFixed(1)}`;
    }).join(' '),
    color: bodeColors[ci % bodeColors.length],
    opacity: ci === bSeqs.length - 1 ? 0.9 : 0.4,
  }));
  // Decade tick marks on Bode x-axis
  const decadeTicks: number[] = [];
  for (let d = Math.ceil(logFLo); d <= Math.floor(logFHi); d++) decadeTicks.push(d);

  // Parameter sparklines with range annotation
  let paramSeries: Array<{ name: string; values: number[]; color: string }> = [];
  if (output.idGt) {
    paramSeries = output.idGt.names.map((n, k) => ({
      name: DATA_PARAM_LABELS[n] ?? n,
      values: output.idGt!.series.map(row => row[k]),
      color: DATA_PARAM_COLORS[n] ?? '#6469a0',
    }));
  } else if (output.groundTruth) {
    const gt = output.groundTruth;
    const entries: Array<[string, ParameterKey]> = [['TER','TER'],['TEC','TEC'],['R2','R2'],['R1','R1']];
    paramSeries = entries
      .filter(([, k]) => (gt[k]?.length ?? 0) > 0)
      .map(([n, k]) => ({ name: DATA_PARAM_LABELS[n] ?? n, values: gt[k], color: DATA_PARAM_COLORS[n] ?? '#6469a0' }));
  }

  return (
    <div className="flex flex-col gap-1 px-2 pt-1 pb-0 overflow-hidden">
      {/* Metadata row */}
      <div className="flex items-center gap-2 text-[7px] font-mono text-[#454549]">
        <span className="tabular-nums">{T}pt</span>
        {dtLabel && <span className="text-[#5a5a65]">Δt {dtLabel}</span>}
        {fLo != null && fHi != null && <span>{fmtHz(fLo)}–{fmtHz(fHi)} Hz</span>}
        {is3PEIS && <span className="text-[#3d7a60]">3PEIS</span>}
      </div>

      {/* Nyquist + Bode side by side */}
      <div className="flex gap-1.5">
        {/* Nyquist */}
        <div className="flex flex-col items-center gap-0.5">
          <span className="text-[6px] font-mono text-[#2a2a33]">Nyquist</span>
          <svg width={NW} height={NH} viewBox={`0 0 ${NW} ${NH}`} className="bg-[#0a0a0c] rounded border border-[#161618]">
            {arcs.map(({ pts, color, opacity }, i) => (
              <polyline key={i} points={pts} fill="none" stroke={color} strokeWidth="1" opacity={opacity} strokeLinejoin="round"/>
            ))}
          </svg>
        </div>
        {/* Bode */}
        <div className="flex flex-col items-center gap-0.5 flex-1">
          <span className="text-[6px] font-mono text-[#2a2a33]">Bode |Z|</span>
          <svg width="100%" height={BH} viewBox={`0 0 ${BW} ${BH}`} preserveAspectRatio="none" className="bg-[#0a0a0c] rounded border border-[#161618]">
            {decadeTicks.map(d => {
              const xd = bx(Math.pow(10, d));
              return <line key={d} x1={xd} y1={bMg} x2={xd} y2={BH - bMg} stroke="#1a1a22" strokeWidth="0.5"/>;
            })}
            {bodeCurves.map(({ pts, color, opacity }, i) => (
              <polyline key={i} points={pts} fill="none" stroke={color} strokeWidth="1" opacity={opacity} strokeLinejoin="round"/>
            ))}
            {fLo != null && <text x={bMg} y={BH - 1} fontSize="3.5" fill="#2a2a33">{fmtHz(fLo)}</text>}
            {fHi != null && <text x={BW - bMg} y={BH - 1} fontSize="3.5" fill="#2a2a33" textAnchor="end">{fmtHz(fHi)}</text>}
          </svg>
        </div>
      </div>

      {/* Parameter statistics table */}
      {paramSeries.length > 0 && (
        <div className="pt-0.5 border-t border-[#111116]">
          <div className="grid gap-x-1 gap-y-0.5" style={{ gridTemplateColumns: 'auto 1fr 1fr 1fr' }}>
            <span className="text-[5.5px] font-mono text-[#1e1e28] uppercase tracking-widest"/>
            <span className="text-[5.5px] font-mono text-[#2a2a38] uppercase tracking-widest text-right">min</span>
            <span className="text-[5.5px] font-mono text-[#2a2a38] uppercase tracking-widest text-right">med</span>
            <span className="text-[5.5px] font-mono text-[#2a2a38] uppercase tracking-widest text-right">max</span>
            {paramSeries.map(s => {
              const sorted = [...s.values].sort((a, b) => a - b);
              const n = sorted.length;
              const med = n % 2 === 1 ? sorted[Math.floor(n / 2)] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
              const fmt = (v: number) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3);
              return (
                <React.Fragment key={s.name}>
                  <span className="text-[7px] font-mono w-7 text-right flex-shrink-0" style={{ color: s.color }}>{s.name}</span>
                  <span className="text-[6.5px] font-mono text-[#454549] text-right tabular-nums">{fmt(sorted[0])}</span>
                  <span className="text-[6.5px] font-mono text-[#7a7a86] text-right tabular-nums">{fmt(med)}</span>
                  <span className="text-[6.5px] font-mono text-[#454549] text-right tabular-nums">{fmt(sorted[n - 1])}</span>
                </React.Fragment>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ---- Source: Data (unified real + synthetic) ----
function SourceDataBody({ node, realSamples, isLoadingSamples, sampleLoadError, onRun, onStop, onConfigChange }: NodeBodyProps) {
  const cfg = node.config as SourceDataCfg;
  const output = node.output?.kind === 'impedance-series' ? node.output : null;

  return (
    <div className="flex flex-col h-full">
      {/* Unified data source dropdown */}
      <div className="flex-shrink-0 px-2 pt-2 pb-1.5">
        <select
          value={cfg.selection}
          onMouseDown={e => e.stopPropagation()}
          onChange={e => onConfigChange({ ...cfg, selection: e.target.value, syntheticTrajIdx: null })}
          className="w-full text-[9px] font-mono bg-[#0d0d0f] border border-[#1e1e24] rounded px-2 py-1 text-[#dddde2] focus:outline-none">
          <optgroup label="Real Data">
            {isLoadingSamples && <option value="real:0" disabled>Loading...</option>}
            {realSamples.map((s, i) => (
              <option key={i} value={`real:${i}`}>
                {s.name} · {s.donor} · {s.timepoints.length}pt{s.atp_hi > s.atp_lo ? ' · 3PEIS' : ''}
              </option>
            ))}
            {!isLoadingSamples && realSamples.length === 0 && (
              <option value="real:0" disabled>{sampleLoadError ?? `No samples`}</option>
            )}
          </optgroup>
          <optgroup label="Synthetic">
            {SYNTHETIC_PATHOLOGIES.map(p => (
              <option key={p.value} value={`synthetic:${p.value}`}>{p.label}</option>
            ))}
          </optgroup>
        </select>
      </div>

      {/* Data preview */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {output && <DataPreview output={output}/>}
        {node.errorMsg && (
          <div className="mx-2 mt-1 text-[9px] text-[#7a3a3a] leading-tight">{node.errorMsg}</div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-2 pb-2 pt-1.5 border-t border-[#1e1e24] flex items-center gap-2">
        {output?.label && (
          <span className="text-[7.5px] font-mono text-[#454549] truncate flex-1 min-w-0">{output.label}</span>
        )}
        <RunBtn status={node.status} onRun={onRun} onStop={onStop} label="Load" className="flex-shrink-0"/>
      </div>
    </div>
  );
}



const KALMAN_STATE_PARAMS: Array<{
  idx: number; label: string; unit: string; toDisplay: (linear: number) => number;
}> = [
  { idx: 2, label: 'TER',     unit: 'kΩ',  toDisplay: v => v / 1000 },
  { idx: 3, label: 'TEC',     unit: 'µF',  toDisplay: v => v * 1e6  },
  { idx: 4, label: 'Rsh',     unit: 'kΩ',  toDisplay: v => v / 1000 },
  { idx: 0, label: 'τ_big',   unit: 's',   toDisplay: v => v        },
  { idx: 1, label: 'τ_small', unit: 's',   toDisplay: v => v        },
];

function InlineSparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null;
  const W = 52; const H = 14;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = W / (values.length - 1);
  const toY = (v: number) => H - ((v - min) / range) * (H - 2) - 1;
  const pts = values.map((v, i) => `${(i * step).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible', flexShrink: 0 }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1" strokeLinejoin="round" opacity="0.7"/>
      <circle cx={W.toString()} cy={toY(values[values.length - 1]).toFixed(1)} r="1.5" fill={color} opacity="0.9"/>
    </svg>
  );
}

// ---- Pipeline mini-graph ----

// Shared stage colors — referenced by both the pipeline mini-graph and the trajectory chart legend
const STAGE_COLOR = {
  mdn:    '#56B4E9',
  rbpf:   '#6469a0',
  rts:    '#009E73',
  ecm:    '#E69F00',
  output: '#c4a040',
} as const;

interface PipelinePos { x: number; y: number; }

interface PipeStage {
  id: string;
  short: string;
  sub: string;
  color: string;
  cfgKey: keyof Pick<ProcessMLCfg, 'useTransformer' | 'useRBPF' | 'useKalman' | 'useEcm'> | null;
}

const PIPE_STAGES: PipeStage[] = [
  { id: 'input',  short: 'Z(ω,t)',  sub: 'input',        color: '#454558',          cfgKey: null            },
  { id: 'mdn',    short: 'MDN',     sub: 'transformer',  color: STAGE_COLOR.mdn,    cfgKey: 'useTransformer' },
  { id: 'rbpf',   short: 'RBPF',    sub: 'particle',     color: STAGE_COLOR.rbpf,   cfgKey: 'useRBPF'        },
  { id: 'rts',    short: 'RTS',     sub: 'smoother',     color: STAGE_COLOR.rts,    cfgKey: 'useKalman'      },
  { id: 'ecm',    short: 'ECM',     sub: 'baseline',     color: STAGE_COLOR.ecm,    cfgKey: 'useEcm'         },
  { id: 'output', short: 'θ̂(t)',    sub: 'ensemble',     color: STAGE_COLOR.output, cfgKey: null             },
];

const PIPE_W = 54;
const PIPE_H = 26;
const PIPE_CW = 308;
const PIPE_CH = 148;

const PIPE_DEFAULT_POS: Record<string, PipelinePos> = {
  input:  { x: 2,   y: 61 },
  mdn:    { x: 66,  y: 10 },
  rbpf:   { x: 135, y: 10 },
  rts:    { x: 204, y: 10 },
  ecm:    { x: 135, y: 112 },
  output: { x: 252, y: 61 },
};

interface PipeDragState {
  id: string; ox: number; oy: number; sx: number; sy: number; moved: boolean;
}

function PipelineView({ cfg, onConfigChange }: { cfg: ProcessMLCfg; onConfigChange: (c: ProcessMLCfg) => void }) {
  const [pos, setPos] = useState<Record<string, PipelinePos>>(() => ({ ...PIPE_DEFAULT_POS }));
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<PipeDragState | null>(null);

  const isOn = (id: string): boolean => {
    const s = PIPE_STAGES.find(x => x.id === id);
    if (!s?.cfgKey) return true;
    if (s.cfgKey === 'useKalman') return !!(cfg.useKalman && cfg.useRBPF);
    return !!(cfg[s.cfgKey]);
  };

  const toggle = (stage: PipeStage) => {
    if (!stage.cfgKey) return;
    if (stage.cfgKey === 'useKalman') {
      if (cfg.useRBPF) onConfigChange({ ...cfg, useKalman: !cfg.useKalman });
    } else if (stage.cfgKey === 'useRBPF') {
      onConfigChange({ ...cfg, useRBPF: !cfg.useRBPF, useKalman: cfg.useRBPF ? false : cfg.useKalman });
    } else {
      onConfigChange({ ...cfg, [stage.cfgKey]: !(cfg[stage.cfgKey] as boolean) });
    }
  };

  const rx = (id: string) => (pos[id]?.x ?? 0) + PIPE_W;
  const lx = (id: string) => (pos[id]?.x ?? 0);
  const my = (id: string) => (pos[id]?.y ?? 0) + PIPE_H / 2;

  const bez = (x1: number, y1: number, x2: number, y2: number) => {
    const cx = (x1 + x2) / 2;
    return `M${x1} ${y1} C${cx} ${y1} ${cx} ${y2} ${x2} ${y2}`;
  };

  const rbpfOn = isOn('rbpf');
  const mdnOn  = isOn('mdn');
  const rtsOn  = isOn('rts');
  const ecmOn  = isOn('ecm');

  const edges: Array<{ from: string; to: string; active: boolean; dashed?: boolean }> = [
    { from: 'input',  to: 'mdn',    active: mdnOn                       },
    { from: 'input',  to: 'rbpf',   active: rbpfOn && !mdnOn, dashed: true },
    { from: 'mdn',    to: 'rbpf',   active: mdnOn && rbpfOn             },
    { from: 'rbpf',   to: 'rts',    active: rbpfOn && rtsOn             },
    { from: 'rbpf',   to: 'output', active: rbpfOn && !rtsOn            },
    { from: 'rts',    to: 'output', active: rtsOn                       },
    { from: 'input',  to: 'ecm',    active: ecmOn                       },
    { from: 'ecm',    to: 'output', active: ecmOn                       },
  ];

  const onMouseDown = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    dragRef.current = { id, ox: pos[id]?.x ?? 0, oy: pos[id]?.y ?? 0, sx: e.clientX - rect.left, sy: e.clientY - rect.top, moved: false };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dx = (e.clientX - rect.left) - dragRef.current.sx;
    const dy = (e.clientY - rect.top)  - dragRef.current.sy;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      dragRef.current.moved = true;
      const nx = Math.max(0, Math.min(PIPE_CW - PIPE_W, dragRef.current.ox + dx));
      const ny = Math.max(0, Math.min(PIPE_CH - PIPE_H, dragRef.current.oy + dy));
      setPos(p => ({ ...p, [dragRef.current!.id]: { x: nx, y: ny } }));
    }
  };

  const onMouseUp = () => { dragRef.current = null; };

  const onNodeMouseUp = (stage: PipeStage) => {
    if (!dragRef.current?.moved) toggle(stage);
    dragRef.current = null;
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] text-[#6b6b76] uppercase tracking-widest">Pipeline</span>
        <button
          onClick={() => setPos({ ...PIPE_DEFAULT_POS })}
          className="font-mono text-[9px] text-[#2a2a38] hover:text-[#4a4a58] transition-colors focus:outline-none"
          title="Reset layout"
        >
          reset
        </button>
      </div>
      <div
        ref={containerRef}
        className="relative rounded-sm bg-[#050508] border border-[#16161e] overflow-hidden"
        style={{ width: PIPE_CW, height: PIPE_CH, cursor: 'default', userSelect: 'none' }}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        <svg className="absolute inset-0 pointer-events-none" width={PIPE_CW} height={PIPE_CH}>
          {edges.map((e, i) => (
            <path
              key={i}
              d={bez(rx(e.from), my(e.from), lx(e.to), my(e.to))}
              fill="none"
              stroke={e.active ? '#33334a' : '#161620'}
              strokeWidth={e.active ? 1.5 : 1}
              strokeDasharray={e.dashed ? '3 2' : undefined}
              strokeLinecap="round"
            />
          ))}
        </svg>
        {PIPE_STAGES.map(stage => {
          const p = pos[stage.id] ?? PIPE_DEFAULT_POS[stage.id];
          const on = isOn(stage.id);
          const toggleable = !!stage.cfgKey;
          return (
            <div
              key={stage.id}
              style={{
                position: 'absolute',
                left: p.x,
                top: p.y,
                width: PIPE_W,
                height: PIPE_H,
                background: on ? '#0e0e14' : '#08080b',
                border: `1px solid ${on ? stage.color + '44' : '#1a1a22'}`,
                borderRadius: 4,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: on ? 1 : 0.3,
                cursor: toggleable ? 'pointer' : 'grab',
                userSelect: 'none',
                transition: 'opacity 0.15s, border-color 0.15s',
              }}
              onMouseDown={e => onMouseDown(e, stage.id)}
              onMouseUp={() => onNodeMouseUp(stage)}
            >
              <span style={{ fontFamily: 'monospace', fontSize: 10, color: on ? stage.color : '#3a3a48', fontWeight: 600, lineHeight: 1 }}>
                {stage.short}
              </span>
              <span style={{ fontFamily: 'monospace', fontSize: 7, color: on ? '#38384a' : '#222230', lineHeight: 1, marginTop: 2 }}>
                {stage.sub}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- Process: Sequential ML ----
function arrayMean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}


const SECTION_LABEL  = 'text-[11px] font-mono font-medium text-[#6b6b76] tracking-[0.1em] uppercase';
const NODE_DIVIDER   = 'h-px bg-[#1a1a22] my-3';

function ProcessMLBody({ node, inputData, onRun, onStop, onConfigChange }: NodeBodyProps) {
  const cfg = node.config as ProcessMLCfg;
  const [activeModel, setActiveModel] = useState<ActiveModelInfo | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [codeError, setCodeError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${ML_API_BASE}/model_info`)
      .then(r => r.ok ? r.json() : null)
      .then((d: ActiveModelInfo | null) => { if (d?.model) setActiveModel(d); })
      .catch(() => {});
    fetch(`${ML_API_BASE}/available_models`)
      .then(r => r.ok ? r.json() : null)
      .then((d: { models?: AvailableModel[] } | null) => { if (d?.models) setAvailableModels(d.models); })
      .catch(() => {});
  }, []);

  const constraints = cfg.constraints ?? { ...DEFAULT_CONSTRAINTS };
  const provenanceLabel = inputData?.kind === 'impedance-series' ? inputData.label : null;

  const switchToCode = () => {
    const code = constraintsToCode(constraints, cfg);
    onConfigChange({ ...cfg, editMode: 'code', modelCode: code });
    setCodeError(null);
  };

  const switchToVisual = () => {
    try {
      const parsed = codeToConstraints(cfg.modelCode ?? '', cfg);
      onConfigChange({ ...cfg, editMode: 'visual', ...parsed });
      setCodeError(null);
    } catch (e) {
      setCodeError(e instanceof Error ? e.message : 'Parse error');
    }
  };

  const updateConstraint = (param: string, field: keyof ParamConstraint, value: ParamConstraint[keyof ParamConstraint]) => {
    const updated = { ...constraints, [param]: { ...constraints[param], [field]: value } };
    onConfigChange({ ...cfg, constraints: updated });
  };

  const PARAM_META: Array<{ name: string; unit: string }> = [
    { name: 'Ra',  unit: 'Ω' },
    { name: 'Rb',  unit: 'Ω' },
    { name: 'Ca',  unit: 'F' },
    { name: 'Cb',  unit: 'F' },
    { name: 'Rsh', unit: 'Ω' },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0 overflow-y-auto px-3 pt-3 pb-1 flex flex-col gap-3">

        {/* Model selector */}
        <div className="relative">
          <button
            onClick={() => setShowModelPicker(v => !v)}
            onBlur={() => setTimeout(() => setShowModelPicker(false), 150)}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded border border-[#1e1e24] bg-[#0d0d0f] hover:border-[#2a2a33] transition-colors focus:outline-none group"
          >
            <span className="font-mono text-[10px] text-[#c4a040] flex-shrink-0">model</span>
            <span className="font-mono text-[10px] text-[#6b6b76] flex-1 truncate text-left">
              {cfg.modelOverride || activeModel?.model || 'auto'}
            </span>
            {activeModel?.rbpf_version && (
              <span className="font-mono text-[9px] text-[#c4a040]/60 border border-[#c4a040]/20 rounded px-1 leading-none flex-shrink-0">
                {activeModel.rbpf_version}
              </span>
            )}
            <svg className="w-3 h-3 text-[#2a2a33] group-hover:text-[#454549] flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
            </svg>
          </button>
          {showModelPicker && (
            <div className="absolute top-full left-0 right-0 mt-0.5 z-50 bg-[#0d0d0f] border border-[#1e1e24] rounded shadow-xl shadow-black/60 overflow-hidden">
              <button
                onMouseDown={() => { onConfigChange({ ...cfg, modelOverride: '' }); setShowModelPicker(false); }}
                className={`w-full text-left px-2.5 py-1.5 font-mono text-[10px] transition-colors hover:bg-[#131316] flex items-center gap-1.5 ${!cfg.modelOverride ? 'text-[#c4a040]' : 'text-[#454549]'}`}>
                <span className="flex-1">auto</span>
                {activeModel?.model && <span className="text-[9px] text-[#2e2e3a]">{activeModel.model}</span>}
              </button>
              {availableModels.map(m => (
                <button key={m.name}
                  onMouseDown={() => { onConfigChange({ ...cfg, modelOverride: m.name }); setShowModelPicker(false); }}
                  className={`w-full text-left px-2.5 py-1.5 font-mono text-[10px] transition-colors hover:bg-[#131316] flex items-center gap-1.5 ${cfg.modelOverride === m.name ? 'text-[#c4a040]' : 'text-[#6b6b76]'}`}>
                  <span className="flex-1 truncate">{m.name}</span>
                  {m.active && <span className="text-[9px] text-[#3a3a48] border border-[#2a2a33] rounded px-1">auto</span>}
                  {m.val_mae != null && <span className="text-[9px] text-[#3a3a48] tabular-nums">{m.val_mae.toFixed(3)}</span>}
                  {m.epoch != null && <span className="text-[9px] text-[#2a2a38]">ep{m.epoch}</span>}
                </button>
              ))}
            </div>
          )}
        </div>

        <PipelineView cfg={cfg} onConfigChange={onConfigChange}/>

        <div className="h-px bg-[#1a1a22]"/>

        {/* Edit mode toggle */}
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] text-[#6b6b76] uppercase tracking-widest">Constraints</span>
          <div className="flex rounded overflow-hidden border border-[#1e1e24]">
            <button onClick={() => cfg.editMode === 'code' ? switchToVisual() : null}
              className={`font-mono text-[9px] px-2 py-0.5 transition-colors focus:outline-none ${cfg.editMode === 'visual' ? 'bg-[#1e1e24] text-[#9a9aa2]' : 'text-[#3a3a48] hover:text-[#6b6b76]'}`}>
              visual
            </button>
            <button onClick={() => cfg.editMode === 'visual' ? switchToCode() : null}
              className={`font-mono text-[9px] px-2 py-0.5 transition-colors focus:outline-none ${cfg.editMode === 'code' ? 'bg-[#1e1e24] text-[#9a9aa2]' : 'text-[#3a3a48] hover:text-[#6b6b76]'}`}>
              code
            </button>
          </div>
        </div>

        {cfg.editMode === 'visual' ? (
          <div className="flex flex-col gap-0">
            {/* Table header */}
            <div className="grid gap-1 mb-1" style={{ gridTemplateColumns: '2rem 1fr 1fr 2.5rem 1.8rem' }}>
              <span/>
              <span className="font-mono text-[8px] text-[#3a3a48] uppercase tracking-widest text-center">min</span>
              <span className="font-mono text-[8px] text-[#3a3a48] uppercase tracking-widest text-center">max</span>
              <span className="font-mono text-[8px] text-[#3a3a48] uppercase tracking-widest text-center">∂/∂t</span>
              <span className="font-mono text-[8px] text-[#3a3a48] uppercase tracking-widest text-center">dir</span>
            </div>
            {PARAM_META.map(({ name, unit }) => {
              const c = constraints[name] ?? DEFAULT_CONSTRAINTS[name];
              return (
                <div key={name} className="grid gap-1 items-center py-0.5" style={{ gridTemplateColumns: '2rem 1fr 1fr 2.5rem 1.8rem' }}>
                  <button
                    onClick={() => updateConstraint(name, 'enabled', !c.enabled)}
                    className="flex items-center gap-1 focus:outline-none group"
                    title={c.enabled ? 'Disable' : 'Enable'}>
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 transition-colors ${c.enabled ? 'bg-[#c4a040]' : 'bg-[#2a2a33]'}`}/>
                    <span className={`font-mono text-[10px] ${c.enabled ? 'text-[#9a9aa2]' : 'text-[#3a3a48]'}`}>{name}</span>
                  </button>
                  <input type="text" defaultValue={fmtPy(c.min)}
                    onBlur={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateConstraint(name, 'min', v); }}
                    disabled={!c.enabled}
                    title={unit}
                    className="font-mono text-[9px] text-center bg-[#0a0a0c] border border-[#1a1a22] rounded px-1 py-0.5 text-[#6b6b76] focus:outline-none focus:border-[#2a2a33] disabled:opacity-30 w-full"/>
                  <input type="text" defaultValue={fmtPy(c.max)}
                    onBlur={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateConstraint(name, 'max', v); }}
                    disabled={!c.enabled}
                    title={unit}
                    className="font-mono text-[9px] text-center bg-[#0a0a0c] border border-[#1a1a22] rounded px-1 py-0.5 text-[#6b6b76] focus:outline-none focus:border-[#2a2a33] disabled:opacity-30 w-full"/>
                  <input type="text" defaultValue={c.ddtMax.toFixed(2)}
                    onBlur={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) updateConstraint(name, 'ddtMax', v); }}
                    disabled={!c.enabled}
                    title="max |Δlog10(param)| per step"
                    className="font-mono text-[9px] text-center bg-[#0a0a0c] border border-[#1a1a22] rounded px-1 py-0.5 text-[#6b6b76] focus:outline-none focus:border-[#2a2a33] disabled:opacity-30 w-full"/>
                  <select value={c.monotone}
                    onChange={e => updateConstraint(name, 'monotone', e.target.value as 'up' | 'down' | 'free')}
                    disabled={!c.enabled}
                    className="font-mono text-[9px] bg-[#0a0a0c] border border-[#1a1a22] rounded px-0.5 py-0.5 text-[#6b6b76] focus:outline-none focus:border-[#2a2a33] disabled:opacity-30 w-full">
                    <option value="free">↕</option>
                    <option value="up">↑</option>
                    <option value="down">↓</option>
                  </select>
                </div>
              );
            })}
            <p className="font-mono text-[8px] text-[#2a2a33] mt-1.5">units: Ω / F · ∂/∂t in log₁₀ decades/step</p>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            <textarea
              value={cfg.modelCode ?? ''}
              onChange={e => onConfigChange({ ...cfg, modelCode: e.target.value })}
              spellCheck={false}
              className="font-mono text-[9px] leading-relaxed text-[#9a9aa2] bg-[#07070a] border border-[#1a1a22] rounded p-2 resize-none focus:outline-none focus:border-[#2a2a33] w-full"
              style={{ height: '280px', tabSize: 4 }}
            />
            {codeError && (
              <span className="font-mono text-[9px] text-[#7a3a3a]">{codeError}</span>
            )}
            <button
              onClick={switchToVisual}
              className="self-end font-mono text-[9px] px-2 py-0.5 rounded border border-[#2a2a33] text-[#6b6b76] hover:text-[#9a9aa2] hover:border-[#3a3a46] transition-colors focus:outline-none">
              apply
            </button>
          </div>
        )}

      </div>

      {/* Footer */}
      <div className="flex-shrink-0 border-t border-[#1e1e24] px-3 pt-2.5 pb-3 flex flex-col gap-1.5">
        <RunBtn
          status={node.status} onRun={onRun} onStop={onStop}
          disabled={!inputData} label="Run inference"
          className="w-full justify-center text-center"
        />
        <div className="font-mono text-[10px] text-[#3d3d48] truncate">
          {node.status === 'running'
            ? <span className="text-[#c4a040]/60">{node.progress ? `${node.progress.done}/${node.progress.total} timepoints` : 'initializing...'}</span>
            : provenanceLabel ?? '← connect dataset'}
        </div>
      </div>
    </div>
  );
}


// ---- Viz: Results ----
function ResultsBody({ node, inputData }: NodeBodyProps) {
  const result = inputData?.kind === 'ml-result' ? inputData.result : null;
  const [showWhy, setShowWhy] = useState(false);

  const smcEss      = result?.smcEss ?? [];
  const meanEss     = smcEss.length > 0 ? arrayMean(smcEss) : null;
  const nClusters   = result?.clusterPaths.length ?? 0;
  const resampCount = smcEss.filter(e => e < 0.5).length;

  const posteriorSummary = result?.posteriorSummary ?? null;
  const mechanism        = result?.mechanism ?? null;

  const fmtV = (v: number) => v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);

  const primaryPs          = posteriorSummary?.TER ?? null;
  const primaryHalfWidth   = primaryPs ? (primaryPs.q95 - primaryPs.q05) / 2 : null;
  const primaryConfidence  = primaryPs ? primaryPs.identifiability > 0.7 ? 'High' : primaryPs.identifiability > 0.4 ? 'Moderate' : 'Low' : null;
  const primaryConfidenceColor = primaryPs ? primaryPs.identifiability > 0.7 ? '#4ade80' : primaryPs.identifiability > 0.4 ? '#c4a040' : '#f97316' : '#5a5a66';

  const terTrend = (() => {
    if (!result) return null;
    const k = result.kalman;
    if (k?.trends && k.param_names) {
      const idx = k.param_names.indexOf('TER');
      if (idx >= 0) {
        const t = k.trends[idx];
        const pct = Math.round((Math.pow(10, Math.abs(t.net_change)) - 1) * 100);
        return { pct, direction: t.direction };
      }
    }
    const vals = result.predictions.TER.mean;
    if (vals.length < 2 || vals[0] <= 0) return null;
    const rawPct = ((vals[vals.length - 1] - vals[0]) / vals[0]) * 100;
    return { pct: Math.round(Math.abs(rawPct)), direction: rawPct > 2 ? 'rising' as const : rawPct < -2 ? 'falling' as const : 'stable' as const };
  })();

  const totalT     = result?.time_min.length ?? 0;
  const resampFrac = totalT > 0 ? resampCount / totalT : 0;
  const filterQuality = !result ? null : (meanEss != null && meanEss > 0.5 && resampFrac < 0.2) ? 'GOOD' : (meanEss != null && meanEss > 0.3) ? 'FAIR' : 'POOR';
  const filterQualityColor = filterQuality === 'GOOD' ? '#4ade80' : filterQuality === 'FAIR' ? '#c4a040' : '#f97316';
  const filterDesc = filterQuality === 'GOOD' ? 'Particle diversity maintained. Posterior stable.' : filterQuality === 'FAIR' ? 'Moderate particle collapse. Treat estimates with caution.' : filterQuality === 'POOR' ? 'Filter degenerated. Results may be unreliable.' : '';

  const POSTERIOR_PARAMS: Array<{ key: string; label: string; color: string; unit: string }> = [
    { key: 'TER', label: 'TER', color: '#009E73', unit: 'kΩ' },
    { key: 'Rsh', label: 'Rsh', color: '#6469a0', unit: 'kΩ' },
    { key: 'TEC', label: 'TEC', color: '#CC79A7', unit: 'µF' },
    { key: 'Ra',  label: 'Ra',  color: '#56B4E9', unit: 'kΩ' },
    { key: 'Rb',  label: 'Rb',  color: '#E69F00', unit: 'kΩ' },
  ];

  if (!result) {
    return (
      <div className="flex flex-col h-full items-center justify-center px-4">
        <span className="font-mono text-[10px] text-[#2a2a33] text-center">connect an Inference node and run to see results</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto px-3 pt-3 pb-4">

      {/* PRIMARY FINDING */}
      <div>
        <div className={SECTION_LABEL}>Primary finding</div>
        {primaryPs ? (
          <div className="mt-3 px-3 py-3 rounded-sm bg-[#0f0f12] border border-[#1e1e24]">
            <div className="font-mono text-[11px] mb-1" style={{ color: '#009E73' }}>TER</div>
            <div className="flex items-baseline gap-2 mb-2">
              <span className="font-mono text-[26px] font-semibold text-[#dddde2] tabular-nums leading-none">{fmtV(primaryPs.median)}</span>
              {primaryHalfWidth != null && <span className="font-mono text-[12px] text-[#4a4a56] tabular-nums">± {fmtV(primaryHalfWidth)}</span>}
              <span className="font-mono text-[11px] text-[#3d3d48]">kΩ</span>
            </div>
            <div className="flex items-center gap-3">
              {terTrend && terTrend.direction !== 'stable' && (
                <span className="font-mono text-[11px] tabular-nums" style={{ color: terTrend.direction === 'rising' ? '#4ade80' : '#f97316' }}>
                  {terTrend.direction === 'rising' ? '↑' : '↓'} {terTrend.pct}%
                </span>
              )}
              <span className="font-mono text-[10px] text-[#3d3d48]">Confidence</span>
              <span className="font-mono text-[10px] font-medium" style={{ color: primaryConfidenceColor }}>{primaryConfidence}</span>
            </div>
          </div>
        ) : (
          <span className="mt-2 block font-mono text-[10px] text-[#2a2a33]">run to populate</span>
        )}
      </div>

      <div className={NODE_DIVIDER}/>

      {/* PARAMETER POSTERIORS */}
      <div>
        <div className={SECTION_LABEL}>Parameter posteriors</div>
        {posteriorSummary ? (
          <div className="mt-3 flex flex-col gap-3">
            {POSTERIOR_PARAMS.map(({ key, label, color, unit }) => {
              const ps = posteriorSummary[key];
              if (!ps) return null;
              const halfWidth = (ps.q95 - ps.q05) / 2;
              const cv = ps.median > 0 ? (ps.q95 - ps.q05) / ps.median : 0;
              const barWidth = Math.min(100, cv * 100);
              const ident = ps.identifiability;
              const barColor = ident > 0.7 ? '#4ade80' : ident > 0.4 ? '#c4a040' : '#f97316';
              return (
                <div key={key} className="flex items-start gap-2">
                  <span className="font-mono text-[11px] w-8 flex-shrink-0 pt-px" style={{ color }}>{label}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-1.5">
                      <span className="font-mono text-[13px] font-semibold text-[#dddde2] tabular-nums">{fmtV(ps.median)}</span>
                      <span className="font-mono text-[10px] text-[#4a4a56] tabular-nums">± {fmtV(halfWidth)}</span>
                      <span className="font-mono text-[9px] text-[#3a3a48]">{unit}</span>
                    </div>
                    <div className="mt-1 h-0.5 bg-[#1a1a22] rounded overflow-hidden">
                      <div style={{ width: `${barWidth}%`, height: '100%', background: barColor, opacity: 0.4 }} className="rounded"/>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <span className="mt-2 block font-mono text-[10px] text-[#2a2a33]">run to populate</span>
        )}
      </div>

      <div className={NODE_DIVIDER}/>

      {/* MECHANISTIC INTERPRETATION */}
      <div>
        <div className={SECTION_LABEL}>Mechanistic interpretation</div>
        {mechanism ? (
          <div className="mt-3">
            {mechanism.hypotheses[0] && (
              <div className="mb-4">
                <div className="font-mono text-[10px] text-[#4a4a56] mb-1.5">Most likely</div>
                <div className="font-mono text-[15px] font-medium text-[#dddde2] leading-tight mb-1">{mechanism.hypotheses[0].label.split(' (')[0]}</div>
                <div className="font-mono text-[11px]" style={{ color: '#4ade80' }}>{Math.round(mechanism.hypotheses[0].probability * 100)}% confidence</div>
              </div>
            )}
            <div className="flex flex-col gap-2.5">
              {mechanism.hypotheses.slice(0, 3).map(h => (
                <div key={h.name}>
                  <div className="flex items-baseline justify-between mb-1">
                    <span className="font-mono text-[10px] text-[#6b6b76] truncate flex-1">{h.label.split(' (')[0]}</span>
                    <span className="font-mono text-[10px] tabular-nums text-[#4a4a56] flex-shrink-0 ml-2">{(h.probability * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1 bg-[#1a1a22] rounded overflow-hidden">
                    <div style={{ width: `${h.probability * 100}%`, height: '100%', background: '#6469a0', opacity: 0.55 }}/>
                  </div>
                </div>
              ))}
            </div>
            {(mechanism.established.length > 0 || mechanism.ambiguous.length > 0) && (
              <div className="mt-3 flex flex-col gap-1.5">
                {mechanism.established.length > 0 && (
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-[9px] text-[#4a4a56] flex-shrink-0 w-16 pt-px">established</span>
                    <span className="font-mono text-[10px] text-[#8a8a96]">{mechanism.established.join(' · ')}</span>
                  </div>
                )}
                {mechanism.ambiguous.length > 0 && (
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-[9px] text-[#4a4a56] flex-shrink-0 w-16 pt-px">ambiguous</span>
                    <span className="font-mono text-[10px] text-[#4a4a56]">{mechanism.ambiguous.join(' · ')}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <span className="mt-2 block font-mono text-[10px] text-[#2a2a33]">run to populate</span>
        )}
      </div>

      <div className={NODE_DIVIDER}/>

      {/* MODEL CONFIDENCE */}
      <div>
        <div className={SECTION_LABEL}>Model confidence</div>
        <div className="mt-3">
          {filterQuality && (
            <>
              <div className="font-mono text-[20px] font-semibold leading-none mb-1.5" style={{ color: filterQualityColor }}>{filterQuality}</div>
              <div className="font-mono text-[10px] text-[#4a4a56] mb-4 leading-relaxed">{filterDesc}</div>
            </>
          )}
          <div className="grid grid-cols-3 gap-3">
            {meanEss != null && (
              <div className="flex flex-col gap-1">
                <span className="font-mono text-[10px] text-[#5a5a66]">ESS</span>
                <span className="font-mono text-[14px] font-semibold text-[#dddde2] tabular-nums">{Math.round(meanEss * 64)}</span>
              </div>
            )}
            {nClusters > 0 && (
              <div className="flex flex-col gap-1">
                <span className="font-mono text-[10px] text-[#5a5a66]">Clusters</span>
                <span className="font-mono text-[14px] font-semibold text-[#dddde2] tabular-nums">{nClusters}</span>
              </div>
            )}
            {smcEss.length > 0 && (
              <div className="flex flex-col gap-1">
                <span className="font-mono text-[10px] text-[#5a5a66]">Resamples</span>
                <span className="font-mono text-[14px] font-semibold text-[#dddde2] tabular-nums">{resampCount}</span>
              </div>
            )}
          </div>
          {result.dlResnorm.length > 0 && (
            <div className="mt-3 flex items-baseline gap-2">
              <span className="font-mono text-[10px] text-[#5a5a66]">Impedance MAE</span>
              <span className="font-mono text-[12px] text-[#dddde2] tabular-nums">{arrayMean(result.dlResnorm).toFixed(3)}</span>
            </div>
          )}
        </div>
      </div>

      {/* VALIDATION */}
      {result.groundTruth && (() => {
        const gt = result.groundTruth!;
        const preds = result.predictions;
        if (result.time_min.length === 0) return null;
        const logMAE = (predArr: number[], gtArr: number[]) => {
          let sum = 0; let n = 0;
          for (let i = 0; i < Math.min(predArr.length, gtArr.length); i++) {
            const p = predArr[i]; const g = gtArr[i];
            if (p > 0 && g > 0) { sum += Math.abs(Math.log10(p) - Math.log10(g)); n++; }
          }
          return n > 0 ? sum / n : null;
        };
        const qualityLabel = (mae: number) => mae < 0.05 ? 'Excellent' : mae < 0.15 ? 'Good' : mae < 0.40 ? 'Moderate' : 'Poor';
        const rows = [
          { label: 'TER', color: '#009E73', mae: logMAE(preds.TER.mean, gt.TER ?? []) },
          { label: 'TEC', color: '#CC79A7', mae: logMAE(preds.TEC.mean, gt.TEC ?? []) },
          { label: 'Rsh', color: '#6469a0', mae: logMAE(preds.R2.mean,  gt.R2  ?? []) },
        ].filter(r => r.mae !== null);
        if (rows.length === 0) return null;
        return (
          <>
            <div className={NODE_DIVIDER}/>
            <div>
              <div className={SECTION_LABEL}>Validation</div>
              <div className="mt-3 flex flex-col gap-3">
                {rows.map(r => (
                  <div key={r.label}>
                    <div className="flex items-baseline justify-between mb-1">
                      <span className="font-mono text-[12px]" style={{ color: r.color }}>{r.label} error</span>
                      <div className="flex items-baseline gap-2">
                        <span className="font-mono text-[10px] tabular-nums text-[#dddde2]">{r.mae!.toFixed(3)}</span>
                        <span className="font-mono text-[9px]" style={{ color: r.mae! < 0.05 ? '#4ade80' : r.mae! < 0.15 ? '#4ade80' : r.mae! < 0.40 ? '#c4a040' : '#f97316' }}>{qualityLabel(r.mae!)}</span>
                      </div>
                    </div>
                    <div className="h-1 bg-[#1a1a22] rounded overflow-hidden">
                      <div style={{ width: `${Math.max(4, (1 - Math.min(r.mae!, 1.0))) * 100}%`, height: '100%', background: r.mae! < 0.15 ? '#4ade80' : r.mae! < 0.40 ? '#c4a040' : '#f97316', opacity: 0.55 }}/>
                    </div>
                  </div>
                ))}
              </div>
              <span className="font-mono text-[9px] text-[#2a2a33] mt-1.5 block">log₁₀ MAE vs ground truth</span>
            </div>
          </>
        );
      })()}

      {/* WHY THIS CONCLUSION */}
      {(mechanism || result.admissibility) && (
        <>
          <div className={NODE_DIVIDER}/>
          <button
            onClick={() => setShowWhy(v => !v)}
            className="flex items-center gap-1.5 group focus:outline-none">
            <span className="font-mono text-[10px] text-[#5a5a66] group-hover:text-[#7a7a86] transition-colors">Why this conclusion?</span>
            <svg className={`w-3 h-3 text-[#3a3a48] group-hover:text-[#5a5a66] transition-transform duration-150 ${showWhy ? 'rotate-180' : ''}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
            </svg>
          </button>
          {showWhy && (
            <div className="flex flex-col gap-5 pb-2">
              {mechanism?.evidence_time && mechanism.evidence_time.length > 0 && (
                <div>
                  <div className="font-mono text-[10px] text-[#5a5a66] mb-2">Evidence windows</div>
                  <div className="flex flex-col gap-2">
                    {mechanism.evidence_time.slice(0, 4).map((ev, i) => (
                      <div key={i}>
                        <div className="flex items-baseline justify-between mb-1">
                          <span className="font-mono text-[10px] text-[#6b6b76] truncate flex-1">{ev.label}</span>
                          <span className="font-mono text-[9px] tabular-nums text-[#4a4a56] flex-shrink-0 ml-2">{ev.t_start.toFixed(0)}–{ev.t_end.toFixed(0)} min</span>
                        </div>
                        <div className="h-0.5 bg-[#1a1a22] rounded overflow-hidden">
                          <div style={{ width: `${Math.min(100, ev.contribution * 100)}%`, height: '100%', background: '#6469a0', opacity: 0.55 }}/>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {result.admissibility?.claim_language && (
                <div>
                  <div className="font-mono text-[10px] text-[#5a5a66] mb-2">Parameter claims</div>
                  <div className="flex flex-col gap-2.5">
                    {Object.entries(result.admissibility.claim_language)
                      .filter(([, v]) => v.tier === 'supported' || v.tier === 'hedged')
                      .slice(0, 5)
                      .map(([param, claim]) => {
                        const tierColor = claim.tier === 'supported' ? '#4ade80' : '#c4a040';
                        return (
                          <div key={param}>
                            <div className="flex items-center gap-1.5 mb-0.5">
                              <span className="font-mono text-[11px] text-[#8a8a96] w-8 flex-shrink-0">{param}</span>
                              <span className="font-mono text-[9px] rounded px-1 py-px leading-tight" style={{ color: tierColor, background: `${tierColor}1a` }}>{claim.tier}</span>
                            </div>
                            <p className="font-mono text-[9px] text-[#4a4a56] leading-relaxed pl-10">{claim.text}</p>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
              {result.changepoints && result.changepoints.length > 0 && (
                <div>
                  <div className="font-mono text-[10px] text-[#5a5a66] mb-2">Detected changes</div>
                  <div className="flex flex-col gap-1.5">
                    {result.changepoints.slice(0, 3).map((cp, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="font-mono text-[9px] text-[#3a3a48] tabular-nums flex-shrink-0 w-12">{cp.time_min != null ? `${cp.time_min.toFixed(0)} min` : `t${cp.timepoint}`}</span>
                        <span className="font-mono text-[9px] text-[#4a4a56] leading-tight">{cp.interpretation}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {node.errorMsg && <p className="text-[10px] text-[#7a3a3a] leading-tight mt-4">{node.errorMsg}</p>}
    </div>
  );
}




// ---- Viz: Trajectory ----
function VizTrajectoryBody({ node, inputData, inputData2, onConfigChange }: NodeBodyProps) {
  const cfg = node.config as VizTrajectoryCfg;
  const result  = inputData?.kind  === 'ml-result' ? inputData.result  : null;
  const result2 = inputData2?.kind === 'ml-result' ? inputData2.result : null;
  const [useLogScale, setUseLogScale] = React.useState(true);

  // Use whichever connected input has trajectory data (in-1 preferred, fall back to in-2)
  const activeResult = result ?? result2;

  // Merge ECM cold path — prefer in-1, fall back to in-2
  const ecmColdPath = (result?.ecmColdPath?.length ? result.ecmColdPath : null)
    ?? result2?.ecmColdPath ?? [];

  // Prefer final cluster paths; fall back to live predictions during streaming
  const trajectories = activeResult
    ? (activeResult.clusterPaths.length > 0 ? activeResult.clusterPaths : predictionsToTrajectory(activeResult))
    : [];

  const ALL_TABLE_PARAMS: TrajectoryParam[] = ['TER', 'TEC', 'Rsh', 'Ra', 'Rb', 'Ca', 'Cb', 'tau_big', 'tau_small'];

  const TABLE_COL_LABELS: Record<TrajectoryParam, string> = {
    TER: 'TER kΩ', TEC: 'TEC µF', Rsh: 'Rsh kΩ',
    Ra: 'Ra kΩ', Rb: 'Rb kΩ', Ca: 'Ca µF', Cb: 'Cb µF',
    tau_big: 'τ_big s', tau_small: 'τ_sm s',
    R1: 'R1 kΩ', C1: 'C1 µF', R2: 'R2 kΩ',
  };

  const VIEWS: { key: TrajView; label: string }[] = [
    { key: 'trajectory',  label: 'Traj'  },
    { key: 'error',       label: 'Err%'  },
    { key: 'derivative',  label: 'd/dt'  },
    { key: 'uncertainty', label: 'IQR'   },
    { key: 'table',       label: 'Table' },
  ];

  // Build table rows from the top trajectory cluster (rank 0) or streaming predictions
  const tableRows: Array<{ time_min: number } & Record<string, number | null>> = React.useMemo(() => {
    if (!activeResult) return [];
    const topPath = activeResult.clusterPaths[0]?.path ?? null;
    const times = activeResult.time_min;
    return times.map((t, i) => {
      const row: Record<string, number | null> = { time_min: t };
      for (const param of ALL_TABLE_PARAMS) {
        if (topPath && topPath[i]) {
          row[param] = trajDisplayValue(param, extractSmoothedValue(param, topPath[i]));
          row[`${param}_q25`] = trajDisplayValue(param, extractSmoothedQ25(param, topPath[i]));
          row[`${param}_q75`] = trajDisplayValue(param, extractSmoothedQ75(param, topPath[i]));
        } else {
          row[param] = null;
        }
        if (activeResult.groundTruth) {
          const gt = activeResult.groundTruth[param as ParameterKey];
          row[`${param}_gt`] = gt?.[i] != null ? gt[i] : null;
        }
        if (activeResult.ecmColdPath[i]) {
          row[`${param}_ecm`] = trajDisplayValue(param, extractEcmValue(param, activeResult.ecmColdPath[i]));
        }
      }
      return row as { time_min: number } & Record<string, number | null>;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeResult]);

  return (
    <div className="flex flex-col h-full">
      {/* Control bar */}
      <div className="flex-shrink-0 flex items-center gap-2 px-2.5 bg-[#131316] border-b border-[#1e1e24]" style={{ height: NODE_CHROME.ctrlBarH }}>
        {/* Param selector — hidden in table view */}
        {cfg.view !== 'table' && (
          <div className="relative flex items-center flex-shrink-0">
            <select value={cfg.param} onChange={e => onConfigChange({ ...cfg, param: e.target.value as TrajectoryParam })}
              className="text-[10px] text-[#dddde2] bg-transparent border-none outline-none cursor-pointer appearance-none pr-4 focus:outline-none">
              {TRAJ_PARAMS_IDENTIFIABLE.map(p => <option key={p} value={p} className="bg-[#131316]">{TRAJ_PARAM_LABELS[p]}</option>)}
              {TRAJ_PARAMS_RAW.map(p => <option key={p} value={p} className="bg-[#131316]">{TRAJ_PARAM_LABELS[p]}</option>)}
            </select>
            <svg className={`absolute right-0 ${ICON.node} text-[#454549] pointer-events-none`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/></svg>
          </div>
        )}

        {/* Scale toggle pill — hidden in table view */}
        {cfg.view !== 'table' && (
          <div className="flex items-center rounded bg-[#18181c] border border-[#1e1e24] p-px gap-px flex-shrink-0">
            {(['log','lin'] as const).map(s => (
              <button key={s} onClick={() => setUseLogScale(s === 'log')}
                className={`px-1.5 py-0.5 text-[9px] rounded font-mono transition-colors duration-100 focus:outline-none ${
                  (s === 'log') === useLogScale ? 'bg-[#2a2a33] text-[#dddde2]' : 'text-[#454549] hover:text-[#7a7a82]'
                }`}>{s}</button>
            ))}
          </div>
        )}

        {/* View segmented control */}
        <div className="flex items-center rounded bg-[#18181c] border border-[#1e1e24] p-px gap-px ml-auto flex-shrink-0">
          {VIEWS.map(({ key, label }) => (
            <button key={key} onClick={() => onConfigChange({ ...cfg, view: key })}
              className={`px-1.5 py-0.5 text-[9px] rounded transition-colors duration-100 focus:outline-none ${
                cfg.view === key ? 'bg-[#2a2a33] text-[#dddde2]' : 'text-[#454549] hover:text-[#7a7a82]'
              }`}>{label}</button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {cfg.view === 'table' ? (
          tableRows.length > 0 ? (
            <div className="h-full overflow-auto">
              <table className="w-full text-[9px] font-mono border-collapse">
                <thead className="sticky top-0 bg-[#0d0d0f] z-10">
                  <tr>
                    <th className="text-left px-2 py-1 text-[#454549] border-b border-[#1e1e24] whitespace-nowrap font-normal">t (min)</th>
                    {ALL_TABLE_PARAMS.map(p => (
                      <th key={p} className="text-right px-2 py-1 text-[#454549] border-b border-[#1e1e24] whitespace-nowrap font-normal">
                        {TABLE_COL_LABELS[p]}
                      </th>
                    ))}
                  </tr>
                  <tr className="bg-[#0a0a0c]">
                    <th className="text-left px-2 py-0.5 text-[#2a2a33] border-b border-[#1e1e24] font-normal">—</th>
                    {ALL_TABLE_PARAMS.map(p => (
                      <th key={p} className="text-right px-2 py-0.5 text-[#2a2a33] border-b border-[#1e1e24] font-normal whitespace-nowrap">
                        mean [q25–q75]
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((row, i) => (
                    <tr key={i} className="border-b border-[#13131a] hover:bg-[#13131a] transition-colors duration-50">
                      <td className="px-2 py-0.5 text-[#9a9aa2] tabular-nums whitespace-nowrap">{row.time_min.toFixed(1)}</td>
                      {ALL_TABLE_PARAMS.map(p => {
                        const val = row[p];
                        const q25 = row[`${p}_q25`];
                        const q75 = row[`${p}_q75`];
                        const gt  = row[`${p}_gt`];
                        const ecm = row[`${p}_ecm`];
                        return (
                          <td key={p} className="px-2 py-0.5 text-right tabular-nums whitespace-nowrap align-top">
                            {val != null ? (
                              <div className="flex flex-col items-end gap-0">
                                <span className="text-[#dddde2]">{val.toPrecision(3)}</span>
                                {q25 != null && q75 != null && (
                                  <span className="text-[#3d3d48] text-[8px]">[{q25.toPrecision(2)}–{q75.toPrecision(2)}]</span>
                                )}
                                {gt != null && (
                                  <span className="text-[#3d7a60] text-[8px]">gt {gt.toPrecision(3)}</span>
                                )}
                                {ecm != null && (
                                  <span className="text-[#6469a0] text-[8px]">ecm {ecm.toPrecision(3)}</span>
                                )}
                              </div>
                            ) : <span className="text-[#2a2a33]">—</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full"><p className="text-[10px] text-[#2a2a33]">Connect a Process node ←</p></div>
          )
        ) : (
          (trajectories.length > 0 || ecmColdPath.length > 0)
            ? <HypothesisTrajectoryChart
                trajectories={trajectories}
                ecmColdPath={ecmColdPath}
                param={cfg.param}
                groundTruth={activeResult?.groundTruth ?? null}
                timeMin={activeResult?.time_min ?? ecmColdPath.map(p => p.time_min)}
                atpLo={activeResult?.atpLo ?? null}
                atpHi={activeResult?.atpHi ?? null}
                ecmResnorm={activeResult?.ecmResnorm}
                colors={HYP_COLORS}
                kalman={activeResult?.kalman}
                activePipeline={activeResult?.activePipeline}
                view={cfg.view}
                useLogScale={useLogScale}
                chartHeight={node.height - NODE_CHROME.headerH - NODE_CHROME.ctrlBarH - NODE_CHROME.chartPadH}/>
            : <div className="flex items-center justify-center h-full"><p className="text-[10px] text-[#2a2a33]">Connect a Process node ←</p></div>
        )}
      </div>
    </div>
  );
}


// Hypothesis color palette (Wong 2011, colorblind-safe): 0=ECM direct, 1=ECM swap, 2=MDN Ca>Cb, 3=Biorealistic
const HYP_COLORS    = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7'] as const;



// ---- KalmanTrendStrip ----
// Compact row of per-param trend indicators drawn from the CV Kalman trend data.
// Shows direction arrow, net log10 displacement, and slope (log10/min).


// Convert streaming predictions (display-unit kΩ/µF) into a SmoothedTrajectory so the
// chart renders live during streaming, before clusterPaths arrives on the done event.
function predictionsToTrajectory(result: TimeSeriesResult): SmoothedTrajectory[] {
  if (!result.time_min.length) return [];
  const pred = result.predictions;
  const toR = (v: number | undefined) => (v ?? 0) * 1000;   // kΩ → Ω
  const toC = (v: number | undefined) => (v ?? 0) / 1e6;    // µF → F
  const path: SmoothedPathPoint[] = result.time_min.map((t, i) => {
    const Ra  = toR(pred.Ra.mean[i]);  const Rb  = toR(pred.Rb.mean[i]);
    const Ca  = toC(pred.Ca.mean[i]);  const Cb  = toC(pred.Cb.mean[i]);
    return {
      t: i, time_min: t,
      TER:       toR(pred.TER.mean[i]), TER_q05: toR(pred.TER.q25[i]), TER_q25: toR(pred.TER.q25[i]), TER_q75: toR(pred.TER.q75[i]), TER_q95: toR(pred.TER.q75[i]),
      TEC:       toC(pred.TEC.mean[i]), TEC_q05: toC(pred.TEC.q25[i]), TEC_q25: toC(pred.TEC.q25[i]), TEC_q75: toC(pred.TEC.q75[i]), TEC_q95: toC(pred.TEC.q75[i]),
      Rsh:       toR(pred.R2.mean[i]),  Rsh_q05: toR(pred.R2.q25[i]),  Rsh_q25: toR(pred.R2.q25[i]),  Rsh_q75: toR(pred.R2.q75[i]),  Rsh_q95: toR(pred.R2.q75[i]),
      Ra,  Ra_q05: toR(pred.Ra.q25[i]),  Ra_q25: toR(pred.Ra.q25[i]),  Ra_q75: toR(pred.Ra.q75[i]),  Ra_q95: toR(pred.Ra.q75[i]),
      Rb,  Rb_q05: toR(pred.Rb.q25[i]),  Rb_q25: toR(pred.Rb.q25[i]),  Rb_q75: toR(pred.Rb.q75[i]),  Rb_q95: toR(pred.Rb.q75[i]),
      Ca,  Ca_q05: toC(pred.Ca.q25[i]),  Ca_q25: toC(pred.Ca.q25[i]),  Ca_q75: toC(pred.Ca.q75[i]),  Ca_q95: toC(pred.Ca.q75[i]),
      Cb,  Cb_q05: toC(pred.Cb.q25[i]),  Cb_q25: toC(pred.Cb.q25[i]),  Cb_q75: toC(pred.Cb.q75[i]),  Cb_q95: toC(pred.Cb.q75[i]),
      tau_big:       Math.max(Ra * Ca, Rb * Cb),
      tau_big_q05:   Math.max(toR(pred.Ra.q25[i]) * toC(pred.Ca.q25[i]), toR(pred.Rb.q25[i]) * toC(pred.Cb.q25[i])),
      tau_big_q25:   Math.max(toR(pred.Ra.q25[i]) * toC(pred.Ca.q25[i]), toR(pred.Rb.q25[i]) * toC(pred.Cb.q25[i])),
      tau_big_q75:   Math.max(toR(pred.Ra.q75[i]) * toC(pred.Ca.q75[i]), toR(pred.Rb.q75[i]) * toC(pred.Cb.q75[i])),
      tau_big_q95:   Math.max(toR(pred.Ra.q75[i]) * toC(pred.Ca.q75[i]), toR(pred.Rb.q75[i]) * toC(pred.Cb.q75[i])),
      tau_small:     Math.min(Ra * Ca, Rb * Cb),
      tau_small_q05: Math.min(toR(pred.Ra.q25[i]) * toC(pred.Ca.q25[i]), toR(pred.Rb.q25[i]) * toC(pred.Cb.q25[i])),
      tau_small_q25: Math.min(toR(pred.Ra.q25[i]) * toC(pred.Ca.q25[i]), toR(pred.Rb.q25[i]) * toC(pred.Cb.q25[i])),
      tau_small_q75: Math.min(toR(pred.Ra.q75[i]) * toC(pred.Ca.q75[i]), toR(pred.Rb.q75[i]) * toC(pred.Cb.q75[i])),
      tau_small_q95: Math.min(toR(pred.Ra.q75[i]) * toC(pred.Ca.q75[i]), toR(pred.Rb.q75[i]) * toC(pred.Cb.q75[i])),
    };
  });
  return [{ rank: 0, probability: 1.0, hypothesis: 0, label: 'MDN/RBPF', path }];
}

// ---- HypothesisTrajectoryChart ----

const W_TRAJ = 680;
const H_TRAJ = 200;
const MG_TRAJ = { top: 12, right: 12, bottom: 34, left: 58 };

// Map a TrajectoryParam to its index in the Kalman identifiable state vector [tau_big, tau_small, TER, TEC, Rsh].
// Returns null for params tracked in raw space (Ra, Rb, Ca, Cb) or needing special computation (R1).
function kalmanParamIndex(param: TrajectoryParam): number | null {
  switch (param) {
    case 'tau_big':   return 0;
    case 'tau_small': return 1;
    case 'TER':       return 2;
    case 'TEC':
    case 'C1':        return 3;  // C1 (series capacitance) = TEC
    case 'Rsh':
    case 'R2':        return 4;
    default:          return null;
  }
}

// Map a TrajectoryParam to its index in the Kalman RAW state vector [Ra, Rb, Ca, Cb, Rsh].
// R1 = Ra+Rb requires special summation (handled separately).
function kalmanRawParamIndex(param: TrajectoryParam): number | null {
  switch (param) {
    case 'Ra': return 0;
    case 'Rb': return 1;
    case 'Ca': return 2;
    case 'Cb': return 3;
    default:   return null;
  }
}

// Convert a log10 Kalman value to display units for a given param.
function kalmanLog10ToDisplay(param: TrajectoryParam, log10Val: number): number {
  const v = Math.pow(10, log10Val);
  if (param === 'TER' || param === 'Rsh' || param === 'R2' || param === 'Ra' || param === 'Rb' || param === 'R1') return v / 1000;
  if (param === 'TEC' || param === 'Ca' || param === 'Cb' || param === 'C1') return v * 1e6;
  return v;  // tau: seconds
}

// Cap std in log10 space so the uncertainty band never explodes during extrapolation.
// ±0.78 log10 ≈ ±6x in linear — beyond that the band is uninformative.
const MAX_STD_LOG10 = 0.78;

interface HypothesisTrajectoryChartProps {
  trajectories: SmoothedTrajectory[];
  ecmColdPath: EcmColdPoint[];
  param: TrajectoryParam;
  groundTruth: Record<ParameterKey, number[]> | null;
  timeMin: number[];
  atpLo: number | null;
  atpHi: number | null;
  ecmResnorm?: number[];
  colors?: readonly string[];
  subtitle?: string;
  kalman?: KalmanResult | null;
  chartHeight?: number;
  view?: TrajView;
  useLogScale: boolean;
  activePipeline?: { useTransformer: boolean; useRBPF: boolean; useKalman: boolean; useEcm: boolean };
}

// Derive stage color from trajectory label so chart colors stay in sync with pipeline nodes
function stageColorForTraj(label: string, fallback: string): string {
  const l = label.toLowerCase();
  if (l.includes('smooth') || l.includes('fitted')) return STAGE_COLOR.rts;
  if (l.includes('rbpf') || l.includes('particle') || l.includes('mdn/rbpf')) return STAGE_COLOR.rbpf;
  if (l.includes('mdn')) return STAGE_COLOR.mdn;
  return fallback;
}

function HypothesisTrajectoryChart({ trajectories, ecmColdPath, param, groundTruth, timeMin, atpLo, atpHi, ecmResnorm, colors = HYP_COLORS, kalman, chartHeight, view = 'trajectory', useLogScale, activePipeline }: HypothesisTrajectoryChartProps) {
  const localH = chartHeight ?? H_TRAJ;
  const CHART_ECM_COLOR = STAGE_COLOR.ecm;
  const CHART_GT_COLOR  = '#4ade80';

  // Filter trajectories whose stage is toggled off in the pipeline
  const visTrajectories = React.useMemo(() => {
    if (!activePipeline) return trajectories;
    return trajectories.filter(traj => {
      const l = traj.label.toLowerCase();
      if ((l.includes('smooth') || l.includes('fitted')) && !activePipeline.useKalman) return false;
      if ((l.includes('rbpf') || l.includes('particle')) && !activePipeline.useRBPF) return false;
      if (l.includes('mdn') && !l.includes('rbpf') && !activePipeline.useTransformer) return false;
      return true;
    });
  }, [trajectories, activePipeline]);

  const showEcmPath = ecmColdPath.length > 0 && (activePipeline?.useEcm !== false);
  // hiddenLines: set of keys ('hyp0'..'hyp3', 'ecm', 'gt') that are toggled off
  const [hiddenLines, setHiddenLines] = React.useState<ReadonlySet<string>>(new Set<string>());
  const [hoverIdx, setHoverIdx] = React.useState<number | null>(null);
  const svgRef = React.useRef<SVGSVGElement>(null);
  const toggleLine = (key: string) => setHiddenLines(prev => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  if (visTrajectories.length === 0 && !showEcmPath) return null;

  const pw = W_TRAJ - MG_TRAJ.left - MG_TRAJ.right;
  const ph = localH - MG_TRAJ.top - MG_TRAJ.bottom;

  // Y-range from trajectory medians only — IQR bounds excluded to prevent axis stretch
  const trajVals: number[] = [];
  for (const traj of visTrajectories) {
    if (hiddenLines.has(`hyp${traj.hypothesis}`)) continue;
    for (const pt of traj.path) {
      const v = trajDisplayValue(param, extractSmoothedValue(param, pt));
      if (isFinite(v) && v > 0) trajVals.push(v);
    }
  }
  // Include ECM cold path values in range when visible
  if (showEcmPath && !hiddenLines.has('ecm')) {
    for (const pt of ecmColdPath) {
      const raw = extractEcmValue(param, pt);
      const v = trajDisplayValue(param, raw);
      if (isFinite(v) && v > 0) trajVals.push(v);
    }
  }

  // Ground truth values — groundTruth already has R1, C1, R2 keys directly
  let gtVals: number[] | null = null;
  if (groundTruth) {
    if      (param === 'TER')       gtVals = groundTruth.TER ?? null;
    else if (param === 'TEC')       gtVals = groundTruth.TEC ?? null;
    else if (param === 'Rsh' || param === 'R2') gtVals = groundTruth.R2 ?? null;
    else if (param === 'Ra')        gtVals = groundTruth.Ra  ?? null;
    else if (param === 'Rb')        gtVals = groundTruth.Rb  ?? null;
    else if (param === 'Ca')        gtVals = groundTruth.Ca  ?? null;
    else if (param === 'Cb')        gtVals = groundTruth.Cb  ?? null;
    else if (param === 'R1')        gtVals = groundTruth.R1  ?? null;
    else if (param === 'C1')        gtVals = groundTruth.C1  ?? null;
    else if (param === 'tau_big' || param === 'tau_small') {
      if (groundTruth.Ra && groundTruth.Rb && groundTruth.Ca && groundTruth.Cb) {
        gtVals = groundTruth.Ra.map((Ra, i) => {
          const Rb = groundTruth.Rb![i] ?? 0;
          const Ca = groundTruth.Ca![i] ?? 0;
          const Cb = groundTruth.Cb![i] ?? 0;
          const tauA = Ra * Ca * 1e-3;
          const tauB = Rb * Cb * 1e-3;
          return param === 'tau_big' ? Math.max(tauA, tauB) : Math.min(tauA, tauB);
        });
      }
    }
    if (gtVals && !hiddenLines.has('gt') && trajVals.length > 0) {
      const tMin2 = Math.min(...trajVals);
      const tMax2 = Math.max(...trajVals);
      for (const v of gtVals) {
        if (isFinite(v) && v > 0 && v >= tMin2 / 4 && v <= tMax2 * 4) trajVals.push(v);
      }
    }
  }

  // ---- Non-trajectory view branches — rendered before expensive Kalman setup ----
  if (view !== 'trajectory') {
    const nT = timeMin.length;
    const altTMin = Math.min(...timeMin);
    const altTMax = Math.max(...timeMin);
    const altXRange = altTMax - altTMin || 1;
    const altSx = (t: number) => MG_TRAJ.left + ((t - altTMin) / altXRange) * pw;
    const xIdxsAlt = nT <= 6 ? timeMin.map((_, i) => i)
      : [0, Math.floor(nT * 0.25), Math.floor(nT * 0.5), Math.floor(nT * 0.75), nT - 1];
    const clipId = `${view}-clip-${param}`;

    const sharedXAxis = (
      <>
        <line x1={MG_TRAJ.left} y1={MG_TRAJ.top + ph} x2={MG_TRAJ.left + pw} y2={MG_TRAJ.top + ph} stroke="#374151" strokeWidth={1} />
        {xIdxsAlt.map(i => (
          <g key={i}>
            <line x1={altSx(timeMin[i])} y1={MG_TRAJ.top + ph} x2={altSx(timeMin[i])} y2={MG_TRAJ.top + ph + 4} stroke="#374151" strokeWidth={0.8} />
            <text x={altSx(timeMin[i])} y={MG_TRAJ.top + ph + 13} fill="#9ca3af" fontSize={9} textAnchor="middle">{timeMin[i].toFixed(0)}</text>
          </g>
        ))}
        <text x={MG_TRAJ.left + pw / 2} y={localH - 2} fill="#6b7280" fontSize={9} textAnchor="middle">time (min)</text>
      </>
    );

    const sharedLegend = (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 pb-3 pt-2 border-t border-neutral-900">
        {visTrajectories.map(traj => {
          const lineKey = `hyp${traj.hypothesis}`;
          const isVis = !hiddenLines.has(lineKey);
          const color = stageColorForTraj(traj.label, stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]));
          const shortLabel = (() => {
            const l = traj.label.toLowerCase();
            if (l.includes('smooth') || l.includes('fitted')) return 'RTS (smoothed)';
            if (l.includes('rbpf') || l.includes('particle') || l.includes('mdn/rbpf')) return 'RBPF';
            if (l.includes('mdn')) return 'MDN';
            return traj.label;
          })();
          return (
            <button key={traj.hypothesis} onClick={() => toggleLine(lineKey)}
              className="flex items-center gap-1.5 select-none focus:outline-none" style={{ opacity: isVis ? 1 : 0.35 }}>
              <div style={{ width: 10, height: 10, flexShrink: 0, border: `1.5px solid ${color}`, borderRadius: 2, backgroundColor: isVis ? color : 'transparent' }} />
              <svg width="20" height="7" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3.5" x2="20" y2="3.5" stroke={color}
                  strokeWidth={traj.rank === 0 ? 2.2 : 1.4} strokeDasharray={traj.rank !== 0 ? '4,2' : undefined} />
              </svg>
              <span className="text-[9.5px] text-neutral-400">
                {shortLabel}
                <span className="ml-1 font-semibold" style={{ color }}>{(traj.probability * 100).toFixed(0)}%</span>
              </span>
            </button>
          );
        })}
        {showEcmPath && (() => {
          const isVis = !hiddenLines.has('ecm');
          return (
            <button onClick={() => toggleLine('ecm')} className="flex items-center gap-1.5 select-none focus:outline-none" style={{ opacity: isVis ? 1 : 0.35 }}>
              <div style={{ width: 10, height: 10, flexShrink: 0, border: `1.5px solid ${CHART_ECM_COLOR}`, borderRadius: 2, backgroundColor: isVis ? CHART_ECM_COLOR : 'transparent' }} />
              <svg width="20" height="7" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3.5" x2="20" y2="3.5" stroke={CHART_ECM_COLOR} strokeWidth={1.4} strokeDasharray="5,3" opacity={0.8} />
              </svg>
              <span className="text-[9.5px]" style={{ color: CHART_ECM_COLOR }}>ECM baseline</span>
            </button>
          );
        })()}
      </div>
    );

    // ---- Error% view ----
    if (view === 'error') {
      if (!gtVals) {
        return (
          <div className="flex items-center justify-center" style={{ height: localH }}>
            <p className="text-neutral-700 text-xs">No ground truth for this parameter</p>
          </div>
        );
      }
      const errRows: Array<{ hyp: number; rank: number; prob: number; color: string; pts: Array<[number,number]> }> = [];
      for (const traj of visTrajectories) {
        if (hiddenLines.has(`hyp${traj.hypothesis}`)) continue;
        const pts: Array<[number,number]> = [];
        for (let i = 0; i < traj.path.length; i++) {
          const rbpf = trajDisplayValue(param, extractSmoothedValue(param, traj.path[i]));
          const gt = gtVals[i];
          if (rbpf > 0 && gt > 0 && isFinite(rbpf) && isFinite(gt))
            pts.push([traj.path[i].time_min, Math.abs(rbpf - gt) / gt * 100]);
        }
        errRows.push({ hyp: traj.hypothesis, rank: traj.rank, prob: traj.probability, color: stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]), pts });
      }
      const ecmErrPts: Array<[number,number]> = [];
      if (showEcmPath && !hiddenLines.has('ecm')) {
        for (let i = 0; i < ecmColdPath.length; i++) {
          const v = trajDisplayValue(param, extractEcmValue(param, ecmColdPath[i]));
          const gt = gtVals[i];
          if (v > 0 && gt > 0 && isFinite(v) && isFinite(gt))
            ecmErrPts.push([ecmColdPath[i].time_min, Math.abs(v - gt) / gt * 100]);
        }
      }
      const allErrVals = [...errRows.flatMap(d => d.pts.map(p => p[1])), ...ecmErrPts.map(p => p[1])];
      const errMax = allErrVals.length > 0 ? Math.max(...allErrVals) * 1.15 : 100;
      const syErr = (v: number) => MG_TRAJ.top + (1 - Math.min(Math.max(v, 0), errMax) / errMax) * ph;
      const fmtErr = (v: number) => v >= 10 ? v.toFixed(0) + '%' : v.toFixed(1) + '%';
      const errTicks = [errMax, errMax * 0.5, 0];
      return (
        <div>
          <svg viewBox={`0 0 ${W_TRAJ} ${localH}`} width="100%" style={{ display: 'block' }}>
            <defs><clipPath id={clipId}><rect x={MG_TRAJ.left} y={MG_TRAJ.top} width={pw} height={ph} /></clipPath></defs>
            <g clipPath={`url(#${clipId})`}>
              {errTicks.map((v, i) => (
                <line key={i} x1={MG_TRAJ.left} y1={syErr(v)} x2={MG_TRAJ.left + pw} y2={syErr(v)} stroke="#1e2430" strokeWidth={0.8} />
              ))}
              {ecmErrPts.length > 0 && (
                <polyline points={ecmErrPts.map(([t,e]) => `${altSx(t)},${syErr(e)}`).join(' ')}
                  fill="none" stroke={CHART_ECM_COLOR} strokeWidth={1.4} strokeDasharray="5,3" opacity={0.65} />
              )}
              {errRows.map(d => (
                <polyline key={d.hyp}
                  points={d.pts.map(([t,e]) => `${altSx(t)},${syErr(e)}`).join(' ')}
                  fill="none" stroke={d.color} strokeWidth={d.rank === 0 ? 1.8 : 1.1}
                  opacity={Math.max(0.4, d.prob)} strokeDasharray={d.rank === 0 ? undefined : '4,2'} />
              ))}
            </g>
            <line x1={MG_TRAJ.left} y1={MG_TRAJ.top} x2={MG_TRAJ.left} y2={MG_TRAJ.top + ph} stroke="#374151" strokeWidth={1} />
            {errTicks.map((v, i) => (
              <g key={i}>
                <line x1={MG_TRAJ.left - 4} y1={syErr(v)} x2={MG_TRAJ.left} y2={syErr(v)} stroke="#374151" strokeWidth={0.8} />
                <text x={MG_TRAJ.left - 5} y={syErr(v) + 3} fill="#6b7280" fontSize={7} textAnchor="end">{fmtErr(v)}</text>
              </g>
            ))}
            <text transform={`translate(${MG_TRAJ.left - 38}, ${MG_TRAJ.top + ph / 2}) rotate(-90)`}
              fill="#6b7280" fontSize={8} textAnchor="middle">% err vs GT</text>
            {sharedXAxis}
          </svg>
          {sharedLegend}
        </div>
      );
    }

    // ---- Derivative view ----
    if (view === 'derivative') {
      if (!showEcmPath || ecmColdPath.length < 2) {
        return (
          <div className="flex items-center justify-center" style={{ height: localH }}>
            <p className="text-neutral-700 text-xs">Enable ECM baseline stage to show derivative residual</p>
          </div>
        );
      }
      const ecmLogByTime = new Map<number, number>();
      for (const pt of ecmColdPath) {
        const v = trajDisplayValue(param, extractEcmValue(param, pt));
        if (v > 0) ecmLogByTime.set(Math.round(pt.time_min * 100) / 100, Math.log(v));
      }
      const ecmDerivByTime = new Map<number, number>();
      for (let i = 0; i < ecmColdPath.length; i++) {
        const tKey = Math.round(ecmColdPath[i].time_min * 100) / 100;
        const prev = i > 0 ? ecmColdPath[i - 1] : null;
        const next = i < ecmColdPath.length - 1 ? ecmColdPath[i + 1] : null;
        if (prev && next) {
          const vP = ecmLogByTime.get(Math.round(prev.time_min * 100) / 100);
          const vN = ecmLogByTime.get(Math.round(next.time_min * 100) / 100);
          const dt = next.time_min - prev.time_min;
          if (vP != null && vN != null && dt > 0) ecmDerivByTime.set(tKey, (vN - vP) / dt);
        } else if (next) {
          const vC = ecmLogByTime.get(tKey);
          const vN = ecmLogByTime.get(Math.round(next.time_min * 100) / 100);
          const dt = next.time_min - ecmColdPath[i].time_min;
          if (vC != null && vN != null && dt > 0) ecmDerivByTime.set(tKey, (vN - vC) / dt);
        } else if (prev) {
          const vC = ecmLogByTime.get(tKey);
          const vP = ecmLogByTime.get(Math.round(prev.time_min * 100) / 100);
          const dt = ecmColdPath[i].time_min - prev.time_min;
          if (vC != null && vP != null && dt > 0) ecmDerivByTime.set(tKey, (vC - vP) / dt);
        }
      }
      const allRes: number[] = [];
      for (const traj of visTrajectories) {
        if (hiddenLines.has(`hyp${traj.hypothesis}`)) continue;
        for (let i = 0; i < traj.path.length; i++) {
          const pt = traj.path[i];
          const v = trajDisplayValue(param, extractSmoothedValue(param, pt));
          if (!(v > 0)) continue;
          const logV = Math.log(v);
          const prev = i > 0 ? traj.path[i - 1] : null;
          const next = i < traj.path.length - 1 ? traj.path[i + 1] : null;
          let rbpfD: number | null = null;
          if (prev && next) {
            const vP = trajDisplayValue(param, extractSmoothedValue(param, prev));
            const vN = trajDisplayValue(param, extractSmoothedValue(param, next));
            if (vP > 0 && vN > 0) rbpfD = (Math.log(vN) - Math.log(vP)) / (next.time_min - prev.time_min);
          } else if (next) {
            const vN = trajDisplayValue(param, extractSmoothedValue(param, next));
            if (vN > 0) rbpfD = (Math.log(vN) - logV) / (next.time_min - pt.time_min);
          } else if (prev) {
            const vP = trajDisplayValue(param, extractSmoothedValue(param, prev));
            if (vP > 0) rbpfD = (logV - Math.log(vP)) / (pt.time_min - prev.time_min);
          }
          const ecmD = ecmDerivByTime.get(Math.round(pt.time_min * 100) / 100) ?? null;
          if (rbpfD !== null && ecmD !== null && isFinite(rbpfD) && isFinite(ecmD)) allRes.push(rbpfD - ecmD);
        }
      }
      if (allRes.length === 0) {
        return <div className="flex items-center justify-center" style={{ height: localH }}><p className="text-neutral-700 text-xs">No derivative data</p></div>;
      }
      const sortedR = [...allRes].sort((a, b) => a - b);
      const absMax = Math.max(Math.abs(sortedR[Math.max(0, Math.floor(sortedR.length * 0.05))]),
        Math.abs(sortedR[Math.min(sortedR.length - 1, Math.floor(sortedR.length * 0.95))]), 0.01) * 1.2;
      const syD = (r: number) => MG_TRAJ.top + (1 - (r + absMax) / (2 * absMax)) * ph;
      const fmtD = (v: number) => Math.abs(v) >= 1 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toFixed(3);
      return (
        <div>
          <svg viewBox={`0 0 ${W_TRAJ} ${localH}`} width="100%" style={{ display: 'block' }}>
            <defs><clipPath id={clipId}><rect x={MG_TRAJ.left} y={MG_TRAJ.top} width={pw} height={ph} /></clipPath></defs>
            <g clipPath={`url(#${clipId})`}>
              <rect x={MG_TRAJ.left} y={syD(absMax * 0.1)} width={pw}
                height={Math.max(0, syD(-absMax * 0.1) - syD(absMax * 0.1))} fill="#374151" opacity={0.22} />
              <line x1={MG_TRAJ.left} y1={syD(0)} x2={MG_TRAJ.left + pw} y2={syD(0)} stroke="#374151" strokeWidth={0.8} />
              {[absMax * 0.5, -absMax * 0.5].map(r => (
                <line key={r} x1={MG_TRAJ.left} y1={syD(r)} x2={MG_TRAJ.left + pw} y2={syD(r)}
                  stroke="#374151" strokeWidth={0.5} strokeDasharray="3,3" />
              ))}
              {visTrajectories.map(traj => {
                if (hiddenLines.has(`hyp${traj.hypothesis}`)) return null;
                const color = stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]);
                const pts = traj.path.map((pt, i) => {
                  const v = trajDisplayValue(param, extractSmoothedValue(param, pt));
                  if (!(v > 0)) return null;
                  const logV = Math.log(v);
                  const prev = i > 0 ? traj.path[i - 1] : null;
                  const next = i < traj.path.length - 1 ? traj.path[i + 1] : null;
                  let rbpfD: number | null = null;
                  if (prev && next) {
                    const vP = trajDisplayValue(param, extractSmoothedValue(param, prev));
                    const vN = trajDisplayValue(param, extractSmoothedValue(param, next));
                    if (vP > 0 && vN > 0) rbpfD = (Math.log(vN) - Math.log(vP)) / (next.time_min - prev.time_min);
                  } else if (next) {
                    const vN = trajDisplayValue(param, extractSmoothedValue(param, next));
                    if (vN > 0) rbpfD = (Math.log(vN) - logV) / (next.time_min - pt.time_min);
                  } else if (prev) {
                    const vP = trajDisplayValue(param, extractSmoothedValue(param, prev));
                    if (vP > 0) rbpfD = (logV - Math.log(vP)) / (pt.time_min - prev.time_min);
                  }
                  const ecmD = ecmDerivByTime.get(Math.round(pt.time_min * 100) / 100) ?? null;
                  if (rbpfD === null || ecmD === null || !isFinite(rbpfD) || !isFinite(ecmD)) return null;
                  return `${altSx(pt.time_min)},${syD(Math.max(-absMax, Math.min(absMax, rbpfD - ecmD)))}`;
                }).filter((s): s is string => s !== null).join(' ');
                return pts.length > 0 ? (
                  <polyline key={traj.hypothesis} points={pts} fill="none" stroke={color}
                    strokeWidth={traj.rank === 0 ? 1.8 : 1.1} opacity={Math.max(0.4, traj.probability)} />
                ) : null;
              })}
            </g>
            <line x1={MG_TRAJ.left} y1={MG_TRAJ.top} x2={MG_TRAJ.left} y2={MG_TRAJ.top + ph} stroke="#374151" strokeWidth={1} />
            {([absMax, 0, -absMax] as number[]).map((r, ri) => (
              <g key={ri}>
                <line x1={MG_TRAJ.left - 4} y1={syD(r)} x2={MG_TRAJ.left} y2={syD(r)} stroke="#374151" strokeWidth={0.8} />
                <text x={MG_TRAJ.left - 5} y={syD(r) + 3} fill="#6b7280" fontSize={7} textAnchor="end">
                  {r === 0 ? '0' : (r > 0 ? '+' : '') + fmtD(r)}
                </text>
              </g>
            ))}
            <text transform={`translate(${MG_TRAJ.left - 38}, ${MG_TRAJ.top + ph / 2}) rotate(-90)`}
              fill="#6b7280" fontSize={8} textAnchor="middle">d(ln)/dt</text>
            {sharedXAxis}
          </svg>
          {sharedLegend}
        </div>
      );
    }

    // ---- Uncertainty (IQR spread) view ----
    const iqrRows: Array<{ hyp: number; rank: number; prob: number; color: string; pts: Array<[number,number]> }> = [];
    for (const traj of visTrajectories) {
      if (hiddenLines.has(`hyp${traj.hypothesis}`)) continue;
      const pts: Array<[number,number]> = [];
      for (const pt of traj.path) {
        const q75 = trajDisplayValue(param, extractSmoothedQ75(param, pt));
        const q25 = trajDisplayValue(param, extractSmoothedQ25(param, pt));
        if (q75 > 0 && q25 > 0 && q75 > q25) {
          const w = Math.log10(q75 / q25);
          if (isFinite(w) && w > 0) pts.push([pt.time_min, w]);
        }
      }
      iqrRows.push({ hyp: traj.hypothesis, rank: traj.rank, prob: traj.probability, color: stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]), pts });
    }
    const allWidths = iqrRows.flatMap(d => d.pts.map(p => p[1]));
    const iqrMax = allWidths.length > 0 ? Math.max(...allWidths) * 1.15 : 1;
    const syIqr = (w: number) => MG_TRAJ.top + (1 - Math.min(Math.max(w, 0), iqrMax) / iqrMax) * ph;
    return (
      <div>
        <svg viewBox={`0 0 ${W_TRAJ} ${localH}`} width="100%" style={{ display: 'block' }}>
          <defs><clipPath id={clipId}><rect x={MG_TRAJ.left} y={MG_TRAJ.top} width={pw} height={ph} /></clipPath></defs>
          <g clipPath={`url(#${clipId})`}>
            <line x1={MG_TRAJ.left} y1={MG_TRAJ.top + ph} x2={MG_TRAJ.left + pw} y2={MG_TRAJ.top + ph} stroke="#374151" strokeWidth={0.8} />
            <line x1={MG_TRAJ.left} y1={syIqr(iqrMax * 0.5)} x2={MG_TRAJ.left + pw} y2={syIqr(iqrMax * 0.5)} stroke="#374151" strokeWidth={0.5} strokeDasharray="3,3" />
            {iqrRows.map(d => (
              <polyline key={d.hyp}
                points={d.pts.map(([t,w]) => `${altSx(t)},${syIqr(w)}`).join(' ')}
                fill="none" stroke={d.color} strokeWidth={d.rank === 0 ? 1.8 : 1.1}
                opacity={Math.max(0.4, d.prob) * 0.85} strokeDasharray={d.rank === 0 ? undefined : '4,2'} />
            ))}
          </g>
          <line x1={MG_TRAJ.left} y1={MG_TRAJ.top} x2={MG_TRAJ.left} y2={MG_TRAJ.top + ph} stroke="#374151" strokeWidth={1} />
          {([iqrMax, iqrMax * 0.5, 0] as number[]).map((v, i) => (
            <g key={i}>
              <line x1={MG_TRAJ.left - 4} y1={syIqr(v)} x2={MG_TRAJ.left} y2={syIqr(v)} stroke="#374151" strokeWidth={0.8} />
              <text x={MG_TRAJ.left - 5} y={syIqr(v) + 3} fill="#6b7280" fontSize={7} textAnchor="end">{v.toFixed(2)}</text>
            </g>
          ))}
          <text transform={`translate(${MG_TRAJ.left - 38}, ${MG_TRAJ.top + ph / 2}) rotate(-90)`}
            fill="#6b7280" fontSize={8} textAnchor="middle">IQR (log dec)</text>
          {sharedXAxis}
        </svg>
        {sharedLegend}
      </div>
    );
  }

  // CV Kalman display points — identifiable space for tau/TER/TEC/Rsh, raw space for Ra/Rb/Ca/Cb
  const kalIdx    = kalmanParamIndex(param);
  const kalRawIdx = kalmanRawParamIndex(param);
  const useRaw    = kalRawIdx !== null && kalIdx === null;

  const kalGridPts: Array<{ t: number; v: number; vHi: number; vLo: number; isExtrap: boolean }> = [];
  const kalMeasPts: Array<{ t: number; v: number }> = [];

  // Helper: push a single grid point, capping std to prevent band explosion in extrapolation
  const pushGridPt = (t: number, log10v: number, log10s: number | null | undefined, isExtrap: boolean) => {
    const v = kalmanLog10ToDisplay(param, log10v);
    if (!(v > 0) || !isFinite(v)) return;
    const cappedS = log10s != null ? Math.min(Math.abs(log10s), MAX_STD_LOG10) : 0;
    const vHi = kalmanLog10ToDisplay(param, log10v + cappedS);
    const vLo = kalmanLog10ToDisplay(param, log10v - cappedS);
    kalGridPts.push({ t, v, vHi: Math.max(vHi, v), vLo: Math.min(vLo, v), isExtrap });
    trajVals.push(v);
  };

  if (kalman) {
    const tGrid      = kalman.t_grid ?? [];
    const tLastMeas  = kalman.time_min[kalman.time_min.length - 1] ?? 0;

    if (param === 'R1' && kalman.mu_grid_raw && kalman.std_grid_raw) {
      // R1 = Ra + Rb — sum from raw-space Kalman. Ra→idx0, Rb→idx1.
      const muR = kalman.mu_grid_raw;
      const sdR = kalman.std_grid_raw;
      for (let i = 0; i < tGrid.length; i++) {
        const t = tGrid[i];
        const muRa = muR[i]?.[0], muRb = muR[i]?.[1];
        const sdRa = sdR[i]?.[0], sdRb = sdR[i]?.[1];
        if (muRa == null || muRb == null || !isFinite(muRa) || !isFinite(muRb)) continue;
        const Ra = Math.pow(10, muRa), Rb = Math.pow(10, muRb);
        const R1_linear = Ra + Rb;  // ohms
        const v = R1_linear / 1000; // kΩ·cm²
        if (!(v > 0) || !isFinite(v)) continue;
        // Approximate std: propagate lognormal uncertainty from each component
        const sdRa_lin = sdRa != null ? Math.min(sdRa, MAX_STD_LOG10) * Ra * Math.LN10 : 0;
        const sdRb_lin = sdRb != null ? Math.min(sdRb, MAX_STD_LOG10) * Rb * Math.LN10 : 0;
        const R1_sd_lin = Math.sqrt(sdRa_lin ** 2 + sdRb_lin ** 2);
        const vHi = (R1_linear + R1_sd_lin) / 1000;
        const vLo = Math.max(R1_linear - R1_sd_lin, 1) / 1000;
        kalGridPts.push({ t, v, vHi, vLo, isExtrap: t > tLastMeas + 0.1 });
        trajVals.push(v);
      }
      // Dots at measured timepoints from raw smoothed
      if (kalman.mu_smoothed_raw) {
        for (let i = 0; i < kalman.time_min.length; i++) {
          const muRa = kalman.mu_smoothed_raw[i]?.[0], muRb = kalman.mu_smoothed_raw[i]?.[1];
          if (muRa == null || muRb == null || !isFinite(muRa) || !isFinite(muRb)) continue;
          const v = (Math.pow(10, muRa) + Math.pow(10, muRb)) / 1000;
          if (v > 0 && isFinite(v)) kalMeasPts.push({ t: kalman.time_min[i], v });
        }
      }
    } else {
      const muGrid     = useRaw ? (kalman.mu_grid_raw  ?? []) : (kalman.mu_grid  ?? []);
      const stdGrid    = useRaw ? (kalman.std_grid_raw ?? []) : (kalman.std_grid ?? []);
      const muSmoothed = useRaw ? (kalman.mu_smoothed_raw ?? kalman.mu_smoothed) : kalman.mu_smoothed;
      const idx        = useRaw ? kalRawIdx! : kalIdx;

      if (idx !== null && tGrid.length > 0 && muGrid.length > 0) {
        for (let i = 0; i < tGrid.length; i++) {
          const t = tGrid[i];
          const log10v = muGrid[i]?.[idx];
          const log10s = stdGrid[i]?.[idx];
          if (log10v == null || !isFinite(log10v)) continue;
          pushGridPt(t, log10v, log10s, t > tLastMeas + 0.1);
        }
        // Dots at measured timepoints
        for (let i = 0; i < kalman.time_min.length; i++) {
          const log10v = muSmoothed[i]?.[idx];
          if (log10v == null || !isFinite(log10v)) continue;
          const v = kalmanLog10ToDisplay(param, log10v);
          if (v > 0 && isFinite(v)) kalMeasPts.push({ t: kalman.time_min[i], v });
        }
      }
    }
  }

  if (trajVals.length === 0) return null;

  const sorted = [...trajVals].sort((a, b) => a - b);
  const p5  = sorted[Math.max(0, Math.floor(sorted.length * 0.05))];
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
  const logLo = Math.log10(Math.max(p5, 1e-12)) - 0.08;
  const logHi = Math.log10(Math.max(p95, 1e-12)) + 0.08;
  const yMin  = 10 ** logLo;
  const yMax  = 10 ** logHi;
  const logRange = logHi - logLo || 1;

  const tMin = Math.min(...timeMin);
  // X-axis spans the measured timepoints only; Kalman predictions within this range
  // are drawn, and any that fall outside are clipped by the SVG viewport.
  const tMax   = Math.max(...timeMin);
  const xRange = tMax - tMin || 1;

  const sx = (t: number) => MG_TRAJ.left + ((t - tMin) / xRange) * pw;
  const syLog = (v: number) => {
    if (!(v > 0)) return MG_TRAJ.top + ph;
    const lv = Math.log10(v);
    return MG_TRAJ.top + (1 - (lv - logLo) / logRange) * ph;
  };
  const linRange = yMax - yMin || 1;
  const syLin = (v: number) => {
    if (!isFinite(v)) return MG_TRAJ.top + ph;
    return MG_TRAJ.top + (1 - (v - yMin) / linRange) * ph;
  };
  const sy = useLogScale ? syLog : syLin;
  const syClip = useLogScale
    ? (v: number) => sy(Math.min(Math.max(v, yMin * 0.5), yMax * 2))
    : (v: number) => sy(Math.max(yMin - linRange * 0.1, Math.min(yMax + linRange * 0.1, v)));

  const yTicksMajor: number[] = [];
  if (useLogScale) {
    for (let e = Math.floor(logLo); e <= Math.ceil(logHi); e++) {
      const decade = 10 ** e;
      if (decade >= yMin * 0.95 && decade <= yMax * 1.05) yTicksMajor.push(decade);
    }
  } else {
    const nTicks = 5;
    for (let i = 0; i < nTicks; i++) {
      yTicksMajor.push(yMin + (i / (nTicks - 1)) * linRange);
    }
  }

  const toSup = (n: number): string =>
    String(n).split('').map(c => ({ '-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹' }[c] ?? c)).join('');
  const formatDecade = (v: number): string => {
    const exp = Math.round(Math.log10(v));
    if (Math.abs(exp) >= 3 || exp <= -2) return `10${toSup(exp)}`;
    if (v >= 1)   return v.toFixed(0);
    if (v >= 0.1) return v.toFixed(1);
    return v.toFixed(2);
  };
  const formatLinearTick = (v: number): string => {
    if (linRange >= 100) return v.toFixed(0);
    if (linRange >= 1)   return v.toFixed(1);
    if (linRange >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  };
  const formatTick = (v: number) => useLogScale ? formatDecade(v) : formatLinearTick(v);

  const nT = timeMin.length;
  const xIdxs = nT <= 6 ? timeMin.map((_, i) => i)
    : [0, Math.floor(nT * 0.25), Math.floor(nT * 0.5), Math.floor(nT * 0.75), nT - 1];

  // Build ECM display points using extractEcmValue
  const ecmDisplayPts = ecmColdPath.map((pt, ptIdx) => {
    const raw = extractEcmValue(param, pt);
    const v   = trajDisplayValue(param, raw);
    const resnormT = ecmResnorm?.[ptIdx] ?? null;
    // Band halfFrac: scale with fitting quality; clamp between 5% and 35%
    const halfFrac = resnormT !== null ? Math.min(Math.max(resnormT * 4, 0.05), 0.35) : 0.15;
    return v > 0 ? { x: sx(pt.time_min), y: syClip(v), yHi: syClip(v * (1 - halfFrac)), yLo: syClip(v * (1 + halfFrac)) } : null;
  }).filter((d): d is NonNullable<typeof d> => d !== null);

  const plotClipId = `plot-clip-${param}`;

  return (
    <div style={{ position: 'relative' }}>
      {/* Hover tooltip — positioned as percentage of SVG viewBox width */}
      {hoverIdx !== null && hoverIdx < timeMin.length && (() => {
        const hx = sx(timeMin[hoverIdx]);
        const leftPct = (hx / W_TRAJ) * 100;
        const gt = gtVals?.[hoverIdx] ?? null;
        const relErr = (val: number) => {
          if (gt === null || !(gt > 0)) return null;
          return Math.abs(val - gt) / gt * 100;
        };
        const fmtVal = (v: number) => {
          if (!isFinite(v) || !(v > 0)) return '—';
          if (v >= 100) return v.toFixed(1);
          if (v >= 1)   return v.toFixed(2);
          if (v >= 0.01) return v.toFixed(4);
          return v.toExponential(2);
        };
        const unit = TRAJ_PARAM_LABELS[param].match(/\(([^)]+)\)/)?.[1] ?? '';
        const rows: { label: string; color: string; val: number | null }[] = [];
        // RBPF trajectories
        for (const traj of visTrajectories) {
          if (hiddenLines.has(`hyp${traj.hypothesis}`)) continue;
          const pt = traj.path[hoverIdx];
          if (!pt) continue;
          const raw = extractSmoothedValue(param, pt);
          const v = trajDisplayValue(param, raw);
          rows.push({ label: traj.label.replace(/ECM/g, 'Fitted'), color: stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]), val: isFinite(v) && v > 0 ? v : null });
        }
        // Baseline
        if (showEcmPath && !hiddenLines.has('ecm') && ecmColdPath[hoverIdx]) {
          const raw = extractEcmValue(param, ecmColdPath[hoverIdx]);
          const v = trajDisplayValue(param, raw);
          rows.push({ label: 'ECM', color: CHART_ECM_COLOR, val: isFinite(v) && v > 0 ? v : null });
        }
        // GT
        if (gtVals && !hiddenLines.has('gt')) {
          const v = gtVals[hoverIdx];
          rows.push({ label: 'GT', color: CHART_GT_COLOR, val: isFinite(v) && v > 0 ? v : null });
        }
        return (
          <div
            style={{
              position: 'absolute',
              left: `${Math.max(2, Math.min(leftPct, 72))}%`,
              top: '4px',
              transform: leftPct > 60 ? 'translateX(-100%)' : 'none',
              pointerEvents: 'none',
              zIndex: 20,
            }}
            className="bg-[#1e1e24] border border-[#2a2a33] rounded px-2 py-1.5 font-mono shadow-lg"
          >
            <div className="text-[8.5px] text-[#454549] mb-1 border-b border-[#2a2a33] pb-1">
              t = {timeMin[hoverIdx].toFixed(1)} min
            </div>
            {rows.map((row, ri) => {
              const err = row.val !== null ? relErr(row.val) : null;
              return (
                <div key={ri} className="flex items-baseline gap-2 text-[8.5px] leading-relaxed">
                  <span style={{ color: row.color }} className="w-[70px] truncate">{row.label}</span>
                  <span className="text-neutral-200 w-[52px] text-right">
                    {row.val !== null ? fmtVal(row.val) : '—'}
                    {unit ? <span className="text-neutral-600 ml-0.5">{unit}</span> : null}
                  </span>
                  {err !== null && row.label !== 'GT' && (
                    <span className={`text-[7.5px] w-[36px] text-right ${err < 20 ? 'text-neutral-400' : err < 50 ? 'text-neutral-500' : 'text-neutral-600'}`}>
                      {err.toFixed(1)}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        );
      })()}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W_TRAJ} ${localH}`}
        width="100%"
        style={{ display: 'block', cursor: 'crosshair' }}
        onMouseMove={(e) => {
          const svg = svgRef.current;
          if (!svg) return;
          const rect = svg.getBoundingClientRect();
          const svgX = (e.clientX - rect.left) * (W_TRAJ / rect.width);
          if (svgX < MG_TRAJ.left - 5 || svgX > MG_TRAJ.left + pw + 5) { setHoverIdx(null); return; }
          const t = tMin + (svgX - MG_TRAJ.left) / pw * xRange;
          let nearestIdx = 0;
          let minDist = Infinity;
          for (let i = 0; i < timeMin.length; i++) {
            const dist = Math.abs(timeMin[i] - t);
            if (dist < minDist) { minDist = dist; nearestIdx = i; }
          }
          setHoverIdx(nearestIdx);
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        <defs>
          {/* Clip everything to the plot area — prevents lines escaping the axes */}
          <clipPath id={plotClipId}>
            <rect x={MG_TRAJ.left} y={MG_TRAJ.top} width={pw} height={ph} />
          </clipPath>
        </defs>

        {/* Wrapped in clipPath group so all chart content stays inside the axes */}
        <g clipPath={`url(#${plotClipId})`}>

        {/* ATP window shading */}
        {atpLo !== null && atpHi !== null && (
          <rect x={sx(atpLo)} y={MG_TRAJ.top} width={Math.max(0, sx(atpHi) - sx(atpLo))} height={ph}
            fill="#f4a7b9" opacity={0.09} />
        )}

        {/* Major grid lines at decades only */}
        {yTicksMajor.map((v, i) => (
          <line key={i} x1={MG_TRAJ.left} y1={sy(v)} x2={MG_TRAJ.left + pw} y2={sy(v)}
            stroke="#1e2430" strokeWidth={0.8} />
        ))}

        {/* Ground truth — white, dominant reference line with dots */}
        {gtVals && !hiddenLines.has('gt') && (
          <g>
            <polyline
              points={gtVals.map((v, i) => `${sx(timeMin[i] ?? tMin)},${syClip(v)}`).join(' ')}
              fill="none" stroke={CHART_GT_COLOR} strokeWidth={2.0} opacity={0.9}
            />
            {gtVals.map((v, i) => {
              const cy = syClip(v);
              if (!isFinite(cy)) return null;
              return <circle key={i} cx={sx(timeMin[i] ?? tMin)} cy={cy} r={2.5} fill={CHART_GT_COLOR} stroke="none" opacity={0.9} />;
            })}
          </g>
        )}

        {/* ECM: probabilistic band + median line */}
        {showEcmPath && ecmDisplayPts.length > 0 && !hiddenLines.has('ecm') && (
          <g>
            {/* Uncertainty band (width ∝ ecmResnorm) */}
            <polygon
              points={[
                ...ecmDisplayPts.map(d => `${d.x},${d.yHi}`),
                ...[...ecmDisplayPts].reverse().map(d => `${d.x},${d.yLo}`),
              ].join(' ')}
              fill={CHART_ECM_COLOR} opacity={0.13} stroke="none"
            />
            {/* Median line */}
            <polyline
              points={ecmDisplayPts.map(d => `${d.x},${d.y}`).join(' ')}
              fill="none" stroke={CHART_ECM_COLOR} strokeWidth={1.4} strokeDasharray="5,3" opacity={0.65}
            />
          </g>
        )}

        {/* Per-hypothesis: envelope band + median line + per-point distribution markers */}
        {visTrajectories.map((traj) => {
          const lineKey = `hyp${traj.hypothesis}`;
          if (hiddenLines.has(lineKey)) return null;

          const color = stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]);
          const baseOpacity = Math.max(0.35, traj.probability);
          const opacity = baseOpacity;
          const strokeW = traj.rank === 0 ? 1.8 : 1.1;

          const safeY = (raw: number) => {
            const v = trajDisplayValue(param, raw);
            return (isFinite(v) && v > 0) ? sy(v) : null;
          };

          const medPts = traj.path
            .map(pt => {
              const y = safeY(extractSmoothedValue(param, pt));
              return y !== null ? `${sx(pt.time_min)},${y}` : null;
            })
            .filter((s): s is string => s !== null)
            .join(' ');

          // q25/q75 inner band
          const bandTopPts = traj.path.map(pt => {
            const y = safeY(extractSmoothedQ75(param, pt));
            return y !== null ? `${sx(pt.time_min)},${y}` : null;
          }).filter((s): s is string => s !== null);
          const bandBotPts = [...traj.path].reverse().map(pt => {
            const y = safeY(extractSmoothedQ25(param, pt));
            return y !== null ? `${sx(pt.time_min)},${y}` : null;
          }).filter((s): s is string => s !== null);
          const bandPts = [...bandTopPts, ...bandBotPts].join(' ') || null;

          // q05/q95 outer credible band
          const outerTopPts = traj.path.map(pt => {
            const y = safeY(extractSmoothedQ95(param, pt));
            return y !== null ? `${sx(pt.time_min)},${y}` : null;
          }).filter((s): s is string => s !== null);
          const outerBotPts = [...traj.path].reverse().map(pt => {
            const y = safeY(extractSmoothedQ05(param, pt));
            return y !== null ? `${sx(pt.time_min)},${y}` : null;
          }).filter((s): s is string => s !== null);
          const outerBandPts = [...outerTopPts, ...outerBotPts].join(' ') || null;

          const step = traj.path.length > 20 ? 2 : 1;

          const outerBandOpacity = traj.rank === 0 ? opacity * 0.07 : opacity * 0.03;
          const bandOpacity = traj.rank === 0 ? opacity * 0.15 : opacity * 0.06;
          return (
            <g key={traj.hypothesis} style={{ cursor: 'default' }}>
              {/* 90% credible band (q05-q95) — outermost, most transparent */}
              {outerBandPts && (
                <polygon points={outerBandPts} fill={color} opacity={outerBandOpacity} stroke="none" />
              )}

              {/* 50% credible band (q25-q75) — inner, more opaque */}
              {bandPts && (
                <polygon points={bandPts} fill={color} opacity={bandOpacity} stroke="none" />
              )}

              {/* Median connecting line */}
              {medPts.length > 0 && (
                <polyline points={medPts} fill="none" stroke={color}
                  strokeWidth={strokeW}
                  opacity={opacity * 0.9}
                  strokeDasharray={traj.rank === 0 ? undefined : '4,2'}
                />
              )}

              {/* Per-point: vertical extent bar (q25→q75) + cap ticks + median dot */}
              {traj.path.filter((_, i) => i % step === 0).map((pt, i) => {
                const cx   = sx(pt.time_min);
                const yMed = safeY(extractSmoothedValue(param, pt));
                const yLo  = safeY(extractSmoothedQ25(param, pt));
                const yHi  = safeY(extractSmoothedQ75(param, pt));
                if (yMed === null || yLo === null || yHi === null) return null;
                const capLen = traj.rank === 0 ? 4 : 3;
                const ibarOpacity = traj.rank === 0 ? 0.65 : 0.45;
                const capW = traj.rank === 0 ? 1.1 : 0.9;
                return (
                  <g key={i} opacity={opacity}>
                    <line x1={cx} y1={yHi} x2={cx} y2={yLo}
                      stroke={color} strokeWidth={traj.rank === 0 ? 1.2 : 0.8} opacity={ibarOpacity} />
                    <line x1={cx - capLen} y1={yHi} x2={cx + capLen} y2={yHi}
                      stroke={color} strokeWidth={capW} opacity={ibarOpacity * 0.85} />
                    <line x1={cx - capLen} y1={yLo} x2={cx + capLen} y2={yLo}
                      stroke={color} strokeWidth={capW} opacity={ibarOpacity * 0.85} />
                    <circle cx={cx} cy={yMed}
                      r={traj.rank === 0 ? 2.2 : 1.6}
                      fill={color} stroke="none" />
                  </g>
                );
              })}
            </g>
          );
        })}


        {/* Hover crosshair + dots */}
        {hoverIdx !== null && hoverIdx < timeMin.length && (() => {
          const hx = sx(timeMin[hoverIdx]);
          return (
            <g>
              {/* Vertical crosshair */}
              <line x1={hx} y1={MG_TRAJ.top} x2={hx} y2={MG_TRAJ.top + ph}
                stroke="#6b7280" strokeWidth={0.8} opacity={0.5} strokeDasharray="3,2" />
              {/* Trajectory dots */}
              {visTrajectories.map(traj => {
                if (hiddenLines.has(`hyp${traj.hypothesis}`)) return null;
                const pt = traj.path[hoverIdx];
                if (!pt) return null;
                const v = trajDisplayValue(param, extractSmoothedValue(param, pt));
                if (!(v > 0)) return null;
                const color = stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]);
                return (
                  <circle key={traj.hypothesis}
                    cx={hx} cy={syClip(v)} r={4}
                    fill={color} stroke="none" opacity={0.95} />
                );
              })}
              {/* ECM dot */}
              {showEcmPath && !hiddenLines.has('ecm') && ecmColdPath[hoverIdx] && (() => {
                const v = trajDisplayValue(param, extractEcmValue(param, ecmColdPath[hoverIdx]));
                if (!(v > 0)) return null;
                return <circle cx={hx} cy={syClip(v)} r={3.5} fill={CHART_ECM_COLOR} stroke="none" opacity={0.9} />;
              })()}
            </g>
          );
        })()}

        </g>{/* end clipPath group */}

        {/* L-shaped axes */}
        <line x1={MG_TRAJ.left} y1={MG_TRAJ.top} x2={MG_TRAJ.left} y2={MG_TRAJ.top + ph}
          stroke="#374151" strokeWidth={1} />
        <line x1={MG_TRAJ.left} y1={MG_TRAJ.top + ph} x2={MG_TRAJ.left + pw} y2={MG_TRAJ.top + ph}
          stroke="#374151" strokeWidth={1} />

        {/* Y-axis: major ticks outward, labeled */}
        {yTicksMajor.map((v, i) => (
          <g key={`maj-${i}`}>
            <line x1={MG_TRAJ.left - 4} y1={sy(v)} x2={MG_TRAJ.left} y2={sy(v)}
              stroke="#374151" strokeWidth={0.8} />
            <text x={MG_TRAJ.left - 6} y={sy(v) + 3.5}
              fill="#9ca3af" fontSize={9} textAnchor="end">
              {formatTick(v)}
            </text>
          </g>
        ))}

        {/* Rotated y-axis label */}
        <text
          transform={`translate(${MG_TRAJ.left - 46}, ${MG_TRAJ.top + ph / 2}) rotate(-90)`}
          fill="#6b7280" fontSize={9} textAnchor="middle">
          {TRAJ_PARAM_LABELS[param]}
        </text>

        {/* X-axis ticks outward */}
        {xIdxs.map(i => (
          <g key={i}>
            <line x1={sx(timeMin[i])} y1={MG_TRAJ.top + ph} x2={sx(timeMin[i])} y2={MG_TRAJ.top + ph + 4}
              stroke="#374151" strokeWidth={0.8} />
            <text x={sx(timeMin[i])} y={MG_TRAJ.top + ph + 13}
              fill="#9ca3af" fontSize={9} textAnchor="middle">
              {timeMin[i].toFixed(0)}
            </text>
          </g>
        ))}

        {/* X-axis label */}
        <text x={MG_TRAJ.left + pw / 2} y={localH - 2}
          fill="#6b7280" fontSize={9} textAnchor="middle">
          time (min)
        </text>

      </svg>

      {/* Checkbox legend */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 pb-3 pt-2">
        {visTrajectories.map(traj => {
          const lineKey = `hyp${traj.hypothesis}`;
          const isVisible = !hiddenLines.has(lineKey);
          const color = stageColorForTraj(traj.label, colors[traj.hypothesis % colors.length]);
          return (
            <button
              key={traj.hypothesis}
              onClick={() => toggleLine(lineKey)}
              className="flex items-center gap-1.5 select-none focus:outline-none"
              style={{ opacity: isVisible ? 1 : 0.35 }}
              title={isVisible ? 'Click to hide' : 'Click to show'}
            >
              {/* Checkbox */}
              <div style={{
                width: 10, height: 10, flexShrink: 0,
                border: `1.5px solid ${color}`, borderRadius: 2,
                backgroundColor: isVisible ? color : 'transparent',
              }} />
              {/* Line swatch */}
              <svg width="20" height="7" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3.5" x2="20" y2="3.5"
                  stroke={color}
                  strokeWidth={traj.rank === 0 ? 2.2 : 1.4}
                  strokeDasharray={traj.rank !== 0 ? '4,2' : undefined} />
              </svg>
              <span className="text-[9.5px] text-neutral-400">
                {traj.label.replace(/ECM/g, 'Fitted')}
                <span className="ml-1 font-semibold" style={{ color }}>
                  {(traj.probability * 100).toFixed(0)}%
                </span>
              </span>
            </button>
          );
        })}

        {/* ECM checkbox */}
        {showEcmPath && ecmDisplayPts.length > 0 && (() => {
          const isVisible = !hiddenLines.has('ecm');
          return (
            <button
              onClick={() => toggleLine('ecm')}
              className="flex items-center gap-1.5 select-none focus:outline-none"
              style={{ opacity: isVisible ? 1 : 0.35 }}
              title={isVisible ? 'Click to hide' : 'Click to show'}
            >
              <div style={{
                width: 10, height: 10, flexShrink: 0,
                border: `1.5px solid ${CHART_ECM_COLOR}`, borderRadius: 2,
                backgroundColor: isVisible ? CHART_ECM_COLOR : 'transparent',
              }} />
              <svg width="20" height="7" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3.5" x2="20" y2="3.5"
                  stroke={CHART_ECM_COLOR} strokeWidth={1.4} strokeDasharray="5,3" opacity={0.8} />
              </svg>
              <span className="text-[9.5px] text-neutral-500">
                Fitted <span className="text-neutral-600">±band</span>
              </span>
            </button>
          );
        })()}

        {/* Ground truth checkbox */}
        {gtVals && (() => {
          const isVisible = !hiddenLines.has('gt');
          return (
            <button
              onClick={() => toggleLine('gt')}
              className="flex items-center gap-1.5 select-none focus:outline-none"
              style={{ opacity: isVisible ? 1 : 0.35 }}
              title={isVisible ? 'Click to hide' : 'Click to show'}
            >
              <div style={{
                width: 10, height: 10, flexShrink: 0,
                border: `1.5px solid ${CHART_GT_COLOR}`, borderRadius: 2,
                backgroundColor: isVisible ? CHART_GT_COLOR : 'transparent',
              }} />
              <svg width="20" height="7" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3.5" x2="20" y2="3.5"
                  stroke={CHART_GT_COLOR} strokeWidth={1.2} strokeDasharray="4,3" opacity={0.8} />
              </svg>
              <span className="text-[9.5px] text-neutral-500">ground truth</span>
            </button>
          );
        })()}

      </div>
    </div>
  );
}

// ---- shared MI helpers ----

function miLinePath(vals: number[], xScale: (i: number) => number, yScale: (v: number) => number): string {
  return vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ');
}

function miAreaPath(vals: number[], xScale: (i: number) => number, yScale: (v: number) => number, yBaseline: number): string {
  const line = miLinePath(vals, xScale, yScale);
  const last = vals.length - 1;
  return `${line} L${xScale(last).toFixed(1)},${yBaseline.toFixed(1)} L${xScale(0).toFixed(1)},${yBaseline.toFixed(1)} Z`;
}

// ---- helper: kalmanPhysFormat ----
function kalmanPhysFormat(name: string, log10v: number): string {
  const v = Math.pow(10, log10v);
  if (name.startsWith('tau')) return v < 0.05 ? `${(v * 60).toFixed(0)}s` : `${v.toFixed(2)}m`;
  if (name === 'TEC') return `${v.toFixed(3)}µF`;
  return v >= 100 ? `${(v / 1000).toFixed(1)}MΩ` : `${v.toFixed(1)}kΩ`;
}

// ---- Viz: Filter Health (Diagnostics) ----
// Combines ESS health, Kalman identifiability, and per-parameter dynamics in one panel.
const DIAG_PARAM_LABELS: Record<string, string> = {
  tau_big: 'τ_big', tau_small: 'τ_sm', TER: 'TER', TEC: 'TEC', Rsh: 'Rsh',
};
const DIAG_PARAM_COLORS = ['#60a5fa', '#4ade80', '#f97316', '#818cf8', '#c4a040'];

function VizDiagnosticsBody({ node, inputData, onConfigChange }: NodeBodyProps) {
  const cfg    = node.config as VizDiagnosticsCfg;
  const result = inputData?.kind === 'ml-result' ? inputData.result : null;
  const kalman = result?.kalman ?? null;

  if (!result) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-[10px] text-[#454549]">Connect a Process node ←</p>
      </div>
    );
  }

  const T      = result.time_min.length;
  const times  = result.time_min;
  const W      = 460;
  const MG     = { left: 10, right: 10, top: 6, bottom: 14 };
  const pw     = W - MG.left - MG.right;
  const tScale = (i: number) => MG.left + (i / Math.max(T - 1, 1)) * pw;

  // ESS panel
  const ess      = result.smcEss;
  const essPct   = ess.map(v => v * 100);
  const avgEss   = essPct.length > 0 ? essPct.reduce((a, v) => a + v, 0) / essPct.length : null;
  const minEss   = essPct.length > 0 ? Math.min(...essPct) : null;
  const essOk    = avgEss != null && avgEss >= 80;
  const essWarn  = avgEss != null && avgEss >= 50 && avgEss < 80;
  const essColor = essOk ? '#4ade80' : essWarn ? '#c4a040' : '#f97316';

  // Identifiability panel (Kalman geometry)
  const geo         = (kalman as unknown as { geometry?: { identifiability: number[]; eigen_vals: number[][] } } | null)?.geometry ?? null;
  const identScore  = geo?.identifiability ?? [];
  const hasIdent    = identScore.length > 1;
  const paramNames  = kalman?.param_names ?? [];
  const nP          = paramNames.length;
  const pIdx        = Math.min(cfg.paramIdx, Math.max(nP - 1, 0));

  const IDENT_PARAMS: TrajectoryParam[] = ['TER', 'TEC', 'tau_big' as TrajectoryParam, 'tau_small' as TrajectoryParam, 'Rsh'];
  const H_IDENT = 80;
  const ph_id   = H_IDENT - MG.top - MG.bottom;

  // Kalman detail panel
  const hasKalman   = kalman != null && kalman.mu_smoothed.length > 0;
  const smoothed    = hasKalman ? kalman!.mu_smoothed.map(row => row[pIdx]) : [];
  const smoothStd   = hasKalman ? kalman!.std_smoothed.map(row => row[pIdx]) : [];
  const filtered    = hasKalman ? kalman!.mu_filtered.map(row => row[pIdx]) : [];
  const allPos      = [...smoothed, ...filtered].filter(isFinite);
  const posLo       = allPos.length > 0 ? Math.min(...allPos) : 0;
  const posHi       = allPos.length > 0 ? Math.max(...allPos) : 1;
  const posRng      = posHi - posLo || 1;
  const H_DET       = 80;
  const ph_d        = H_DET - MG.top - MG.bottom;
  const py          = (v: number) => MG.top + (1 - (v - posLo) / posRng) * ph_d;
  const gx          = (t: number) => MG.left + ((t - (times[0] ?? 0)) / Math.max((times[T - 1] ?? 1) - (times[0] ?? 0), 1)) * pw;

  const bandPath = hasKalman && smoothed.length > 1 ? [
    smoothed.map((v, i) => `${i === 0 ? 'M' : 'L'}${gx(times[i]).toFixed(1)},${py(v + smoothStd[i]).toFixed(1)}`).join(' '),
    smoothed.map((v, i) => `${i === 0 ? 'M' : 'L'}${gx(times[i]).toFixed(1)},${py(v - smoothStd[i]).toFixed(1)}`).reverse().join(' '),
    'Z',
  ].join(' ') : '';
  const smoothPath = hasKalman && smoothed.length > 1 ? miLinePath(smoothed, i => gx(times[i]), py) : '';
  const filterPath = hasKalman && filtered.length > 1 ? miLinePath(filtered, i => gx(times[i]), py) : '';

  const changepoints   = result.changepoints ?? [];
  const paramColor     = DIAG_PARAM_COLORS[pIdx % DIAG_PARAM_COLORS.length];
  const paramLabel     = DIAG_PARAM_LABELS[paramNames[pIdx] ?? ''] ?? paramNames[pIdx] ?? '';
  const atpLo = (result as unknown as { atpLo?: number }).atpLo;
  const atpHi = (result as unknown as { atpHi?: number }).atpHi;

  return (
    <div className="flex flex-col h-full text-[10px] text-[#9a9aa2]">
      {/* Header — param selector */}
      <div className="flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 border-b border-[#1e1e24]">
        <span className="text-[#454549] uppercase tracking-widest text-[9px] flex-shrink-0">Filter Health</span>
        {hasKalman && (
          <div className="flex ml-2 gap-0.5 flex-wrap">
            {paramNames.map((name, i) => (
              <button
                key={i}
                onClick={() => onConfigChange({ ...cfg, paramIdx: i })}
                className={`px-1.5 py-0.5 text-[8px] rounded focus:outline-none transition-colors ${pIdx === i ? 'bg-[#1e1e24] text-[#dddde2] border border-[#3a3a46]' : 'text-[#454549] hover:text-[#9a9aa2]'}`}
                style={{ borderColor: pIdx === i ? DIAG_PARAM_COLORS[i % DIAG_PARAM_COLORS.length] : undefined }}
              >
                {DIAG_PARAM_LABELS[name] ?? name}
              </button>
            ))}
          </div>
        )}
        {avgEss != null && (
          <span className="ml-auto text-[8.5px] font-mono flex-shrink-0" style={{ color: essColor }}>
            ESS {avgEss.toFixed(0)}%
          </span>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">

        {/* Panel 0: Cluster paths */}
        {result.clusterPaths.length > 0 && (
          <div className="flex-shrink-0 px-3 pt-2 pb-1 border-b border-[#1e1e24]">
            <div className="text-[9px] text-[#454549] mb-1.5 uppercase tracking-wider">particle clusters</div>
            {result.clusterPaths.map((cp, i) => {
              const color = HYP_COLORS[cp.hypothesis % 4];
              return (
                <div key={i} className="flex items-center gap-1.5 py-0.5">
                  <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }}/>
                  <span className="text-[8px] text-[#7a7a82] flex-1 truncate">{cp.label.replace(/ECM/g, 'Fitted')}</span>
                  {cp.pCaGtCb != null && (
                    <span className="font-mono text-[7px] text-[#3d3d48]">Ca{cp.pCaGtCb > 0.5 ? '>' : '<'}Cb</span>
                  )}
                  <div className="w-16 h-0.5 rounded-full bg-[#1a1a1e] overflow-hidden flex-shrink-0">
                    <div className="h-full rounded-full transition-all" style={{ width: `${cp.probability * 100}%`, backgroundColor: color }}/>
                  </div>
                  <span className="font-mono text-[7px] tabular-nums w-6 text-right flex-shrink-0" style={{ color }}>
                    {(cp.probability * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Panel 0a: Nonlinear mode posterior — the RBPF-specific visualization */}
        {result.smcPCaGtCb.length > 1 && (
          <div className="flex-shrink-0 px-3 pt-2 pb-1.5 border-b border-[#1e1e24]">
            <div className="text-[9px] text-[#454549] mb-0.5 uppercase tracking-wider">nonlinear mode posterior θ_nl</div>
            <div className="text-[7.5px] font-mono text-[#2a2a33] mb-1.5">P(Ca&gt;Cb | Z₁:t) · discrete state marginal from particle ensemble</div>
            <svg viewBox={`0 0 ${W} 50`} width="100%" height={50} style={{ display: 'block' }}>
              <line x1={MG.left} y1={MG.top + (50 - MG.top - MG.bottom) / 2} x2={MG.left + pw} y2={MG.top + (50 - MG.top - MG.bottom) / 2}
                stroke="#1e1e24" strokeWidth={0.8} strokeDasharray="2,3"/>
              <line x1={MG.left} y1={MG.top} x2={MG.left + pw} y2={MG.top} stroke="#0d0d10" strokeWidth={0.5}/>
              <line x1={MG.left} y1={50 - MG.bottom} x2={MG.left + pw} y2={50 - MG.bottom} stroke="#1e1e24" strokeWidth={1}/>
              {(() => {
                const vals = result.smcPCaGtCb;
                const pH = 50 - MG.top - MG.bottom;
                const sx = (i: number) => MG.left + (i / Math.max(vals.length - 1, 1)) * pw;
                const sy = (v: number) => MG.top + (1 - Math.min(Math.max(v, 0), 1)) * pH;
                const pts = vals.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
                const areaBot = 50 - MG.bottom;
                return (
                  <>
                    <polygon points={`${sx(0).toFixed(1)},${sy(vals[0]).toFixed(1)} ${vals.map((v,i)=>`${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')} ${sx(vals.length-1).toFixed(1)},${areaBot} ${sx(0).toFixed(1)},${areaBot}`}
                      fill="#56B4E9" opacity={0.07}/>
                    <polyline points={pts} fill="none" stroke="#56B4E9" strokeWidth={1.4} opacity={0.85}/>
                    {result.smcPRaLtRb.length > 1 && (
                      <polyline
                        points={result.smcPRaLtRb.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')}
                        fill="none" stroke="#E69F00" strokeWidth={1.0} opacity={0.6} strokeDasharray="4,2"/>
                    )}
                  </>
                );
              })()}
              <text x={MG.left - 2} y={MG.top + 4} fontSize={5.5} fill="#454549" textAnchor="end">1</text>
              <text x={MG.left - 2} y={50 - MG.bottom + 1} fontSize={5.5} fill="#454549" textAnchor="end">0</text>
              <text x={MG.left} y={50 - 1} fontSize={5.5} fill="#454549">{times[0]?.toFixed(0)}m</text>
              <text x={MG.left + pw} y={50 - 1} fontSize={5.5} fill="#454549" textAnchor="end">{times[T-1]?.toFixed(0)}m</text>
            </svg>
            <div className="flex gap-3 mt-1">
              <span className="text-[7px] font-mono text-[#56B4E9]">— P(Ca&gt;Cb | Z₁:t)</span>
              {result.smcPRaLtRb.length > 1 && <span className="text-[7px] font-mono text-[#E69F00]">- - P(Ra&lt;Rb | Z₁:t)</span>}
              <span className="text-[7px] font-mono text-[#2a2a33] ml-auto">0.5 = max uncertainty</span>
            </div>
          </div>
        )}

        {/* Panel 0b: Kalman parameter state — log₁₀ → display with evolution sparklines */}
        {kalman && kalman.mu_smoothed.length > 0 && (
          <div className="flex-shrink-0 px-3 pt-2 pb-1 border-b border-[#1e1e24]">
            <div className="text-[9px] text-[#454549] mb-1 uppercase tracking-wider">log₁₀ state → display · evolution</div>
            {KALMAN_STATE_PARAMS.map(({ idx, label, unit, toDisplay }) => {
              const nT = kalman.mu_smoothed.length;
              const finalLog = kalman.mu_smoothed[nT - 1][idx];
              const finalStd = kalman.std_smoothed[nT - 1][idx];
              const evolution = kalman.mu_smoothed.map(row => row[idx]);
              const dispVal = toDisplay(Math.pow(10, finalLog));
              return (
                <div key={idx} className="flex items-center gap-1 py-0.5 border-b border-[#0d0d10] last:border-0">
                  <span className="font-mono text-[8px] text-[#7a7a82] w-12 flex-shrink-0">{label}</span>
                  <span className="font-mono text-[8px] text-[#6469a0] tabular-nums flex-shrink-0 w-16">
                    {finalLog.toFixed(2)}<span className="text-[#2a2a33] text-[7px]"> ±{finalStd.toFixed(2)}</span>
                  </span>
                  <span className="font-mono text-[7px] text-[#9a9aa2] tabular-nums flex-1 text-right pr-1">
                    {dispVal.toPrecision(3)} {unit}
                  </span>
                  <InlineSparkline values={evolution} color="#6469a0" />
                </div>
              );
            })}
          </div>
        )}

        {/* Panel 1: Particle filter health — ESS + resampling events */}
        {essPct.length > 1 && (() => {
          const resampleEvents: number[] = [];
          for (let i = 1; i < essPct.length; i++) {
            if (essPct[i - 1] < 50 && essPct[i] >= 50) resampleEvents.push(i);
          }
          const belowThreshPct = essPct.filter(v => v < 50).length / essPct.length * 100;
          const PW = 440; const PH = 56;
          const PMG = { left: 8, right: 8, top: 6, bottom: 14 };
          const ppw = PW - PMG.left - PMG.right;
          const pph = PH - PMG.top - PMG.bottom;
          const sx = (i: number) => PMG.left + (i / Math.max(essPct.length - 1, 1)) * ppw;
          const sy = (v: number) => PMG.top + (1 - Math.min(Math.max(v, 0), 100) / 100) * pph;
          const linePts = essPct.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
          return (
            <div className="flex-shrink-0 px-3 pt-2 pb-1 border-b border-[#1e1e24]">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[9px] text-[#454549] uppercase tracking-wider">Particle Filter Health</span>
                <span className="font-mono text-[7px]" style={{ color: essColor }}>avg {avgEss?.toFixed(0)}%</span>
                <span className="font-mono text-[7px] text-[#454549]">min {minEss?.toFixed(0)}%</span>
                <span className="font-mono text-[7px] text-[#c4a040] ml-auto">{resampleEvents.length} resamples</span>
              </div>
              <svg viewBox={`0 0 ${PW} ${PH}`} width="100%" height={PH} style={{ display: 'block', overflow: 'visible' }}>
                {/* Shaded below-threshold region */}
                <defs>
                  <linearGradient id="ess-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={essColor} stopOpacity={0.18}/>
                    <stop offset="100%" stopColor={essColor} stopOpacity={0.03}/>
                  </linearGradient>
                </defs>
                {/* Danger zone: below 50% */}
                <rect x={PMG.left} y={sy(50)} width={ppw} height={PH - PMG.bottom - sy(50)} fill="#f97316" opacity={0.04}/>
                {/* Threshold line */}
                <line x1={PMG.left} y1={sy(50)} x2={PMG.left + ppw} y2={sy(50)} stroke="#f97316" strokeWidth={0.6} strokeDasharray="3,2" opacity={0.4}/>
                {/* ESS area fill */}
                <polygon
                  points={`${sx(0).toFixed(1)},${(PH - PMG.bottom).toFixed(1)} ${linePts} ${sx(essPct.length - 1).toFixed(1)},${(PH - PMG.bottom).toFixed(1)}`}
                  fill={`url(#ess-grad)`}/>
                {/* ESS line — color by health */}
                {essPct.map((v, i) => {
                  if (i === 0) return null;
                  const ok = v >= 50;
                  const c = ok ? '#4ade80' : '#f97316';
                  return <line key={i} x1={sx(i-1).toFixed(1)} y1={sy(essPct[i-1]).toFixed(1)} x2={sx(i).toFixed(1)} y2={sy(v).toFixed(1)} stroke={c} strokeWidth={1.3} opacity={0.9}/>;
                })}
                {/* Resampling event markers — vertical lines where filter recovered */}
                {resampleEvents.map(i => (
                  <g key={i}>
                    <line x1={sx(i)} y1={PMG.top} x2={sx(i)} y2={PH - PMG.bottom} stroke="#c4a040" strokeWidth={0.8} strokeDasharray="2,2" opacity={0.6}/>
                    <circle cx={sx(i)} cy={sy(essPct[i])} r={2} fill="#c4a040" opacity={0.8}/>
                  </g>
                ))}
                {/* Axes */}
                <line x1={PMG.left} y1={PH - PMG.bottom} x2={PMG.left + ppw} y2={PH - PMG.bottom} stroke="#1e1e24" strokeWidth={1}/>
                <text x={2} y={sy(100) + 4} fontSize={5} fill="#454549">100%</text>
                <text x={2} y={sy(50) + 4} fontSize={5} fill="#f97316" opacity={0.6}>50%</text>
                <text x={PMG.left} y={PH - 1} fontSize={5} fill="#454549">{times[0]?.toFixed(0)}m</text>
                <text x={PMG.left + ppw} y={PH - 1} textAnchor="end" fontSize={5} fill="#454549">{times[T-1]?.toFixed(0)}m</text>
              </svg>
              <div className="flex items-center gap-3 mt-0.5">
                <div className="flex items-center gap-1">
                  <span className="w-2 h-0.5 rounded bg-[#4ade80] inline-block"/>
                  <span className="text-[7px] text-[#454549] font-mono">ESS &gt; 50% (healthy)</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-0.5 rounded bg-[#f97316] inline-block"/>
                  <span className="text-[7px] text-[#454549] font-mono">ESS &lt; 50% (degraded)</span>
                </div>
                <div className="flex items-center gap-1 ml-auto">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#c4a040] inline-block"/>
                  <span className="text-[7px] text-[#454549] font-mono">resample</span>
                </div>
              </div>
              {belowThreshPct > 5 && (
                <div className="mt-1 text-[7px] font-mono text-[#c4a040]">
                  {belowThreshPct.toFixed(0)}% of steps below resampling threshold — particle diversity strained
                </div>
              )}
            </div>
          );
        })()}

        {/* Panel 2: Identifiability (Kalman geometry or IQR fallback) */}
        <div className="flex-shrink-0 px-3 pt-3">
          <div className="text-[9px] text-[#454549] mb-1 uppercase tracking-wider">
            {hasIdent ? 'identifiability λ_min/λ_max · 0=degenerate 1=isotropic' : 'posterior IQR by parameter · tighter = better identified'}
          </div>
          {hasIdent ? (
            <svg viewBox={`0 0 ${W} ${H_IDENT}`} width="100%" height={H_IDENT} style={{ display: 'block' }}>
              <path d={miAreaPath(identScore, tScale, v => MG.top + (1 - v) * ph_id, MG.top + ph_id)} fill="#4ade80" opacity={0.1} />
              <path d={miLinePath(identScore, tScale, v => MG.top + (1 - v) * ph_id)} fill="none" stroke="#4ade80" strokeWidth={1.3} />
              {changepoints.map((cp, ci) => {
                const ti = times.findIndex(t => t >= (cp.time_min ?? 0));
                if (ti < 0) return null;
                return <line key={ci} x1={tScale(ti)} y1={MG.top} x2={tScale(ti)} y2={MG.top + ph_id} stroke="#c4a040" strokeWidth={0.8} strokeDasharray="3,2" opacity={0.7} />;
              })}
              <line x1={MG.left} y1={MG.top + ph_id * 0.5} x2={MG.left + pw} y2={MG.top + ph_id * 0.5} stroke="#2a2a33" strokeWidth={0.5} strokeDasharray="2,3" />
              <line x1={MG.left} y1={MG.top + ph_id} x2={MG.left + pw} y2={MG.top + ph_id} stroke="#1e1e24" strokeWidth={1} />
              <text x={2} y={MG.top + 6} fontSize={5} fill="#454549">1</text>
              <text x={2} y={MG.top + ph_id} fontSize={5} fill="#454549">0</text>
              <text x={MG.left} y={H_IDENT - 1} fontSize={5.5} fill="#454549">{times[0]?.toFixed(0)}min</text>
              <text x={MG.left + pw} y={H_IDENT - 1} textAnchor="end" fontSize={5.5} fill="#454549">{times[T - 1]?.toFixed(0)}min</text>
            </svg>
          ) : (
            <div className="flex flex-col gap-1">
              {IDENT_PARAMS.map(param => {
                const key = (param === 'Rsh' ? 'R2' : param) as ParameterKey;
                const s   = result.predictions[key];
                if (!s) return null;
                const iqrs = Array.from({ length: T }, (_, i) => Math.abs((s.q75[i] ?? 0) - (s.q25[i] ?? 0)));
                const maxIqr = Math.max(...iqrs, 1e-9);
                const avgIqr = iqrs.reduce((a, v) => a + v, 0) / Math.max(iqrs.length, 1);
                const frac = Math.min(avgIqr / maxIqr, 1);
                const col = frac < 0.33 ? '#4ade80' : frac < 0.66 ? '#c4a040' : '#f97316';
                return (
                  <div key={param} className="flex items-center gap-2">
                    <span className="w-10 text-right text-[8px] flex-shrink-0" style={{ color: col }}>{DIAG_PARAM_LABELS[param as string] ?? param}</span>
                    <div className="flex-1 h-2 bg-[#0d0d0f] rounded overflow-hidden">
                      <div style={{ width: `${Math.max(frac * 100, 3)}%`, height: '100%', background: col, opacity: 0.7 }} />
                    </div>
                    <span className="text-[7px] font-mono text-[#454549] w-10 flex-shrink-0">IQR {avgIqr.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Panel 3: Kalman detail — smoothed vs filtered */}
        {hasKalman && smoothed.length > 1 && (
          <div className="flex-shrink-0 px-3 pt-3 pb-2">
            <div className="text-[9px] text-[#454549] mb-1 uppercase tracking-wider">
              {paramLabel} · smoothed ±1σ vs causal filtered
            </div>
            <svg viewBox={`0 0 ${W} ${H_DET}`} width="100%" height={H_DET} style={{ display: 'block' }}>
              {atpLo != null && atpHi != null && (
                <rect
                  x={gx(atpLo)} y={MG.top}
                  width={Math.max(gx(atpHi) - gx(atpLo), 1)} height={ph_d}
                  fill="#c4a040" opacity={0.07}
                />
              )}
              {bandPath && <path d={bandPath} fill={paramColor} opacity={0.12} />}
              {filterPath && <path d={filterPath} fill="none" stroke="#454549" strokeWidth={0.9} strokeDasharray="4,3" opacity={0.55} />}
              {smoothPath && <path d={smoothPath} fill="none" stroke={paramColor} strokeWidth={1.5} />}
              {changepoints.filter(cp => cp.dominant_param === paramNames[pIdx]).map((cp, ci) => (
                <line key={ci} x1={gx(cp.time_min ?? 0)} y1={MG.top} x2={gx(cp.time_min ?? 0)} y2={MG.top + ph_d} stroke="#c4a040" strokeWidth={0.9} strokeDasharray="3,2" opacity={0.7} />
              ))}
              <line x1={MG.left} y1={MG.top + ph_d} x2={MG.left + pw} y2={MG.top + ph_d} stroke="#1e1e24" strokeWidth={1} />
              <text x={2} y={MG.top + 7} fontSize={5.5} fill="#454549">{hasKalman && kalman ? kalmanPhysFormat(paramNames[pIdx], posHi) : ''}</text>
              <text x={2} y={MG.top + ph_d} fontSize={5.5} fill="#454549">{hasKalman && kalman ? kalmanPhysFormat(paramNames[pIdx], posLo) : ''}</text>
              <text x={MG.left} y={H_DET - 1} fontSize={5.5} fill="#454549">{times[0]?.toFixed(0)}min</text>
              <text x={MG.left + pw} y={H_DET - 1} textAnchor="end" fontSize={5.5} fill="#454549">{times[T - 1]?.toFixed(0)}min</text>
            </svg>
            <div className="flex gap-3 mt-0.5">
              <span className="text-[7.5px]" style={{ color: paramColor }}>— smoothed ±1σ</span>
              <span className="text-[7.5px] text-[#454549]">- - causal</span>
              {changepoints.length > 0 && <span className="text-[7.5px] text-[#c4a040]">| changepoints</span>}
            </div>
          </div>
        )}

        {/* Panel 4: Changepoints summary */}
        {changepoints.length > 0 && (
          <div className="flex-shrink-0 px-3 pt-2 pb-2 border-t border-[#1e1e24]">
            <div className="text-[9px] text-[#454549] mb-1 uppercase tracking-wider">detected changepoints</div>
            <div className="flex flex-col gap-1">
              {changepoints.map((cp, ci) => (
                <div key={ci} className="flex items-start gap-1.5">
                  <span className="text-[8px] font-mono text-[#c4a040] flex-shrink-0">{cp.time_min?.toFixed(0)}m</span>
                  <span className="text-[8px] text-[#9a9aa2] leading-tight">
                    {cp.interpretation ?? `${DIAG_PARAM_LABELS[cp.dominant_param] ?? cp.dominant_param} ${cp.direction}`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>{/* end scroll */}
    </div>
  );
}


// ---- ObservabilityBody ----
function ObservabilityBody({ node: _node, inputData, onRun: _onRun, onStop: _onStop }: NodeBodyProps) {
  const [obsResult, setObsResult] = React.useState<ObservabilityResult | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);

  const sequences = inputData?.kind === 'impedance-series' ? inputData.sequences : null;

  const runAnalysis = React.useCallback(async () => {
    if (!sequences || sequences.length === 0) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${ML_API_BASE}/observability_analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequences }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json() as ObservabilityResult;
      setObsResult(data);
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [sequences]);

  React.useEffect(() => { return () => { abortRef.current?.abort(); }; }, []);

  const fmtScore = (s: number) => `${Math.round(s * 100)}%`;
  const scoreColor = (s: number) => s > 0.7 ? '#4ade80' : s > 0.4 ? '#c4a040' : '#f97316';

  const IDENT_PARAMS = ['TER', 'Rsh', 'TEC', 'Ra', 'Rb'] as const;
  const IDENT_COLORS: Record<string, string> = { TER: '#009E73', Rsh: '#6469a0', TEC: '#CC79A7', Ra: '#56B4E9', Rb: '#E69F00' };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0 overflow-y-auto px-2 pt-1.5 pb-1 flex flex-col">

        <div className={SECTION_LABEL}>Data Quality</div>
        {obsResult ? (
          <>
            <div className="mt-1 flex flex-col gap-1">
              {([
                { label: 'SNR',       score: obsResult.snr_score },
                { label: 'Freq span', score: obsResult.freq_coverage_score },
                { label: 'Drift',     score: 1 - Math.min(obsResult.drift_score * 3, 1) },
              ] as Array<{ label: string; score: number }>).map(({ label, score }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <span className="font-mono text-[7.5px] text-[#454549] w-14 flex-shrink-0">{label}</span>
                  <div className="flex-1 h-1 bg-[#1e1e24] rounded overflow-hidden">
                    <div style={{ width: `${score * 100}%`, height: '100%', background: scoreColor(score), opacity: 0.7 }}/>
                  </div>
                  <span className="font-mono text-[8px] tabular-nums w-8 text-right flex-shrink-0" style={{ color: scoreColor(score) }}>{fmtScore(score)}</span>
                </div>
              ))}
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="font-mono text-[7.5px] text-[#454549] w-14 flex-shrink-0">Drift</span>
                <span className={`font-mono text-[7.5px] px-1 py-0.5 rounded ${obsResult.drift_label === 'stable' ? 'text-[#4ade80] bg-[#4ade80]/10' : obsResult.drift_label === 'event-driven' ? 'text-[#c4a040] bg-[#c4a040]/10' : 'text-[#f97316] bg-[#f97316]/10'}`}>
                  {obsResult.drift_label}
                </span>
                <span className="font-mono text-[7px] text-[#3d3d48] ml-auto">{obsResult.n_sweeps} sweeps</span>
              </div>
              <div className="font-mono text-[6.5px] text-[#454549]">
                {obsResult.freq_range[0].toExponential(1)} – {obsResult.freq_range[1].toExponential(1)} Hz
              </div>
            </div>

            <div className={NODE_DIVIDER}/>
            <div className={SECTION_LABEL}>Identifiability</div>
            <div className="mt-1 flex flex-col gap-1">
              {IDENT_PARAMS.map(p => {
                const entry = obsResult.identifiability[p];
                if (!entry) return null;
                const score = entry.score;
                const color = IDENT_COLORS[p] ?? '#6b6b76';
                return (
                  <div key={p} className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[8px] w-6 flex-shrink-0" style={{ color }}>{p}</span>
                      <div className="flex-1 h-1 bg-[#1e1e24] rounded overflow-hidden">
                        <div style={{ width: `${score * 100}%`, height: '100%', background: scoreColor(score), opacity: 0.6 }}/>
                      </div>
                      <span className="font-mono text-[7.5px] tabular-nums w-8 text-right flex-shrink-0" style={{ color: scoreColor(score) }}>{fmtScore(score)}</span>
                    </div>
                    {entry.note && (
                      <div className="font-mono text-[6.5px] text-[#454549] pl-6">{entry.note}{entry.structurally_ambiguous ? ' · structurally ambiguous' : ''}</div>
                    )}
                  </div>
                );
              })}
            </div>

            {obsResult.event_windows.length > 0 && (
              <>
                <div className={NODE_DIVIDER}/>
                <div className={SECTION_LABEL}>Events</div>
                <div className="mt-1 flex flex-col gap-0.5">
                  {obsResult.event_windows.map((w, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="font-mono text-[7px] text-[#c4a040]">{w.t_start.toFixed(0)}–{w.t_end.toFixed(0)} min</span>
                      <div className="flex-1 h-0.5 bg-[#1e1e24] rounded">
                        <div style={{ width: `${Math.round(w.confidence * 100)}%`, height: '100%', background: '#c4a040', opacity: 0.5 }}/>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        ) : (
          <span className="mt-1 font-mono text-[8px] text-[#2a2a33]">{inputData ? 'click Analyze' : 'connect data'}</span>
        )}

        {error && <p className="text-[8px] text-[#7a3a3a] mt-1.5 font-mono">{error}</p>}
      </div>

      <div className="flex-shrink-0 border-t border-[#1e1e24] px-2.5 pt-2 pb-2.5 flex flex-col gap-1.5">
        {loading ? (
          <button onClick={() => { abortRef.current?.abort(); setLoading(false); }}
            className="w-full text-center px-2.5 py-1 text-[10px] rounded border border-[#7a3a3a]/60 text-[#7a3a3a] hover:text-[#a05050] transition-colors duration-100 focus:outline-none">
            Stop
          </button>
        ) : (
          <button onClick={runAnalysis} disabled={!sequences || sequences.length === 0}
            className="w-full text-center px-2.5 py-1 text-[10px] rounded border border-[#2a2a33] bg-[#131316] text-[#dddde2] hover:border-[#3a3a46] hover:text-white transition-colors duration-100 focus:outline-none disabled:opacity-30 disabled:cursor-not-allowed">
            {obsResult ? 'Reanalyze' : 'Analyze'}
          </button>
        )}
        <div className="font-mono text-[7.5px] text-[#3d3d48] truncate">
          {inputData?.kind === 'impedance-series' ? inputData.label : '← connect dataset'}
        </div>
      </div>
    </div>
  );
}

// ---- Viz: Spider Plot (tornado — circuit parameters over time) ----

const ANNOT_COLORS = ['#56B4E9', '#009E73', '#E69F00', '#CC79A7', '#D55E00', '#7b68ee'];
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

function hexToRGBA(hex: string, alpha: number): string {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

// Viridis colormap (10 stops, t ∈ [0,1] → RGB)
function viridisRGB(t: number): [number, number, number] {
  const stops: Array<[number, number, number]> = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
    [181, 222, 43], [253, 231, 37],
  ];
  const idx = Math.min(t * (stops.length - 1), stops.length - 1 - 1e-9);
  const lo = Math.floor(idx);
  const f = idx - lo;
  const [r1, g1, b1] = stops[lo];
  const [r2, g2, b2] = stops[Math.min(lo + 1, stops.length - 1)];
  return [r1 + f * (r2 - r1), g1 + f * (g2 - g1), b1 + f * (b2 - b1)].map(Math.round) as [number, number, number];
}

type SpiderParamKey = 'Rsh' | 'Ra' | 'Ca' | 'Rb' | 'Cb';
const SPIDER_PARAM_KEYS: SpiderParamKey[] = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
const SPIDER_DISPLAY: Record<SpiderParamKey, { label: string; unit: string; factor: number }> = {
  Rsh: { label: 'Rsh', unit: 'kΩ', factor: 1 / 1000 },
  Ra:  { label: 'Ra',  unit: 'kΩ', factor: 1 / 1000 },
  Ca:  { label: 'Ca',  unit: 'µF', factor: 1e6 },
  Rb:  { label: 'Rb',  unit: 'kΩ', factor: 1 / 1000 },
  Cb:  { label: 'Cb',  unit: 'µF', factor: 1e6 },
};
const PENTA_ANGLES = SPIDER_PARAM_KEYS.map((_, i) => -Math.PI / 2 + i * (2 * Math.PI / 5));

function buildSpiderModels(
  output: Extract<NodeOutput, { kind: 'impedance-series' }>
): ModelSnapshot[] | null {
  const gt = output.groundTruth;
  if (!gt || !gt.Ra?.length || !gt.Rb?.length || !gt.Ca?.length || !gt.Cb?.length) return null;
  const T = gt.Ra.length;
  const maxModels = 100;
  const step = T > maxModels ? Math.ceil(T / maxModels) : 1;
  const freqs = output.sequences[0]?.frequencies ?? [];
  const freqRange: [number, number] = freqs.length ? [freqs[0], freqs[freqs.length - 1]] : [0.1, 100000];
  const models: ModelSnapshot[] = [];
  for (let i = 0; i < T; i += step) {
    const timeMin = output.sequences[i]?.time_min ?? i;
    models.push({
      id: `tp-${i}`,
      name: `t=${timeMin.toFixed(1)}min`,
      timestamp: timeMin,
      parameters: {
        Rsh: (gt.R2?.[i] ?? gt.TER?.[i] ?? 1) * 1000,
        Ra:  gt.Ra[i] * 1000,
        Ca:  gt.Ca[i] * 1e-6,
        Rb:  gt.Rb[i] * 1000,
        Cb:  gt.Cb[i] * 1e-6,
        frequency_range: freqRange,
      },
      resnorm: T > 1 ? i / (T - 1) : 0,
      color: '#6469a0',
      isVisible: true,
      opacity: 1.0,
      data: [],
    });
  }
  return models.length > 0 ? models : null;
}

function buildSpiderAxisConfig(models: ModelSnapshot[]): SpiderAxisConfig[] {
  const buckets: Record<SpiderParamKey, number[]> = { Rsh: [], Ra: [], Ca: [], Rb: [], Cb: [] };
  for (const m of models) {
    for (const k of SPIDER_PARAM_KEYS) buckets[k].push(m.parameters[k as keyof typeof m.parameters] as number);
  }
  return SPIDER_PARAM_KEYS.map(key => {
    const arr = buckets[key].filter(v => v > 0);
    const logVals = arr.map(v => Math.log10(v));
    return {
      key,
      name: `${SPIDER_DISPLAY[key].label} (${SPIDER_DISPLAY[key].unit})`,
      normRange: {
        min: Math.pow(10, Math.min(...logVals) - 0.35),
        max: Math.pow(10, Math.max(...logVals) + 0.35),
      },
    };
  });
}

// ---- 3D tornado canvas renderer ----
interface SpiderTornadoProps {
  models: ModelSnapshot[];
  axisConfig: SpiderAxisConfig[];
  showVertGrid: boolean;
  annotations: TimeAnnotation[];
  timeUnit: 'h' | 'm';
  highlightRange?: [number, number]; // normalized [0,1] zFrac range to brighten
  hoverFrac?: number; // time fraction [0,1] under cursor (0=tMin/early, 1=tMax/late)
}

function SpiderTornadoCanvas({ models, axisConfig, showVertGrid, annotations, timeUnit, highlightRange, hoverFrac }: SpiderTornadoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 520, h: 380 });
  const rotRef  = useRef({ az: -0.55, el: 0.40 });
  const [rot, setRot]   = useState(rotRef.current);
  const dragRef = useRef<{ sx: number; sy: number; az0: number; el0: number } | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([e]) => {
      if (e) setSize({ w: Math.max(240, e.contentRect.width), h: Math.max(180, e.contentRect.height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const draw = useCallback(() => {
    const visibleAnnotations = annotations.filter(a => !a.isHidden);
    const fmtTime = (min: number) =>
      timeUnit === 'h' ? `${(min / 60).toFixed(1)}h` : (min < 60 ? `${min.toFixed(0)}m` : `${(min / 60).toFixed(1)}h`);
    const canvas = canvasRef.current;
    if (!canvas || models.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = size.w, H = size.h;
    const dpr = window.devicePixelRatio ?? 1;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // ── Layout ──
    const CBAR_W   = 14;
    const CBAR_GAP = 80;
    const PLOT_W   = W - CBAR_W - CBAR_GAP - 4;
    const CX = PLOT_W * 0.50;  // true center of plot area
    const CY = H * 0.40;
    // SC: derived from safe viewport after reserving LABEL_MARGIN px for label text overhang
    // at every edge. This prevents axis labels clipping the canvas boundary at any rotation.
    const LABEL_R      = 1.18;  // label placement radius (fraction of SC)
    const LABEL_MARGIN = 32;    // px reserved at each edge beyond the label tip
    const safeSC = Math.min(
      (CX - LABEL_MARGIN) / LABEL_R,
      (PLOT_W - CX - LABEL_MARGIN) / LABEL_R,
      (CY - LABEL_MARGIN) / LABEL_R,
      (H - CY - LABEL_MARGIN) / LABEL_R,
    );
    const SC      = Math.max(60, Math.min(250, safeSC));
    const fSize   = Math.max(9, Math.min(11, Math.round(SC * 0.088)));
    const fSizeSm = Math.max(8, fSize - 1);
    const { az, el } = rotRef.current;

    const proj = (x: number, y: number, z: number) => {
      const cosAz = Math.cos(az), sinAz = Math.sin(az);
      const rx = x * cosAz - y * sinAz;
      const ry = x * sinAz + y * cosAz;
      const cosEl = Math.cos(el), sinEl = Math.sin(el);
      const ex = rx;
      const ey = ry * cosEl - z * sinEl;
      const ez = ry * sinEl + z * cosEl;
      const dp = 4.0 / (4.0 + ez);
      return { sx: CX + ex * dp * SC, sy: CY - ey * dp * SC, depth: ez };
    };

    const T = models.length;
    const tMin = models[0].timestamp;
    const tMax = models[T - 1].timestamp;
    const tRange = tMax - tMin || 1;

    // ── Compute pentagon rings ──
    // zFrac: color fraction (0=early/purple, 1=late/yellow); zPos: 3D height (1=bottom/early, 0=top/late)
    type Ring = { i: number; zFrac: number; verts: Array<{ sx: number; sy: number; depth: number }>; avgDepth: number };
    const rings: Ring[] = models.map((m, i) => {
      const zFrac = T > 1 ? i / (T - 1) : 0;
      const zPos  = 1 - zFrac; // flip: early at bottom (z=1), late at top (z=0)
      const verts = axisConfig.map((ax, j) => {
        const val = m.parameters[ax.key as keyof typeof m.parameters] as number;
        const logV = Math.log10(Math.max(val, ax.normRange.min * 1e-9));
        const logMin = Math.log10(ax.normRange.min);
        const logMax = Math.log10(ax.normRange.max);
        const r = Math.max(0, Math.min(1, (logV - logMin) / (logMax - logMin)));
        return proj(Math.cos(PENTA_ANGLES[j]) * r, Math.sin(PENTA_ANGLES[j]) * r, zPos);
      });
      return { i, zFrac, verts, avgDepth: verts.reduce((s, v) => s + v.depth, 0) / verts.length };
    });
    const sortedRings = [...rings].sort((a, b) => a.avgDepth - b.avgDepth);

    // ── Base grid ──
    ctx.save();
    // Bottom pentagon rings (more prominent — these anchor the space)
    for (const [r, opacity] of [[0.25, 0.6], [0.5, 0.65], [0.75, 0.7], [1.0, 0.8]] as Array<[number, number]>) {
      ctx.strokeStyle = `rgba(62, 64, 84, ${opacity})`;
      ctx.lineWidth = r === 1.0 ? 0.8 : 0.5;
      ctx.beginPath();
      for (let j = 0; j <= 5; j++) {
        const p = proj(Math.cos(PENTA_ANGLES[j % 5]) * r, Math.sin(PENTA_ANGLES[j % 5]) * r, 0);
        if (j === 0) { ctx.moveTo(p.sx, p.sy); } else { ctx.lineTo(p.sx, p.sy); }
      }
      ctx.stroke();
    }
    // Radial axis lines on base
    ctx.strokeStyle = 'rgba(58, 60, 78, 0.65)'; ctx.lineWidth = 0.55;
    for (let j = 0; j < 5; j++) {
      const c = proj(0, 0, 0);
      const v = proj(Math.cos(PENTA_ANGLES[j]), Math.sin(PENTA_ANGLES[j]), 0);
      ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(v.sx, v.sy); ctx.stroke();
    }

    // Vertical cage lines from ALL grid intersections (toggled)
    if (showVertGrid) {
      ctx.strokeStyle = 'rgba(38, 40, 55, 0.32)'; ctx.lineWidth = 0.4;
      for (const r of [0.25, 0.5, 0.75]) {
        for (let j = 0; j < 5; j++) {
          const bot = proj(Math.cos(PENTA_ANGLES[j]) * r, Math.sin(PENTA_ANGLES[j]) * r, 0);
          const top = proj(Math.cos(PENTA_ANGLES[j]) * r, Math.sin(PENTA_ANGLES[j]) * r, 1);
          ctx.beginPath(); ctx.moveTo(bot.sx, bot.sy); ctx.lineTo(top.sx, top.sy); ctx.stroke();
        }
      }
    }
    // Outer vertex vertical edges (always shown, slightly brighter than cage)
    ctx.strokeStyle = 'rgba(50, 52, 70, 0.50)'; ctx.lineWidth = 0.5;
    for (let j = 0; j < 5; j++) {
      const bot = proj(Math.cos(PENTA_ANGLES[j]), Math.sin(PENTA_ANGLES[j]), 0);
      const top = proj(Math.cos(PENTA_ANGLES[j]), Math.sin(PENTA_ANGLES[j]), 1);
      ctx.beginPath(); ctx.moveTo(bot.sx, bot.sy); ctx.lineTo(top.sx, top.sy); ctx.stroke();
    }
    // Central Z axis
    ctx.strokeStyle = 'rgba(65, 67, 88, 0.6)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(proj(0,0,0).sx, proj(0,0,0).sy); ctx.lineTo(proj(0,0,1).sx, proj(0,0,1).sy); ctx.stroke();
    // Dashed Z-plane rings at intermediate heights
    ctx.strokeStyle = 'rgba(38, 40, 56, 0.45)'; ctx.lineWidth = 0.4; ctx.setLineDash([2, 3]);
    for (let ti = 1; ti <= 4; ti++) {
      const gz = ti / 5;
      ctx.beginPath();
      for (let j = 0; j <= 5; j++) {
        const p = proj(Math.cos(PENTA_ANGLES[j % 5]), Math.sin(PENTA_ANGLES[j % 5]), gz);
        if (j === 0) { ctx.moveTo(p.sx, p.sy); } else { ctx.lineTo(p.sx, p.sy); }
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();

    // ── Annotation bands (draw before pentagons so they sit behind) ──
    for (const ann of visibleAnnotations) {
      const zFracS = Math.max(0, Math.min(1, (ann.tStart - tMin) / tRange));
      const zFracE = Math.max(0, Math.min(1, (ann.tEnd   - tMin) / tRange));
      const isPoint = ann.tStart === ann.tEnd;

      if (isPoint) {
        // Point annotation: filled translucent pentagon slice at this z
        const zPos = 1 - zFracS;
        ctx.beginPath();
        for (let j = 0; j <= 5; j++) {
          const p = proj(Math.cos(PENTA_ANGLES[j % 5]), Math.sin(PENTA_ANGLES[j % 5]), zPos);
          if (j === 0) { ctx.moveTo(p.sx, p.sy); } else { ctx.lineTo(p.sx, p.sy); }
        }
        ctx.closePath();
        ctx.fillStyle = hexToRGBA(ann.color, 0.28);
        ctx.fill();
        ctx.strokeStyle = hexToRGBA(ann.color, 0.75);
        ctx.lineWidth = 1.1;
        ctx.stroke();
      } else {
        if (zFracS >= zFracE) continue;
        // flip to match ring orientation: early=bottom(z=1), late=top(z=0)
        const zS = 1 - zFracE;
        const zE = 1 - zFracS;
        // Draw 5 quad faces between the outer pentagon at zS and zE
        for (let j = 0; j < 5; j++) {
          const a1 = PENTA_ANGLES[j], a2 = PENTA_ANGLES[(j + 1) % 5];
          const p1 = proj(Math.cos(a1), Math.sin(a1), zS);
          const p2 = proj(Math.cos(a2), Math.sin(a2), zS);
          const p3 = proj(Math.cos(a2), Math.sin(a2), zE);
          const p4 = proj(Math.cos(a1), Math.sin(a1), zE);
          ctx.beginPath();
          ctx.moveTo(p1.sx, p1.sy); ctx.lineTo(p2.sx, p2.sy);
          ctx.lineTo(p3.sx, p3.sy); ctx.lineTo(p4.sx, p4.sy);
          ctx.closePath();
          ctx.fillStyle = hexToRGBA(ann.color, 0.10);
          ctx.strokeStyle = hexToRGBA(ann.color, 0.30);
          ctx.lineWidth = 0.7;
          ctx.fill(); ctx.stroke();
        }
        // Outer pentagon ring at boundaries
        for (const gz of [zS, zE]) {
          ctx.strokeStyle = hexToRGBA(ann.color, 0.50);
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          for (let j = 0; j <= 5; j++) {
            const p = proj(Math.cos(PENTA_ANGLES[j % 5]), Math.sin(PENTA_ANGLES[j % 5]), gz);
            if (j === 0) { ctx.moveTo(p.sx, p.sy); } else { ctx.lineTo(p.sx, p.sy); }
          }
          ctx.stroke();
        }
      }
    }

    // ── Pentagon tornado (back → front) ──
    for (const ring of sortedRings) {
      const { verts, zFrac, i } = ring;
      const [r, g, b] = viridisRGB(zFrac);
      const inHL = !highlightRange || (zFrac >= highlightRange[0] && zFrac <= highlightRange[1]);
      const dim = inHL ? 1.0 : 0.12;
      if (i < T - 1) {
        const next = rings[i + 1];
        const [r2, g2, b2] = viridisRGB(next.zFrac);
        for (let j = 0; j < 5; j++) {
          const p1 = verts[j], p2 = next.verts[j];
          const gr = ctx.createLinearGradient(p1.sx, p1.sy, p2.sx, p2.sy);
          gr.addColorStop(0, `rgba(${r},${g},${b},${0.22 * dim})`);
          gr.addColorStop(1, `rgba(${r2},${g2},${b2},${0.22 * dim})`);
          ctx.beginPath(); ctx.moveTo(p1.sx, p1.sy); ctx.lineTo(p2.sx, p2.sy);
          ctx.strokeStyle = gr; ctx.lineWidth = 0.6; ctx.stroke();
        }
      }
      ctx.beginPath();
      verts.forEach((p, j) => (j === 0 ? ctx.moveTo(p.sx, p.sy) : ctx.lineTo(p.sx, p.sy)));
      ctx.closePath();
      ctx.fillStyle   = `rgba(${r},${g},${b},${0.38 * dim})`;
      ctx.strokeStyle = `rgba(${r},${g},${b},${0.72 * dim})`;
      ctx.lineWidth   = 0.55;
      ctx.fill(); ctx.stroke();
    }

    // ── Axis labels + scale ticks ──
    const pCenter = proj(0, 0, 0);
    ctx.textBaseline = 'middle';
    axisConfig.forEach((ax, j) => {
      const disp  = SPIDER_DISPLAY[ax.key as SpiderParamKey];
      const angle = PENTA_ANGLES[j];

      // Projected axis direction (determines left vs right text alignment)
      const pTip = proj(Math.cos(angle), Math.sin(angle), 0);
      const axDx = pTip.sx - pCenter.sx;
      const axDy = pTip.sy - pCenter.sy;
      const axLen = Math.sqrt(axDx * axDx + axDy * axDy) || 1;
      // Perpendicular unit vector (rotate 90° CW) — tick labels offset in this direction
      const perpX = axDy / axLen;
      const perpY = -axDx / axLen;
      const tAlign: CanvasTextAlign = Math.abs(axDx) < axLen * 0.2 ? 'center' : axDx > 0 ? 'left' : 'right';

      // Axis label at LABEL_R × tip
      const pLabel = proj(Math.cos(angle) * LABEL_R, Math.sin(angle) * LABEL_R, 0);
      ctx.font = `bold ${fSize}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillStyle = '#d8d8e8';
      ctx.fillText(`${disp.label} (${disp.unit})`, pLabel.sx, pLabel.sy);

      // Scale ticks at 0.5 and 1.0 only — enough to read the range without clutter
      const logMin = Math.log10(ax.normRange.min);
      const logMax = Math.log10(ax.normRange.max);
      [0.5, 1.0].forEach(frac => {
        const rawVal  = Math.pow(10, logMin + frac * (logMax - logMin));
        const dispVal = rawVal * disp.factor;
        const valStr  = dispVal >= 100 ? dispVal.toFixed(0) : dispVal >= 1 ? dispVal.toFixed(1) : dispVal.toFixed(2);
        const pTick   = proj(Math.cos(angle) * frac, Math.sin(angle) * frac, 0);
        ctx.strokeStyle = '#404050'; ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(pTick.sx + perpX * 3, pTick.sy + perpY * 3);
        ctx.lineTo(pTick.sx - perpX * 3, pTick.sy - perpY * 3);
        ctx.stroke();
        const side = axDx > 0 ? 1 : axDx < 0 ? -1 : 0;
        ctx.font = `${fSizeSm - 1}px monospace`;
        ctx.textAlign = tAlign;
        ctx.fillStyle = '#c0c0d2';
        ctx.fillText(valStr, pTick.sx + side * 5 + perpX * 2, pTick.sy + perpY * 2 - 4);
      });
    });

    // Z-axis time ticks — zPos flipped so early=bottom(z=1), late=top(z=0)
    ctx.font = `${fSizeSm}px monospace`; ctx.textAlign = 'right';
    const N_Z = Math.min(5, T);
    for (let ti = 0; ti <= N_Z; ti++) {
      const frac = ti / N_Z;
      const p = proj(0, 0, 1 - frac); // flip z
      const tm = tMin + frac * tRange;
      const ts = fmtTime(tm);
      ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(p.sx, p.sy, 2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#9090a8'; ctx.fillText(ts, p.sx - 4, p.sy);
    }

    // Hover crosshair ring in 3D space
    if (hoverFrac !== undefined) {
      const zPos = 1 - hoverFrac;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.65)';
      ctx.lineWidth = 1.0;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      for (let j = 0; j <= 5; j++) {
        const p = proj(Math.cos(PENTA_ANGLES[j % 5]), Math.sin(PENTA_ANGLES[j % 5]), zPos);
        if (j === 0) { ctx.moveTo(p.sx, p.sy); } else { ctx.lineTo(p.sx, p.sy); }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      const pc = proj(0, 0, zPos);
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.beginPath(); ctx.arc(pc.sx, pc.sy, 1.8, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    // ── Colorbar ──
    const CB_LEFT = W - CBAR_W - CBAR_GAP + 2;
    const CB_TOP  = H * 0.07, CB_H = H * 0.86;
    const cbGrad  = ctx.createLinearGradient(0, CB_TOP + CB_H, 0, CB_TOP);
    for (let ti = 0; ti <= 10; ti++) {
      const t = ti / 10;
      const [r, g, b] = viridisRGB(t);
      cbGrad.addColorStop(t, `rgb(${r},${g},${b})`);
    }
    ctx.fillStyle = cbGrad; ctx.fillRect(CB_LEFT, CB_TOP, CBAR_W, CB_H);
    ctx.strokeStyle = 'rgba(55, 57, 75, 0.5)'; ctx.lineWidth = 0.5;
    ctx.strokeRect(CB_LEFT, CB_TOP, CBAR_W, CB_H);

    // Annotation bracket marks + inline labels on colorbar
    ctx.save();
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    for (const ann of visibleAnnotations) {
      const zS = Math.max(0, Math.min(1, (ann.tStart - tMin) / tRange));
      const zE = Math.max(0, Math.min(1, (ann.tEnd   - tMin) / tRange));
      const isPoint = ann.tStart === ann.tEnd;

      if (isPoint) {
        const y = CB_TOP + CB_H - zS * CB_H;
        // Diamond tick on the bar
        ctx.strokeStyle = hexToRGBA(ann.color, 0.90);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(CB_LEFT - 4, y); ctx.lineTo(CB_LEFT + CBAR_W + 4, y); ctx.stroke();
        ctx.font = `${fSizeSm - 1}px monospace`;
        ctx.fillStyle = ann.color;
        const maxChars = Math.floor((CBAR_GAP - 10) / (fSizeSm * 0.6));
        const label = ann.label.length > maxChars ? ann.label.slice(0, maxChars - 1) + '…' : ann.label;
        ctx.fillText(label, CB_LEFT + CBAR_W + 8, y);
      } else {
        if (zS >= zE) continue;
        const yTop = CB_TOP + CB_H - zE * CB_H;
        const yBot = CB_TOP + CB_H - zS * CB_H;
        const yMid = (yTop + yBot) / 2;
        ctx.fillStyle = hexToRGBA(ann.color, 0.22);
        ctx.fillRect(CB_LEFT - 2, yTop, CBAR_W + 4, yBot - yTop);
        ctx.strokeStyle = hexToRGBA(ann.color, 0.75);
        ctx.lineWidth = 1.0;
        ctx.beginPath(); ctx.moveTo(CB_LEFT - 3, yTop); ctx.lineTo(CB_LEFT + CBAR_W + 6, yTop); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(CB_LEFT - 3, yBot); ctx.lineTo(CB_LEFT + CBAR_W + 6, yBot); ctx.stroke();
        if (yBot - yTop > 8) {
          ctx.font = `${fSizeSm - 1}px monospace`;
          ctx.fillStyle = ann.color;
          const maxChars = Math.floor((CBAR_GAP - 10) / (fSizeSm * 0.6));
          const label = ann.label.length > maxChars ? ann.label.slice(0, maxChars - 1) + '…' : ann.label;
          ctx.fillText(label, CB_LEFT + CBAR_W + 8, yMid);
        }
      }
    }
    ctx.restore();

    // Colorbar tick labels
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.font = `${fSizeSm}px monospace`;
    for (let ti = 0; ti <= 5; ti++) {
      const t = ti / 5;
      const cy2 = CB_TOP + CB_H - t * CB_H;
      const tm  = tMin + t * tRange;
      const ts  = fmtTime(tm);
      ctx.fillStyle = '#404055'; ctx.fillRect(CB_LEFT + CBAR_W, cy2 - 0.4, 3, 0.8);
      ctx.fillStyle = '#b0b0c8'; ctx.fillText(ts, CB_LEFT + CBAR_W + 6, cy2);
    }
    ctx.save();
    ctx.translate(CB_LEFT + CBAR_W / 2, CB_TOP - 10);
    ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#606070'; ctx.font = `${fSizeSm}px monospace`;
    ctx.fillText('time', 0, 0);
    ctx.restore();

    // Hover cursor line on colorbar
    if (hoverFrac !== undefined) {
      const cy2 = CB_TOP + CB_H - hoverFrac * CB_H;
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1.0;
      ctx.beginPath(); ctx.moveTo(CB_LEFT - 5, cy2); ctx.lineTo(CB_LEFT + CBAR_W + 5, cy2); ctx.stroke();
      ctx.restore();
    }

  }, [models, axisConfig, showVertGrid, annotations, timeUnit, highlightRange, hoverFrac, rot, size]);

  useEffect(() => { draw(); }, [draw]);

  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { sx: e.clientX, sy: e.clientY, az0: rotRef.current.az, el0: rotRef.current.el };
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const { sx, sy, az0, el0 } = dragRef.current;
    const az = az0 + (e.clientX - sx) * 0.006;
    const el = Math.max(-1.4, Math.min(1.4, el0 - (e.clientY - sy) * 0.006));
    rotRef.current = { az, el };
    setRot({ az, el });
  };
  const onMouseUp = () => { dragRef.current = null; };

  return (
    <div ref={containerRef} className="w-full h-full select-none"
      style={{ cursor: dragRef.current ? 'grabbing' : 'grab' }}
      onMouseDown={onMouseDown} onMouseMove={onMouseMove}
      onMouseUp={onMouseUp} onMouseLeave={onMouseUp}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }}/>
    </div>
  );
}

// ---- VizSpiderplotBody ----

function VizSpiderplotBody({ node, inputData, onConfigChange }: NodeBodyProps) {
  const rawCfg = node.config as VizSpiderplotCfg & { _placeholder?: boolean };
  const cfg: VizSpiderplotCfg = {
    kind: 'viz-spiderplot',
    showVertGrid: rawCfg.showVertGrid ?? false,
    annotations: rawCfg.annotations ?? [],
    timeUnit: rawCfg.timeUnit ?? 'h',
  };
  const output = inputData?.kind === 'impedance-series' ? inputData : null;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [inlineLabel, setInlineLabel] = useState('');
  const [hoverFrac, setHoverFrac] = useState<number | undefined>(undefined);
  const [dragPreview, setDragPreview] = useState<[number, number] | undefined>(undefined);
  const autoLabelRef = useRef<string | null>(null);

  const { models, axisConfig } = useMemo(() => {
    if (!output) return { models: null, axisConfig: undefined };
    const m = buildSpiderModels(output);
    if (!m) return { models: null, axisConfig: undefined };
    return { models: m, axisConfig: buildSpiderAxisConfig(m) };
  }, [output]);

  const tMin = models?.[0]?.timestamp ?? 0;
  const tMax = models?.[models.length - 1]?.timestamp ?? 1;
  const tRange = tMax - tMin || 1;

  useEffect(() => {
    if (!output || !models || cfg.annotations.length > 0) return;
    if (autoLabelRef.current === output.label) return;
    autoLabelRef.current = output.label;
    const autoLabel = output.label.split(' · ')[0].replace(/_/g, ' ');
    onConfigChange({ ...cfg, annotations: [{ id: uid(), label: autoLabel, tStart: tMin, tEnd: tMax, color: ANNOT_COLORS[0] }] });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [output?.label, models]);

  const selectedAnnotation = cfg.annotations.find(a => a.id === selectedId) ?? null;

  // ROI analysis: per-param delta inside vs outside selected annotation time range
  const roiAnalysis = useMemo(() => {
    if (!selectedAnnotation || selectedAnnotation.tStart === selectedAnnotation.tEnd) return null;
    if (!models || models.length === 0) return null;
    const { tStart, tEnd } = selectedAnnotation;
    const inRange  = models.filter(m => m.timestamp >= tStart && m.timestamp <= tEnd);
    const outRange = models.filter(m => m.timestamp < tStart  || m.timestamp  > tEnd);
    if (inRange.length === 0 || outRange.length === 0) return null;
    const median = (vals: number[]) => {
      const s = [...vals].sort((a, b) => a - b);
      const n = s.length;
      return n % 2 === 1 ? s[Math.floor(n / 2)] : (s[n / 2 - 1] + s[n / 2]) / 2;
    };
    return SPIDER_PARAM_KEYS.map(key => {
      const disp    = SPIDER_DISPLAY[key];
      const inVals  = inRange.map(m  => (m.parameters[key as keyof typeof m.parameters]  as number) * disp.factor);
      const outVals = outRange.map(m => (m.parameters[key as keyof typeof m.parameters] as number) * disp.factor);
      const inMed   = median(inVals);
      const outMed  = median(outVals);
      const deltaPct = outMed !== 0 ? ((inMed - outMed) / outMed) * 100 : 0;
      return { key, label: disp.label, unit: disp.unit, deltaPct };
    });
  }, [selectedAnnotation, models]);

  const fmtT = (t: number) =>
    cfg.timeUnit === 'h' ? `${(t / 60).toFixed(2)}h` : (t < 60 ? `${t.toFixed(0)}m` : `${(t / 60).toFixed(1)}h`);

  const deleteAnnotation = (id: string) => {
    onConfigChange({ ...cfg, annotations: cfg.annotations.filter(a => a.id !== id) });
    if (selectedId === id) setSelectedId(null);
  };
  const updateLabel = (id: string, label: string) =>
    onConfigChange({ ...cfg, annotations: cfg.annotations.map(a => a.id === id ? { ...a, label } : a) });
  const updateColor = (id: string, color: string) =>
    onConfigChange({ ...cfg, annotations: cfg.annotations.map(a => a.id === id ? { ...a, color } : a) });

  // Normalized [0,1] zFrac range for ring highlight
  const highlightRange = useMemo((): [number, number] | undefined => {
    if (!selectedAnnotation) return undefined;
    const zS = Math.max(0, Math.min(1, (selectedAnnotation.tStart - tMin) / tRange));
    const zE = Math.max(0, Math.min(1, (selectedAnnotation.tEnd   - tMin) / tRange));
    if (selectedAnnotation.tStart === selectedAnnotation.tEnd) {
      // point: highlight a small window so nearby rings dim correctly
      return [Math.max(0, zS - 0.04), Math.min(1, zS + 0.04)];
    }
    return [zS, zE];
  }, [selectedAnnotation, tMin, tRange]);

  if (!output) {
    return <div className="flex items-center justify-center h-full"><span className="font-mono text-[9px] text-[#2a2a33]">connect data node (out-viz port)</span></div>;
  }
  if (!models || !axisConfig) {
    return <div className="flex items-center justify-center h-full px-4 text-center"><span className="font-mono text-[9px] text-[#454549]">no raw circuit params in ground truth</span></div>;
  }

  // dragPreview takes priority over selectedAnnotation highlight while dragging
  const effectiveHighlight = dragPreview ?? highlightRange;

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">

      {/* Control bar */}
      <div className="flex-shrink-0 flex items-center gap-2 px-3 py-1.5 border-b border-[#111116]">
        <div className="flex items-center rounded border border-[#1a1a22] overflow-hidden">
          {(['h', 'm'] as const).map(u => (
            <button key={u} onClick={() => onConfigChange({ ...cfg, timeUnit: u })}
              className={`font-mono text-[7.5px] px-1.5 py-0.5 transition-colors duration-100 focus:outline-none ${cfg.timeUnit === u ? 'bg-[#1e1e2a] text-[#7a7a8a]' : 'text-[#2e2e3a]'}`}>
              {u}
            </button>
          ))}
        </div>
        <span className="font-mono text-[7px] text-[#1e1e28] ml-auto tabular-nums">{models.length}pt · drag to rotate</span>
      </div>

      {/* Canvas + overlays — annotation panel is absolute so it never resizes the canvas */}
      <div className="flex-1 min-h-0 relative">
        <SpiderTornadoCanvas
          models={models}
          axisConfig={axisConfig}
          showVertGrid={cfg.showVertGrid}
          annotations={cfg.annotations}
          timeUnit={cfg.timeUnit}
          highlightRange={effectiveHighlight}
          hoverFrac={hoverFrac}
        />
        <TimelineOverlay
          tMin={tMin} tMax={tMax}
          annotations={cfg.annotations}
          selectedId={selectedId}
          onSelect={id => {
            setSelectedId(id);
            const a = cfg.annotations.find(x => x.id === id);
            if (a) setInlineLabel(a.label);
          }}
          onAdd={(tStart, tEnd) => {
            const newAnnot: TimeAnnotation = {
              id: uid(), label: 'New region', tStart, tEnd,
              color: ANNOT_COLORS[cfg.annotations.length % ANNOT_COLORS.length],
            };
            onConfigChange({ ...cfg, annotations: [...cfg.annotations, newAnnot] });
            setSelectedId(newAnnot.id);
            setInlineLabel('New region');
          }}
          onHover={f => setHoverFrac(f ?? undefined)}
          onDragPreview={range => setDragPreview(range ?? undefined)}
        />

        {/* Selected annotation panel — floats over canvas bottom, never causes layout reflow */}
        {selectedAnnotation && (
          <div className="absolute bottom-0 left-0 right-0 z-10 bg-[#07070b]/95 border-t border-[#111116] px-3 pt-1.5 pb-2">
            <div className="flex items-center gap-1.5 mb-1">
              <input
                value={inlineLabel}
                onMouseDown={e => e.stopPropagation()}
                onChange={e => setInlineLabel(e.target.value)}
                onBlur={() => updateLabel(selectedAnnotation.id, inlineLabel)}
                onKeyDown={e => { if (e.key === 'Enter') updateLabel(selectedAnnotation.id, inlineLabel); }}
                className="flex-1 min-w-0 font-mono text-[10px] bg-transparent border-b border-[#2a2a33] text-[#c5c5c8] focus:outline-none focus:border-[#454549] pb-0.5"
                placeholder="label"
              />
              <span className="font-mono text-[8px] text-[#3a3a4a] tabular-nums flex-shrink-0">
                {selectedAnnotation.tStart === selectedAnnotation.tEnd
                  ? fmtT(selectedAnnotation.tStart)
                  : `${fmtT(selectedAnnotation.tStart)}–${fmtT(selectedAnnotation.tEnd)}`}
              </span>
              <div className="flex items-center gap-1 flex-shrink-0">
                {ANNOT_COLORS.map(c => (
                  <button key={c} onMouseDown={e => e.stopPropagation()}
                    onClick={() => updateColor(selectedAnnotation.id, c)}
                    className="w-2.5 h-2.5 rounded-full focus:outline-none flex-shrink-0"
                    style={{ background: c, boxShadow: selectedAnnotation.color === c ? `0 0 0 2px ${c}44, 0 0 0 1px ${c}` : 'none' }}/>
                ))}
              </div>
              <button
                onClick={() => deleteAnnotation(selectedAnnotation.id)}
                className="text-[11px] leading-none text-[#3a3a48] hover:text-[#7a3a3a] transition-colors focus:outline-none flex-shrink-0"
                title="Delete">×</button>
            </div>
            {roiAnalysis && (
              <div className="grid gap-x-1.5 gap-y-px" style={{ gridTemplateColumns: '1.8rem 1fr 2.8rem' }}>
                {roiAnalysis.map(r => {
                  const abs   = Math.min(100, Math.abs(r.deltaPct));
                  const color = abs < 5 ? '#3a3a4a' : r.deltaPct > 0 ? '#4ade80' : '#f97316';
                  const pct   = `${r.deltaPct > 0 ? '+' : ''}${r.deltaPct.toFixed(0)}%`;
                  return (
                    <React.Fragment key={r.key}>
                      <span className="font-mono text-[9px] text-[#5a5a66] self-center">{r.label}</span>
                      <div className="h-px self-center bg-[#1a1a22] rounded overflow-hidden">
                        <div style={{ width: `${abs}%`, height: '100%', background: color, opacity: 0.6,
                          marginLeft: r.deltaPct < 0 ? `${100 - abs}%` : '0' }}/>
                      </div>
                      <span className="font-mono text-[9px] tabular-nums text-right" style={{ color }}>{pct}</span>
                    </React.Fragment>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TimelineOverlay({ tMin, tMax, annotations, selectedId, onSelect, onAdd, onHover, onDragPreview }: {
  tMin: number; tMax: number;
  annotations: TimeAnnotation[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onAdd: (tStart: number, tEnd: number) => void;
  onHover: (frac: number | null) => void;
  onDragPreview: (range: [number, number] | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startFrac: number } | null>(null);
  const [dragFrac, setDragFrac] = useState<{ lo: number; hi: number } | null>(null);
  const tRange = tMax - tMin || 1;

  const fracToTime = (frac: number) => tMin + (1 - frac) * tRange;
  // CSS frac: 0=top=tMax, 1=bottom=tMin → time fraction: 0=tMin, 1=tMax
  const cssToTimeFrac = (cssFrac: number) => 1 - cssFrac;

  const onMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const frac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    dragRef.current = { startFrac: frac };
    setDragFrac({ lo: frac, hi: frac });

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current || !ref.current) return;
      const r = ref.current.getBoundingClientRect();
      const cur = Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height));
      const lo = Math.min(dragRef.current.startFrac, cur);
      const hi = Math.max(dragRef.current.startFrac, cur);
      setDragFrac({ lo, hi });
      // live-preview in canvas: convert CSS fracs → time fracs for highlightRange
      if (hi - lo > 0.01) onDragPreview([cssToTimeFrac(hi), cssToTimeFrac(lo)]);
    };
    const onUp = (ev: MouseEvent) => {
      if (dragRef.current && ref.current) {
        const r = ref.current.getBoundingClientRect();
        const cur = Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height));
        const lo = Math.min(dragRef.current.startFrac, cur);
        const hi = Math.max(dragRef.current.startFrac, cur);
        if (hi - lo > 0.02) {
          onAdd(fracToTime(hi), fracToTime(lo));
        }
      }
      onDragPreview(null);
      setDragFrac(null);
      dragRef.current = null;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div className="absolute top-0 bottom-0 right-0 pointer-events-none" style={{ width: 94 }}>
      {/* Interactive zone matches canvas colorbar: top 7%, height 86% */}
      <div
        ref={ref}
        className="absolute cursor-crosshair"
        style={{ top: '7%', height: '86%', left: 0, right: 0, pointerEvents: 'auto' }}
        onMouseDown={onMouseDown}
        onDoubleClick={e => {
          e.stopPropagation();
          const rect = ref.current?.getBoundingClientRect();
          if (!rect) return;
          const frac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          const t = fracToTime(frac);
          onAdd(t, t);
        }}
        onMouseMove={e => {
          const rect = ref.current?.getBoundingClientRect();
          if (!rect) return;
          const cssFrac = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
          onHover(cssToTimeFrac(cssFrac));
        }}
        onMouseLeave={() => onHover(null)}
      >
        {/* Existing annotation regions */}
        {annotations.filter(a => !a.isHidden).map(ann => {
          const zS  = Math.max(0, Math.min(1, (ann.tStart - tMin) / tRange));
          const zE  = Math.max(0, Math.min(1, (ann.tEnd   - tMin) / tRange));
          const isPoint = ann.tStart === ann.tEnd;
          const isSel = ann.id === selectedId;

          if (isPoint) {
            return (
              <div
                key={ann.id}
                data-annot-id={ann.id}
                style={{
                  position: 'absolute',
                  top: `${(1 - zS) * 100}%`,
                  height: 2,
                  transform: 'translateY(-50%)',
                  left: 0, right: 0,
                  background: isSel ? ann.color : `${ann.color}99`,
                  cursor: 'pointer', pointerEvents: 'auto',
                }}
                onClick={e => { e.stopPropagation(); onSelect(ann.id === selectedId ? null : ann.id); }}
              />
            );
          }

          const top = `${(1 - zE) * 100}%`;
          const height = `${(zE - zS) * 100}%`;
          return (
            <div
              key={ann.id}
              data-annot-id={ann.id}
              style={{
                position: 'absolute', top, height, left: 16, right: 0,
                background: `${ann.color}${isSel ? '28' : '12'}`,
                borderLeft: `1.5px solid ${ann.color}${isSel ? 'dd' : '55'}`,
                cursor: 'pointer', pointerEvents: 'auto',
              }}
              onClick={e => { e.stopPropagation(); onSelect(ann.id === selectedId ? null : ann.id); }}
            />
          );
        })}
        {/* Drag preview */}
        {dragFrac && dragFrac.hi - dragFrac.lo > 0.01 && (
          <div style={{
            position: 'absolute',
            top: `${dragFrac.lo * 100}%`,
            height: `${(dragFrac.hi - dragFrac.lo) * 100}%`,
            left: 16, right: 0, pointerEvents: 'none',
            background: 'rgba(100,105,160,0.18)',
            border: '1px dashed rgba(100,105,160,0.55)',
          }}/>
        )}
      </div>
    </div>
  );
}


// ---- MechanismBody ----
function MechanismBody({ node, inputData }: NodeBodyProps) {
  const result = inputData?.kind === 'ml-result' ? inputData.result : null;
  const mechanism = result?.mechanism ?? null;
  const posteriorSummary = result?.posteriorSummary ?? null;

  const MECH_COLORS = ['#56B4E9', '#E69F00', '#009E73'] as const;

  const fmtV = (v: number) => v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
  const POSTERIOR_PARAMS: Array<{ key: string; label: string; color: string; unit: string }> = [
    { key: 'TER', label: 'TER', color: '#009E73', unit: 'kΩ' },
    { key: 'Rsh', label: 'Rsh', color: '#6469a0', unit: 'kΩ' },
    { key: 'TEC', label: 'TEC', color: '#CC79A7', unit: 'µF' },
    { key: 'Ra',  label: 'Ra',  color: '#56B4E9', unit: 'kΩ' },
    { key: 'Rb',  label: 'Rb',  color: '#E69F00', unit: 'kΩ' },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0 overflow-y-auto px-2 pt-1.5 pb-1 flex flex-col">

        {/* ─── HYPOTHESES ─── */}
        <div className={SECTION_LABEL}>Hypotheses</div>
        {mechanism ? (
          <div className="mt-1 flex flex-col gap-2">
            {mechanism.hypotheses.map((h, i) => {
              const color = MECH_COLORS[i % MECH_COLORS.length];
              const barW = Math.round(h.probability * 100);
              return (
                <div key={h.name} className="flex flex-col gap-0.5">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-[8px] tabular-nums font-semibold w-7 flex-shrink-0" style={{ color }}>
                      {barW}%
                    </span>
                    <span className="font-mono text-[8px] text-[#9a9aa2] truncate">{h.label}</span>
                  </div>
                  <div className="h-1.5 bg-[#1e1e24] rounded overflow-hidden">
                    <div style={{ width: `${barW}%`, height: '100%', background: color, opacity: 0.65 }}/>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <span className="mt-1 font-mono text-[8px] text-[#2a2a33]">{result ? 'run inference to populate' : 'connect a Process node'}</span>
        )}

        {mechanism && (
          <>
            <div className={NODE_DIVIDER}/>
            <div className={SECTION_LABEL}>Evidence windows</div>
            <div className="mt-1 flex flex-col gap-0.5">
              {mechanism.evidence_time.length > 0 ? mechanism.evidence_time.map((w, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <span className="font-mono text-[7px] text-[#454549] w-20 flex-shrink-0">{w.t_start.toFixed(0)}–{w.t_end.toFixed(0)} min</span>
                  <div className="flex-1 h-1 bg-[#1e1e24] rounded overflow-hidden">
                    <div style={{ width: `${Math.min(w.contribution * 100, 100)}%`, height: '100%', background: '#c4a040', opacity: 0.55 }}/>
                  </div>
                </div>
              )) : <span className="font-mono text-[7.5px] text-[#2a2a33]">—</span>}
            </div>
          </>
        )}

        {(mechanism?.established.length ?? 0) > 0 || (mechanism?.ambiguous.length ?? 0) > 0 ? (
          <>
            <div className={NODE_DIVIDER}/>
            <div className={SECTION_LABEL}>Parameter status</div>
            <div className="mt-1 flex flex-col gap-1">
              {mechanism!.established.length > 0 && (
                <div className="flex flex-wrap gap-1 items-center">
                  <span className="font-mono text-[6.5px] text-[#4ade80]/60 w-14 flex-shrink-0">established</span>
                  {mechanism!.established.map(p => (
                    <span key={p} className="font-mono text-[6.5px] text-[#4ade80] bg-[#4ade80]/10 rounded px-1 py-0.5">{p}</span>
                  ))}
                </div>
              )}
              {mechanism!.ambiguous.length > 0 && (
                <div className="flex flex-wrap gap-1 items-center">
                  <span className="font-mono text-[6.5px] text-[#f97316]/60 w-14 flex-shrink-0">ambiguous</span>
                  {mechanism!.ambiguous.map(p => (
                    <span key={p} className="font-mono text-[6.5px] text-[#f97316] bg-[#f97316]/10 rounded px-1 py-0.5">{p}?</span>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : null}

        {posteriorSummary && (
          <>
            <div className={NODE_DIVIDER}/>
            <div className={SECTION_LABEL}>Final posterior</div>
            <div className="mt-1 flex flex-col gap-0.5">
              {POSTERIOR_PARAMS.map(({ key, label, color, unit }) => {
                const ps = posteriorSummary[key];
                if (!ps) return null;
                const ident = ps.identifiability;
                const identColor = ident > 0.7 ? '#4ade80' : ident > 0.4 ? '#c4a040' : '#f97316';
                return (
                  <div key={key} className="flex items-baseline gap-1">
                    <span className="font-mono text-[7.5px] w-6 flex-shrink-0" style={{ color }}>{label}</span>
                    <span className="font-mono text-[8px] text-[#dddde2] tabular-nums flex-1">{fmtV(ps.median)}</span>
                    <span className="font-mono text-[6.5px] text-[#2a2a33]">[{fmtV(ps.q05)}–{fmtV(ps.q95)}]</span>
                    <span className="font-mono text-[6.5px] flex-shrink-0" style={{ color: identColor }}>{Math.round(ident * 100)}%</span>
                    <span className="font-mono text-[6.5px] text-[#454549] flex-shrink-0">{unit}</span>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {node.errorMsg && <p className="text-[8px] text-[#7a3a3a] mt-1.5 font-mono">{node.errorMsg}</p>}
      </div>
    </div>
  );
}
