Quick Wins You Can Land Today (no redesign)
1) Kill the cascading effects with a single reducer + debounced intent

Replace the intertwined useEffects with one reducer that owns the “reference model” lifecycle and is only triggered by a debounced parameter update.

// ReferenceModel.tsx (extract from CircuitSimulator)
import { useReducer, useMemo, useEffect, useRef } from "react";

type State =
  | { status: "idle"; snapshot?: ModelSnapshot }
  | { status: "computing"; snapshot?: ModelSnapshot }
  | { status: "ready"; snapshot: ModelSnapshot };

type Action =
  | { type: "PARAMS_CHANGED"; params: ParamsKey }
  | { type: "COMPUTED"; snapshot: ModelSnapshot }
  | { type: "RESET" };

function refModelReducer(state: State, action: Action): State {
  switch (action.type) {
    case "PARAMS_CHANGED":
      return { status: "computing", snapshot: state.snapshot };
    case "COMPUTED":
      return { status: "ready", snapshot: action.snapshot };
    case "RESET":
      return { status: "idle" };
    default:
      return state;
  }
}

// stable key to avoid stale closures
const paramsKey = (p: CircuitParameters) =>
  `${p.Rsh}|${p.Ra}|${p.Ca}|${p.Rb}|${p.Cb}|${p.minFreq}|${p.maxFreq}|${p.numPoints}`;

export function useReferenceModel(parameters: CircuitParameters, createReferenceModel: (p: CircuitParameters) => Promise<ModelSnapshot>) {
  const [state, dispatch] = useReducer(refModelReducer, { status: "idle" });
  const key = useMemo(() => paramsKey(parameters), [parameters]); // stable
  const inFlight = useRef<AbortController | null>(null);

  // Debounce param changes (no re-renders on every keystroke)
  useEffect(() => {
    const handle = setTimeout(() => dispatch({ type: "PARAMS_CHANGED", params: key }), 300);
    return () => clearTimeout(handle);
  }, [key]);

  // Single place that performs the expensive work
  useEffect(() => {
    if (state.status !== "computing") return;

    // cancel prior compute
    inFlight.current?.abort();
    const ac = new AbortController();
    inFlight.current = ac;

    createReferenceModel(parameters).then((snapshot) => {
      if (!ac.signal.aborted) dispatch({ type: "COMPUTED", snapshot });
    }).catch(() => {/* swallow or surface error UX */});

    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.status, key]); // depend on key, not on parameters object fan-out

  return state;
}


How to use: call useReferenceModel(parameters, createReferenceModel) inside your (now smaller) simulator; render from state.snapshot. This removes the “set X → triggers Y → recompute” loop entirely.

2) Fix worker stalls with sane concurrency, chunking, and heartbeats

Drop this into workerManager.ts. It auto-sizes chunking and keeps workers alive via heartbeats instead of huge timeouts.

const CPU = Math.max(1, Math.min(navigator.hardwareConcurrency ?? 4, 8));

function calcChunkSize(totalPoints: number) {
  // Aim ~40–80 chunks total; cap per chunk
  const targetChunks = Math.min(Math.max(Math.floor(totalPoints / 5000), 40), 80);
  const raw = Math.ceil(totalPoints / targetChunks);
  return Math.min(20_000, Math.max(1_000, raw));
}

export async function distributeWork(points: Float64Array, payload: Payload, signal?: AbortSignal) {
  const chunkSize = calcChunkSize(points.length);
  const chunks: Array<{start:number; end:number}> = [];
  for (let i = 0; i < points.length; i += chunkSize) chunks.push({ start: i, end: Math.min(i + chunkSize, points.length) });

  const pool = makeWorkerPool(CPU);
  const results = new Float64Array(points.length * payload.outStride);

  // heartbeat instead of huge timeouts
  const HEARTBEAT_MS = 15_000;

  let nextChunk = 0;
  const runners = Array.from({ length: CPU }).map(async () => {
    while (nextChunk < chunks.length && !signal?.aborted) {
      const { start, end } = chunks[nextChunk++];
      const hb = setInterval(() => pool.post({ type: "ping" }), HEARTBEAT_MS);
      try {
        // Transfer views for speed
        const slice = points.subarray(start, end);
        const res: Float64Array = await pool.compute({ type: "compute", payload, points: slice }, [slice.buffer]);
        results.set(res, start * payload.outStride);
      } finally {
        clearInterval(hb);
      }
      // Yield back to UI
      await Promise.resolve();
    }
  });

  await Promise.all(runners);
  return results;
}


Key changes

Concurrency = min(hardwareConcurrency, 8).

Chunk sizing targets 40–80 chunks total (prevents micro-chunks & massive single chunks).

Heartbeats replace 3–5 minute timeouts.

Always yield to UI between chunks.

3) Move the heavy math off the main thread and cache ω

Small but valuable micro-opt.

// impedance.ts
const omegaCache = new Map<number, number>();
export const omega = (f: number) => {
  let v = omegaCache.get(f);
  if (v === undefined) { v = 2 * Math.PI * f; omegaCache.set(f, v); }
  return v;
};

// use typed arrays for complex math across the sweep


If you can compile a tiny WASM kernel later, swap the inner loop behind the same interface—no UI code changes needed.

4) Cap canvas work to “only when dirty”, 60fps max, with viewport culling

Your loop is trying to be ultra-smooth at ~240fps. Make it deterministic and cheap.

class OptimizedSpiderRenderer {
  private animationFrameId: number | null = null;
  private needsRender = false;

  markDirty() { this.needsRender = true; if (!this.animationFrameId) this.queue(); }

  private queue() {
    this.animationFrameId = requestAnimationFrame(() => {
      if (this.needsRender) {
        this.needsRender = false;
        this.performRenderWithCulling();
      }
      this.animationFrameId = null;
      if (this.needsRender) this.queue(); // coalesce bursts into 1 frame
    });
  }

  private performRenderWithCulling() {
    const visible = this.models.filter(m => intersects(m.bounds, this.viewport));
    // draw only visible; reuse Path2D per model id; avoid context switches
  }
}


Bonus: if you already use workers, try OffscreenCanvas for background rasterization on Chrome; feature-detect gracefully.

5) Frequency generation that doesn’t churn

Cache the logspace by a stable key and use typed arrays.

const freqCache = new Map<string, Float64Array>();
export function makeLogspace(minF: number, maxF: number, n: number) {
  const key = `${minF}|${maxF}|${n}`;
  const cached = freqCache.get(key);
  if (cached) return cached;
  const out = new Float64Array(n);
  const logMin = Math.log10(minF);
  const logMax = Math.log10(maxF);
  const step = (logMax - logMin) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = Math.pow(10, logMin + i * step);
  freqCache.set(key, out);
  return out;
}

6) Memory pressure hygiene (common offenders)

Workers: always terminate() on unmount and when pool shrinks; clear maps/queues.

Event listeners: prefer useEffect with return cleanup; avoid recreating handlers in deps—wrap with useEvent or refs.

Canvas: reuse one CanvasRenderingContext2D; do not create per frame.

Large arrays: store only the current page/selection; persist full arrays in the worker and request slices.

Minimal worker pool cleanup:

function makeWorkerPool(n: number) {
  const workers = Array.from({ length: n }, () => new Worker(new URL("./worker.js", import.meta.url)));
  const free: Worker[] = [...workers];

  function destroy() {
    workers.forEach(w => w.terminate());
    free.length = 0;
  }

  // ...
  return { compute, post, destroy };
}


Call pool.destroy() in the component’s cleanup.

Near-Term Refactor (2–4 days)

Split the simulator exactly as you proposed (Params | ComputeEngine | Viz | ReferenceModelProvider).

Lift all compute orchestration (chunking, pooling, progress events, cancellation) into ComputationEngine context; UI subscribes to derived state only.

Adopt a single source of truth with useReducer for simulator state + derived selectors (memoized).

Introduce a ComputeStream interface now (even without SharedArrayBuffer), so streaming & WASM drop in later without UI churn.

export interface ComputeStream<T> {
  start(params: CircuitParameters): void;
  onPartial(cb: (partial: T) => void): () => void;
  onProgress(cb: (p: number) => void): () => void;
  cancel(): void;
}

Validation Checklist (turn this into CI)

Parameter tweak → first paint of new curve < 100ms for small sweeps (devtools “Interactions”).

Large grid (≥ 200k pts) → main thread long tasks < 50ms each; no frame drops below 55fps while panning.

Worker tasks finish without timeouts; no stalled promises after abort.

Heap snapshot before/after repeated runs: < 10% growth and drops back after GC.

Add your monitor wrapper (nice!) around:

parameter → compute start/end

worker chunk start/end

render start/end

…and log to a ring buffer exposed by a hidden /perf route for field data.

“Paste-This” Replacements for Your Hotspots
A) The two cascading effects → single hook usage
// CircuitSimulator.tsx (now smaller)
const refModel = useReferenceModel(parameters, createReferenceModelInWorker);
// render uses refModel.snapshot when status === "ready"

B) Worker timeout block → heartbeat + abortable compute

Remove the 3–5 min setTimeout; adopt the heartbeat pattern in the pool (above). Also wire AbortController from UI (route change, param change) to cancel inflight.

C) SpiderPlot loop → on-demand render

Replace the perpetual loop with markDirty() on:

new data in

viewport change

style/theme change
Everything else does nothing.

Risk Notes (so nothing bites you)

AbortController in Safari < 15: guard with feature check; fallback to a boolean “cancelled” flag.

OffscreenCanvas: feature-detect; never require it.

SharedArrayBuffer: only if cross-origin isolation is configured; keep a non-SAB code path ready.

WASM: verify numeric parity within 1e-9 before switching.