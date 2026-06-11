"""
Posterior Trajectory Graph (PTG)

Post-processes FFBS output (smoothed particle clouds) into a temporal graph whose
nodes are mechanistic modes at each timestep and whose edges encode the process-model
transition probability.  The graph is a scientific object: its topology reveals when
mechanistic ambiguity emerges (forks) and when it resolves (merges).

Interface
---------
    result = build_ptg(positions_history, smoothed_weights, dt_history)
    print(result.summary())

Inputs come directly from the GPF + FFBS pipeline:
    positions_history  : list of T (N, 5) arrays  — log10 [Ra, Rb, Ca, Cb, Rsh]
    smoothed_weights   : list of T (N,) arrays    — FFBS-smoothed normalized weights
    dt_history         : list of T floats         — Δt in minutes per step

Design decisions
----------------
- Clustering is performed in *identifiable* space [tau_big, tau_small, TER, TEC, Rsh]
  (all log10) to eliminate the Ra↔Rb degeneracy from cluster assignment.
- Weighted resampling (resample_n draws ~ smoothed_weights) feeds sklearn.cluster.HDBSCAN,
  so the cluster structure reflects the FFBS posterior, not the raw forward filter.
- Edge weights are -log p(center_{t+1} | center_t) under the GPF process model,
  reusing _Q_POS_DIAG directly — no new thresholds or free parameters.
- MAP path is the minimum-cost path through the graph (Dijkstra / Viterbi decoding),
  which equals the maximum-posterior trajectory through the mode sequence.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN

from src.pipeline.gpf import _Q_POS_DIAG, BOUNDS_LOW, BOUNDS_HIGH

_PARAM_NAMES_RAW = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
_PARAM_NAMES_ID  = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def _to_identifiable(pos_log10: np.ndarray) -> np.ndarray:
    """
    (N, 5) log10 raw [Ra, Rb, Ca, Cb, Rsh] -> (N, 5) log10 identifiable
    [tau_big, tau_small, TER, TEC, Rsh].

    Handles both single-row (1, 5) and batch (N, 5) inputs.
    """
    p = 10.0 ** np.atleast_2d(pos_log10).astype(np.float64)
    Ra, Rb, Ca, Cb, Rsh = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]

    tau_a = Ra * Ca
    tau_b = Rb * Cb
    TER   = Rsh * (Ra + Rb) / (Rsh + Ra + Rb + _EPS)
    TEC   = Ca * Cb / (Ca + Cb + _EPS)

    tau_big   = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)

    out = np.column_stack([
        np.log10(np.maximum(tau_big,   _EPS)),
        np.log10(np.maximum(tau_small, _EPS)),
        np.log10(np.maximum(TER,       _EPS)),
        np.log10(np.maximum(TEC,       _EPS)),
        np.log10(np.maximum(Rsh,       _EPS)),
    ])
    return out if pos_log10.ndim == 2 else out[0]


def _weighted_mean_raw(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of (N, 5) particle positions given (N,) weights."""
    w = weights / (weights.sum() + _EPS)
    return (positions * w[:, np.newaxis]).sum(axis=0)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PTGNode:
    t: int
    cluster_id: int
    center_raw: np.ndarray   # (5,) log10 [Ra, Rb, Ca, Cb, Rsh]
    center_id: np.ndarray    # (5,) log10 [tau_big, tau_small, TER, TEC, Rsh]
    std_id: np.ndarray       # (5,) within-cluster std in identifiable space
    weight: float            # total smoothed weight assigned to this cluster
    n_particles: int         # number of original particles assigned to this cluster


@dataclass
class ForkEvent:
    t: int        # timestep where the fork occurs (cluster count increased)
    n_before: int
    n_after: int


@dataclass
class MergeEvent:
    t: int        # timestep where the merge occurs (cluster count decreased)
    n_before: int
    n_after: int


@dataclass
class PTGResult:
    nodes: dict                              # (t, cluster_id) -> PTGNode
    adjacency: dict                          # (t, c) -> list[(cost, (t+1, c'))]
    cluster_counts: list                     # T ints
    fork_events: list                        # ForkEvent list
    merge_events: list                       # MergeEvent list
    map_path_nodes: list                     # [(t, cluster_id), ...]
    map_trajectory: np.ndarray               # (T, 5) log10 raw centers along MAP path
    T: int
    N: int

    def summary(self) -> str:
        lines = [f"PTG  T={self.T}  N={self.N}"]
        if not self.fork_events and not self.merge_events:
            lines.append("  Topology: single branch (no forks or merges)")
        else:
            for ev in self.fork_events:
                lines.append(
                    f"  Fork  at t={ev.t:4d}  ({ev.n_before} -> {ev.n_after} clusters)"
                )
            for ev in self.merge_events:
                lines.append(
                    f"  Merge at t={ev.t:4d}  ({ev.n_before} -> {ev.n_after} clusters)"
                )
        lines.append(f"  MAP path length: {len(self.map_path_nodes)} nodes")
        return "\n".join(lines)

    def fork_times(self, min_duration: int = 3) -> list[int]:
        """
        Timesteps where a sustained fork begins.

        A fork is sustained if the cluster count stays above its pre-fork value
        for at least `min_duration` consecutive steps.  This suppresses ephemeral
        HDBSCAN splits caused by finite-sample noise.
        """
        return [t for t in self._sustained_fork_times(min_duration)]

    def first_fork_time(self, min_duration: int = 3) -> Optional[int]:
        times = self.fork_times(min_duration)
        return times[0] if times else None

    def _sustained_fork_times(self, min_duration: int) -> list[int]:
        counts = self.cluster_counts
        result = []
        t = 0
        while t < len(counts):
            if t > 0 and counts[t] > counts[t - 1]:
                new_level = counts[t]
                run = 1
                for t2 in range(t + 1, len(counts)):
                    if counts[t2] >= new_level:
                        run += 1
                    else:
                        break
                if run >= min_duration:
                    result.append(t)
            t += 1
        return result

    def mode_separation_series(self) -> np.ndarray:
        """
        D_t = max pairwise cluster-center distance / pooled within-cluster std
        in identifiable log10 space, at each timestep.  Analogous to Cohen's d.

        D_t = 0  when there is one cluster (unambiguous).
        D_t > 1  means mode separation exceeds the pooled intra-cluster spread
                 — the hypotheses are geometrically distinct.
        D_t_abs (raw center distance in log10 decades) is also recoverable:
                 multiply D_t by the pooled std.
        """
        D = np.zeros(self.T)
        for t in range(self.T):
            cluster_ids = [c for (tt, c) in self.nodes if tt == t]
            if len(cluster_ids) < 2:
                continue
            centers = np.stack([self.nodes[(t, c)].center_id for c in cluster_ids])
            stds    = np.stack([self.nodes[(t, c)].std_id     for c in cluster_ids])
            # Pooled within-cluster L2 spread (scalar)
            pooled_std = float(np.sqrt((stds ** 2).mean()))
            # Max pairwise center distance
            max_dist = 0.0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    max_dist = max(max_dist, float(np.linalg.norm(centers[i] - centers[j])))
            D[t] = max_dist / max(pooled_std, 1e-12)
        return D

    def mode_separation_abs_series(self) -> np.ndarray:
        """
        Absolute max pairwise cluster-center distance (log10 decades) at each timestep.
        This is D_true's observable counterpart — the actual measured hypothesis separation.
        """
        D = np.zeros(self.T)
        for t in range(self.T):
            cluster_ids = [c for (tt, c) in self.nodes if tt == t]
            if len(cluster_ids) < 2:
                continue
            centers = np.stack([self.nodes[(t, c)].center_id for c in cluster_ids])
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    D[t] = max(D[t], float(np.linalg.norm(centers[i] - centers[j])))
        return D

    def cluster_entropy_series(self) -> np.ndarray:
        """
        H_t = Shannon entropy of the cluster weight distribution at each timestep.

        H_t = 0 when all weight is on one cluster (no ambiguity).
        H_t = log(k) when k clusters have equal weight (maximum ambiguity for k modes).

        This is the information-theoretic complement to mode_separation_series:
        D measures geometric separation, H measures weight balance.
        """
        H = np.zeros(self.T)
        for t in range(self.T):
            weights = np.array([
                self.nodes[(t, c)].weight
                for c in [c for (tt, c) in self.nodes if tt == t]
            ])
            weights = weights / (weights.sum() + _EPS)
            H[t] = -float(np.sum(weights * np.log(weights + _EPS)))
        return H

    def first_merge_time(self, min_duration: int = 3) -> Optional[int]:
        """
        First timestep where cluster count drops and stays lower for min_duration steps.

        The merge time is the point at which the posterior resolved competing hypotheses.
        """
        counts = self.cluster_counts
        for t in range(1, len(counts)):
            if counts[t] < counts[t - 1]:
                run = 1
                for t2 in range(t + 1, len(counts)):
                    if counts[t2] <= counts[t]:
                        run += 1
                    else:
                        break
                if run >= min_duration:
                    return t
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _resample_weighted(positions: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    """
    Draw n samples from positions according to weights (with replacement).

    Adds small Gaussian jitter (10% of each dimension's std) to avoid
    HDBSCAN seeing aliased point clouds when n >> len(positions).
    """
    w = np.maximum(weights, 0.0)
    total = w.sum()
    if total <= 0:
        idx = np.random.randint(len(positions), size=n)
    else:
        idx = np.random.choice(len(positions), size=n, replace=True, p=w / total)
    samples = positions[idx]
    jitter_scale = positions.std(axis=0).clip(1e-6) * 0.1
    samples = samples + np.random.randn(*samples.shape) * jitter_scale[np.newaxis, :]
    return samples


def _cluster_at_timestep(
    positions_log10: np.ndarray,
    weights: np.ndarray,
    min_cluster_size: int,
    resample_n: int,
) -> tuple[np.ndarray, dict, dict]:
    """
    Cluster particles at a single timestep.

    Resamples `resample_n` particles from the smoothed posterior, clusters them
    in identifiable space with HDBSCAN, then maps each original particle to its
    nearest cluster center.

    Returns
    -------
    labels       : (N,) int cluster assignment for each original particle (-1 = noise)
    centers_raw  : {cluster_id: (5,) log10 raw center}
    centers_id   : {cluster_id: (5,) log10 identifiable center}
    """
    N = len(positions_log10)

    # --- Resample from FFBS posterior ---
    resampled = _resample_weighted(positions_log10, weights, resample_n)
    resampled_id = _to_identifiable(resampled)

    # Scale min_cluster_size relative to resample_n so that meaningful modes
    # require at least ~10% of the resampled mass — prevents micro-cluster explosion.
    effective_min = max(min_cluster_size, resample_n // 10)

    # cluster_selection_epsilon merges adjacent clusters within this distance (log10
    # decades in identifiable space).  Chosen to be small compared to genuine
    # mechanistic separation (~0.5+ decades) but large enough to absorb jitter-induced
    # density peaks (~0.05 decades).
    epsilon = 0.15

    # --- HDBSCAN in identifiable space ---
    clusterer = HDBSCAN(
        min_cluster_size=effective_min,
        min_samples=max(2, effective_min // 4),
        cluster_selection_method='eom',
        cluster_selection_epsilon=epsilon,
    )
    sample_labels = clusterer.fit_predict(resampled_id)

    unique_clusters = sorted(set(sample_labels) - {-1})

    # Fallback: if everything is noise, treat all as cluster 0
    if not unique_clusters:
        unique_clusters = [0]
        sample_labels = np.zeros(len(resampled_id), dtype=int)

    # Cluster centers in identifiable space (from resampled cloud)
    centers_id_resampled: dict[int, np.ndarray] = {}
    for c in unique_clusters:
        mask = sample_labels == c
        centers_id_resampled[c] = resampled_id[mask].mean(axis=0)

    # Assign each *original* particle to nearest cluster by distance in identifiable space
    orig_id = _to_identifiable(positions_log10)
    center_matrix = np.stack([centers_id_resampled[c] for c in unique_clusters])  # (K, 5)

    diffs = orig_id[:, np.newaxis, :] - center_matrix[np.newaxis, :, :]  # (N, K, 5)
    dist_sq = (diffs ** 2).sum(axis=2)                                    # (N, K)
    nearest = np.argmin(dist_sq, axis=1)                                  # (N,)
    labels = np.array([unique_clusters[i] for i in nearest], dtype=int)

    # Compute final cluster centers as weighted mean of *original* particles per cluster
    w = np.maximum(weights, 0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum

    centers_raw: dict[int, np.ndarray] = {}
    centers_id: dict[int, np.ndarray] = {}

    for c in unique_clusters:
        mask = labels == c
        if mask.sum() == 0:
            centers_raw[c] = positions_log10.mean(axis=0)
        else:
            wc = w[mask]
            wc_sum = wc.sum()
            if wc_sum > 0:
                centers_raw[c] = _weighted_mean_raw(positions_log10[mask], wc)
            else:
                centers_raw[c] = positions_log10[mask].mean(axis=0)
        centers_id[c] = _to_identifiable(centers_raw[c][np.newaxis, :])[0]

    return labels, centers_raw, centers_id


# ---------------------------------------------------------------------------
# Edge weights (process model)
# ---------------------------------------------------------------------------

def _edge_neg_log_prob(center_t: np.ndarray, center_t1: np.ndarray, dt: float) -> float:
    """
    -log p(center_t1 | center_t) under the GPF random-walk process model.

    Uses _Q_POS_DIAG from gpf directly — no new free parameters.
    Both centers are in raw log10 space.
    """
    var = _Q_POS_DIAG * max(dt, 1e-3)
    diff = center_t1 - center_t
    return float(0.5 * (diff ** 2 / var).sum())


# ---------------------------------------------------------------------------
# Dijkstra for MAP path
# ---------------------------------------------------------------------------

def _dijkstra_map_path(
    nodes: dict,
    adjacency: dict,
    T: int,
) -> list[tuple[int, int]]:
    """
    Minimum-cost path from any node at t=0 to any node at t=T-1.

    Node cost at t=0: -log(cluster_weight) — more probable clusters have lower entry cost.
    Edge cost: already -log p(transition) stored in adjacency.

    Returns a list of (t, cluster_id) tuples of length T.
    """
    # Priority queue: (cumulative_cost, (t, cluster_id), path_so_far)
    heap: list = []

    for (t, c), node in nodes.items():
        if t != 0:
            continue
        entry_cost = -np.log(max(node.weight, 1e-300))
        heapq.heappush(heap, (entry_cost, (t, c), [(t, c)]))

    best_cost = np.inf
    best_path: list[tuple[int, int]] = []

    visited: set = set()

    while heap:
        cost, node_key, path = heapq.heappop(heap)

        if node_key in visited:
            continue
        visited.add(node_key)

        t, c = node_key
        if t == T - 1:
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue

        for edge_cost, neighbor in adjacency.get(node_key, []):
            if neighbor not in visited:
                new_cost = cost + edge_cost
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))

    # Fallback: if no complete path found (disconnected graph), greedily pick
    # the highest-weight cluster at each timestep
    if not best_path:
        best_path = []
        for t in range(T):
            best_c = max(
                (c for (tt, c) in nodes if tt == t),
                key=lambda c: nodes[(t, c)].weight,
                default=0,
            )
            best_path.append((t, best_c))

    return best_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_ptg(
    positions_history: list,
    smoothed_weights: list,
    dt_history: list,
    min_cluster_size: int = 6,
    resample_n: int = 256,
) -> PTGResult:
    """
    Build a Posterior Trajectory Graph from FFBS output.

    Parameters
    ----------
    positions_history : list of T (N, 5) arrays, log10 [Ra, Rb, Ca, Cb, Rsh]
    smoothed_weights  : list of T (N,) arrays, FFBS-smoothed normalized weights
    dt_history        : list of T floats, Δt in minutes per step
    min_cluster_size  : minimum particles per HDBSCAN cluster (tune for N)
    resample_n        : number of weighted resamples fed to HDBSCAN per timestep

    Returns
    -------
    PTGResult with nodes, adjacency, topology events, and MAP path
    """
    T = len(positions_history)
    N = len(positions_history[0]) if T > 0 else 0

    nodes: dict = {}
    all_labels: list = []
    all_centers_raw: list = []
    all_centers_id: list = []

    # --- Pass 1: cluster each timestep independently ---
    for t in range(T):
        pos = positions_history[t]
        w   = smoothed_weights[t]

        finite = np.all(np.isfinite(pos), axis=1)
        pos_f  = pos[finite]
        w_f    = w[finite]

        if w_f.sum() > 0:
            w_f = w_f / w_f.sum()

        labels, centers_raw, centers_id = _cluster_at_timestep(
            pos_f, w_f, min_cluster_size, resample_n
        )

        all_labels.append(labels)
        all_centers_raw.append(centers_raw)
        all_centers_id.append(centers_id)

        # Compute cluster weights and within-cluster spread
        orig_id = _to_identifiable(pos_f)
        for c, center_raw in centers_raw.items():
            mask = labels == c
            cluster_weight = float(w_f[mask].sum()) if mask.sum() > 0 else 0.0
            # Within-cluster std in identifiable space
            if mask.sum() >= 2:
                std_id = orig_id[mask].std(axis=0)
            else:
                std_id = np.zeros(5)
            nodes[(t, c)] = PTGNode(
                t=t,
                cluster_id=c,
                center_raw=center_raw.copy(),
                center_id=centers_id[c].copy(),
                std_id=std_id,
                weight=cluster_weight,
                n_particles=int(mask.sum()),
            )

    # --- Pass 2: build directed edges between adjacent timesteps ---
    adjacency: dict = {}

    for t in range(T - 1):
        dt = max(float(dt_history[t]), 1e-3)
        for c_t, center_t in all_centers_raw[t].items():
            edges = []
            for c_t1, center_t1 in all_centers_raw[t + 1].items():
                cost = _edge_neg_log_prob(center_t, center_t1, dt)
                edges.append((cost, (t + 1, c_t1)))
            adjacency[(t, c_t)] = edges

    # --- Pass 3: topology analysis ---
    cluster_counts = [len(all_centers_raw[t]) for t in range(T)]

    fork_events: list[ForkEvent] = []
    merge_events: list[MergeEvent] = []

    for t in range(1, T):
        before = cluster_counts[t - 1]
        after  = cluster_counts[t]
        if after > before:
            fork_events.append(ForkEvent(t=t, n_before=before, n_after=after))
        elif after < before:
            merge_events.append(MergeEvent(t=t, n_before=before, n_after=after))

    # --- Pass 4: MAP path ---
    map_path_nodes = _dijkstra_map_path(nodes, adjacency, T)

    map_trajectory = np.stack([
        nodes[(t, c)].center_raw for t, c in map_path_nodes
        if (t, c) in nodes
    ])

    return PTGResult(
        nodes=nodes,
        adjacency=adjacency,
        cluster_counts=cluster_counts,
        fork_events=fork_events,
        merge_events=merge_events,
        map_path_nodes=map_path_nodes,
        map_trajectory=map_trajectory,
        T=T,
        N=N,
    )
