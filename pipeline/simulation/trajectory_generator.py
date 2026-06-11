"""
Human RPE trajectory generator.

Samples initial biological states from literature-informed priors and evolves
them forward in time using the SDE dynamics defined in pathology_models.py.

Literature priors
-----------------
All values are for Ussing chamber measurements with cross-section A = 0.11409 cm².
Raw Ω and F values are derived by dividing Ω·cm² by A and multiplying µF/cm² by A.

TER (Ω·cm²)       : healthy ~300, range 150–800
Ra/Rb ratio        : apical resistance 2–8x basolateral (classical RPE asymmetry)
Rsh (Ω·cm²)       : healthy 500–5000, stressed 100–1000, pathological 5–100
τa = Ra*Ca (s)     : 0.01–1 s
τb = Rb*Cb (s)     : 0.005–0.5 s

References
----------
- Stern et al. (2015), PMC4640474: adult hRPE TER ~179 Ω·cm²
- Hartzell et al. (1990), PubMed 2170293: fetal hRPE TER ~330 Ω·cm²
- Maruotti et al. (2013), PMC3916756: mature hRPE TER 500–1400 Ω·cm²
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from .pathology_models import (
    PathologyType, PATHOLOGY_WEIGHTS, DYNAMICS_MAP,
    Ra, Rb, Ca, Cb, Rsh,
)

# Ussing chamber cross-sectional area [cm²]
CROSS_SECTION_CM2 = 0.11409


# ---------------------------------------------------------------------------
# Biological bounds in raw Ω/F (wider than the existing GPF training bounds).
# These represent true physiological extremes, not optimization constraints.
# The GPF bounds (src/pipeline/gpf.py) should be updated to match.
# ---------------------------------------------------------------------------
SIM_BOUNDS_LOW_LOG10  = np.array([1.699, 1.699, -6.5, -6.5, 1.699],  dtype=np.float64)
SIM_BOUNDS_HIGH_LOG10 = np.array([4.602, 4.398, -2.5, -2.5, 4.602],  dtype=np.float64)
# Ra/Rsh upper: 40000 Ω (4.602), Rb: 25000 Ω (4.398)
# Ca/Cb: 3e-7 to 3e-3 F (captures τ = 0.001–3 s across the full R range)


class Trajectory(NamedTuple):
    """One simulated RPE trajectory."""
    params_log10:  np.ndarray   # (T, 5)  log10[Ra, Rb, Ca, Cb, Rsh]
    time_minutes:  np.ndarray   # (T,)
    pathology:     PathologyType
    initial_state: np.ndarray   # (5,) log10 — initial healthy state before perturbation


class TrajectoryGenerator:
    """
    Generates stochastic biological trajectories for synthetic EIS dataset creation.

    Parameters
    ----------
    cross_section_cm2 : float
        Ussing chamber area.  Used to convert literature Ω·cm² → raw Ω.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, cross_section_cm2: float = CROSS_SECTION_CM2,
                 seed: int | None = None):
        self.area = cross_section_cm2
        self.rng  = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_timepoints:     int   = 200,
        dt_minutes:       float = 3.0,
        pathology_type:   PathologyType | None = None,
    ) -> Trajectory:
        """
        Generate a single trajectory.

        Parameters
        ----------
        n_timepoints : int
            Number of observations per trajectory.
        dt_minutes : float
            Interval between observations in minutes.
        pathology_type : PathologyType | None
            Force a specific pathology; None samples from PATHOLOGY_WEIGHTS.

        Returns
        -------
        Trajectory
            Named tuple with params_log10 (T, 5), time_minutes (T,),
            pathology type, and the initial healthy state.
        """
        if pathology_type is None:
            pathology_type = self._sample_pathology_type()

        x0_log10 = self._sample_healthy_state()
        x_log10  = self._perturb_for_pathology(x0_log10.copy(), pathology_type)

        dyn_params = self._make_dynamics_params(x0_log10, x_log10, pathology_type)
        dynamics   = DYNAMICS_MAP[pathology_type]

        times  = np.arange(n_timepoints, dtype=np.float64) * dt_minutes
        states = np.empty((n_timepoints, 5), dtype=np.float64)
        states[0] = x_log10

        for t_idx in range(1, n_timepoints):
            x_new = dynamics.step(
                states[t_idx - 1], times[t_idx - 1], dt_minutes,
                dyn_params, self.rng,
            )
            states[t_idx] = np.clip(x_new, SIM_BOUNDS_LOW_LOG10, SIM_BOUNDS_HIGH_LOG10)

        return Trajectory(
            params_log10  = states,
            time_minutes  = times,
            pathology     = pathology_type,
            initial_state = x0_log10,
        )

    def generate_batch(
        self,
        n_trajectories: int,
        n_timepoints:   int   = 200,
        dt_minutes:     float = 3.0,
        pathology_type: PathologyType | None = None,
    ) -> list[Trajectory]:
        """Generate multiple trajectories."""
        return [
            self.generate(n_timepoints, dt_minutes, pathology_type)
            for _ in range(n_trajectories)
        ]

    # ------------------------------------------------------------------
    # Initial state sampling (human RPE priors)
    # ------------------------------------------------------------------

    def _sample_healthy_state(self) -> np.ndarray:
        """
        Sample one healthy RPE state from literature-informed priors.

        Strategy: sample Rb, Ra/Rb ratio, Rsh independently in log10 space,
        then derive Ca/Cb from sampled time constants (τ = R * C).
        This produces physically consistent capacitances automatically.
        """
        rng = self.rng

        # Basolateral resistance: 20–150 Ω·cm² → raw Ω
        rb_specific_log10 = rng.normal(loc=np.log10(60.0), scale=0.35)
        rb_log10  = rb_specific_log10 + np.log10(1.0 / self.area)
        # Ra/Rb ratio: 2–8 (apical > basolateral in RPE)
        ratio_log10 = rng.normal(loc=np.log10(4.0), scale=0.30)
        ra_log10  = rb_log10 + ratio_log10

        # Shunt resistance: healthy tight junction 500–5000 Ω·cm²
        rsh_specific_log10 = rng.normal(loc=np.log10(1500.0), scale=0.40)
        rsh_log10 = rsh_specific_log10 + np.log10(1.0 / self.area)

        # Time constants → capacitances
        tau_a_log10 = rng.normal(loc=np.log10(0.15), scale=0.5)   # τa: 0.01–1 s
        tau_b_log10 = rng.normal(loc=np.log10(0.05), scale=0.5)   # τb: 0.005–0.5 s
        ca_log10 = tau_a_log10 - ra_log10   # log10(Ca) = log10(τa) - log10(Ra)
        cb_log10 = tau_b_log10 - rb_log10

        state = np.array([ra_log10, rb_log10, ca_log10, cb_log10, rsh_log10])
        return np.clip(state, SIM_BOUNDS_LOW_LOG10, SIM_BOUNDS_HIGH_LOG10)

    def _perturb_for_pathology(
        self, x: np.ndarray, ptype: PathologyType
    ) -> np.ndarray:
        """
        Apply an initial offset to the healthy state so the trajectory
        starts at the right biological condition for each pathology.
        """
        rng = self.rng

        if ptype == PathologyType.HEALTHY:
            pass  # no perturbation

        elif ptype == PathologyType.MATURATION:
            # Start immature: lower Rsh, slightly higher Ra/Rb
            x[Rsh] -= rng.uniform(0.4, 0.8)
            x[Ra]  += rng.uniform(0.1, 0.3)
            x[Rb]  += rng.uniform(0.05, 0.2)

        elif ptype == PathologyType.BARRIER_BREAKDOWN:
            # Start healthy but about to degrade — no offset, dynamics drive it
            pass

        elif ptype == PathologyType.APICAL_INJURY:
            pass

        elif ptype == PathologyType.BASOLATERAL_INJURY:
            pass

        elif ptype == PathologyType.OXIDATIVE_STRESS:
            pass

        elif ptype == PathologyType.RECOVERY:
            # Start stressed: Rsh is depressed, Ra may be lower
            x[Rsh] -= rng.uniform(0.5, 1.2)
            x[Ra]  -= rng.uniform(0.2, 0.6)

        elif ptype == PathologyType.MIXED_PATHOLOGY:
            pass

        elif ptype == PathologyType.UNKNOWN:
            # Shift into unusual region of parameter space
            delta = rng.uniform(-0.5, 0.5, size=5)
            x = x + delta

        return np.clip(x, SIM_BOUNDS_LOW_LOG10, SIM_BOUNDS_HIGH_LOG10)

    # ------------------------------------------------------------------
    # Dynamics parameter construction
    # ------------------------------------------------------------------

    def _make_dynamics_params(
        self,
        x0: np.ndarray,
        x_start: np.ndarray,
        ptype: PathologyType,
    ) -> dict:
        """
        Build the per-trajectory parameter dict consumed by dynamics.step().
        This encodes trajectory-specific targets and rates, adding variation
        across instances of the same pathology type.
        """
        rng = self.rng

        # Pathological floors (specific Ω·cm² → raw Ω)
        def rsh_floor():
            floor_specific = 10 ** rng.uniform(0.7, 1.5)  # 5–32 Ω·cm²
            return np.log10(floor_specific / self.area)

        def ra_floor():
            floor_specific = 10 ** rng.uniform(0.9, 1.5)  # 8–32 Ω·cm²
            return np.log10(floor_specific / self.area)

        def rb_floor():
            floor_specific = 10 ** rng.uniform(0.9, 1.4)  # 8–25 Ω·cm²
            return np.log10(floor_specific / self.area)

        params: dict = {"x0": x0.copy()}

        if ptype == PathologyType.HEALTHY:
            pass

        elif ptype == PathologyType.MATURATION:
            mature_rsh = x0[Rsh] + rng.uniform(0.3, 0.7)   # target: higher Rsh
            params["rsh_mature_log10"] = min(mature_rsh, SIM_BOUNDS_HIGH_LOG10[Rsh])

        elif ptype == PathologyType.BARRIER_BREAKDOWN:
            params["kappa_rsh"]       = rng.uniform(1.0e-3, 2.5e-3)
            params["rsh_floor_log10"] = rsh_floor()

        elif ptype == PathologyType.APICAL_INJURY:
            params["kappa_ra"]       = rng.uniform(0.8e-3, 2.0e-3)
            params["ra_floor_log10"] = ra_floor()
            params["dca_rate"]       = rng.uniform(1e-4, 4e-4)

        elif ptype == PathologyType.BASOLATERAL_INJURY:
            params["kappa_rb"]       = rng.uniform(0.8e-3, 2.0e-3)
            params["rb_floor_log10"] = rb_floor()
            params["dcb_rate"]       = rng.uniform(1e-4, 4e-4)

        elif ptype == PathologyType.OXIDATIVE_STRESS:
            params["t_switch_min"]   = rng.uniform(60.0, 180.0)
            params["kappa_ra"]       = rng.uniform(8e-4, 1.5e-3)
            params["ra_floor_log10"] = ra_floor()
            params["dcb_rate"]       = rng.uniform(2e-4, 5e-4)
            params["kappa_rsh"]      = rng.uniform(4e-4, 1.0e-3)
            params["rsh_floor_log10"] = rsh_floor()

        elif ptype == PathologyType.RECOVERY:
            # Asymptotic target: close to healthy but not quite fully recovered
            fraction = rng.uniform(0.6, 0.95)   # recovery fraction
            target   = x0 + fraction * (x0 - x_start)
            params["recovery_target_log10"] = np.clip(
                target, SIM_BOUNDS_LOW_LOG10, SIM_BOUNDS_HIGH_LOG10
            )
            params["kappa_recovery"] = rng.uniform(5e-4, 1.5e-3)

        elif ptype == PathologyType.MIXED_PATHOLOGY:
            params["kappa_rsh"]       = rng.uniform(1.0e-3, 2.0e-3)
            params["rsh_floor_log10"] = rsh_floor()
            params["kappa_ra"]        = rng.uniform(6e-4, 1.5e-3)
            params["ra_floor_log10"]  = ra_floor()
            params["dcb_rate"]        = rng.uniform(1e-4, 3e-4)
            params["t_second_min"]    = rng.uniform(60.0, 180.0)

        elif ptype == PathologyType.UNKNOWN:
            # Random drift direction — fixed per trajectory
            direction = rng.standard_normal(5)
            direction /= np.linalg.norm(direction)
            params["drift_direction"] = direction

        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_pathology_type(self) -> PathologyType:
        types   = list(PATHOLOGY_WEIGHTS.keys())
        weights = np.array([PATHOLOGY_WEIGHTS[t] for t in types])
        idx = self.rng.choice(len(types), p=weights / weights.sum())
        return types[idx]
