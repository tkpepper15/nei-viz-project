"""
Stochastic dynamical systems for each RPE pathology trajectory type.

All dynamics operate in log10 parameter space:
    x = [log10(Ra), log10(Rb), log10(Ca), log10(Cb), log10(Rsh)]

Time units: minutes.  Euler-Maruyama discretization.

Each class defines:
    drift(x, t, p)      -> (5,) vector   d(log10 x)/dt
    diffusion(x, t, p)  -> (5,) vector   sigma_i  (diagonal noise)
"""

from __future__ import annotations
import numpy as np
from enum import Enum

# Indices into the 5-dim state vector
Ra, Rb, Ca, Cb, Rsh = 0, 1, 2, 3, 4


class PathologyType(Enum):
    HEALTHY           = "healthy"
    MATURATION        = "maturation"
    BARRIER_BREAKDOWN = "barrier_breakdown"
    APICAL_INJURY     = "apical_injury"
    BASOLATERAL_INJURY = "basolateral_injury"
    OXIDATIVE_STRESS  = "oxidative_stress"
    RECOVERY          = "recovery"
    MIXED_PATHOLOGY   = "mixed_pathology"
    UNKNOWN           = "unknown"


# Population mixture for sampling pathology at trajectory generation time.
# 70% normal physiology (healthy + maturation), 25% pathological, 5% unknown.
PATHOLOGY_WEIGHTS: dict[PathologyType, float] = {
    PathologyType.HEALTHY:            0.40,
    PathologyType.MATURATION:         0.15,
    PathologyType.BARRIER_BREAKDOWN:  0.10,
    PathologyType.APICAL_INJURY:      0.05,
    PathologyType.BASOLATERAL_INJURY: 0.05,
    PathologyType.OXIDATIVE_STRESS:   0.07,
    PathologyType.RECOVERY:           0.08,
    PathologyType.MIXED_PATHOLOGY:    0.05,
    PathologyType.UNKNOWN:            0.05,
}


def _ou_drift(x_log10: np.ndarray, targets_log10: np.ndarray,
              kappas: np.ndarray) -> np.ndarray:
    """
    Ornstein-Uhlenbeck drift toward targets.
    d(log10 x)/dt = kappa * (target - x)
    """
    return kappas * (targets_log10 - x_log10)


class BaseDynamics:
    """Euler-Maruyama SDE step in log10 parameter space."""

    # Biological noise floor (log10/sqrt(min)) applied to all parameters.
    # A value of 0.002 produces ~11% random variation over 10 hours.
    BASE_SIGMA = np.array([0.002, 0.002, 0.0015, 0.0015, 0.002], dtype=np.float64)

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        raise NotImplementedError

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        return self.BASE_SIGMA.copy()

    def step(self, x: np.ndarray, t: float, dt: float,
             p: dict, rng: np.random.Generator) -> np.ndarray:
        mu    = self.drift(x, t, p)
        sigma = self.diffusion(x, t, p)
        dW    = rng.standard_normal(5) * np.sqrt(dt)
        return x + mu * dt + sigma * dW


class HealthyDynamics(BaseDynamics):
    """
    Stable barrier with slow physiological drift.

    All parameters undergo a very slow random walk around their initial values.
    Rsh drifts very slightly upward (healthy tissue maintains its junction).
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        # Gentle mean-reversion toward the initial healthy state
        d += _ou_drift(x, p["x0"], np.full(5, 3e-4))
        # Rsh has a small upward bias — healthy junctions tighten slowly
        d[Rsh] += 5e-5
        return d


class MaturationDynamics(BaseDynamics):
    """
    Progressive barrier formation.

    Rsh rises toward a mature ceiling as tight junctions form.
    Ra and Rb decrease moderately as transcellular conductance increases.
    Ca and Cb increase slightly as membrane surface area grows.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        # Rsh increases toward mature ceiling
        d[Rsh] = _ou_drift(x[[Rsh]], p["rsh_mature_log10"], np.array([5e-4]))[0]
        # Resistance decreases — maturing cells have more channels
        d[Ra] = _ou_drift(x[[Ra]], p["x0"][[Ra]] - 0.3, np.array([2e-4]))[0]
        d[Rb] = _ou_drift(x[[Rb]], p["x0"][[Rb]] - 0.2, np.array([2e-4]))[0]
        # Capacitance grows with membrane surface area
        d[Ca] += 1e-4
        d[Cb] += 8e-5
        return d


class BarrierBreakdownDynamics(BaseDynamics):
    """
    Paracellular barrier failure (tight junction disruption).

    Rsh decays toward a pathological floor. Ra and Rb are secondarily
    affected as the epithelium responds to junction disruption.

    dR_shunt = -kappa * (log10(Rsh) - log10(Rsh_floor)) dt  (log-space OU)
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        kappa = p.get("kappa_rsh", 1.5e-3)
        d[Rsh] = -kappa * (x[Rsh] - p["rsh_floor_log10"])
        # Secondary: mild apical resistance decrease as cytoskeleton remodels
        d[Ra] -= 3e-4
        return d

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma[Rsh] = 0.004   # Rsh is noisy during breakdown
        return sigma


class ApicalInjuryDynamics(BaseDynamics):
    """
    Damage to the apical (photoreceptor-facing) membrane.

    Ra decreases as apical channels become disrupted.
    Ca may shift as membrane composition changes.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        kappa = p.get("kappa_ra", 1.2e-3)
        d[Ra] = -kappa * (x[Ra] - p["ra_floor_log10"])
        # Secondary Ca shift (lipid remodeling)
        d[Ca] += p.get("dca_rate", 2e-4)
        return d

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma[Ra] = 0.0035
        return sigma


class BasolateralInjuryDynamics(BaseDynamics):
    """
    Damage to the basolateral (choroidal-facing) membrane.

    Rb decreases. Cb may shift. Rsh is relatively preserved early.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        kappa = p.get("kappa_rb", 1.2e-3)
        d[Rb] = -kappa * (x[Rb] - p["rb_floor_log10"])
        d[Cb] += p.get("dcb_rate", 1.5e-4)
        return d

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma[Rb] = 0.0035
        return sigma


class OxidativeStressDynamics(BaseDynamics):
    """
    Oxidative damage: two-phase response.

    Phase 1 (t < t_switch): Ra decreases from oxidative membrane disruption.
    Phase 2 (t >= t_switch): Cb shifts as lipid peroxidation affects
                              basolateral capacitance; Rsh begins to fall.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        t_switch = p.get("t_switch_min", 120.0)
        if t < t_switch:
            d[Ra] -= p.get("kappa_ra", 1e-3) * (x[Ra] - p["ra_floor_log10"])
        else:
            d[Cb] += p.get("dcb_rate", 3e-4)
            d[Rsh] -= p.get("kappa_rsh", 5e-4) * (x[Rsh] - p["rsh_floor_log10"])
        return d

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma[[Ra, Cb, Rsh]] = 0.003
        return sigma


class RecoveryDynamics(BaseDynamics):
    """
    Partial recovery after a prior perturbation.

    All parameters mean-revert toward a healthy target, but may not fully
    recover — the asymptotic target is sampled slightly below healthy.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        kappas = np.full(5, p.get("kappa_recovery", 8e-4))
        return _ou_drift(x, p["recovery_target_log10"], kappas)

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma *= 1.3   # slightly noisier during recovery
        return sigma


class MixedPathologyDynamics(BaseDynamics):
    """
    Compound failure: barrier breakdown followed by transcellular injury.

    Rsh falls first, then Ra decreases as the apical membrane is exposed.
    Represents the most common real-world failure mode.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        d = np.zeros(5)
        t_second = p.get("t_second_min", 90.0)
        # Continuous Rsh decay
        d[Rsh] = -p.get("kappa_rsh", 1.2e-3) * (x[Rsh] - p["rsh_floor_log10"])
        if t >= t_second:
            d[Ra] = -p.get("kappa_ra", 8e-4) * (x[Ra] - p["ra_floor_log10"])
            d[Cb] += p.get("dcb_rate", 1.5e-4)
        return d

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        sigma = super().diffusion(x, t, p).copy()
        sigma[[Ra, Rsh]] = 0.004
        return sigma


class UnknownDynamics(BaseDynamics):
    """
    Novel / out-of-distribution biology.

    Larger diffusion, uncorrelated random walks with occasional sudden shifts.
    Forces the inference system to learn "I don't know" rather than always
    mapping observations onto one of the named pathology modes.
    """

    def drift(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        # Slow drift in a random direction fixed at trajectory initialization
        return p["drift_direction"] * 3e-4

    def diffusion(self, x: np.ndarray, t: float, p: dict) -> np.ndarray:
        # 2.5x larger noise than baseline
        return self.BASE_SIGMA * 2.5


# Registry — indexed by PathologyType
DYNAMICS_MAP: dict[PathologyType, BaseDynamics] = {
    PathologyType.HEALTHY:            HealthyDynamics(),
    PathologyType.MATURATION:         MaturationDynamics(),
    PathologyType.BARRIER_BREAKDOWN:  BarrierBreakdownDynamics(),
    PathologyType.APICAL_INJURY:      ApicalInjuryDynamics(),
    PathologyType.BASOLATERAL_INJURY: BasolateralInjuryDynamics(),
    PathologyType.OXIDATIVE_STRESS:   OxidativeStressDynamics(),
    PathologyType.RECOVERY:           RecoveryDynamics(),
    PathologyType.MIXED_PATHOLOGY:    MixedPathologyDynamics(),
    PathologyType.UNKNOWN:            UnknownDynamics(),
}
