#!/usr/bin/env python3
"""
Symmetric Parameter Metrics for EIS Parameter Extraction

Handles the fundamental degeneracy in the RCRC circuit model:
- Ra and Rb are interchangeable (apical vs basolateral symmetry)
- Ca and Cb are interchangeable
- Rsh and Ra+Rb represent different pathways (paracellular vs transcellular)

Instead of forcing the model to assign "apical" vs "basolateral" labels,
we predict:
- R1 (transcellular membrane resistance): best match to Ra OR Rb
- C1 (membrane capacitance): best match to Ca OR Cb
- R2 (paracellular/total resistance): best match to Rsh OR Ra+Rb

This removes the arbitrary labeling degeneracy and evaluates what the
model can actually identify from the impedance spectrum.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


def compute_symmetric_mae(
    pred_R1: np.ndarray,
    pred_C1: np.ndarray,
    pred_R2: np.ndarray,
    true_Ra: np.ndarray,
    true_Rb: np.ndarray,
    true_Ca: np.ndarray,
    true_Cb: np.ndarray,
    true_Rsh: np.ndarray,
    log_space: bool = True
) -> Dict[str, float]:
    """
    Compute symmetric MAE metrics that handle parameter degeneracy.

    For each prediction, we compute the minimum error against both
    possible assignments:

    - R1: min(|R1 - Ra|, |R1 - Rb|) - transcellular membrane resistance
    - C1: min(|C1 - Ca|, |C1 - Cb|) - membrane capacitance
    - R2: min(|R2 - Rsh|, |R2 - (Ra+Rb)|) - paracellular pathway

    Args:
        pred_R1: Predicted transcellular resistance (N,)
        pred_C1: Predicted membrane capacitance (N,)
        pred_R2: Predicted paracellular resistance (N,)
        true_Ra, true_Rb: True apical/basolateral resistances (N,)
        true_Ca, true_Cb: True apical/basolateral capacitances (N,)
        true_Rsh: True shunt resistance (N,)
        log_space: If True, compute error in log10 space (decades)

    Returns:
        Dict with:
            - 'R1_mae': MAE for R1 (min against Ra, Rb)
            - 'C1_mae': MAE for C1 (min against Ca, Cb)
            - 'R2_mae': MAE for R2 (min against Rsh, Ra+Rb)
            - 'R1_assignment': Fraction assigned to Ra vs Rb
            - 'C1_assignment': Fraction assigned to Ca vs Cb
            - 'R2_assignment': Fraction assigned to Rsh vs Ra+Rb
    """
    if log_space:
        # Convert to log10 space
        pred_R1_log = np.log10(pred_R1 + 1e-12)
        pred_C1_log = np.log10(pred_C1 + 1e-12)
        pred_R2_log = np.log10(pred_R2 + 1e-12)

        true_Ra_log = np.log10(true_Ra + 1e-12)
        true_Rb_log = np.log10(true_Rb + 1e-12)
        true_Ca_log = np.log10(true_Ca + 1e-12)
        true_Cb_log = np.log10(true_Cb + 1e-12)
        true_Rsh_log = np.log10(true_Rsh + 1e-12)
        true_RaRb_log = np.log10(true_Ra + true_Rb + 1e-12)
    else:
        pred_R1_log, pred_C1_log, pred_R2_log = pred_R1, pred_C1, pred_R2
        true_Ra_log, true_Rb_log = true_Ra, true_Rb
        true_Ca_log, true_Cb_log = true_Ca, true_Cb
        true_Rsh_log = true_Rsh
        true_RaRb_log = true_Ra + true_Rb

    # R1: min error between R1 and either Ra or Rb
    err_R1_vs_Ra = np.abs(pred_R1_log - true_Ra_log)
    err_R1_vs_Rb = np.abs(pred_R1_log - true_Rb_log)
    R1_min_errors = np.minimum(err_R1_vs_Ra, err_R1_vs_Rb)
    R1_assigned_to_Ra = (err_R1_vs_Ra <= err_R1_vs_Rb)

    # C1: min error between C1 and either Ca or Cb
    err_C1_vs_Ca = np.abs(pred_C1_log - true_Ca_log)
    err_C1_vs_Cb = np.abs(pred_C1_log - true_Cb_log)
    C1_min_errors = np.minimum(err_C1_vs_Ca, err_C1_vs_Cb)
    C1_assigned_to_Ca = (err_C1_vs_Ca <= err_C1_vs_Cb)

    # R2: min error between R2 and either Rsh or (Ra+Rb)
    err_R2_vs_Rsh = np.abs(pred_R2_log - true_Rsh_log)
    err_R2_vs_RaRb = np.abs(pred_R2_log - true_RaRb_log)
    R2_min_errors = np.minimum(err_R2_vs_Rsh, err_R2_vs_RaRb)
    R2_assigned_to_Rsh = (err_R2_vs_Rsh <= err_R2_vs_RaRb)

    return {
        'R1_mae': float(np.mean(R1_min_errors)),
        'C1_mae': float(np.mean(C1_min_errors)),
        'R2_mae': float(np.mean(R2_min_errors)),
        'R1_assignment_Ra_fraction': float(np.mean(R1_assigned_to_Ra)),
        'C1_assignment_Ca_fraction': float(np.mean(C1_assigned_to_Ca)),
        'R2_assignment_Rsh_fraction': float(np.mean(R2_assigned_to_Rsh)),
        # Individual errors for analysis
        'R1_errors': R1_min_errors,
        'C1_errors': C1_min_errors,
        'R2_errors': R2_min_errors,
    }


def compute_symmetric_mae_from_model_output(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    log_space: bool = True
) -> Dict[str, float]:
    """
    Compute symmetric MAE from model predictions dict.

    Maps model outputs to symmetric parameters:
    - R1 from Ra (or whichever the model predicts as "first" resistance)
    - C1 from Ca
    - R2 from Rsh

    Then evaluates using symmetric metrics.

    Args:
        predictions: Dict with 'Ra', 'Rb', 'Ca', 'Cb', 'Rsh' predictions
        targets: Dict with 'Ra', 'Rb', 'Ca', 'Cb', 'Rsh' true values
        log_space: Compute in log10 space

    Returns:
        Symmetric MAE metrics
    """
    # Use model's Ra prediction as R1, Ca as C1, Rsh as R2
    # The symmetric metric will find the best assignment
    return compute_symmetric_mae(
        pred_R1=predictions['Ra'],
        pred_C1=predictions['Ca'],
        pred_R2=predictions['Rsh'],
        true_Ra=targets['Ra'],
        true_Rb=targets['Rb'],
        true_Ca=targets['Ca'],
        true_Cb=targets['Cb'],
        true_Rsh=targets['Rsh'],
        log_space=log_space
    )


def compute_optimal_assignment(
    pred_R1: np.ndarray,
    pred_R2: np.ndarray,
    pred_C1: np.ndarray,
    pred_C2: np.ndarray,
    pred_Rsh: np.ndarray,
    true_Ra: np.ndarray,
    true_Rb: np.ndarray,
    true_Ca: np.ndarray,
    true_Cb: np.ndarray,
    true_Rsh: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Find optimal assignment of predicted parameters to true parameters.

    Given predictions (R1, R2, C1, C2, Rsh), find the assignment to
    (Ra, Rb, Ca, Cb, Rsh) that minimizes total error.

    This handles the full permutation symmetry:
    - (R1, C1) can match (Ra, Ca) or (Rb, Cb)
    - (R2, C2) matches the remaining pair
    - Rsh matches Rsh

    Returns:
        assigned: Dict mapping pred names to assigned target names
        metrics: Dict with assignment statistics
    """
    n_samples = len(pred_R1)

    # Convert to log space for fair comparison
    pred_R1_log = np.log10(pred_R1 + 1e-12)
    pred_R2_log = np.log10(pred_R2 + 1e-12)
    pred_C1_log = np.log10(pred_C1 + 1e-12)
    pred_C2_log = np.log10(pred_C2 + 1e-12)

    true_Ra_log = np.log10(true_Ra + 1e-12)
    true_Rb_log = np.log10(true_Rb + 1e-12)
    true_Ca_log = np.log10(true_Ca + 1e-12)
    true_Cb_log = np.log10(true_Cb + 1e-12)

    # Assignment 1: (R1, C1) -> (Ra, Ca), (R2, C2) -> (Rb, Cb)
    err_assign1 = (
        np.abs(pred_R1_log - true_Ra_log) +
        np.abs(pred_C1_log - true_Ca_log) +
        np.abs(pred_R2_log - true_Rb_log) +
        np.abs(pred_C2_log - true_Cb_log)
    )

    # Assignment 2: (R1, C1) -> (Rb, Cb), (R2, C2) -> (Ra, Ca)
    err_assign2 = (
        np.abs(pred_R1_log - true_Rb_log) +
        np.abs(pred_C1_log - true_Cb_log) +
        np.abs(pred_R2_log - true_Ra_log) +
        np.abs(pred_C2_log - true_Ca_log)
    )

    # Choose best assignment per sample
    use_assign1 = (err_assign1 <= err_assign2)

    # Compute errors with optimal assignment
    R1_errors = np.where(use_assign1,
                         np.abs(pred_R1_log - true_Ra_log),
                         np.abs(pred_R1_log - true_Rb_log))
    R2_errors = np.where(use_assign1,
                         np.abs(pred_R2_log - true_Rb_log),
                         np.abs(pred_R2_log - true_Ra_log))
    C1_errors = np.where(use_assign1,
                         np.abs(pred_C1_log - true_Ca_log),
                         np.abs(pred_C1_log - true_Cb_log))
    C2_errors = np.where(use_assign1,
                         np.abs(pred_C2_log - true_Cb_log),
                         np.abs(pred_C2_log - true_Ca_log))

    # Rsh is always matched to Rsh
    Rsh_errors = np.abs(np.log10(pred_Rsh + 1e-12) - np.log10(true_Rsh + 1e-12))

    return {
        'assignment_1_fraction': float(np.mean(use_assign1)),
        'R1_mae': float(np.mean(R1_errors)),
        'R2_mae': float(np.mean(R2_errors)),
        'C1_mae': float(np.mean(C1_errors)),
        'C2_mae': float(np.mean(C2_errors)),
        'Rsh_mae': float(np.mean(Rsh_errors)),
        'total_mae': float(np.mean(R1_errors + R2_errors + C1_errors + C2_errors + Rsh_errors) / 5),
    }


class SymmetricEvaluator:
    """
    Evaluator that handles parameter symmetry for EIS models.

    Provides fair evaluation metrics that don't penalize models
    for arbitrary apical/basolateral labeling choices.
    """

    def __init__(self, log_space: bool = True):
        self.log_space = log_space
        self.results = []

    def evaluate_batch(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions with symmetric metrics.
        """
        # Convert to numpy
        preds_np = {k: v.detach().cpu().numpy() for k, v in predictions.items()}
        targs_np = {k: v.detach().cpu().numpy() for k, v in targets.items()}

        # Compute symmetric metrics
        metrics = compute_symmetric_mae_from_model_output(
            preds_np, targs_np, self.log_space
        )

        self.results.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get aggregate metrics across all batches."""
        if not self.results:
            return {}

        summary = {
            'R1_mae_mean': np.mean([r['R1_mae'] for r in self.results]),
            'C1_mae_mean': np.mean([r['C1_mae'] for r in self.results]),
            'R2_mae_mean': np.mean([r['R2_mae'] for r in self.results]),
            'R1_assignment_Ra_mean': np.mean([r['R1_assignment_Ra_fraction'] for r in self.results]),
        }

        return summary

    def reset(self):
        """Reset accumulated results."""
        self.results = []


def compute_standard_vs_symmetric_comparison(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compare standard MAE (fixed assignment) vs symmetric MAE (optimal assignment).

    This shows how much of the apparent "error" is due to arbitrary labeling.

    Returns:
        Dict with:
            - 'standard_Ra_mae': Error assuming pred_Ra should match true_Ra
            - 'symmetric_R1_mae': Error with optimal Ra/Rb assignment
            - 'improvement_R1': Reduction in MAE from symmetric evaluation
            (same for C, Rsh)
    """
    # Standard MAE (fixed assignment)
    std_Ra_mae = np.mean(np.abs(
        np.log10(predictions['Ra'] + 1e-12) - np.log10(targets['Ra'] + 1e-12)
    ))
    std_Rb_mae = np.mean(np.abs(
        np.log10(predictions['Rb'] + 1e-12) - np.log10(targets['Rb'] + 1e-12)
    ))
    std_Ca_mae = np.mean(np.abs(
        np.log10(predictions['Ca'] + 1e-12) - np.log10(targets['Ca'] + 1e-12)
    ))
    std_Cb_mae = np.mean(np.abs(
        np.log10(predictions['Cb'] + 1e-12) - np.log10(targets['Cb'] + 1e-12)
    ))
    std_Rsh_mae = np.mean(np.abs(
        np.log10(predictions['Rsh'] + 1e-12) - np.log10(targets['Rsh'] + 1e-12)
    ))

    # Symmetric MAE
    sym_metrics = compute_symmetric_mae_from_model_output(predictions, targets)

    return {
        'standard_Ra_mae': float(std_Ra_mae),
        'standard_Rb_mae': float(std_Rb_mae),
        'standard_Ca_mae': float(std_Ca_mae),
        'standard_Cb_mae': float(std_Cb_mae),
        'standard_Rsh_mae': float(std_Rsh_mae),
        'symmetric_R1_mae': sym_metrics['R1_mae'],
        'symmetric_C1_mae': sym_metrics['C1_mae'],
        'symmetric_R2_mae': sym_metrics['R2_mae'],
        'R1_improvement': float(std_Ra_mae - sym_metrics['R1_mae']),
        'C1_improvement': float(std_Ca_mae - sym_metrics['C1_mae']),
    }
