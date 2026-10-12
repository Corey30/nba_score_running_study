"""Focal Loss custom objective for XGBoost."""

import numpy as np
from config import *

def focal_loss_objective(preds: np.ndarray, dtrain, gamma: float, alpha: float):
    """XGBoost custom objective: Focal Loss with (gamma, alpha).

    Returns (grad, hess) for Newton-Raphson leaf-weight optimization.
    Uses a heuristic positive curvature surrogate for numerical stability.
    """
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))          # sigmoid
    p = np.clip(p, 1e-16, 1.0 - 1e-16)
    p_neg = 1.0 - p

    log_p     = np.log(p)
    log_1_minus_p = np.log(p_neg)

    g_pos = alpha * np.power(p_neg, gamma) * (p + gamma * p * log_p - 1.0)
    g_neg = (1.0 - alpha) * np.power(p, gamma) * (p - gamma * p_neg * log_1_minus_p)
    grad  = np.where(labels == 1, g_pos, g_neg)
    grad  = np.clip(grad, -5.0, 5.0)

    hess_fl = np.abs(grad) * np.abs(1.0 - 2.0 * p)
    hess_ce = p * p_neg
    hess    = np.maximum(hess_fl, 0.3 * hess_ce)
    hess    = np.maximum(hess, 1e-4)

    return grad, hess

def make_focal_loss(gamma: float, alpha: float):
    """Factory returning a callable ``f(preds, dtrain)`` for XGBoost ``obj``."""
    def _obj(preds, dtrain):
        return focal_loss_objective(preds, dtrain, gamma, alpha)
    return _obj
