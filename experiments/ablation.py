"""Ablation experiment (alias for cross-validation with tuned params)."""
from experiments.cross_validation import _run_cross_validation

def _run_ablation(df, args):
    """Feature ablation via 5-fold CV — delegates to _run_cross_validation."""
    _run_cross_validation(df, args)
