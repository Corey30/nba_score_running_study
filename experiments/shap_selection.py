"""SHAP feature selection: SHAP top-26 (selected) vs legacy-26."""
import json
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from config import *
from split import split_train_val_test_by_season
from feature_engineering import resolve_feature_set, _build_all_features
from models.training import train_focal_loss
from experiments.grid_search import (
    GRID_SEARCH_SPACES,
    _exp_prefix,
    _random_sample,
    _save_best_params,
    _grid_tune_on_fold1,
    _evaluate_params_on_cv,
)

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    shap = None
    SHAP_AVAILABLE = False

N_SELECTED = 26
SHAP_SAMPLE = 5000  # subsample rows for SHAP to avoid OOM


def _load_stage7_params(out_dir: Path, prefix: str):
    path = out_dir / f"{prefix}grid_best_params_stage7.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    hp = dict(payload.get("params", {}))
    if "refit_rounds" in payload:
        hp["refit_rounds"] = payload["refit_rounds"]
    print(f"[params] loaded from {path}")
    return hp


def _append_arm_rows(rows, arm, fold_results, mean, std):
    for fi, pr in enumerate(fold_results, 1):
        rows.append({"arm": arm, "fold": fi, "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr})
    rows.append({"arm": arm, "fold": "mean", "test_season": "all", "pr_auc": mean})
    rows.append({"arm": arm, "fold": "std", "test_season": "all", "pr_auc": std})


def _run_shap_selection(df, args) -> None:
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    prefix = _exp_prefix(args)
    full_cols, _ = resolve_feature_set("stage7")
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=6)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trials = getattr(args, "n_trials", None) or 50
    grid_mode = getattr(args, "grid_mode", "quick")
    n_estimators = getattr(args, "n_estimators", 500)
    early_stopping = getattr(args, "grid_early_stop", 30)

    hp = _load_stage7_params(out_dir, prefix)

    print("\n" + "=" * 60)
    print("SHAP FEATURE SELECTION — stage7 -> top-26 (selected) vs legacy-26 vs full")
    print("=" * 60)
    print(f"  Full features: {len(full_cols)}  |  select top: {N_SELECTED}")
    print(f"  SHAP: TreeExplainer on {SHAP_SAMPLE} test rows/fold (subsampled to avoid OOM)")

    importance_rows = []
    per_fold_importance = []
    for fold_idx, (train_seasons, val_season, test_season) in enumerate(CV_FOLDS, 1):
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_by_season(
            df_feat, full_cols, train_seasons, val_season, test_season
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        booster, _ = train_focal_loss(
            X_train_s, y_train, X_test_s, y_test,
            X_val=X_val_s, y_val=y_val,
            tuned_hp=hp,
        )

        n = min(SHAP_SAMPLE, len(X_test_s))
        sample = X_test_s[:n]
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class-1 for binary
        shap_vals = shap_vals.reshape(n, len(full_cols))
        imp = np.abs(shap_vals).mean(axis=0)
        per_fold_importance.append(imp)
        for name, v in zip(full_cols, imp):
            importance_rows.append({"fold": fold_idx, "feature": name, "importance": float(v)})
        print(f"  Fold {fold_idx}: SHAP importance computed on {n} samples")

    importance_out = out_dir / f"{prefix}shap_feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(importance_out, index=False)
    print(f"\nSHAP importance -> {importance_out}")

    imp_matrix = np.vstack(per_fold_importance)
    mean_imp = imp_matrix.mean(axis=0)
    order = np.argsort(mean_imp)[::-1]
    selected = [full_cols[i] for i in order[:N_SELECTED]]

    selected_out = out_dir / f"{prefix}selected_26_features.json"
    selected_out.write_text(
        json.dumps({"n_features": len(selected), "features": selected}, indent=2),
        encoding="utf-8",
    )
    print(f"Selected top-{N_SELECTED} features -> {selected_out}")

    print("\n" + "=" * 60)
    print("selected-26: grid search + 5-fold CV")
    print("=" * 60)
    space = GRID_SEARCH_SPACES[grid_mode]
    sel_best, sel_val = _grid_tune_on_fold1(
        df_feat, selected, _random_sample(space, n_trials), args,
        n_estimators, early_stopping,
        out_dir / f"{prefix}grid_search_trials_selected.csv",
    )
    sel_hp = {**sel_best, "refit_rounds": max(sel_best.get("refit_rounds", 500), 50)}
    if "refit_rounds" not in sel_hp:
        sel_hp["refit_rounds"] = 500
    sel_folds, sel_mean, sel_std = _evaluate_params_on_cv(df_feat, selected, sel_hp, args)
    _save_best_params(
        out_dir / f"{prefix}grid_best_params_selected.json", sel_best,
        tune_arm="selected", best_val_pr_auc=sel_val,
        refit_rounds=sel_hp["refit_rounds"], experiment_tag=args.experiment_tag,
    )

    print("\n" + "=" * 60)
    print("legacy-26: grid search + 5-fold CV")
    print("=" * 60)
    legacy_cols, _ = resolve_feature_set("legacy")
    leg_best, leg_val = _grid_tune_on_fold1(
        df_feat, legacy_cols, _random_sample(space, n_trials), args,
        n_estimators, early_stopping,
        out_dir / f"{prefix}grid_search_trials_legacy.csv",
    )
    leg_hp = {**leg_best, "refit_rounds": max(leg_best.get("refit_rounds", 500), 50)}
    if "refit_rounds" not in leg_hp:
        leg_hp["refit_rounds"] = 500
    leg_folds, leg_mean, leg_std = _evaluate_params_on_cv(df_feat, legacy_cols, leg_hp, args)

    print("\n" + "=" * 60)
    print("full (stage7-141): 5-fold CV (reuse tuned params)")
    print("=" * 60)
    if hp is None:
        raise FileNotFoundError(
            "stage7 tuned params not found; run grid search / model-compare first"
        )
    full_hp = {**hp, "refit_rounds": hp.get("refit_rounds", 500)}
    full_folds, full_mean, full_std = _evaluate_params_on_cv(df_feat, full_cols, full_hp, args)

    comp_rows = []
    _append_arm_rows(comp_rows, "selected", sel_folds, sel_mean, sel_std)
    _append_arm_rows(comp_rows, "legacy", leg_folds, leg_mean, leg_std)
    _append_arm_rows(comp_rows, "full", full_folds, full_mean, full_std)

    comp_out = out_dir / f"{prefix}shap_selection_results.csv"
    pd.DataFrame(comp_rows).to_csv(comp_out, index=False)
    print(f"\nSHAP selection results -> {comp_out}")
    print(f"  selected (top-{N_SELECTED}): {sel_mean:.6f} +/- {sel_std:.6f}")
    print(f"  legacy  (26-dim):          {leg_mean:.6f} +/- {leg_std:.6f}")
    print(f"  full    (stage7-141):      {full_mean:.6f} +/- {full_std:.6f}")
