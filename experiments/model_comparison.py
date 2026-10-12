"""Three-model comparison: XGBoost(FL) vs LightGBM(FL) vs CatBoost(scale_pos_weight).

All three models are independently tuned (random search on Fold 1, 50 trials by
default) and then evaluated with 5-fold temporal CV on stage7 (full + team-prior) features.
"""
import json
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

from config import *
from feature_engineering import resolve_feature_set, _build_all_features
from experiments.grid_search import (
    GRID_SEARCH_SPACES,
    _exp_prefix,
    _random_sample,
    _save_best_params,
    _grid_tune_on_fold1,
    _evaluate_params_on_cv,
)
from models.lgb_trainer import lgb_grid_search, lgb_cv_evaluate
from models.cb_trainer import cb_grid_search, cb_cv_evaluate

warnings.filterwarnings("ignore")


def _save_model_params(path: Path, hp: dict, best_val_pr_auc=None):
    payload = {"params": hp}
    if best_val_pr_auc is not None:
        payload["best_val_pr_auc"] = best_val_pr_auc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[params] saved -> {path}")


def _append_summary(rows, model, fold_results, mean, std):
    for fi, pr in enumerate(fold_results, 1):
        rows.append({"model": model, "fold": fi, "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr})
    rows.append({"model": model, "fold": "mean", "test_season": "all", "pr_auc": mean})
    rows.append({"model": model, "fold": "std", "test_season": "all", "pr_auc": std})


def _run_model_comparison(df, args) -> None:
    prefix = _exp_prefix(args)
    cols, desc = resolve_feature_set("stage7")
    # max_stage=6 computes all in-game rolling features; the team-prior
    # stage7 features are already attached to df during preprocessing.
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=6)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trials = getattr(args, "n_trials", None) or 50
    grid_mode = getattr(args, "grid_mode", "quick")
    n_estimators = getattr(args, "n_estimators", 500)
    early_stopping = getattr(args, "grid_early_stop", 30)

    print("\n" + "=" * 60)
    print("THREE-MODEL COMPARISON — XGBoost vs LightGBM vs CatBoost (stage7)")
    print("=" * 60)
    print(f"  Features: {len(cols)} ({desc})")
    print(f"  Tuning: randomized search, {n_trials} trials/model  |  mode: {grid_mode}")

    summary_rows = []

    print("\n" + "=" * 60)
    print("XGBoost (Focal Loss)")
    print("=" * 60)
    space = GRID_SEARCH_SPACES[grid_mode]
    xgb_combos = _random_sample(space, n_trials)
    trials_out = out_dir / f"{prefix}grid_search_trials_stage7.csv"
    best_trial, best_val_pr = _grid_tune_on_fold1(
        df_feat, cols, xgb_combos, args, n_estimators, early_stopping, trials_out
    )
    hp = {**best_trial, "refit_rounds": max(best_trial.get("refit_rounds", 500), 50)}
    if "refit_rounds" not in hp:
        hp["refit_rounds"] = 500
    xgb_folds, xgb_mean, xgb_std = _evaluate_params_on_cv(df_feat, cols, hp, args)
    _save_best_params(
        out_dir / f"{prefix}grid_best_params_stage7.json", best_trial,
        tune_arm="stage7", best_val_pr_auc=best_val_pr,
        refit_rounds=hp["refit_rounds"], experiment_tag=args.experiment_tag,
    )
    _append_summary(summary_rows, "XGBoost", xgb_folds, xgb_mean, xgb_std)
    print(f"\n  XGBoost tuned CV: {xgb_mean:.6f} +/- {xgb_std:.6f}")

    print("\n" + "=" * 60)
    print("LightGBM (Focal Loss)")
    print("=" * 60)
    lgb_hp = lgb_grid_search(
        df_feat, cols, n_trials, args,
        trials_out=out_dir / f"{prefix}lgb_search_trials_stage7.csv",
    )
    lgb_folds, lgb_mean, lgb_std = lgb_cv_evaluate(df_feat, cols, lgb_hp, args)
    _save_model_params(out_dir / f"{prefix}lgb_best_params_stage7.json", lgb_hp)
    _append_summary(summary_rows, "LightGBM", lgb_folds, lgb_mean, lgb_std)
    print(f"\n  LightGBM tuned CV: {lgb_mean:.6f} +/- {lgb_std:.6f}")

    print("\n" + "=" * 60)
    print("CatBoost (Logloss + scale_pos_weight)")
    print("=" * 60)
    cb_hp = cb_grid_search(
        df_feat, cols, n_trials, args,
        trials_out=out_dir / f"{prefix}cb_search_trials_stage7.csv",
    )
    cb_folds, cb_mean, cb_std = cb_cv_evaluate(df_feat, cols, cb_hp, args)
    _save_model_params(out_dir / f"{prefix}cb_best_params_stage7.json", cb_hp)
    _append_summary(summary_rows, "CatBoost", cb_folds, cb_mean, cb_std)
    print(f"\n  CatBoost tuned CV: {cb_mean:.6f} +/- {cb_std:.6f}")

    summary = pd.DataFrame(summary_rows)
    out = out_dir / f"{prefix}model_comparison_results.csv"
    summary.to_csv(out, index=False)
    print(f"\nModel comparison results -> {out}")

    pivot = summary[summary["fold"].isin(["mean", "std"])].pivot(
        index="model", columns="fold", values="pr_auc"
    )
    print("\nModel Comparison (PR-AUC mean +/- std):")
    print(pivot.to_string())
