"""Grid search for hyperparameter tuning on CV Fold 1, evaluated on all 5 folds."""
import itertools
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import *
from models.training import (
    train_focal_loss,
    _safe_best_iteration,
    _fit_focal_loss_grid,
)
from split import split_train_val_test_by_season
from feature_engineering import resolve_feature_set, _build_all_features

def _exp_prefix(args) -> str:
    tag = (getattr(args, "experiment_tag", None) or "").strip()
    return f"exp_{tag}_" if tag else ""

def _save_best_params(path, params, tune_arm, best_val_pr_auc, refit_rounds, experiment_tag=None):
    payload = {
        "experiment_tag": experiment_tag,
        "tune_arm": tune_arm,
        "best_val_pr_auc": best_val_pr_auc,
        "refit_rounds": refit_rounds,
        "params": params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[params] saved -> {path}")

GRID_SEARCH_SPACES = {
    "quick": {
        "max_depth": [2, 3],
        "learning_rate": [0.01, 0.03, 0.05],
        "reg_lambda": [8.0, 16.0, 32.0, 64.0],
        "reg_alpha": [0.5, 1.0, 2.0, 4.0],
        "min_child_weight": [10, 20, 50, 100],
        "subsample": [0.5, 0.6, 0.7, 0.8],
        "colsample_bytree": [0.3, 0.5],
        "gamma_fl": [1.5, 2.0, 2.5],
        "alpha_fl": [0.75, 0.9],
    },
    "full": {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_lambda": [1.0, 2.0, 4.0, 8.0, 16.0],
        "reg_alpha": [0.1, 0.5, 1.0, 2.0],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.5, 0.6, 0.7, 0.8],
        "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
        "gamma_fl": [1.0, 1.5, 2.0, 2.5],
        "alpha_fl": [0.5, 0.75, 0.9],
    },
}

def _grid_product(space: dict) -> list[dict]:
    keys = list(space.keys())
    combos = []
    for values in itertools.product(*(space[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos

def _random_sample(space: dict, n: int, seed: int = 42) -> list[dict]:
    """Randomly sample n combinations from the search space."""
    import random
    random.seed(seed)
    keys = list(space.keys())
    combos = []
    seen = set()
    max_attempts = n * 20
    attempts = 0
    while len(combos) < n and attempts < max_attempts:
        combo = tuple(random.choice(space[k]) for k in keys)
        if combo not in seen:
            seen.add(combo)
            combos.append(dict(zip(keys, combo)))
        attempts += 1
    return combos

def _arm_to_max_stage(arm: str) -> int:
    """Map arm name to the max feature stage needed."""
    if arm == "legacy":
        return 4  # legacy uses features from L0-L3
    if arm.startswith("stage"):
        return int(arm.replace("stage", ""))
    return 6

def _grid_tune_on_fold1(df, cols, combos, args, n_estimators, early_stopping, trials_out):
    """在 Fold 1 训练数据上做网格搜索（赛季级 train/val）。"""
    train_seasons, val_season, test_season = CV_FOLDS[0]

    # 按赛季切分 train/val（test 不用于调参）
    X_tr, X_val, _, y_tr, y_val, _ = split_train_val_test_by_season(
        df, cols, train_seasons, val_season, test_season
    )

    # StandardScaler（与 CV 保持一致）
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    print(f"  train_inner={len(X_tr):,}  val={len(X_val):,}")
    print(f"  grid combos: {len(combos)}")

    trial_rows = []
    best_val_pr = -1.0
    best_trial = None
    best_booster = None
    t0 = time.time()

    for i, combo in enumerate(combos, 1):
        booster, val_pr = _fit_focal_loss_grid(
            X_tr, y_tr, X_val, y_val, combo,
            n_estimators=n_estimators,
            early_stopping=early_stopping,
            warmup_rounds=args.fl_warmup,
        )
        row = {**combo, "val_pr_auc": val_pr, "trial": i}
        if booster is not None:
            row["best_iteration"] = _safe_best_iteration(booster)
        trial_rows.append(row)
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_trial = combo
            best_booster = booster
        if i % max(1, len(combos) // 10) == 0 or i == len(combos):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(combos) - i) / max(rate, 1e-6)
            print(f"  [{i}/{len(combos)}] best_val_pr={best_val_pr:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    trials_df = pd.DataFrame(trial_rows).sort_values("val_pr_auc", ascending=False)
    trials_df.to_csv(trials_out, index=False)
    print(f"  all trials -> {trials_out}")

    if best_trial is None or best_booster is None:
        print("  [grid] no successful trial; returning defaults")
        return {"gamma_fl": 2.0, "alpha_fl": 0.75}, best_val_pr

    n_rounds = max(_safe_best_iteration(best_booster) + 1, 50)
    best_trial["refit_rounds"] = n_rounds
    print(f"\n  best val PR-AUC={best_val_pr:.4f}  rounds={n_rounds}")
    print(f"  best params: {best_trial}")

    return best_trial, best_val_pr

def _evaluate_params_on_cv(df, cols, hp, args):
    """用给定超参数在所有 5 折上评估 PR-AUC（赛季级 val）。"""
    fold_results = []
    for fold_idx, (train_seasons, val_season, test_season) in enumerate(CV_FOLDS, 1):
        X_train, X_val, X_test, y_train, y_val, y_test = \
            split_train_val_test_by_season(
                df, cols, train_seasons, val_season, test_season
            )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s   = scaler.transform(X_val).astype(np.float32)
        X_test_s  = scaler.transform(X_test).astype(np.float32)

        _, metrics = train_focal_loss(
            X_train_s, y_train, X_test_s, y_test,
            X_val=X_val_s, y_val=y_val,
            gamma=args.fl_gamma, alpha=args.fl_alpha,
            warmup_rounds=args.fl_warmup,
            tuned_hp=hp,
        )
        fold_results.append(metrics["pr_auc"])
        print(f"  Fold {fold_idx} PR-AUC: {metrics['pr_auc']:.6f}")

    mean_pr = np.mean(fold_results)
    std_pr = np.std(fold_results, ddof=1)
    return fold_results, mean_pr, std_pr

def _run_grid_search(df: pd.DataFrame, args) -> None:
    """Grid search on Fold 1, evaluate best params on all 5 CV folds."""
    prefix = _exp_prefix(args)

    if args.tune_on:
        arms = [args.tune_on]
    else:
        arms = ["legacy", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]

    grid_mode = getattr(args, "grid_mode", "quick")
    space = GRID_SEARCH_SPACES[grid_mode]
    n_trials = getattr(args, "n_trials", None)

    if n_trials:
        combos = _random_sample(space, n_trials)
        search_mode = f"randomized ({len(combos)} trials)"
    else:
        combos = _grid_product(space)
        search_mode = f"grid ({len(combos)} combos)"

    n_estimators = getattr(args, "n_estimators", 500)
    early_stopping = getattr(args, "grid_early_stop", 30)

    print("\n" + "=" * 60)
    print(f"GRID SEARCH — Focal Loss ({grid_mode}, {search_mode}/arm)")
    print("=" * 60)
    train_s, val_s, test_s = CV_FOLDS[0]
    print(f"  Tuning on Fold 1: train {train_s}  |  val {val_s}")
    print(f"  Evaluating on all 5 CV folds")
    print(f"  Arms: {arms}")

    max_st = max(_arm_to_max_stage(a) for a in arms)
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=max_st)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for arm in arms:
        cols, desc = resolve_feature_set(arm)
        print(f"\n{'=' * 60}")
        print(f"GRID -- {arm} ({len(cols)} features: {desc})")
        print("=" * 60)

        # Phase 1: Grid search on Fold 1
        trials_out = out_dir / f"{prefix}grid_search_trials_{arm}.csv"
        best_trial, best_val_pr = _grid_tune_on_fold1(
            df_feat, cols, combos, args,
            n_estimators, early_stopping, trials_out,
        )

        # Phase 2: Evaluate on all 5 folds
        hp = {**best_trial, "refit_rounds": max(
            best_trial.get("refit_rounds", 500), 50
        )}
        if "refit_rounds" not in hp:
            hp["refit_rounds"] = 500

        print(f"\n  Evaluating tuned params on all 5 CV folds...")
        fold_results, mean_pr, std_pr = _evaluate_params_on_cv(
            df_feat, cols, hp, args
        )

        print(f"\n  {arm} tuned CV: {mean_pr:.6f} +/- {std_pr:.6f}")
        for fi, pr in enumerate(fold_results, 1):
            summary_rows.append({
                "arm": arm, "n_features": len(cols), "fold": fi,
                "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr,
            })
        summary_rows.append({
            "arm": arm, "n_features": len(cols), "fold": "mean",
            "test_season": "all", "pr_auc": mean_pr,
        })
        summary_rows.append({
            "arm": arm, "n_features": len(cols), "fold": "std",
            "test_season": "all", "pr_auc": std_pr,
        })

        # Save best params
        params_path = out_dir / f"{prefix}grid_best_params_{arm}.json"
        _save_best_params(
            params_path, best_trial,
            tune_arm=arm,
            best_val_pr_auc=best_val_pr,
            refit_rounds=hp["refit_rounds"],
            experiment_tag=args.experiment_tag,
        )

    summary_path = out_dir / f"{prefix}grid_search_cv_results.csv"
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print(f"\nGrid CV summary -> {summary_path}")

    pivot = summary[summary["fold"].isin(["mean", "std"])].pivot(
        index="arm", columns="fold", values="pr_auc"
    )
    print("\nGrid Search CV Summary (PR-AUC mean +/- std):")
    print(pivot.to_string())
