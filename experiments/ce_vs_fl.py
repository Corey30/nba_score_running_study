"""Compare independently tuned CE and Focal Loss on stage7 features.

Both objectives receive the same trial budget and tree search space. Each tunes
its own loss parameters on Fold 1's validation season, then uses validation-only
early stopping in temporal CV. Export pooled test predictions for PR plotting.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from config import CV_FOLDS, ROLLING_WINDOW, USE_GPU
from split import split_train_val_test_by_season
from feature_engineering import resolve_feature_set, _build_all_features
from models.training import (
    _xgb_params,
    _booster_pr_auc,
    pr_auc_eval,
    _fit_focal_loss_grid,
    SHARED_PARAMS,
    _safe_best_iteration,
)
from models.focal_loss import make_focal_loss
from experiments.grid_search import _random_sample

warnings.filterwarnings("ignore")

# Shared tree search space (same for both objectives -> fair comparison).
TREE_SPACE = {
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.03, 0.05],
    "reg_lambda": [8.0, 16.0, 32.0, 64.0],
    "reg_alpha": [0.5, 1.0, 2.0, 4.0],
    "min_child_weight": [10, 20, 50, 100],
    "subsample": [0.5, 0.6, 0.7, 0.8],
    "colsample_bytree": [0.3, 0.5],
}
FL_OBJ_SPACE = {"gamma_fl": [1.5, 2.0, 2.5], "alpha_fl": [0.75, 0.9]}
CE_OBJ_SPACE = {"scale_pos_weight": [30, 40, 50, 65, 80, 100]}

TREE_KEYS = ("max_depth", "learning_rate", "reg_lambda", "reg_alpha",
             "min_child_weight", "subsample", "colsample_bytree")


def _exp_prefix(args) -> str:
    tag = (getattr(args, "experiment_tag", None) or "").strip()
    return f"exp_{tag}_" if tag else ""


def _tree_overrides(trial: dict) -> dict:
    o = {k: trial[k] for k in TREE_KEYS if k in trial}
    o.setdefault("subsample", SHARED_PARAMS["subsample"])
    o.setdefault("colsample_bytree", SHARED_PARAMS["colsample_bytree"])
    return o


def _fit_fl_trial(X_train, y_train, X_val, y_val, trial, *, n_estimators, early_stopping):
    return _fit_focal_loss_grid(
        X_train, y_train, X_val, y_val, trial,
        n_estimators=n_estimators, early_stopping=early_stopping, warmup_rounds=0,
    )


def _fit_ce_trial(X_train, y_train, X_val, y_val, trial, *, n_estimators, early_stopping):
    pos_rate = float(y_train.mean())
    params = _xgb_params({
        **_tree_overrides(trial),
        "objective": "binary:logistic",
        "scale_pos_weight": trial["scale_pos_weight"],
        "base_score": pos_rate,
    }, use_gpu=USE_GPU)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    try:
        booster = xgb.train(
            params, dtrain, num_boost_round=n_estimators,
            evals=[(dval, "val")], custom_metric=pr_auc_eval,
            early_stopping_rounds=early_stopping, maximize=True, verbose_eval=False,
        )
    except Exception:
        return None, 0.0
    return booster, _booster_pr_auc(booster, X_val, y_val)


def _tune_on_fold1(X_tr, y_tr, X_val, y_val, combos, fit_fn, n_estimators, early_stopping, label):
    best_val, best_trial = -1.0, None
    t0 = time.time()
    for i, combo in enumerate(combos, 1):
        booster, val_pr = fit_fn(
            X_tr, y_tr, X_val, y_val, combo,
            n_estimators=n_estimators, early_stopping=early_stopping,
        )
        if booster is not None and np.isfinite(val_pr) and val_pr > best_val:
            best_val, best_trial = val_pr, combo
        if i % 10 == 0 or i == len(combos):
            print(f"  [{label}] {i}/{len(combos)}  best_val={best_val:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    if best_trial is None:
        raise RuntimeError(f"All {label} tuning trials failed; check data and device configuration.")
    print(f"  [{label}] best val PR-AUC={best_val:.4f}  params={best_trial}", flush=True)
    return best_trial, best_val


def _predict_proba(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    # Both CE and FL use sigmoid(raw margin); retain the validation-best trees.
    margins = booster.predict(
        xgb.DMatrix(X), output_margin=True,
        iteration_range=(0, _safe_best_iteration(booster) + 1),
    )
    return 1.0 / (1.0 + np.exp(-np.clip(margins, -35.0, 35.0)))


def _eval_fl_fold(X_train, y_train, X_val, y_val, X_test, y_test, trial, n_rounds, early_stopping):
    pos_rate = float(y_train.mean())
    params = _xgb_params({**_tree_overrides(trial), "base_score": pos_rate}, use_gpu=USE_GPU)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params, dtrain, num_boost_round=n_rounds,
        evals=[(dval, "val")],
        obj=make_focal_loss(gamma=trial["gamma_fl"], alpha=trial["alpha_fl"]),
        custom_metric=pr_auc_eval, early_stopping_rounds=early_stopping,
        maximize=True, verbose_eval=False,
    )
    probabilities = _predict_proba(booster, X_test)
    return float(average_precision_score(y_test, probabilities)), probabilities


def _eval_ce_fold(X_train, y_train, X_val, y_val, X_test, y_test, trial, n_rounds, early_stopping):
    pos_rate = float(y_train.mean())
    params = _xgb_params({
        **_tree_overrides(trial),
        "objective": "binary:logistic",
        "scale_pos_weight": trial["scale_pos_weight"],
        "base_score": pos_rate,
    }, use_gpu=USE_GPU)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    booster = xgb.train(
        params, dtrain, num_boost_round=n_rounds,
        evals=[(dval, "val")], custom_metric=pr_auc_eval,
        early_stopping_rounds=early_stopping, maximize=True, verbose_eval=False,
    )
    probabilities = _predict_proba(booster, X_test)
    return float(average_precision_score(y_test, probabilities)), probabilities


def _run_ce_vs_fl(df, args):
    prefix = _exp_prefix(args)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cols, desc = resolve_feature_set("stage7")
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=6)

    n_trials = int(getattr(args, "n_trials", None) or 50)
    n_estimators = int(getattr(args, "n_estimators", 500))
    early_stopping = int(getattr(args, "grid_early_stop", 30))

    print("\n" + "=" * 64)
    print("E1 INDEPENDENT TUNING — CE vs FOCAL LOSS (stage7)")
    print("=" * 64)
    print(f"  features: {len(cols)} ({desc})")
    print(f"  n_trials={n_trials}/objective  n_estimators={n_estimators}  "
          f"early_stop={early_stopping}")

    tr_s, val_s, te_s = CV_FOLDS[0]
    X_tr, X_val, _, y_tr, y_val, _ = split_train_val_test_by_season(
        df_feat, cols, tr_s, val_s, te_s
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_val_s = sc.transform(X_val).astype(np.float32)

    fl_combos = _random_sample({**TREE_SPACE, **FL_OBJ_SPACE}, n_trials, seed=42)
    ce_combos = _random_sample({**TREE_SPACE, **CE_OBJ_SPACE}, n_trials, seed=42)

    print(f"\n  --- Fold-1 tuning (train {tr_s} | val {val_s}) ---", flush=True)
    fl_trial, fl_val = _tune_on_fold1(
        X_tr_s, y_tr, X_val_s, y_val, fl_combos, _fit_fl_trial,
        n_estimators, early_stopping, "FL")
    ce_trial, ce_val = _tune_on_fold1(
        X_tr_s, y_tr, X_val_s, y_val, ce_combos, _fit_ce_trial,
        n_estimators, early_stopping, "CE")

    print("\n  --- 5-fold held-out evaluation ---", flush=True)
    rows = []
    fl_vals, ce_vals = [], []
    fl_probs, ce_probs, y_trues = [], [], []
    for fold_idx, (train_s, val_s, test_s) in enumerate(CV_FOLDS, 1):
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_by_season(
            df_feat, cols, train_s, val_s, test_s
        )
        sc2 = StandardScaler()
        X_train_s = sc2.fit_transform(X_train).astype(np.float32)
        X_val_s = sc2.transform(X_val).astype(np.float32)
        X_test_s = sc2.transform(X_test).astype(np.float32)

        fl_pr, fl_prob = _eval_fl_fold(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                              fl_trial, n_estimators, early_stopping)
        ce_pr, ce_prob = _eval_ce_fold(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                              ce_trial, n_estimators, early_stopping)
        fl_probs.append(fl_prob)
        ce_probs.append(ce_prob)
        y_trues.append(np.asarray(y_test, dtype=np.int64))
        fl_vals.append(fl_pr)
        ce_vals.append(ce_pr)
        rows.append({"fold": fold_idx, "test_season": test_s,
                     "FL_pr_auc": fl_pr, "CE_pr_auc": ce_pr})
        print(f"  Fold {fold_idx} (test {test_s}): FL={fl_pr:.6f}  CE={ce_pr:.6f}", flush=True)

    fl_mean, fl_std = float(np.mean(fl_vals)), float(np.std(fl_vals, ddof=1))
    ce_mean, ce_std = float(np.mean(ce_vals)), float(np.std(ce_vals, ddof=1))
    rows.append({"fold": "mean", "test_season": "all", "FL_pr_auc": fl_mean, "CE_pr_auc": ce_mean})
    rows.append({"fold": "std", "test_season": "all", "FL_pr_auc": fl_std, "CE_pr_auc": ce_std})

    out = out_dir / f"{prefix}ce_vs_fl_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    pred_out = out_dir / f"{prefix}ce_vs_fl_predictions.npz"
    np.savez_compressed(
        pred_out, y_true=np.concatenate(y_trues),
        ce_prob=np.concatenate(ce_probs), fl_prob=np.concatenate(fl_probs),
    )

    meta = {
        "features": desc,
        "n_trials": n_trials,
        "n_estimators": n_estimators,
        "early_stopping": early_stopping,
        "FL_best_val_pr_auc": fl_val,
        "FL_best_params": fl_trial,
        "CE_best_val_pr_auc": ce_val,
        "CE_best_params": ce_trial,
    }
    (out_dir / f"{prefix}ce_vs_fl_best_params.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"  FL (gamma/alpha tuned): {fl_mean:.6f} +/- {fl_std:.6f}")
    print(f"  CE (spw tuned):         {ce_mean:.6f} +/- {ce_std:.6f}")
    print(f"  FL - CE = {fl_mean - ce_mean:+.6f}")
    print(f"  results -> {out}")
    print(f"  predictions -> {pred_out}")
    print("=" * 64)
