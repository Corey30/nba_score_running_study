"""Model training: baseline CE, Focal Loss XGBoost."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from config import *
from models.focal_loss import make_focal_loss

SHARED_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "verbosity": 0,
    "seed": 42,
    "disable_default_eval_metric": 1,
}

XGB_NUM_BOOST_ROUND = 500
XGB_EARLY_STOPPING = 30

def _xgb_params(extra: dict | None = None, use_gpu: bool = True) -> dict:
    p = {**SHARED_PARAMS, **xgb_device_params(use_gpu)}
    if extra:
        p.update(extra)
    return p

def _logloss_eval(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    ll = -(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean()
    return "logloss", ll

def _safe_best_iteration(booster: xgb.Booster) -> int:
    try:
        return int(booster.best_iteration)
    except AttributeError:
        return max(0, int(booster.num_boosted_rounds()) - 1)

def evaluate_model(booster: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray,
                   objective_name: str) -> dict:
    y_score = booster.predict(xgb.DMatrix(X_test))
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=10.0, neginf=-10.0)
    if y_score.min() < -0.05 or y_score.max() > 1.05:
        y_prob = 1.0 / (1.0 + np.exp(-y_score))
    else:
        y_prob = y_score
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_thresh = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "model": objective_name,
        "objective": objective_name,
        "best_threshold": best_thresh,
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "recall_pos": recall_score(y_test, y_pred, pos_label=1),
        "f1_pos": f1_score(y_test, y_pred, pos_label=1),
        "y_prob_mean": float(y_prob.mean()),
        "y_pred_pos_rate": float(y_pred.mean()),
        "best_iteration": _safe_best_iteration(booster),
    }

    print(f"\n  Best threshold:     {best_thresh:.4f}")
    print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Recall (y=1):       {metrics['recall_pos']:.4f}")
    print(f"  F1 (y=1):           {metrics['f1_pos']:.4f}")
    print(f"  Predicted pos rate: {metrics['y_pred_pos_rate']:.4f}")
    print(f"  y_prob mean:        {metrics['y_prob_mean']:.4f}")
    print(f"  Trees (iteration):  {metrics['best_iteration']}")
    print("\n" + classification_report(
        y_test, y_pred, target_names=["No Run (0)", "Run (1)"], digits=4
    ))
    return metrics

def _parse_tuned_hp(hp: dict) -> tuple[dict, float, float, int]:
    """Extract tree overrides, FL gamma/alpha, and round count from a param dict."""
    gamma = float(hp.get("gamma_fl", hp.get("gamma", 2.0)))
    alpha = float(hp.get("alpha_fl", hp.get("alpha", 0.75)))
    n_rounds = int(hp.get("refit_rounds", XGB_NUM_BOOST_ROUND))
    tree_keys = (
        "max_depth", "learning_rate", "reg_lambda", "reg_alpha",
        "min_child_weight", "subsample", "colsample_bytree",
    )
    tree_overrides = {k: hp[k] for k in tree_keys if k in hp}
    tree_overrides.setdefault("subsample", SHARED_PARAMS["subsample"])
    tree_overrides.setdefault("colsample_bytree", SHARED_PARAMS["colsample_bytree"])
    return tree_overrides, gamma, alpha, n_rounds

def _diagnose_fit(booster: xgb.Booster, X_train, y_train, X_val, y_val, X_test, y_test):
    """诊断过拟合/欠拟合：对比 train / val / test 的 PR-AUC。"""
    train_pr = _booster_pr_auc(booster, X_train, y_train)
    test_pr  = _booster_pr_auc(booster, X_test, y_test)

    print(f"\n  {'='*50}")
    print(f"  FIT DIAGNOSIS — PR-AUC Comparison")
    print(f"  {'='*50}")
    print(f"  Train PR-AUC : {train_pr:.6f}")

    if X_val is not None and y_val is not None:
        val_pr = _booster_pr_auc(booster, X_val, y_val)
        print(f"  Val   PR-AUC : {val_pr:.6f}")
        print(f"  Test  PR-AUC : {test_pr:.6f}")
        print(f"  {'-'*50}")

        train_val_gap = train_pr - val_pr
        val_test_gap  = val_pr - test_pr

        if train_val_gap > 0.002:
            severity = "SEVERE" if train_val_gap > 0.005 else "MILD"
            print(f"  OVERFITTING ({severity}): Train-Val gap = {train_val_gap:.6f}")
        elif train_pr < 0.003:
            print(f"  UNDERFITTING: Train PR-AUC ({train_pr:.6f}) is very low")
        else:
            print(f"  GOOD FIT: Train-Val gap = {train_val_gap:.6f} (negligible)")

        if val_test_gap > 0.002:
            print(f"  NOTE: Val-Test gap = {val_test_gap:.6f} — possible season shift")
        return train_pr, val_pr, test_pr
    else:
        print(f"  Test  PR-AUC : {test_pr:.6f}")
        print(f"  {'-'*50}")
        train_test_gap = train_pr - test_pr
        if train_test_gap > 0.002:
            print(f"  OVERFITTING: Train-Test gap = {train_test_gap:.6f}")
        elif train_pr < 0.003:
            print(f"  UNDERFITTING: Train PR-AUC ({train_pr:.6f}) is very low")
        else:
            print(f"  GOOD FIT: Train-Test gap = {train_test_gap:.6f} (negligible)")
        return train_pr, None, test_pr

def train_focal_loss(
    X_train, y_train, X_test, y_test,
    X_val=None, y_val=None,
    gamma: float = 2.0,
    alpha: float = 0.75,
    warmup_rounds: int = 0,
    tuned_hp: dict | None = None,
) -> tuple[xgb.Booster, dict]:
    """Train XGBoost with Focal Loss.

    If X_val/y_val provided, use for early stopping.
    Otherwise falls back to X_test/y_test (with a warning).
    """
    pos_rate = y_train.mean()

    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        eval_set = [(dval, "val")]
        print(f"  Early stopping on: val ({len(y_val):,} samples)")
    else:
        dval = xgb.DMatrix(X_test, label=y_test)
        eval_set = [(dval, "test")]
        print("  [WARNING] No validation set provided; early stopping on test set.")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    if tuned_hp is not None:
        tree_overrides, gamma, alpha, n_rounds = _parse_tuned_hp(tuned_hp)
        fl_params = _xgb_params({**tree_overrides, "base_score": float(pos_rate)}, use_gpu=USE_GPU)
        label = f"Focal Loss tuned (gamma={gamma}, alpha={alpha}, max_rounds={n_rounds})"
        print(f"\n--- {label} ---")
        print(f"  tree params: {tree_overrides}")

        booster = xgb.train(
            fl_params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train")] + eval_set,
            obj=make_focal_loss(gamma=gamma, alpha=alpha),
            custom_metric=pr_auc_eval,
            early_stopping_rounds=XGB_EARLY_STOPPING,
            maximize=True,
            verbose_eval=100,
        )
        metrics = evaluate_model(booster, X_test, y_test, label)
        train_pr, val_pr, _ = _diagnose_fit(booster, X_train, y_train, X_val, y_val, X_test, y_test)
        metrics["train_pr_auc"] = train_pr
        if val_pr is not None:
            metrics["val_pr_auc"] = val_pr
        return booster, metrics

    ce_params = _xgb_params({"objective": "binary:logistic",
                 "scale_pos_weight": min((1.0 - pos_rate) / pos_rate, 10.0)}, use_gpu=USE_GPU)
    fl_params = _xgb_params({"base_score": float(pos_rate)}, use_gpu=USE_GPU)

    label = f"Focal Loss (gamma={gamma}, alpha={alpha}, warmup={warmup_rounds})"
    print(f"\n--- {label} ---")

    fl_boost_rounds = XGB_NUM_BOOST_ROUND - warmup_rounds
    if warmup_rounds > 0:
        print(f"  Phase 1: CE warm-up for {warmup_rounds} rounds ...")
        booster = xgb.train(
            ce_params, dtrain,
            num_boost_round=warmup_rounds,
            evals=[(dtrain, "train")] + eval_set,
            verbose_eval=100,
        )
        xgb_model = booster
        print(f"  Phase 2: Focal Loss fine-tune for up to {fl_boost_rounds} rounds ...")
    else:
        print(f"  Direct Focal Loss (no warm-up) for up to {fl_boost_rounds} rounds ...")
        xgb_model = None

    booster = xgb.train(
        fl_params, dtrain,
        num_boost_round=fl_boost_rounds,
        evals=[(dtrain, "train")] + eval_set,
        obj=make_focal_loss(gamma=gamma, alpha=alpha),
        custom_metric=pr_auc_eval,
        xgb_model=xgb_model,
        early_stopping_rounds=XGB_EARLY_STOPPING,
        maximize=True,
        verbose_eval=100,
    )
    metrics = evaluate_model(booster, X_test, y_test, label)
    train_pr, val_pr, _ = _diagnose_fit(booster, X_train, y_train, X_val, y_val, X_test, y_test)
    metrics["train_pr_auc"] = train_pr
    if val_pr is not None:
        metrics["val_pr_auc"] = val_pr
    return booster, metrics

def pr_auc_eval(preds: np.ndarray, dtrain) -> tuple[str, float]:
    """Validation metric: Precision-Recall AUC."""
    labels = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    try:
        score = average_precision_score(labels, p)
    except ValueError:
        score = 0.0
    return "pr_auc", score

def _booster_pr_auc(booster: xgb.Booster, X: np.ndarray, y: np.ndarray) -> float:
    y_score = booster.predict(xgb.DMatrix(X))
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=10.0, neginf=-10.0)
    if y_score.min() < -0.05 or y_score.max() > 1.05:
        y_prob = 1.0 / (1.0 + np.exp(-y_score))
    else:
        y_prob = y_score
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    try:
        return float(average_precision_score(y, y_prob))
    except ValueError:
        return 0.0

def _fit_focal_loss_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trial: dict,
    *,
    n_estimators: int,
    early_stopping: int,
    warmup_rounds: int = 0,
) -> tuple[xgb.Booster | None, float]:
    """Train one FL config; early-stop on val PR-AUC. Returns (booster, val_pr_auc)."""
    pos_rate = float(y_train.mean())
    gamma = trial["gamma_fl"]
    alpha = trial["alpha_fl"]
    tree_keys = (
        "max_depth", "learning_rate", "reg_lambda", "reg_alpha",
        "min_child_weight", "subsample", "colsample_bytree",
    )
    tree_overrides = {k: trial[k] for k in tree_keys if k in trial}
    tree_overrides.setdefault("subsample", SHARED_PARAMS["subsample"])
    tree_overrides.setdefault("colsample_bytree", SHARED_PARAMS["colsample_bytree"])
    fl_params = _xgb_params({**tree_overrides, "base_score": pos_rate}, use_gpu=USE_GPU)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    try:
        n_fine = n_estimators - warmup_rounds
        xgb_model = None
        if warmup_rounds > 0:
            ce_params = _xgb_params(
                {**tree_overrides, "objective": "binary:logistic",
                 "scale_pos_weight": min((1.0 - pos_rate) / max(pos_rate, 1e-9), 10.0)},
                use_gpu=USE_GPU,
            )
            xgb_model = xgb.train(
                ce_params, dtrain,
                num_boost_round=warmup_rounds,
                evals=[(dval, "val")],
                verbose_eval=False,
            )
        if n_fine <= 0:
            return None, 0.0
        booster = xgb.train(
            fl_params, dtrain,
            num_boost_round=n_fine,
            evals=[(dval, "val")],
            obj=make_focal_loss(gamma=gamma, alpha=alpha),
            custom_metric=pr_auc_eval,
            xgb_model=xgb_model,
            early_stopping_rounds=early_stopping,
            maximize=True,
            verbose_eval=False,
        )
    except Exception:
        return None, 0.0

    val_pr = _booster_pr_auc(booster, X_val, y_val)
    return booster, val_pr
