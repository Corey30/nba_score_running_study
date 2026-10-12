"""CatBoost trainer (Logloss + scale_pos_weight) with hyperparameter search."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import *
from split import split_train_val_test_by_season

warnings.filterwarnings("ignore")

_EARLY_STOPPING = 30

try:
    from catboost import CatBoostClassifier
    CB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    CatBoostClassifier = None
    CB_AVAILABLE = False


def _require_cb():
    if not CB_AVAILABLE:
        raise ImportError("CatBoost is not installed. Run: pip install catboost")


def _pr_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, preds))
    except ValueError:
        return 0.0


def _random_sample(space: dict, n: int, seed: int = 42) -> list[dict]:
    """Randomly sample n combinations from the search space."""
    import random
    random.seed(seed)
    keys = list(space.keys())
    combos = []
    seen = set()
    attempts = 0
    while len(combos) < n and attempts < n * 20:
        combo = tuple(random.choice(space[k]) for k in keys)
        if combo not in seen:
            seen.add(combo)
            combos.append(dict(zip(keys, combo)))
        attempts += 1
    return combos


def _build_cb_params(hp: dict, n_estimators: int) -> dict:
    """Merge a search combo with CatBoost defaults; set iterations."""
    hp = dict(hp)
    params = {**CB_DEFAULT_PARAMS}
    for k in CB_DEFAULT_PARAMS:
        if k in hp:
            params[k] = hp[k]
    # ensure scale_pos_weight is present (default for class imbalance)
    params.setdefault("scale_pos_weight", 50.0)
    params["iterations"] = int(hp.get("iterations", hp.get("refit_rounds", n_estimators)))
    return params


def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test, params: dict):
    """Train CatBoost (Logloss + scale_pos_weight). Returns (model, metrics_dict)."""
    _require_cb()
    cb_params = _build_cb_params(params, int(params.get("iterations", params.get("refit_rounds", 500))))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        early_stopping_rounds=_EARLY_STOPPING,
        **cb_params,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    metrics = {
        "pr_auc": _pr_auc(y_test, model.predict_proba(X_test)[:, 1]),
        "val_pr_auc": _pr_auc(y_val, model.predict_proba(X_val)[:, 1]),
        "train_pr_auc": _pr_auc(y_train, model.predict_proba(X_train)[:, 1]),
    }
    return model, metrics


def _fit_cb_fold(X_tr, y_tr, X_val, y_val, hp: dict, n_estimators: int) -> float:
    """Single-trial fit on Fold 1 inner (train/val). Returns val PR-AUC."""
    cb_params = _build_cb_params(hp, n_estimators)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        early_stopping_rounds=_EARLY_STOPPING,
        **cb_params,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return _pr_auc(y_val, model.predict_proba(X_val)[:, 1])


def cb_grid_search(df, cols, n_trials: int, args, trials_out=None) -> dict:
    """Random search on Fold 1; return best param dict (incl. iterations)."""
    _require_cb()
    train_seasons, val_season, test_season = CV_FOLDS[0]
    X_tr, X_val, _, y_tr, y_val, _ = split_train_val_test_by_season(
        df, cols, train_seasons, val_season, test_season
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    n_estimators = getattr(args, "n_estimators", 500)
    combos = _random_sample(CB_SEARCH_SPACE, n_trials)

    print(f"  train_inner={len(X_tr):,}  val={len(X_val):,}  trials={len(combos)}")

    trial_rows = []
    best_val_pr = -1.0
    best_combo = None
    for i, combo in enumerate(combos, 1):
        val_pr = _fit_cb_fold(X_tr, y_tr, X_val, y_val, combo, n_estimators)
        trial_rows.append({**combo, "val_pr_auc": val_pr, "trial": i})
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_combo = combo
        print(f"  [{i}/{len(combos)}] best_val_pr={best_val_pr:.4f}")

    if trials_out is not None:
        pd.DataFrame(trial_rows).sort_values("val_pr_auc", ascending=False).to_csv(
            trials_out, index=False
        )
        print(f"  all trials -> {trials_out}")

    if best_combo is None:
        return {"scale_pos_weight": 65, "iterations": n_estimators}
    return {**best_combo, "iterations": n_estimators}


def cb_cv_evaluate(df, cols, hp: dict, args) -> tuple[list, float, float]:
    """5-fold CV evaluation. Returns (fold_results, mean, std)."""
    _require_cb()
    fold_results = []
    for train_seasons, val_season, test_season in CV_FOLDS:
        X_tr, X_val, X_test, y_tr, y_val, y_test = split_train_val_test_by_season(
            df, cols, train_seasons, val_season, test_season
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        _, metrics = train_catboost(X_tr, y_tr, X_val, y_val, X_test, y_test, hp)
        fold_results.append(metrics["pr_auc"])
        print(f"  Fold PR-AUC: {metrics['pr_auc']:.6f}")

    mean_pr = float(np.mean(fold_results))
    std_pr = float(np.std(fold_results, ddof=1))
    return fold_results, mean_pr, std_pr
