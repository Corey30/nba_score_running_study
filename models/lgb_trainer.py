"""LightGBM trainer with focal loss + hyperparameter search."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import *
from split import split_train_val_test_by_season
from models.focal_loss import focal_loss_objective

warnings.filterwarnings("ignore")

_EARLY_STOPPING = 30

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on env
    lgb = None
    LGB_AVAILABLE = False


def _require_lgb():
    if not LGB_AVAILABLE:
        raise ImportError("LightGBM is not installed. Run: pip install lightgbm")


def _pr_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """PR-AUC from raw margins (sigmoid -> probability)."""
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    try:
        return float(average_precision_score(labels, p))
    except ValueError:
        return 0.0


def _lgb_focal(gamma: float, alpha: float):
    """Wrap the shared XGBoost focal objective for LightGBM (grad/hess)."""
    def _fobj(preds: np.ndarray, train_data):
        return focal_loss_objective(preds, train_data, gamma, alpha)
    return _fobj


def _lgb_pr_auc(preds: np.ndarray, dataset):
    """LightGBM `feval` for early stopping on PR-AUC."""
    return "pr_auc", _pr_auc(dataset.get_label(), preds), True


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


def _split_lgb_params(hp: dict, n_estimators: int = 500) -> tuple[dict, float, float, int]:
    """Split a search combo into (lgb_params, gamma, alpha, n_rounds)."""
    hp = dict(hp)
    gamma = float(hp.pop("gamma_fl", 2.0))
    alpha = float(hp.pop("alpha_fl", 0.75))
    n_rounds = int(hp.pop("iterations", hp.pop("refit_rounds", n_estimators)))

    params = {**LGB_DEFAULT_PARAMS}
    for k in LGB_DEFAULT_PARAMS:
        if k in hp:
            params[k] = hp[k]
    params["verbosity"] = -1
    return params, gamma, alpha, n_rounds


def train_lgb(X_train, y_train, X_val, y_val, X_test, y_test, params: dict):
    """Train LightGBM with focal loss. Returns (booster, metrics_dict)."""
    _require_lgb()
    lgb_params, gamma, alpha, n_rounds = _split_lgb_params(params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    lgb_params["objective"] = _lgb_focal(gamma, alpha)
    booster = lgb.train(
        lgb_params, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        feval=_lgb_pr_auc,
        callbacks=[lgb.early_stopping(_EARLY_STOPPING, first_metric_only=True)],
    )

    metrics = {
        "pr_auc": _pr_auc(y_test, booster.predict(X_test)),
        "val_pr_auc": _pr_auc(y_val, booster.predict(X_val)),
        "train_pr_auc": _pr_auc(y_train, booster.predict(X_train)),
    }
    return booster, metrics


def _fit_lgb_fold(X_tr, y_tr, X_val, y_val, hp: dict, n_estimators: int) -> float:
    """Single-trial fit on Fold 1 inner (train/val). Returns val PR-AUC."""
    lgb_params, gamma, alpha, n_rounds = _split_lgb_params(hp, n_estimators)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    lgb_params["objective"] = _lgb_focal(gamma, alpha)
    booster = lgb.train(
        lgb_params, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval],
        feval=_lgb_pr_auc,
        callbacks=[lgb.early_stopping(_EARLY_STOPPING, first_metric_only=True)],
    )
    return _pr_auc(y_val, booster.predict(X_val))


def lgb_grid_search(df, cols, n_trials: int, args, trials_out=None) -> dict:
    """Random search on Fold 1; return best param dict (incl. iterations)."""
    _require_lgb()
    train_seasons, val_season, test_season = CV_FOLDS[0]
    X_tr, X_val, _, y_tr, y_val, _ = split_train_val_test_by_season(
        df, cols, train_seasons, val_season, test_season
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    n_estimators = getattr(args, "n_estimators", 500)
    combos = _random_sample(LGB_SEARCH_SPACE, n_trials)

    print(f"  train_inner={len(X_tr):,}  val={len(X_val):,}  trials={len(combos)}")

    trial_rows = []
    best_val_pr = -1.0
    best_combo = None
    for i, combo in enumerate(combos, 1):
        val_pr = _fit_lgb_fold(X_tr, y_tr, X_val, y_val, combo, n_estimators)
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
        return {"gamma_fl": 2.0, "alpha_fl": 0.75, "iterations": n_estimators}
    return {**best_combo, "iterations": n_estimators}


def lgb_cv_evaluate(df, cols, hp: dict, args) -> tuple[list, float, float]:
    """5-fold CV evaluation. Returns (fold_results, mean, std)."""
    _require_lgb()
    fold_results = []
    for train_seasons, val_season, test_season in CV_FOLDS:
        X_tr, X_val, X_test, y_tr, y_val, y_test = split_train_val_test_by_season(
            df, cols, train_seasons, val_season, test_season
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        _, metrics = train_lgb(X_tr, y_tr, X_val, y_val, X_test, y_test, hp)
        fold_results.append(metrics["pr_auc"])
        print(f"  Fold PR-AUC: {metrics['pr_auc']:.6f}")

    mean_pr = float(np.mean(fold_results))
    std_pr = float(np.std(fold_results, ddof=1))
    return fold_results, mean_pr, std_pr
