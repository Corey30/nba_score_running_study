"""Fixed-parameter evaluation runner (no grid search).

Reuses the tuned parameters already saved in
``output_result/exp_formal_grid_best_params_*.json`` and runs only the 5-fold
temporal CV to quickly produce results:

  * E2  feature-stage ablation  -> exp_formal_grid_search_cv_results.csv
  * E4  SHAP feature selection  -> exp_formal_shap_selection_results.csv
                                   + shap_feature_importance.csv
                                   + selected_26_features.json

E3 (three-model comparison) already exists as exp_exog_model_comparison_results.csv.
E1 (CE vs FL) is produced separately via ``server_pipeline.py --ce-vs-fl``.
"""
import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import *
from preprocessing import run_preprocessing
from split import split_train_val_test_by_season
from feature_engineering import resolve_feature_set, _build_all_features
from models.training import train_focal_loss
from experiments.grid_search import _evaluate_params_on_cv

warnings.filterwarnings("ignore")

N_SELECTED = 26
SHAP_SAMPLE = 5000


def _hp_from(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hp = dict(payload.get("params", {}))
    if "refit_rounds" not in hp:
        hp["refit_rounds"] = payload.get("refit_rounds", 500)
    return hp


def _append_e2(rows, arm, n_features, fold_results, mean, std):
    for fi, pr in enumerate(fold_results, 1):
        rows.append({"arm": arm, "n_features": n_features, "fold": fi,
                     "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr})
    rows.append({"arm": arm, "n_features": n_features, "fold": "mean",
                 "test_season": "all", "pr_auc": mean})
    rows.append({"arm": arm, "n_features": n_features, "fold": "std",
                 "test_season": "all", "pr_auc": std})


def _append_e4(rows, arm, fold_results, mean, std):
    for fi, pr in enumerate(fold_results, 1):
        rows.append({"arm": arm, "fold": fi,
                     "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr})
    rows.append({"arm": arm, "fold": "mean", "test_season": "all", "pr_auc": mean})
    rows.append({"arm": arm, "fold": "std", "test_season": "all", "pr_auc": std})


def run_e2(df_feat, out_dir, prefix, eval_args):
    arms = ["legacy", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]
    rows = []
    for arm in arms:
        cols, _ = resolve_feature_set(arm)
        p = out_dir / f"{prefix}grid_best_params_{arm}.json"
        if not p.exists():
            print(f"[E2] missing {p.name}; skipping {arm}")
            continue
        hp = _hp_from(p)
        folds, mean, std = _evaluate_params_on_cv(df_feat, cols, hp, eval_args)
        _append_e2(rows, arm, len(cols), folds, mean, std)
        print(f"[E2] {arm}: {mean:.6f} +/- {std:.6f}")
    out = out_dir / f"{prefix}grid_search_cv_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[E2] -> {out}")


def run_e4(df_feat, out_dir, prefix, eval_args):
    import shap

    full_cols, _ = resolve_feature_set("stage7")
    stage7_hp = _hp_from(out_dir / f"{prefix}grid_best_params_stage7.json")
    legacy_hp = _hp_from(out_dir / f"{prefix}grid_best_params_legacy.json")

    # 1) per-fold SHAP importance (fixed stage7 params)
    imp_rows = []
    per_fold_imp = []
    for fold_idx, (tr, va, te) in enumerate(CV_FOLDS, 1):
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test_by_season(
            df_feat, full_cols, tr, va, te)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        booster, _ = train_focal_loss(
            X_train_s, y_train, X_test_s, y_test,
            X_val=X_val_s, y_val=y_val, tuned_hp=stage7_hp)

        n = min(SHAP_SAMPLE, len(X_test_s))
        sample = X_test_s[:n]
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).reshape(n, len(full_cols))
        imp = np.abs(sv).mean(axis=0)
        per_fold_imp.append(imp)
        for name, v in zip(full_cols, imp):
            imp_rows.append({"fold": fold_idx, "feature": name, "importance": float(v)})
        print(f"[E4] SHAP fold {fold_idx}/{len(CV_FOLDS)} done")

    pd.DataFrame(imp_rows).to_csv(out_dir / f"{prefix}shap_feature_importance.csv", index=False)

    # 2) average importance -> top-26
    mean_imp = np.vstack(per_fold_imp).mean(axis=0)
    order = np.argsort(mean_imp)[::-1]
    selected = [full_cols[i] for i in order[:N_SELECTED]]
    (out_dir / f"{prefix}selected_26_features.json").write_text(
        json.dumps({"n_features": len(selected), "features": selected}, indent=2),
        encoding="utf-8")
    print(f"[E4] selected top-{N_SELECTED} saved")

    # 3) fixed-param evaluation. selected & legacy share legacy params to isolate
    #    the feature-choice effect; full uses stage7 params.
    legacy_cols, _ = resolve_feature_set("legacy")
    arms = [
        ("selected", selected, legacy_hp),
        ("legacy", legacy_cols, legacy_hp),
        ("full", full_cols, stage7_hp),
    ]
    rows = []
    for name, cols, hp in arms:
        folds, mean, std = _evaluate_params_on_cv(df_feat, cols, hp, eval_args)
        _append_e4(rows, name, folds, mean, std)
        print(f"[E4] {name}: {mean:.6f} +/- {std:.6f}")

    out = out_dir / f"{prefix}shap_selection_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[E4] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-tag", default="formal")
    ap.add_argument("--output", default="output_result/server_pipeline_results.csv")
    ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--skip-e4", action="store_true")
    a = ap.parse_args()

    tag = a.experiment_tag.strip() or "formal"
    prefix = f"exp_{tag}_"
    data_dir = Path(LOCAL_DATA_DIR)
    seasons = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]

    print("Preprocessing...")
    df = run_preprocessing(data_dir, seasons=seasons)
    print("Building features (max_stage=6)...")
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=6)

    out_dir = Path(a.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_args = SimpleNamespace(fl_gamma=2.0, fl_alpha=0.75, fl_warmup=0)

    if not a.skip_e2:
        run_e2(df_feat, out_dir, prefix, eval_args)
    if not a.skip_e4:
        run_e4(df_feat, out_dir, prefix, eval_args)


if __name__ == "__main__":
    main()
