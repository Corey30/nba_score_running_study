"""5-fold temporal cross-validation with feature ablation."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from config import *
from models.training import train_focal_loss
from split import split_train_val_test_by_season
from feature_engineering import resolve_feature_set, _build_all_features

def _exp_prefix(args) -> str:
    tag = (getattr(args, "experiment_tag", None) or "").strip()
    return f"exp_{tag}_" if tag else ""

def _load_best_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hp = dict(payload.get("params", {}))
    if "refit_rounds" in payload:
        hp["refit_rounds"] = payload["refit_rounds"]
    print(f"[params] loaded from {path}")
    print(f"  tune_arm={payload.get('tune_arm')}  "
          f"val_pr={payload.get('best_val_pr_auc')}  "
          f"refit_rounds={payload.get('refit_rounds')}")
    return hp

def _load_best_params_for_arm(arm: str, params_dir: Path) -> dict | None:
    """按 arm 加载对应的 grid_best_params_{arm}.json，不存在则返回 None。"""
    path = params_dir / f"grid_best_params_{arm}.json"
    if path.exists():
        return _load_best_params(path)
    print(f"  [params] no tuned params for {arm}, using defaults")
    return None

def _run_cross_validation(df, args):
    """5-fold temporal CV across 8 feature sets (legacy, stage1-7).

    若 args.tuned 为 True，自动从 output_result/ 加载各 arm 的调优参数。
    若 args.load_grid_params 指定了文件，则所有 arm 共用该参数。
    """
    prefix = _exp_prefix(args)
    arms = ["legacy", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]

    params_dir = Path(args.output).parent
    use_tuned = (
        getattr(args, "tuned", False)
        or (args.load_grid_params is not None)
        or any((params_dir / f"grid_best_params_{arm}.json").exists() for arm in arms)
    )

    print("\n" + "=" * 60)
    print("5-FOLD TEMPORAL CROSS-VALIDATION + FEATURE ABLATION")
    print("=" * 60)
    for i, (train_s, val_s, test_s) in enumerate(CV_FOLDS, 1):
        print(f"  Fold {i}: train {train_s}  |  val {val_s}  |  test {test_s}")
    print(f"  Arms: {arms}")
    print(f"  Split: season-level (no cross-season leakage)")
    print(f"  Hyperparams: {'per-arm tuned' if use_tuned else 'defaults'}")

    max_st = max(
        int(a.replace("stage", "")) if a.startswith("stage") else 4
        for a in arms
    )
    df_feat, _ = _build_all_features(df, window_seconds=ROLLING_WINDOW, stage=None, max_stage=max_st)

    all_rows = []
    for arm in arms:
        cols, desc = resolve_feature_set(arm)

        tuned_hp = None
        if use_tuned:
            if args.load_grid_params:
                tuned_hp = _load_best_params(Path(args.load_grid_params))
            else:
                tuned_hp = _load_best_params_for_arm(arm, params_dir)

        print(f"\n{'=' * 60}")
        hp_info = "tuned" if tuned_hp else "default"
        print(f"CV - {arm} ({len(cols)} features: {desc}) [{hp_info}]")
        print("=" * 60)

        fold_results = []
        for fold_idx, (train_seasons, val_season, test_season) in enumerate(CV_FOLDS, 1):
            # 1. 按赛季切分 train/val/test
            X_train, X_val, X_test, y_train, y_val, y_test = \
                split_train_val_test_by_season(
                    df_feat, cols, train_seasons, val_season, test_season
                )

            # 2. 标准化
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train).astype(np.float32)
            X_val_s   = scaler.transform(X_val).astype(np.float32)
            X_test_s  = scaler.transform(X_test).astype(np.float32)

            pos_train = y_train.mean() * 100
            pos_val   = y_val.mean() * 100
            pos_test  = y_test.mean() * 100
            print(f"\n  Fold {fold_idx}: train {len(X_train_s):,} (pos {pos_train:.2f}%)"
                  f"  val {len(X_val_s):,} (pos {pos_val:.2f}%)"
                  f"  -> test {len(X_test_s):,} (pos {pos_test:.2f}%)")

            # 3. 训练（val 用于 early stopping，test 仅用于最终评估）
            _, metrics = train_focal_loss(
                X_train_s, y_train, X_test_s, y_test,
                X_val=X_val_s, y_val=y_val,
                gamma=args.fl_gamma, alpha=args.fl_alpha,
                warmup_rounds=args.fl_warmup,
                tuned_hp=tuned_hp,
            )
            fold_results.append(metrics["pr_auc"])
            print(f"  Fold {fold_idx} PR-AUC: {metrics['pr_auc']:.6f}")

        mean_pr = np.mean(fold_results)
        std_pr = np.std(fold_results, ddof=1)
        print(f"\n  {arm} - 5-fold PR-AUC: {mean_pr:.6f} +/- {std_pr:.6f}")
        for fi, pr in enumerate(fold_results, 1):
            all_rows.append({
                "arm": arm, "n_features": len(cols), "fold": fi,
                "test_season": CV_FOLDS[fi - 1][2], "pr_auc": pr,
            })
        all_rows.append({
            "arm": arm, "n_features": len(cols), "fold": "mean",
            "test_season": "all", "pr_auc": mean_pr,
        })
        all_rows.append({
            "arm": arm, "n_features": len(cols), "fold": "std",
            "test_season": "all", "pr_auc": std_pr,
        })

    out = Path(args.output).parent / f"{prefix}cross_validation_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary.to_csv(out, index=False)
    print(f"\nCV results saved -> {out}")

    pivot = summary[summary["fold"].isin(["mean", "std"])].pivot(
        index="arm", columns="fold", values="pr_auc"
    )
    print("\n5-Fold CV Summary (PR-AUC mean +/- std):")
    print(pivot.to_string())
