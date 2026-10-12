"""Generate standardized figures for the four formal experiments (E1-E4).

Reads the result CSVs / prediction npz produced by the experiment scripts and
writes four PNG figures under figures/:

  fig_ablation_stage.png   (E2: feature-stage ablation bar chart)
  fig_ce_vs_fl_pr.png      (E1: CE vs Focal Loss PR curves)
  fig_model_comparison.png (E3: three-model comparison bar chart)
  fig_feature_selection.png(E4: legacy vs SHAP-selected vs full bar chart)

Usage:
  python experiments/plot_figures.py --experiment-tag formal --output-dir output_result
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Random baseline: positive rate ~2.124% under the peak-based +10/180s label.
RANDOM_BASELINE = 0.0212

# Feature-stage metadata for nice x-axis labels (dims mirror config.ABLATION_ARMS).
STAGE_LABELS = {
    "legacy": ("Legacy", 26),
    "stage1": ("S1\nContext", 10),
    "stage2": ("S2\nRaw counts", 89),
    "stage3": ("S3\n+Eff./pace", 107),
    "stage4": ("S4\n+Momentum", 112),
    "stage5": ("S5\n+Spatial", 118),
    "stage6": ("S6\n+Continuity", 122),
    "stage7": ("S7\n+Team priors", 141),
}
STAGE_ORDER = ["legacy", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]

FEATURE_SEL_ORDER = ["legacy", "selected", "full"]
FEATURE_SEL_LABELS = {
    "legacy": "Legacy\n(26 prior)",
    "selected": "SHAP-selected\n(top-26)",
    "full": "Stage7 full\n(141)",
}


def _exp_prefix(args) -> str:
    tag = (getattr(args, "experiment_tag", None) or "").strip()
    return f"exp_{tag}_" if tag else ""


def _mean_std(df: pd.DataFrame, group_col: str, value_col: str = "pr_auc"):
    """Extract (mean, std) indexed by group from a summary CSV."""
    means = df[df["fold"] == "mean"].set_index(group_col)[value_col]
    stds = df[df["fold"] == "std"].set_index(group_col)[value_col]
    return means, stds


def _bar_chart(labels, means, stds, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85,
           error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
    ax.axhline(RANDOM_BASELINE, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Overall corpus prevalence ({RANDOM_BASELINE:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    # Zoom the y-axis away from zero so differences in the 0.024-0.028 range are
    # legible; the corpus prevalence (0.0212) remains inside the view.
    ax.set_ylim(0.0195, 0.0340)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figure] {out_path}")


def _line_chart(labels, means, stds, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=stds, capsize=4, color="#4C72B0",
                marker="o", markersize=5, linewidth=1.8, linestyle="-",
                elinewidth=1.2, ecolor="#333333")
    ax.axhline(RANDOM_BASELINE, color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"Overall corpus prevalence ({RANDOM_BASELINE:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    # Zoom the y-axis away from zero so differences in the 0.024-0.028 
    # range are legible; the corpus prevalence (0.0212) remains inside the view.
    ax.set_ylim(0.0195, 0.0340)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figure] {out_path}")


def plot_ablation(csv_path: Path, out_path: Path):
    if not csv_path.exists():
        print(f"[skip] missing {csv_path}")
        return
    df = pd.read_csv(csv_path)
    means, stds = _mean_std(df, "arm")
    arms = [a for a in STAGE_ORDER if a in means.index]
    labels = ["Legacy" if a == "legacy" else a.replace("stage", "S") for a in arms]
    _line_chart(labels, means[arms].values, stds[arms].values, out_path,
                "Feature-stage ablation (XGBoost + Focal Loss)", "Test Average Precision (AP)")


def plot_model_comparison(csv_path: Path, out_path: Path):
    if not csv_path.exists():
        print(f"[skip] missing {csv_path}")
        return
    df = pd.read_csv(csv_path)
    means, stds = _mean_std(df, "model")
    labels = list(means.index)
    _bar_chart(labels, means.values, stds.values, out_path,
               "Three-model comparison (stage7 features)", "Test Average Precision (AP)")


def plot_feature_selection(csv_path: Path, out_path: Path):
    if not csv_path.exists():
        print(f"[skip] missing {csv_path}")
        return
    df = pd.read_csv(csv_path)
    means, stds = _mean_std(df, "arm")
    arms = [a for a in FEATURE_SEL_ORDER if a in means.index]
    labels = [FEATURE_SEL_LABELS[a] for a in arms]
    _bar_chart(labels, means[arms].values, stds[arms].values, out_path,
               "Feature selection: SHAP top-26 vs legacy vs full", "Test Average Precision (AP)")


def plot_ce_vs_fl_pr(npz_path: Path, out_path: Path):
    if not npz_path.exists():
        print(f"[skip] missing {npz_path}")
        return
    data = np.load(npz_path)
    y_true = data["y_true"]
    ce_prob = data["ce_prob"]
    fl_prob = data["fl_prob"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for prob, color, label in [(ce_prob, "#C44E52", "Cross-entropy"),
                               (fl_prob, "#4C72B0", "Focal Loss")]:
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(recall, precision, color=color, linewidth=1.8,
                label=f"{label} (AP={ap:.4f})")

    ax.axhline(RANDOM_BASELINE, color="#777777", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({RANDOM_BASELINE:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curves: CE vs Focal Loss (stage7, 5-fold pooled)")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figure] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot formal experiment figures")
    parser.add_argument("--experiment-tag", type=str, default="", help="Tag for input files")
    parser.add_argument("--output-dir", type=str, default="output_result",
                        help="Directory containing result CSVs/npz")
    parser.add_argument("--fig-dir", type=str, default="figures",
                        help="Directory to write PNG figures")
    args = parser.parse_args()

    prefix = _exp_prefix(args)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation(
        out_dir / f"{prefix}grid_search_cv_results.csv",
        fig_dir / "fig_ablation_stage.png",
    )
    plot_model_comparison(
        out_dir / f"{prefix}model_comparison_results.csv",
        fig_dir / "fig_model_comparison.png",
    )
    plot_feature_selection(
        out_dir / f"{prefix}shap_selection_results.csv",
        fig_dir / "fig_feature_selection.png",
    )
    plot_ce_vs_fl_pr(
        out_dir / f"{prefix}ce_vs_fl_predictions.npz",
        fig_dir / "fig_ce_vs_fl_pr.png",
    )


if __name__ == "__main__":
    main()
