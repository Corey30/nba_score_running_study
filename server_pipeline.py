"""NBA Scoring Run Prediction - Main Entry Point"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from config import *
from data_loader import download_kaggle_data
from preprocessing import run_preprocessing
from experiments.cross_validation import _run_cross_validation
from experiments.grid_search import _run_grid_search
from experiments.ablation import _run_ablation

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="NBA Scoring Run Prediction Pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download")
    parser.add_argument("--data-dir", type=str, default=str(LOCAL_DATA_DIR), help="Data directory")
    parser.add_argument("--seasons", type=str, default="2020,2021,2022,2023,2024,2025,2026",
                        help="Comma-separated season codes")
    parser.add_argument("--output", type=str, default="output_result/server_pipeline_results.csv",
                        help="Output CSV")
    parser.add_argument("--ablation", action="store_true", help="Run ablation experiment")
    parser.add_argument("--tuned", action="store_true",
                        help="Load per-arm tuned params from output_result/")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search")
    parser.add_argument("--tune-on", type=str, default="", choices=TUNE_ON_CHOICES,
                        help="Feature set for grid search tuning (empty = run all arms)")
    parser.add_argument("--load-grid-params", type=str, default=None,
                        help="JSON file with pre-tuned params")
    parser.add_argument("--experiment-tag", type=str, default="", help="Tag for output files")

    parser.add_argument("--grid-mode", type=str, default="quick",
                        choices=["quick", "full"], help="Grid search scope")

    parser.add_argument("--n-trials", type=int, default=None,
                        help="Random search trials (per objective for CE vs FL)")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Number of boosting rounds")
    parser.add_argument("--grid-early-stop", type=int, default=30,
                        help="Early stopping rounds for grid search")
    parser.add_argument("--fl-gamma", type=float, default=2.0, help="Focal Loss gamma")
    parser.add_argument("--fl-alpha", type=float, default=0.75, help="Focal Loss alpha")
    parser.add_argument("--fl-warmup", type=int, default=0, help="FL warmup rounds")
    parser.add_argument("--ce-vs-fl", action="store_true",
                        help="Independently tuned CE vs Focal Loss comparison on stage7")
    parser.add_argument("--model-compare", action="store_true",
                        help="XGBoost vs LightGBM vs CatBoost comparison")
    parser.add_argument("--shap-selection", action="store_true",
                        help="SHAP feature selection: selected vs legacy")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    if not args.skip_download:
        download_kaggle_data(data_dir)

    df = run_preprocessing(data_dir, seasons=seasons)
    if df is None or len(df) == 0:
        print("ERROR: No data loaded.")
        sys.exit(1)


    if args.ce_vs_fl:
        from experiments.ce_vs_fl import _run_ce_vs_fl
        _run_ce_vs_fl(df, args)
    elif args.model_compare:
        from experiments.model_comparison import _run_model_comparison
        _run_model_comparison(df, args)
    elif args.shap_selection:
        from experiments.shap_selection import _run_shap_selection
        _run_shap_selection(df, args)
    elif args.grid_search:
        _run_grid_search(df, args)
    elif args.ablation:
        _run_ablation(df, args)
    else:
        _run_cross_validation(df, args)

if __name__ == "__main__":
    main()
