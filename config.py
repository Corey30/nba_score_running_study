"""Project configuration: constants, feature groups, hyperparameters."""
import numpy as np
from pathlib import Path

LOCAL_DATA_DIR = Path("./nba_pbp_data")

ROLLING_WINDOW = 120

CV_FOLDS = [
    (["2020"], "2021", "2022"),
    (["2020", "2021"], "2022", "2023"),
    (["2020", "2021", "2022"], "2023", "2024"),
    (["2020", "2021", "2022", "2023"], "2024", "2025"),
    (["2020", "2021", "2022", "2023", "2024"], "2025", "2026"),
]

FEATURE_GROUPS = {
    "L0_context": [
        "f_period", "f_time_remaining_game", "f_score_margin", "f_is_home",
        "f_score_margin_abs", "f_is_close", "f_is_blowout", "f_is_clutch",
        "f_is_overtime", "f_time_in_period",
    ],
    "L1_counts": (
        ["f_home_tov", "f_away_tov", "f_home_fouls", "f_away_fouls",
         "f_home_made", "f_home_miss", "f_away_made", "f_away_miss",
         "f_home_reb", "f_away_reb", "f_home_ft", "f_away_ft"]
        + ["f_home_tov_60s", "f_away_tov_60s", "f_home_fouls_60s", "f_away_fouls_60s",
           "f_home_made_60s", "f_home_miss_60s", "f_away_made_60s", "f_away_miss_60s",
           "f_home_reb_60s", "f_away_reb_60s", "f_home_ft_60s", "f_away_ft_60s"]
        + ["f_home_tov_180s", "f_away_tov_180s", "f_home_fouls_180s", "f_away_fouls_180s",
           "f_home_made_180s", "f_home_miss_180s", "f_away_made_180s", "f_away_miss_180s",
           "f_home_reb_180s", "f_away_reb_180s", "f_home_ft_180s", "f_away_ft_180s"]
        + ["f_home_3pm", "f_away_3pm", "f_home_3pa", "f_away_3pa",
           "f_home_oreb", "f_away_oreb", "f_home_dreb", "f_away_dreb",
           "f_home_timeout", "f_away_timeout", "f_home_subs", "f_away_subs"]
        + ["f_home_3pm_60s", "f_away_3pm_60s", "f_home_3pa_60s", "f_away_3pa_60s",
           "f_home_oreb_60s", "f_away_oreb_60s", "f_home_dreb_60s", "f_away_dreb_60s",
           "f_home_timeout_60s", "f_away_timeout_60s", "f_home_subs_60s", "f_away_subs_60s"]
        + ["f_home_3pm_180s", "f_away_3pm_180s", "f_home_3pa_180s", "f_away_3pa_180s",
           "f_home_oreb_180s", "f_away_oreb_180s", "f_home_dreb_180s", "f_away_dreb_180s",
           "f_home_timeout_180s", "f_away_timeout_180s", "f_home_subs_180s", "f_away_subs_180s"]
        + ["f_home_pts_60s", "f_away_pts_60s", "f_home_pts_180s", "f_away_pts_180s"]
        + ["f_home_3pt_pct", "f_away_3pt_pct", "f_3pt_pct_diff"]
    ),
    "L2_efficiency_pace": [
        # 效率指标 (14)
        "f_home_fga", "f_away_fga", "f_home_fg_pct", "f_away_fg_pct",
        "f_fg_pct_diff", "f_home_efg_pct", "f_away_efg_pct", "f_efg_diff",
        "f_home_pts", "f_away_pts", "f_home_poss_est", "f_away_poss_est",
        "f_poss_diff", "f_net_rating",
        # 比赛节奏 (4)
        "f_event_pace", "f_home_event_pace", "f_away_event_pace", "f_pace_diff",
    ],
    "L3_momentum": [
        "f_score_slope", "efg_volatility", "f_home_pts_slope",
        "f_away_pts_slope", "f_slope_diff",
    ],
    "L4_spatial": [
        "f_avg_shot_dist_home", "f_avg_shot_dist_away",
        "f_rim_rate_home", "f_rim_rate_away",
        "f_three_rate_home", "f_three_rate_away",
    ],
    "L5_continuity": [
        "f_secs_since_last_home_score", "f_secs_since_last_away_score",
        "f_home_score_streak", "f_away_score_streak",
    ],
    "L6_exogenous": [
        # season-to-date team priors (computed in preprocessing, no future leakage)
        "f_home_prior_pts_for", "f_home_prior_pts_against", "f_home_prior_net",
        "f_home_prior_win_pct", "f_home_prior_fg_pct", "f_home_prior_pace",
        "f_home_prior_games",
        "f_away_prior_pts_for", "f_away_prior_pts_against", "f_away_prior_net",
        "f_away_prior_win_pct", "f_away_prior_fg_pct", "f_away_prior_pace",
        "f_away_prior_games",
        "f_prior_net_diff", "f_prior_win_pct_diff", "f_prior_pts_for_diff",
        "f_prior_def_diff", "f_rest_diff",
    ],
}

FEATURE_STAGES = {
    "stage1": FEATURE_GROUPS["L0_context"],
    "stage2": FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"],
    "stage3": (FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"]
               + FEATURE_GROUPS["L2_efficiency_pace"]),
    "stage4": (FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"]
               + FEATURE_GROUPS["L2_efficiency_pace"] + FEATURE_GROUPS["L3_momentum"]),
    "stage5": (FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"]
               + FEATURE_GROUPS["L2_efficiency_pace"] + FEATURE_GROUPS["L3_momentum"]
               + FEATURE_GROUPS["L4_spatial"]),
    "stage6": (FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"]
               + FEATURE_GROUPS["L2_efficiency_pace"] + FEATURE_GROUPS["L3_momentum"]
               + FEATURE_GROUPS["L4_spatial"] + FEATURE_GROUPS["L5_continuity"]),
    "stage7": (FEATURE_GROUPS["L0_context"] + FEATURE_GROUPS["L1_counts"]
               + FEATURE_GROUPS["L2_efficiency_pace"] + FEATURE_GROUPS["L3_momentum"]
               + FEATURE_GROUPS["L4_spatial"] + FEATURE_GROUPS["L5_continuity"]
               + FEATURE_GROUPS["L6_exogenous"]),
}

LEGACY_V1_FEATURES = (
    ["f_period", "f_time_remaining_game", "f_score_margin", "f_is_home"]
    + FEATURE_GROUPS["L1_counts"][:12]          # counting 部分 (12)
    + ["f_home_fga", "f_away_fga", "f_home_fg_pct", "f_away_fg_pct",
       "f_fg_pct_diff", "f_home_pts", "f_away_pts", "f_net_rating"]
    + ["f_score_slope", "efg_volatility"]
)

ABLATION_ARMS = {
    "legacy": (LEGACY_V1_FEATURES, "26-dim legacy (independent baseline)"),
    "stage1": (FEATURE_STAGES["stage1"], "10-dim L0: Context"),
    "stage2": (FEATURE_STAGES["stage2"], "89-dim L0+L1: 3-window Raw Counts"),
    "stage3": (FEATURE_STAGES["stage3"], "107-dim L0-L2: +Efficiency & Pace"),
    "stage4": (FEATURE_STAGES["stage4"], "112-dim L0-L3: +Momentum"),
    "stage5": (FEATURE_STAGES["stage5"], "118-dim L0-L4: +Spatial"),
    "stage6": (FEATURE_STAGES["stage6"], "122-dim L0-L5: +Continuity (full)"),
    "stage7": (FEATURE_STAGES["stage7"], "141-dim L0-L6: +Team priors (full)"),
}

TUNE_ON_CHOICES = ["legacy", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7"]

USE_GPU = True

HORIZON_SECONDS = 180.0
RUN_THRESHOLD = 10

def xgb_device_params(use_gpu: bool = True) -> dict:
    """Return XGBoost device parameters (GPU or CPU)."""
    if use_gpu:
        return {"device": "cuda", "tree_method": "hist"}
    return {"tree_method": "hist"}

LGB_DEFAULT_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.05,
    "min_data_in_leaf": 50, "feature_fraction": 0.5,
    "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l1": 0.5, "lambda_l2": 4.0,
    "verbosity": -1, "seed": 42,
}

LGB_SEARCH_SPACE = {
    "num_leaves": [8, 16, 31, 63],
    "learning_rate": [0.01, 0.03, 0.05],
    "min_data_in_leaf": [20, 50, 100, 200],
    "feature_fraction": [0.3, 0.5, 0.7],
    "bagging_fraction": [0.5, 0.6, 0.7, 0.8],
    "bagging_freq": [1, 5],
    "lambda_l1": [0.5, 1.0, 2.0, 4.0],
    "lambda_l2": [8.0, 16.0, 32.0, 64.0],
    "gamma_fl": [1.5, 2.0, 2.5],
    "alpha_fl": [0.75, 0.9],
}

CB_DEFAULT_PARAMS = {
    "depth": 3, "learning_rate": 0.05,
    "l2_leaf_reg": 8.0, "random_strength": 1.0,
    "bagging_temperature": 0.5, "min_data_in_leaf": 50,
    "border_count": 128, "verbose": 0, "random_seed": 42,
}

CB_SEARCH_SPACE = {
    "depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05],
    "l2_leaf_reg": [8.0, 16.0, 32.0, 64.0],
    "random_strength": [1.0, 2.0, 5.0],
    "bagging_temperature": [0.0, 0.5, 1.0],
    "min_data_in_leaf": [20, 50, 100],
    "scale_pos_weight": [30, 50, 65, 80],
    "border_count": [128, 254],
}
