"""Feature engineering: rolling windows, slope, all features."""

import warnings

import numpy as np
import pandas as pd

from config import *  # noqa: F403

warnings.filterwarnings("ignore")

# Additional constants (not in config.py)

STAGE_DESCRIPTIONS: dict[str, str] = {
    "stage1": "L0: Static game context",
    "stage2": "L0+L1: 3-window raw event counts",
    "stage3": "L0-L2: +Efficiency & Pace",
    "stage4": "L0-L3: +Momentum",
    "stage5": "L0-L4: +Spatial shot profile",
    "stage6": "L0-L5: +Scoring continuity (full)",
    "stage7": "L0-L6: +Team priors (full)",
}

FEATURE_SET_CHOICES: dict[str, tuple[list[str], str]] = {
    "legacy": (LEGACY_V1_FEATURES, "Original v1 baseline (26 features, independent)"),
    "stage1": (FEATURE_STAGES["stage1"], STAGE_DESCRIPTIONS["stage1"]),
    "stage2": (FEATURE_STAGES["stage2"], STAGE_DESCRIPTIONS["stage2"]),
    "stage3": (FEATURE_STAGES["stage3"], STAGE_DESCRIPTIONS["stage3"]),
    "stage4": (FEATURE_STAGES["stage4"], STAGE_DESCRIPTIONS["stage4"]),
    "stage5": (FEATURE_STAGES["stage5"], STAGE_DESCRIPTIONS["stage5"]),
    "stage6": (FEATURE_STAGES["stage6"], STAGE_DESCRIPTIONS["stage6"]),
    "stage7": (FEATURE_STAGES["stage7"], STAGE_DESCRIPTIONS["stage7"]),
}
# Time helpers (used in _build_all_features)

def _period_start_seconds(period: float) -> float:
    return sum(720.0 if p < 5 else 300.0 for p in range(1, int(period)))

def _period_duration_seconds(period: float) -> float:
    return 720.0 if period <= 4 else 300.0

# Rolling helpers — O(n log n) per game via cumsum + binary search

def _rolling_count_at_times(
    times: np.ndarray,
    binary_series: np.ndarray,
    window: float,
) -> np.ndarray:
    cum = np.cumsum(binary_series)
    start_indices = np.searchsorted(times, times - window, side="left")
    return cum - np.where(start_indices > 0, cum[start_indices - 1], 0)

def _rolling_count_df(
    df: pd.DataFrame,
    col: str,
    window: float,
    new_name: str,
) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        idx = grp.index.values
        times = grp["abs_time"].values
        vals = grp[col].values.astype(float)
        results.iloc[idx] = _rolling_count_at_times(times, vals, window)
    return results.rename(new_name)

def _rolling_mean_df(
    df: pd.DataFrame,
    col: str,
    window: float,
    weight_col: str | None = None,
) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        idx = grp.index.values
        times = grp["abs_time"].values
        vals = grp[col].values.astype(float)
        w = grp[weight_col].values.astype(float) if weight_col else np.ones(len(vals))

        vw = vals * w
        cum_v = np.cumsum(vw)
        cum_w = np.cumsum(w)
        starts = np.searchsorted(times, times - window, side="left")
        n = cum_w - np.where(starts > 0, cum_w[starts - 1], 0)
        sum_v = cum_v - np.where(starts > 0, cum_v[starts - 1], 0)
        results.iloc[idx] = np.where(n > 0, sum_v / n, 0.0)
    return results

def _rolling_slope_col_df(
    df: pd.DataFrame,
    col: str,
    window: float,
    new_name: str,
) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        idx = grp.index.values
        times = grp["abs_time"].values.astype(float)
        scores = grp[col].values.astype(float)
        slopes = np.zeros(len(times))
        for i in range(len(times)):
            past_idx = np.searchsorted(times, times[i] - window, side="left")
            if past_idx < i:
                dt = times[i] - times[past_idx]
                if dt > 1e-6:
                    slopes[i] = (scores[i] - scores[past_idx]) / dt
        results.iloc[idx] = slopes
    return results.rename(new_name)

def _rolling_slope_df(df: pd.DataFrame, window: float) -> pd.Series:
    return _rolling_slope_col_df(df, "net_score", window, "f_score_slope")

def _rolling_std_df(
    df: pd.DataFrame,
    col: str,
    window: float,
    weight_col: str | None = None,
) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        idx = grp.index.values
        times = grp["abs_time"].values
        vals = grp[col].values.astype(float)
        w = grp[weight_col].values.astype(float) if weight_col else np.ones(len(vals))

        vw = vals * w
        cum_v = np.cumsum(vw)
        cum_sq = np.cumsum(vals * vals * w)
        cum_w = np.cumsum(w)

        starts = np.searchsorted(times, times - window, side="left")
        n = cum_w - np.where(starts > 0, cum_w[starts - 1], 0)
        sum_v = cum_v - np.where(starts > 0, cum_v[starts - 1], 0)
        sum_sq = cum_sq - np.where(starts > 0, cum_sq[starts - 1], 0)
        mean = np.where(n > 1, sum_v / n, 0.0)
        variance = np.where(n > 1, np.maximum(0.0, sum_sq / n - mean * mean), 0.0)
        results.iloc[idx] = np.sqrt(variance)
    return results

def _seconds_since_last_score(df: pd.DataFrame, delta_col: str, name: str) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        times = grp["abs_time"].values.astype(float)
        deltas = grp[delta_col].values.astype(float)
        secs = np.zeros(len(times))
        last_time = times[0]
        for i in range(len(times)):
            if deltas[i] > 0:
                last_time = times[i]
            secs[i] = times[i] - last_time
        results.iloc[grp.index] = secs
    return results.rename(name)

def _consecutive_score_streak(df: pd.DataFrame, delta_col: str, name: str) -> pd.Series:
    results = pd.Series(0.0, index=df.index, dtype=float)
    for _game_id, grp in df.groupby("gameid"):
        scored = (grp[delta_col].values > 0).astype(int)
        streak = np.zeros(len(scored))
        count = 0
        for i, flag in enumerate(scored):
            count = count + 1 if flag else 0
            streak[i] = count
        results.iloc[grp.index] = streak
    return results.rename(name)

# Feature set resolution

def resolve_feature_set(name: str) -> tuple[list[str], str]:
    """Return (column list, description) for a named feature set."""
    key = name.lower().strip()
    aliases = {"stage1": "stage1", "stage2": "stage2", "stage3": "stage3",
               "stage4": "stage4", "stage5": "stage5", "stage6": "stage6",
               "stage7": "stage7",
               "1": "stage1", "2": "stage2", "3": "stage3",
               "4": "stage4", "5": "stage5", "6": "stage6", "7": "stage7"}
    key = aliases.get(key, key)
    if key not in FEATURE_SET_CHOICES:
        opts = ", ".join(FEATURE_SET_CHOICES)
        raise ValueError(f"Unknown feature set {name!r}. Choose: {opts}")
    cols, desc = FEATURE_SET_CHOICES[key]
    return list(cols), desc

def get_feature_cols(stage: int | str | None = None) -> list[str]:
    """Return feature columns for an ablation arm or named feature set."""
    if stage is None:
        return list(FEATURE_STAGES["stage6"])
    if isinstance(stage, str) and stage in FEATURE_SET_CHOICES:
        return list(FEATURE_SET_CHOICES[stage][0])
    key = str(stage) if stage != "legacy" else "legacy"
    if key == "legacy":
        return list(LEGACY_V1_FEATURES)
    if isinstance(stage, int) and str(stage) in {"1", "2", "3", "4", "5", "6", "7"}:
        return list(FEATURE_STAGES[f"stage{stage}"])
    if key in ABLATION_ARMS:
        return list(ABLATION_ARMS[key][0])
    raise ValueError(f"stage must be legacy, 1-6, or None; got {stage}")


# Public API

def _build_all_features(
    df: pd.DataFrame,
    window_seconds: float = 120.0,
    stage: int | None = None,
    max_stage: int = 6,
) -> tuple[pd.DataFrame, list[str]]:
    """Create rolling momentum features and append them to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`preprocessing.run_preprocessing`.
    window_seconds : float
        Primary rolling window (default 120 s).
    stage : int or None
        If given (1–6), return only that stage's feature columns.
        All features are still computed regardless.

    Returns
    -------
    (df, feature_cols)
    """
    df = df.copy()
    eps = 1e-8
    w = window_seconds

    is_turnover = (df["type"] == "Turnover").astype(float)
    is_foul = (df["type"] == "Foul").astype(float)
    is_made_shot = (df["type"] == "Made Shot").astype(float)
    is_missed_shot = (df["type"] == "Missed Shot").astype(float)
    is_rebound = (df["type"] == "Rebound").astype(float)
    is_free_throw = (df["type"] == "Free Throw").astype(float)
    is_timeout = (df["type"] == "Timeout").astype(float)
    is_sub = (df["type"] == "Substitution").astype(float)

    is_home = df["is_home"].astype(float)
    is_away = 1.0 - is_home
    is_home_team_row = (df["team"] == df["home_team"]).astype(float)
    is_away_team_row = 1.0 - is_home_team_row

    subtype_lower = df["subtype"].astype(str).str.lower()
    subtype_3p = subtype_lower.str.contains("3pt|three", na=False)
    result_3p = df["result"].astype(str).str.strip().eq("3")
    is_three_made = (is_made_shot * (subtype_3p | result_3p)).astype(float)
    is_two_made = is_made_shot - is_three_made
    is_three_attempt = ((is_made_shot + is_missed_shot) * subtype_3p).astype(float)

    is_oreb = (is_rebound * subtype_lower.str.contains("offensive", na=False)).astype(float)
    is_dreb = (is_rebound * subtype_lower.str.contains("defensive", na=False)).astype(float)

    masks = {
        "_h_tov": is_turnover * is_home,
        "_h_foul": is_foul * is_home,
        "_h_made": is_made_shot * is_home,
        "_h_miss": is_missed_shot * is_home,
        "_h_reb": is_rebound * is_home,
        "_h_ft": is_free_throw * is_home,
        "_a_tov": is_turnover * is_away,
        "_a_foul": is_foul * is_away,
        "_a_made": is_made_shot * is_away,
        "_a_miss": is_missed_shot * is_away,
        "_a_reb": is_rebound * is_away,
        "_a_ft": is_free_throw * is_away,
        "_h_3pm": is_three_made * is_home,
        "_a_3pm": is_three_made * is_away,
        "_h_3pa": is_three_attempt * is_home,
        "_a_3pa": is_three_attempt * is_away,
        "_h_oreb": is_oreb * is_home,
        "_a_oreb": is_oreb * is_away,
        "_h_dreb": is_dreb * is_home,
        "_a_dreb": is_dreb * is_away,
        "_h_timeout": is_timeout * is_home_team_row,
        "_a_timeout": is_timeout * is_away_team_row,
        "_h_sub": is_sub * is_home_team_row,
        "_a_sub": is_sub * is_away_team_row,
        "_home_event": is_home_team_row,
        "_away_event": is_away_team_row,
        "_any_event": np.ones(len(df)),
    }
    for col, values in masks.items():
        df[col] = values

    df["f_period"] = df["period"].astype(int)
    # Regulation-time remaining (48 min = 2880 s). Using the per-game observed
    # end time would leak whether/how long the game goes to OT; the regulation
    # clock is always known in advance, so this removes a small look-ahead leak.
    df["f_time_remaining_game"] = (2880.0 - df["abs_time"]).clip(lower=0.0)
    df["f_score_margin"] = df["net_score"]
    df["f_is_home"] = df["is_home"].astype(int)
    df["f_score_margin_abs"] = df["net_score"].abs()
    df["f_is_close"] = (df["f_score_margin_abs"] <= 5).astype(int)
    df["f_is_blowout"] = (df["f_score_margin_abs"] > 15).astype(int)
    df["f_is_overtime"] = (df["period"] >= 5).astype(int)

    period_start = df["period"].map(_period_start_seconds)
    period_dur = df["period"].map(_period_duration_seconds)
    df["f_time_in_period"] = df["abs_time"] - period_start
    period_remaining = period_dur - df["f_time_in_period"]
    df["f_is_clutch"] = (
        (df["period"] >= 4) & (period_remaining <= 300)
    ).astype(int)


    if max_stage <= 1:
        df.drop(columns=[c for c in list(masks.keys()) if c in df.columns], inplace=True)
        feature_cols = get_feature_cols("stage1")
        print(f"Feature engineering: L0 only ({len(feature_cols)} features), "
              f"skipped all rolling windows  [max_stage={max_stage}]")
        return df, feature_cols

    count_specs = [
        # basic (12)
        ("_h_tov", "f_home_tov"), ("_a_tov", "f_away_tov"),
        ("_h_foul", "f_home_fouls"), ("_a_foul", "f_away_fouls"),
        ("_h_made", "f_home_made"), ("_h_miss", "f_home_miss"),
        ("_a_made", "f_away_made"), ("_a_miss", "f_away_miss"),
        ("_h_reb", "f_home_reb"), ("_a_reb", "f_away_reb"),
        ("_h_ft", "f_home_ft"), ("_a_ft", "f_away_ft"),
        # fine-grained (12)
        ("_h_3pm", "f_home_3pm"), ("_a_3pm", "f_away_3pm"),
        ("_h_3pa", "f_home_3pa"), ("_a_3pa", "f_away_3pa"),
        ("_h_oreb", "f_home_oreb"), ("_a_oreb", "f_away_oreb"),
        ("_h_dreb", "f_home_dreb"), ("_a_dreb", "f_away_dreb"),
        ("_h_timeout", "f_home_timeout"), ("_a_timeout", "f_away_timeout"),
        ("_h_sub", "f_home_subs"), ("_a_sub", "f_away_subs"),
    ]
    for src_col, feat_name in count_specs:
        df[feat_name] = _rolling_count_df(df, src_col, 120.0, feat_name)
        df[feat_name + "_60s"] = _rolling_count_df(df, src_col, 60.0, feat_name + "_60s")
        df[feat_name + "_180s"] = _rolling_count_df(df, src_col, 180.0, feat_name + "_180s")

    df["f_home_3pt_pct"] = df["f_home_3pm"] / (df["f_home_3pa"] + eps)
    df["f_away_3pt_pct"] = df["f_away_3pm"] / (df["f_away_3pa"] + eps)
    df["f_3pt_pct_diff"] = df["f_home_3pt_pct"] - df["f_away_3pt_pct"]

    df["f_event_pace"] = _rolling_count_df(df, "_any_event", w, "f_event_pace_temp") / w
    df["f_home_event_pace"] = _rolling_count_df(df, "_home_event", w, "f_home_event_pace_temp") / w
    df["f_away_event_pace"] = _rolling_count_df(df, "_away_event", w, "f_away_event_pace_temp") / w
    df["f_pace_diff"] = df["f_home_event_pace"] - df["f_away_event_pace"]

    df["_h_pts_delta"] = df.groupby("gameid")["h_pts"].diff().fillna(0).clip(lower=0)
    df["_a_pts_delta"] = df.groupby("gameid")["a_pts"].diff().fillna(0).clip(lower=0)

    for window, suffix in ((60.0, "_60s"), (180.0, "_180s")):
        df[f"f_home_pts{suffix}"] = _rolling_count_df(
            df.assign(_h_pts_delta_val=df["_h_pts_delta"]),
            "_h_pts_delta_val", window, f"f_home_pts{suffix}_temp",
        )
        df[f"f_away_pts{suffix}"] = _rolling_count_df(
            df.assign(_a_pts_delta_val=df["_a_pts_delta"]),
            "_a_pts_delta_val", window, f"f_away_pts{suffix}_temp",
        )


    if max_stage <= 2:
        df.drop(columns=[c for c in list(masks.keys()) + ["_h_pts_delta", "_a_pts_delta"]
                         if c in df.columns], inplace=True)
        feature_cols = get_feature_cols("stage2")
        print(f"Feature engineering: L0+L1 ({len(feature_cols)} features), "
              f"skipped L2-L5  [max_stage={max_stage}]")
        return df, feature_cols

    df["f_home_fga"] = df["f_home_made"] + df["f_home_miss"]
    df["f_away_fga"] = df["f_away_made"] + df["f_away_miss"]
    df["f_home_fg_pct"] = df["f_home_made"] / (df["f_home_fga"] + eps)
    df["f_away_fg_pct"] = df["f_away_made"] / (df["f_away_fga"] + eps)
    df["f_fg_pct_diff"] = df["f_home_fg_pct"] - df["f_away_fg_pct"]

    df["f_home_efg_pct"] = (df["f_home_made"] + 0.5 * df["f_home_3pm"]) / (df["f_home_fga"] + eps)
    df["f_away_efg_pct"] = (df["f_away_made"] + 0.5 * df["f_away_3pm"]) / (df["f_away_fga"] + eps)
    df["f_efg_diff"] = df["f_home_efg_pct"] - df["f_away_efg_pct"]

    df["f_home_poss_est"] = df["f_home_fga"] + 0.44 * df["f_home_ft"] + df["f_home_tov"]
    df["f_away_poss_est"] = df["f_away_fga"] + 0.44 * df["f_away_ft"] + df["f_away_tov"]
    df["f_poss_diff"] = df["f_home_poss_est"] - df["f_away_poss_est"]

    df["f_home_pts"] = _rolling_count_df(
        df.assign(_h_pts_delta_val=df["_h_pts_delta"]),
        "_h_pts_delta_val", w, "f_home_pts_temp",
    )
    df["f_away_pts"] = _rolling_count_df(
        df.assign(_a_pts_delta_val=df["_a_pts_delta"]),
        "_a_pts_delta_val", w, "f_away_pts_temp",
    )
    df["f_net_rating"] = df["f_home_pts"] - df["f_away_pts"]

    if max_stage <= 3:
        df.drop(columns=[c for c in list(masks.keys()) + ["_h_pts_delta", "_a_pts_delta"]
                         if c in df.columns], inplace=True)
        feature_cols = get_feature_cols("stage3")
        print(f"Feature engineering: L0-L2 ({len(feature_cols)} features), "
              f"skipped L3-L5  [max_stage={max_stage}]")
        return df, feature_cols

    df["f_score_slope"] = _rolling_slope_df(df, w)
    df["f_home_pts_slope"] = _rolling_slope_col_df(df, "h_pts", w, "f_home_pts_slope")
    df["f_away_pts_slope"] = _rolling_slope_col_df(df, "a_pts", w, "f_away_pts_slope")
    df["f_slope_diff"] = df["f_home_pts_slope"] - df["f_away_pts_slope"]

    df["_shot_value"] = is_two_made * 2.0 + is_three_made * 3.0
    df["_is_shot"] = (is_made_shot + is_missed_shot).astype(float)
    df["efg_volatility"] = _rolling_std_df(df, "_shot_value", 180.0, "_is_shot") / 2.0

    if max_stage <= 4:
        df.drop(columns=[c for c in list(masks.keys()) + [
            "_h_pts_delta", "_a_pts_delta", "_shot_value", "_is_shot",
        ] if c in df.columns], inplace=True)
        feature_cols = get_feature_cols("stage4")
        print(f"Feature engineering: L0-L3 ({len(feature_cols)} features), "
              f"skipped L4-L5  [max_stage={max_stage}]")
        return df, feature_cols

    if "dist" in df.columns:
        shot_dist = pd.to_numeric(df["dist"], errors="coerce").fillna(0.0)
    else:
        shot_dist = pd.Series(0.0, index=df.index)

    df["_h_shot_dist"] = shot_dist * (is_made_shot + is_missed_shot) * is_home
    df["_a_shot_dist"] = shot_dist * (is_made_shot + is_missed_shot) * is_away
    df["_h_shot_flag"] = (is_made_shot + is_missed_shot) * is_home
    df["_a_shot_flag"] = (is_made_shot + is_missed_shot) * is_away
    df["_h_rim"] = ((shot_dist < 4.0) & (df["_h_shot_flag"] > 0)).astype(float)
    df["_a_rim"] = ((shot_dist < 4.0) & (df["_a_shot_flag"] > 0)).astype(float)

    df["f_avg_shot_dist_home"] = _rolling_mean_df(df, "_h_shot_dist", w, "_h_shot_flag")
    df["f_avg_shot_dist_away"] = _rolling_mean_df(df, "_a_shot_dist", w, "_a_shot_flag")
    df["f_rim_rate_home"] = _rolling_count_df(df, "_h_rim", w, "f_rim_rate_home_temp") / (
        _rolling_count_df(df, "_h_shot_flag", w, "f_h_shots_temp") + eps
    )
    df["f_rim_rate_away"] = _rolling_count_df(df, "_a_rim", w, "f_rim_rate_away_temp") / (
        _rolling_count_df(df, "_a_shot_flag", w, "f_a_shots_temp") + eps
    )
    df["f_three_rate_home"] = df["f_home_3pa"] / (df["f_home_fga"] + eps)
    df["f_three_rate_away"] = df["f_away_3pa"] / (df["f_away_fga"] + eps)

    if max_stage <= 5:
        df.drop(columns=[c for c in list(masks.keys()) + [
            "_h_pts_delta", "_a_pts_delta", "_shot_value", "_is_shot",
            "_h_shot_dist", "_a_shot_dist", "_h_shot_flag", "_a_shot_flag",
            "_h_rim", "_a_rim",
        ] if c in df.columns], inplace=True)
        feature_cols = get_feature_cols("stage5")
        print(f"Feature engineering: L0-L4 ({len(feature_cols)} features), "
              f"skipped L5  [max_stage={max_stage}]")
        return df, feature_cols

    df["f_secs_since_last_home_score"] = _seconds_since_last_score(
        df, "_h_pts_delta", "f_secs_since_last_home_score"
    )
    df["f_secs_since_last_away_score"] = _seconds_since_last_score(
        df, "_a_pts_delta", "f_secs_since_last_away_score"
    )
    df["f_home_score_streak"] = _consecutive_score_streak(
        df, "_h_pts_delta", "f_home_score_streak"
    )
    df["f_away_score_streak"] = _consecutive_score_streak(
        df, "_a_pts_delta", "f_away_score_streak"
    )

    temp_cols = list(masks.keys()) + [
        "_h_pts_delta", "_a_pts_delta",
        "_shot_value", "_is_shot",
        "_h_shot_dist", "_a_shot_dist",
        "_h_shot_flag", "_a_shot_flag",
        "_h_rim", "_a_rim",
    ]
    df.drop(columns=[c for c in temp_cols if c in df.columns], inplace=True)

    feature_cols = get_feature_cols(stage)
    stage_label = f"stage {stage}" if stage is not None else "all stages (stage6)"
    print(f"Feature engineering complete.  {len(feature_cols)} features selected ({stage_label}).")
    print(f"  Full catalog: {len(FEATURE_STAGES['stage6'])} features across 6 nested stages")
    print(f"    Stage1 (L0): {len(FEATURE_STAGES['stage1'])}  |  "
          f"Stage2 (L0+L1): {len(FEATURE_STAGES['stage2'])}  |  "
          f"Stage3 (L0-L2): {len(FEATURE_STAGES['stage3'])}  |  "
          f"Stage4 (L0-L3): {len(FEATURE_STAGES['stage4'])}  |  "
          f"Stage5 (L0-L4): {len(FEATURE_STAGES['stage5'])}  |  "
          f"Stage6 (L0-L5): {len(FEATURE_STAGES['stage6'])}")
    print(f"  Rolling window: {window_seconds:.0f} s")
    print(f"  Baseline positive rate: {df['y'].mean() * 100:.2f}%")

    return df, feature_cols
