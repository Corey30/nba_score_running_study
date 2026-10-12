"""Preprocessing: load, clock parsing, score reconstruction, label generation."""
import re, warnings
from collections import deque
import numpy as np, pandas as pd
from pathlib import Path
from config import *

CLOCK_PATTERN = re.compile(r"PT(\d+)M(\d+\.?\d*)S")
PERIOD_DURATION = {p: 720 for p in range(1, 5)}


def _period_dur(period: int) -> int:
    return PERIOD_DURATION.get(period, 300)

def clock_to_elapsed(clock_str: str, period: int) -> float:
    """Convert PTmmMss.ssS to seconds elapsed since tip-off."""
    m = CLOCK_PATTERN.match(str(clock_str))
    if not m:
        return np.nan
    minutes = float(m.group(1))
    seconds = float(m.group(2))
    remaining = minutes * 60 + seconds
    elapsed_in_period = _period_dur(period) - remaining
    prior = sum(_period_dur(p) for p in range(1, int(period)))
    return prior + elapsed_in_period


def load_pbp_files(data_dir: Path, seasons: list[str] | None = None) -> pd.DataFrame:
    """Read all pbp*.csv files from data_dir. Filter by seasons if given."""
    files = sorted(data_dir.glob("pbp*.csv"))
    if seasons is not None:
        files = [f for f in files if f.stem.replace("pbp", "") in seasons]
    if not files:
        raise FileNotFoundError(f"No pbp CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            dtype={
                "gameid": str, "period": float, "h_pts": float, "a_pts": float,
                "team": str, "player": str, "type": str, "subtype": str,
                "result": str, "season": str,
            },
            low_memory=False,
        )
        # 如果 CSV 缺少 season 列，从文件名推断
        if "season" not in df.columns:
            season_str = f.stem.replace("pbp", "")
            df["season"] = season_str
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(files)} season files -> {len(df):,} rows, {df['gameid'].nunique():,} games")
    return df


def reconstruct_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill h_pts and a_pts within each game."""
    df = df.copy()
    df["h_pts"] = pd.to_numeric(df["h_pts"], errors="coerce")
    df["a_pts"] = pd.to_numeric(df["a_pts"], errors="coerce")
    df["h_pts"] = df.groupby("gameid")["h_pts"].ffill().fillna(0)
    df["a_pts"] = df.groupby("gameid")["a_pts"].ffill().fillna(0)
    df["net_score"] = df["h_pts"] - df["a_pts"]
    return df


def identify_home_team(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row as home-team action by detecting which team increases h_pts."""
    df = df.copy()
    df["_h_pts_diff"] = df.groupby("gameid")["h_pts"].diff().fillna(0)
    home_mask = df["team"].notna() & (df["_h_pts_diff"] > 0)
    home_teams = df.loc[home_mask].groupby("gameid")["team"].first()
    df["home_team"] = df["gameid"].map(home_teams)
    df["is_home"] = df["team"] == df["home_team"]
    df.drop(columns=["_h_pts_diff"], inplace=True)
    return df


def add_team_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add game-level team-prior features describing each team's season-to-date
    performance, computed strictly from PRIOR games in the same season.

    For every event we attach, per team (home & away): points scored / allowed,
    net rating, win %, field-goal %, pace, and games played — all averaged over
    the team's earlier games only (earlier ``gameid`` within the season). This
    injects an external "team strength / form" signal and introduces no future
    leakage. Teams with no earlier games are filled with 0.0.
    """
    df = df.copy()

    made = (df["type"] == "Made Shot").astype(float)
    miss = (df["type"] == "Missed Shot").astype(float)
    tov = (df["type"] == "Turnover").astype(float)
    ft = (df["type"] == "Free Throw").astype(float)
    is_home_row = (df["team"] == df["home_team"]).astype(float)
    is_away_row = 1.0 - is_home_row

    g = df.groupby("gameid")
    game = pd.DataFrame({
        "season": g["season"].first(),
        "home_team": g["home_team"].first(),
        "home_pts": g["h_pts"].max(),
        "away_pts": g["a_pts"].max(),
        "home_fgm": (made * is_home_row).groupby(df["gameid"]).sum(),
        "home_fga": ((made + miss) * is_home_row).groupby(df["gameid"]).sum(),
        "away_fgm": (made * is_away_row).groupby(df["gameid"]).sum(),
        "away_fga": ((made + miss) * is_away_row).groupby(df["gameid"]).sum(),
        "home_pace": ((made + miss + tov + ft) * is_home_row).groupby(df["gameid"]).sum(),
        "away_pace": ((made + miss + tov + ft) * is_away_row).groupby(df["gameid"]).sum(),
    })
    game["home_pts"] = pd.to_numeric(game["home_pts"], errors="coerce")
    game["away_pts"] = pd.to_numeric(game["away_pts"], errors="coerce")

    # away team = the non-home team in each game
    teams_by_game = df.groupby("gameid")["team"].agg(
        lambda s: sorted({x for x in s.dropna() if x})
    )
    away_team = {}
    for gid, ts in teams_by_game.items():
        ht = game.loc[gid, "home_team"]
        away_team[gid] = next((t for t in ts if t != ht), ht)
    game["away_team"] = game.index.map(away_team)

    # team-game long table (one row per team per game)
    home_side = pd.DataFrame({
        "season": game["season"].values,
        "gameid": game.index.values,
        "team": game["home_team"].values,
        "pts_for": game["home_pts"].values,
        "pts_against": game["away_pts"].values,
        "fgm": game["home_fgm"].values,
        "fga": game["home_fga"].values,
        "pace": game["home_pace"].values,
    })
    away_side = pd.DataFrame({
        "season": game["season"].values,
        "gameid": game.index.values,
        "team": game["away_team"].values,
        "pts_for": game["away_pts"].values,
        "pts_against": game["home_pts"].values,
        "fgm": game["away_fgm"].values,
        "fga": game["away_fga"].values,
        "pace": game["away_pace"].values,
    })
    team_game = pd.concat([home_side, away_side], ignore_index=True)
    team_game["fg_pct"] = team_game["fgm"] / team_game["fga"].replace(0, np.nan)
    team_game["win"] = (team_game["pts_for"] > team_game["pts_against"]).astype(float)

    # chronological prior aggregates (earlier gameid within season ≈ earlier game)
    stats_cols = ["pts_for", "pts_against", "fg_pct", "win", "pace"]
    prior_cols = ["prior_pts_for", "prior_pts_against", "prior_fg_pct",
                  "prior_win_pct", "prior_pace", "prior_net", "prior_games"]
    team_game = team_game.sort_values(["season", "gameid"]).reset_index(drop=True)

    parts = []
    for (_season, team), grp in team_game.groupby(["season", "team"], sort=False):
        grp = grp.sort_values("gameid")
        prior_mean = grp[stats_cols].shift(1).expanding(min_periods=1).mean()
        grp = grp.copy()
        grp["prior_pts_for"] = prior_mean["pts_for"].values
        grp["prior_pts_against"] = prior_mean["pts_against"].values
        grp["prior_fg_pct"] = prior_mean["fg_pct"].values
        grp["prior_win_pct"] = prior_mean["win"].values
        grp["prior_pace"] = prior_mean["pace"].values
        grp["prior_net"] = grp["prior_pts_for"] - grp["prior_pts_against"]
        grp["prior_games"] = np.arange(len(grp))
        parts.append(grp)
    team_prior = pd.concat(parts, ignore_index=True)

    lookup = team_prior.set_index(["gameid", "team"])
    game = game.reset_index()  # gameid becomes a regular column
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays(
            [game["gameid"], game[f"{side}_team"]], names=["gameid", "team"]
        )
        for pc in prior_cols:
            game[f"{side}_{pc}"] = lookup[pc].reindex(idx).to_numpy()

    # broadcast game-level values to every event, then derive diffs
    feat_map = game.set_index("gameid")
    for side in ("home", "away"):
        for pc in prior_cols:
            df[f"f_{side}_{pc}"] = df["gameid"].map(feat_map[f"{side}_{pc}"]).fillna(0.0)

    df["f_prior_net_diff"] = df["f_home_prior_net"] - df["f_away_prior_net"]
    df["f_prior_win_pct_diff"] = df["f_home_prior_win_pct"] - df["f_away_prior_win_pct"]
    df["f_prior_pts_for_diff"] = df["f_home_prior_pts_for"] - df["f_away_prior_pts_for"]
    df["f_prior_def_diff"] = df["f_away_prior_pts_against"] - df["f_home_prior_pts_against"]
    df["f_rest_diff"] = df["f_home_prior_games"] - df["f_away_prior_games"]

    return df


def _future_peak_label(times: np.ndarray, values: np.ndarray,
                       horizon: float, threshold: float) -> np.ndarray:
    """For each event i, return 1 if ``values`` (= net_score, home - away) gains
    >= threshold anywhere within (t_i, t_i + horizon]; NaN if the game ends
    before the full horizon is observable.

    ``values`` is non-monotonic, so the label is computed from the window PEAK
    (running max), not the value at the horizon endpoint. O(n) per game.
    """
    n = len(times)
    end = times + horizon
    right = np.searchsorted(times, end, side="right") - 1  # last idx <= t_i+horizon
    valid = times[-1] >= end                              # game extends full horizon

    labels = np.where(valid, 0.0, np.nan)
    dq = deque()
    j = 1
    for i in range(n):
        if not valid[i]:
            continue
        # push candidate future events up to the horizon boundary
        while j <= right[i]:
            v = values[j]
            while dq and values[dq[-1]] <= v:
                dq.pop()
            dq.append(j)
            j += 1
        # drop events no longer in the forward window (index <= i)
        while dq and dq[0] <= i:
            dq.popleft()
        if dq and values[dq[0]] - values[i] >= threshold:
            labels[i] = 1.0
    return labels

def generate_labels(df: pd.DataFrame,
                    horizon_seconds: float = HORIZON_SECONDS,
                    run_threshold: float = RUN_THRESHOLD) -> pd.DataFrame:
    """Create binary label y: +10 net-score run within horizon_seconds.

    y[i] = 1 iff max(net_score[j] - net_score[i]) >= run_threshold over all
    j with t_i < t_j <= t_i + horizon_seconds (window peak, not endpoint).
    """
    df = df.copy()
    df = df.sort_values(["gameid", "abs_time"]).reset_index(drop=True)
    results = []

    for game_id, grp in df.groupby("gameid", sort=False):
        grp = grp.sort_values("abs_time").reset_index(drop=True)
        times = grp["abs_time"].to_numpy(dtype=float)
        nets = grp["net_score"].to_numpy(dtype=float)
        grp["y"] = _future_peak_label(times, nets, horizon_seconds, run_threshold)
        results.append(grp)

    merged = pd.concat(results, ignore_index=True)
    merged = merged.dropna(subset=["y"])
    merged["y"] = merged["y"].astype(int)
    merged = merged.sort_values(["gameid", "abs_time"]).reset_index(drop=True)
    return merged


def run_preprocessing(data_dir: Path, seasons: list[str] | None = None) -> pd.DataFrame:
    """Full Phase-1: load -> clock -> scores -> home team -> labels."""
    print("\n" + "=" * 60)
    print("MODULE 1 - Multi-Season Data Preprocessing & Label Generation")
    print("=" * 60)

    print("\n[1/5] Loading CSV files ...")
    df = load_pbp_files(data_dir, seasons=seasons)

    print("\n[2/5] Parsing clock -> absolute seconds ...")
    df["abs_time"] = df.apply(lambda r: clock_to_elapsed(r["clock"], int(r["period"])), axis=1)
    bad = df["abs_time"].isna().sum()
    if bad:
        print(f"      Dropping {bad} rows with unparseable clock.")
        df = df.dropna(subset=["abs_time"])
    print(f"      Clock range: {df['abs_time'].min():.0f} - {df['abs_time'].max():.0f} s")

    print("\n[3/5] Reconstructing scores (forward-fill within game) ...")
    df = reconstruct_scores(df)

    print("\n[4/5] Identifying home/away teams ...")
    df = identify_home_team(df)
    print(f"      Home-team action rows: {df['is_home'].mean() * 100:.1f}%")

    print("\n[4b/5] Adding season-to-date team-prior features ...")
    df = add_team_prior_features(df)
    print(f"      Added {len(FEATURE_GROUPS['L6_exogenous'])} team-prior features")

    print(f"\n[5/5] Generating scoring-run labels (horizon={HORIZON_SECONDS}s, threshold>={RUN_THRESHOLD:+} pts) ...")
    df = generate_labels(df, HORIZON_SECONDS, RUN_THRESHOLD)
    pos_rate = df["y"].mean() * 100
    print(f"      Positive rate: {pos_rate:.2f}%  ({df['y'].sum():,} / {len(df):,} events)")
    print(f"      Final dataset: {len(df):,} rows, {df['gameid'].nunique():,} games")
    print("=" * 60 + "\n")
    return df
