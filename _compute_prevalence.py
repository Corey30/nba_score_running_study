import numpy as np, pandas as pd
from pathlib import Path
from config import LOCAL_DATA_DIR, HORIZON_SECONDS, RUN_THRESHOLD
from preprocessing import load_pbp_files, reconstruct_scores, generate_labels


def elapsed_v(clock, period):
    period = pd.to_numeric(period, errors="coerce").astype(float)
    m = clock.astype(str).str.extract(r"PT(\d+)M(\d+\.?\d*)S")
    minutes = pd.to_numeric(m[0], errors="coerce")
    seconds = pd.to_numeric(m[1], errors="coerce")
    remaining = minutes * 60.0 + seconds
    dur = np.where(period <= 4, 720.0, 300.0)
    prior = np.where(period <= 5, (period - 1) * 720.0, 2880.0 + (period - 5) * 300.0)
    return prior + (dur - remaining)


data_dir = Path(LOCAL_DATA_DIR)
tot_pos = 0
tot_n = 0
for s in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]:
    df = load_pbp_files(data_dir, seasons=[s])
    df["abs_time"] = elapsed_v(df["clock"], df["period"])
    df = df.dropna(subset=["abs_time"])
    df = reconstruct_scores(df)
    df = generate_labels(df, HORIZON_SECONDS, RUN_THRESHOLD)
    pos = int(df["y"].sum())
    n = len(df)
    tot_pos += pos
    tot_n += n
    gids = df["gameid"].astype(str)
    n_games = gids.nunique()
    n_playoff = gids[gids.str.startswith("004")].nunique()
    print(f"{s}: games={n_games} playoff_games={n_playoff} pos={pos} total={n} prev={pos/n:.6f}", flush=True)
print(f"TOTAL: pos={tot_pos} total={tot_n} prev={tot_pos/tot_n:.6f}", flush=True)
