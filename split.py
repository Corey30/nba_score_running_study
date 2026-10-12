"""Dataset splitting: season-level train/val/test splits."""
import numpy as np
import pandas as pd
from config import *

def split_train_val_test_by_season(df, feature_cols, train_seasons, val_season, test_season):
    """按赛季划分 train/val/test，保证同一场比赛不跨集。
    """
    df_sorted = df.sort_values(["gameid", "abs_time"]).reset_index(drop=True)

    train_df = df_sorted[df_sorted["season"].astype(str).isin(train_seasons)]
    val_df   = df_sorted[df_sorted["season"].astype(str) == str(val_season)]
    test_df  = df_sorted[df_sorted["season"].astype(str) == str(test_season)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["y"].values.astype(int)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["y"].values.astype(int)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["y"].values.astype(int)

    print(f"  [split] train: {len(X_train):,} ({train_df['gameid'].nunique():,} games)"
          f"  |  val: {len(X_val):,} ({val_df['gameid'].nunique():,} games)"
          f"  |  test: {len(X_test):,} ({test_df['gameid'].nunique():,} games)")
    return X_train, X_val, X_test, y_train, y_val, y_test
