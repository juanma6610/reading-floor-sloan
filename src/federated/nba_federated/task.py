"""
task.py — Data Loading & Partitioning for Federated NBA Shot Prediction

Supports two partitioning strategies:
  "team" — split by team_id (30 Non-IID clients) [PRIMARY EXPERIMENT]
  "iid"  — stratified random split (30 IID clients) [CONTROL EXPERIMENT]

All feature columns and preprocessing are kept identical to train_xgboost.py
so that federated results are directly comparable to the centralized baseline.

Two important properties are preserved:
  1. A SINGLE global train/test split (group-aware by game_id) is computed
     ONCE and shared by every caller, so the global test set is NEVER seen
     by any client (no data leakage between client training and centralised
     server-side evaluation).
  2. Each client's local train/eval split is also group-aware by game_id,
     mirroring train_xgboost.py.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Constants — must match train_xgboost.py exactly
# ──────────────────────────────────────────────────────────────
METADATA_COLS = ['player_name', 'game_time', 'quarter', 'score_margin',
                 'team_id', 'game_id', 'description', "closest_def_name"]
TARGET_COL = 'made_shot'
GROUP_COL  = 'game_id'

GLOBAL_TEST_SIZE = 0.15        # held-out for server-side evaluation
RANDOM_STATE     = 42

# Default data path (relative to the federated/ directory when running flwr)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / 'data' / 'shot_features_valid2_type.csv'
if not DEFAULT_DATA_PATH.parent.exists():
    DEFAULT_DATA_PATH = Path('/mnt/c/Users/juanm/Documents/KUL_MAI/reading-floor-sloan/data/shot_features_valid2_type.csv')

# Where every federated run writes its outputs: <project>/results/federated. Derived
# from the data path so it is right both in the repo and in Flower's bundled app copy.
OUTPUT_DIR = DEFAULT_DATA_PATH.parents[1] / "results" / "federated"


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return model input columns — everything except metadata and target."""
    return [c for c in df.columns if c not in METADATA_COLS + [TARGET_COL]]


# ──────────────────────────────────────────────────────────────
# Caches — ensure server and all clients see the SAME splits
# ──────────────────────────────────────────────────────────────
_CACHE_DF           = None
_CACHE_GLOBAL_SPLIT = None       # (train_idx_array, test_idx_array)


def load_full_dataset(data_path: str = None) -> pd.DataFrame:
    """Load the full shot features CSV once."""
    global _CACHE_DF
    if _CACHE_DF is not None:
        return _CACHE_DF
    path = data_path or str(DEFAULT_DATA_PATH)
    df = pd.read_csv(path)
    _CACHE_DF = df
    return df


def _get_global_split(data_path: str = None):
    """
    Compute (and cache) the group-aware global train/test split.

    Using GroupShuffleSplit on `game_id` guarantees that no game appears in
    both the federated training pool and the held-out global test set —
    matches the protocol used in train_xgboost.py.

    Falls back to a stratified random split if `game_id` is not available.
    """
    global _CACHE_GLOBAL_SPLIT
    if _CACHE_GLOBAL_SPLIT is not None:
        return _CACHE_GLOBAL_SPLIT

    df = load_full_dataset(data_path)

    if GROUP_COL in df.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=GLOBAL_TEST_SIZE, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(
            splitter.split(df, df[TARGET_COL], groups=df[GROUP_COL])
        )
    else:
        all_idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=GLOBAL_TEST_SIZE,
            random_state=RANDOM_STATE, stratify=df[TARGET_COL]
        )

    _CACHE_GLOBAL_SPLIT = (np.asarray(train_idx), np.asarray(test_idx))
    return _CACHE_GLOBAL_SPLIT


def get_global_make_rate(data_path: str = None) -> float:
    """
    Global positive-class rate, computed from TRAINING data only.
    Used to set xgboost `base_score` so every client/round starts from the
    correct prior — and so calibration is preserved across rounds.
    """
    df = load_full_dataset(data_path)
    train_idx, _ = _get_global_split(data_path)
    return float((df.iloc[train_idx][TARGET_COL] == 1).mean())


def get_team_ids(data_path: str = None) -> list:
    """Sorted list of team_ids present in the federated training pool."""
    df = load_full_dataset(data_path)
    train_idx, _ = _get_global_split(data_path)
    return sorted(df.iloc[train_idx]['team_id'].unique().tolist())


def partition_frames(
    partition_id: int,
    num_partitions: int = 30,
    strategy: str = "team",
    test_size: float = 0.20,
    data_path: str = None,
    random_state: int = 42,
) -> tuple:
    """
    A single federated client's local train / local-eval split as DataFrames.

    Important: the client's data is drawn ONLY from the global TRAIN portion.
    The held-out global test set is invisible to every client.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_full_dataset(data_path)
    feature_cols = _get_feature_cols(df)

    train_idx, _ = _get_global_split(data_path)
    train_pool = df.iloc[train_idx].reset_index(drop=True)

    if strategy == "team":
        team_ids = sorted(train_pool['team_id'].unique().tolist())
        if partition_id >= len(team_ids):
            raise ValueError(f"partition_id {partition_id} >= num teams {len(team_ids)}")
        team_id = team_ids[partition_id]
        local_df = train_pool[train_pool['team_id'] == team_id].copy()

    elif strategy == "iid":
        df_shuffled = train_pool.sample(frac=1, random_state=random_state).reset_index(drop=True)
        chunks = np.array_split(df_shuffled.index, num_partitions)
        local_df = df_shuffled.loc[chunks[partition_id]].copy()

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'team' or 'iid'.")

    X = local_df[feature_cols]
    y = local_df[TARGET_COL]

    # Group-aware local train/eval split when game_id is present and we have
    # >1 game; otherwise fall back to a stratified random split.
    used_group = False
    if GROUP_COL in local_df.columns and local_df[GROUP_COL].nunique() > 1:
        try:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            tr, te = next(splitter.split(X, y, groups=local_df[GROUP_COL]))
            X_train, X_test = X.iloc[tr], X.iloc[te]
            y_train, y_test = y.iloc[tr], y.iloc[te]
            used_group = True
        except ValueError:
            pass

    if not used_group:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    return X_train, X_test, y_train, y_test


def load_partition(
    partition_id: int,
    num_partitions: int = 30,
    strategy: str = "team",
    test_size: float = 0.20,
    data_path: str = None,
    random_state: int = 42,
) -> tuple:
    """
    Load training and local-evaluation data for a single federated client.

    Returns:
        train_dmatrix, test_dmatrix, num_train, num_test
    """
    X_train, X_test, y_train, y_test = partition_frames(
        partition_id, num_partitions, strategy, test_size, data_path, random_state
    )
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_dmatrix  = xgb.DMatrix(X_test,  label=y_test)

    return train_dmatrix, test_dmatrix, len(X_train), len(X_test)


def load_global_test_set(data_path: str = None, **_legacy_kwargs) -> tuple:
    """
    Held-out global test set for centralised server-side evaluation.

    Group-aware (by game_id) so test games are completely disjoint from every
    client's training data. Accepts (and ignores) legacy kwargs like
    `test_size` for backwards compatibility.

    Returns:
        test_dmatrix, y_test (numpy array)
    """
    df = load_full_dataset(data_path)
    feature_cols = _get_feature_cols(df)
    _, test_idx = _get_global_split(data_path)

    test_df = df.iloc[test_idx]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL].values

    test_dmatrix = xgb.DMatrix(X_test, label=y_test)
    return test_dmatrix, y_test
