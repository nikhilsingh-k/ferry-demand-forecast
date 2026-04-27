import pandas as pd
from typing import Tuple


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-series train-test split.

    - No shuffling
    - Chronological order preserved
    - Target = 'Sales Count'
    """


    # VALIDATIONS --------------------------------------------------------------

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    if "Sales Count" not in df.columns:
        raise ValueError("Target column 'Sales Count' not found")

    if len(df) < 10:
        raise ValueError("Dataset too small for splitting")

    # Ensure sorted
    df = df.sort_index()

    # SPLIT ------------------------------------------------------------------------------
 
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    
    # FEATURES & TARGET -------------------------------------------------------------------------
   
    target_col = "Sales Count"
    feature_cols = [col for col in df.columns if col != target_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

 
    # LOGGING --------------------------------------------------------------------------------------

    print("✅ Time split complete")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Train range: {X_train.index.min()} → {X_train.index.max()}")
    print(f"Test range : {X_test.index.min()} → {X_test.index.max()}")

    return X_train, X_test, y_train, y_test


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_ferry_data
    from src.features import create_features

    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_features = create_features(df_raw)

    X_train, X_test, y_train, y_test = time_split(df_features)

    print(X_train.head())
    print(y_train.head())
