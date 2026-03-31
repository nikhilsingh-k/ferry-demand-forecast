import pandas as pd
from typing import List


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for ferry demand forecasting.

    - Uses only past data (no leakage)
    - Preserves original columns
    """

    df = df.copy()

    # ======================================================
    # 1. LAG FEATURES
    # ======================================================
    lags: List[int] = [1, 2, 4, 8]

    for lag in lags:
        df[f"Sales Count_lag_{lag}"] = df["Sales Count"].shift(lag)
        df[f"Redemption Count_lag_{lag}"] = df["Redemption Count"].shift(lag)

    # ======================================================
    # 2. ROLLING FEATURES
    # ======================================================
    windows: List[int] = [4, 8]

    for window in windows:
        # Sales
        df[f"Sales Count_rolling_mean_{window}"] = df["Sales Count"].rolling(window).mean()
        df[f"Sales Count_rolling_std_{window}"] = df["Sales Count"].rolling(window).std()
        df[f"Sales Count_rolling_max_{window}"] = df["Sales Count"].rolling(window).max()

        # Redemption
        df[f"Redemption Count_rolling_mean_{window}"] = df["Redemption Count"].rolling(window).mean()
        df[f"Redemption Count_rolling_std_{window}"] = df["Redemption Count"].rolling(window).std()
        df[f"Redemption Count_rolling_max_{window}"] = df["Redemption Count"].rolling(window).max()

    # ======================================================
    # 3. TIME FEATURES
    # ======================================================
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    # ======================================================
    # DROP NaNs
    # ======================================================
    initial_rows = len(df)
    df = df.dropna()

    final_rows = len(df)

    print("✅ Feature engineering complete")
    print(f"Rows before: {initial_rows}, after: {final_rows}")
    print(f"Total columns: {len(df.columns)}")

    return df


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_ferry_data

    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_features = create_features(df_raw)

    print(df_features.head())
