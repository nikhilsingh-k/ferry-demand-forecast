import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # SORT 

    df = df.sort_index()

    # TIME FEATURES
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # LAG FEATURES
    df["lag_1"] = df["Sales Count"].shift(1)
    df["lag_2"] = df["Sales Count"].shift(2)
    df["lag_4"] = df["Sales Count"].shift(4)
    df["lag_8"] = df["Sales Count"].shift(8)

    # DAILY SEASONALITY 
    df["lag_96"] = df["Sales Count"].shift(96)

    # ROLLING FEATURES
    df["rolling_mean_4"] = df["Sales Count"].rolling(4).mean()
    df["rolling_mean_96"] = df["Sales Count"].rolling(96).mean()

    df["rolling_std_4"] = df["Sales Count"].rolling(4).std()
    df["rolling_max_4"] = df["Sales Count"].rolling(4).max()

    # DROP NaNs 
    
    df = df.dropna()

    return df
