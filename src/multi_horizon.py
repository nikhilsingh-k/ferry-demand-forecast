import pandas as pd


def create_multi_horizon_targets(df, horizons=[1, 2, 4, 8]):
    """
    horizons:
    1 = 15 min
    2 = 30 min
    4 = 1 hour
    8 = 2 hour
    """

    df = df.copy()

    for h in horizons:
        df[f"target_t+{h}"] = df["Sales Count"].shift(-h)

    df = df.dropna()

    return df
