from pathlib import Path
import pandas as pd


def load_ferry_data(file_path: str) -> pd.DataFrame:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load CSV
    df = pd.read_csv(path)

    # Check required columns
    required_cols = ["Timestamp", "Sales Count", "Redemption Count"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Converted timestamp 
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Removed invalid timestamps
    df = df.dropna(subset=["Timestamp"])

    # Set index
    df = df.set_index("Timestamp")

    # Sorted for time series
    df = df.sort_index()

    # Removed duplicates
    df = df[~df.index.duplicated(keep="first")]

    # If empty → stop early
    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    # Created FULL 15-min timeline
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min"
    )

    df = df.reindex(full_index)

    # ================== HANDLE MISSING ==================

    # Interpolate (middle gaps)
    df["Sales Count"] = df["Sales Count"].interpolate(method="time")
    df["Redemption Count"] = df["Redemption Count"].interpolate(method="time")

    # Filled edges
    df["Sales Count"] = df["Sales Count"].bfill().ffill()
    df["Redemption Count"] = df["Redemption Count"].bfill().ffill()

    # Final safety
    if df.isna().sum().sum() > 0:
        print("⚠️ Warning: Remaining NaNs filled with 0")
        df = df.fillna(0)

    return df
