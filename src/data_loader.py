from pathlib import Path
import pandas as pd


def load_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load and preprocess ferry demand dataset.

    Steps:
    - Validate required columns
    - Convert Timestamp to datetime
    - Set as index
    - Sort chronologically
    - Enforce 15-minute frequency
    - Handle missing timestamps using interpolation

    Returns:
    - Cleaned DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load data
    df = pd.read_csv(path)

    # Validate columns
    required_cols = ["Timestamp", "Sales Count", "Redemption Count"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Set index
    df.set_index("Timestamp", inplace=True)

    # Sort
    df.sort_index(inplace=True)

    # Create full 15-min index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="15min"
    )

    # Reindex
    df = df.reindex(full_index)

    # Interpolate missing values (time-based)
    df["Sales Count"] = df["Sales Count"].interpolate(method="time")
    df["Redemption Count"] = df["Redemption Count"].interpolate(method="time")

    # Final check
    if df.isna().sum().sum() > 0:
        raise ValueError("NaN values still present after interpolation")

    return df


# ================= TEST =================
if __name__ == "__main__":
    df = load_data("data/ferry_demand_data.csv")
    print(df.head())
    print(df.tail())
