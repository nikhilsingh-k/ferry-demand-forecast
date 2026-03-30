from pathlib import Path
import pandas as pd


def load_ferry_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load ferry dataset strictly following project constraints.

    Steps:
    - No column renaming
    - Strict datetime conversion
    - Chronological ordering
    - Enforce 15-minute intervals
    - Handle missing timestamps via interpolation ONLY
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load
    df = pd.read_csv(path)

    # Validate required columns
    required_cols = ["Timestamp", "Sales Count", "Redemption Count"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert timestamp (STRICT)
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

    # Final validation
    if df.isna().sum().sum() > 0:
        raise ValueError("NaNs remain after interpolation")

    return df


# TEST
if __name__ == "__main__":
    df = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    print(df.head())
    print(df.tail())
