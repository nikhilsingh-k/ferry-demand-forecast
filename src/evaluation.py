import pandas as pd
import numpy as np
from typing import Dict


def evaluate_all(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    """
    Evaluate forecasting performance.

    Metrics:
    - MAE
    - RMSE
    - MAPE
    - Accuracy (%)
    """

    # ===============================
    # VALIDATION
    # ===============================
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Align index (important)
    y_true, y_pred = y_true.align(y_pred, join="inner")

    # ===============================
    # ERROR CALCULATIONS
    # ===============================
    errors = y_true - y_pred
    abs_errors = errors.abs()

    # MAE
    mae = abs_errors.mean()

    # RMSE
    rmse = np.sqrt((errors ** 2).mean())

    # MAPE (safe)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = (abs_errors[mask] / y_true[mask]).mean() * 100
    else:
        mape = np.nan

    # Accuracy
    accuracy = 100 - mape if not np.isnan(mape) else np.nan

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4) if not np.isnan(mape) else None,
        "Accuracy": round(float(accuracy), 2) if not np.isnan(accuracy) else None,
    }


# ======================================================
# HORIZON ERROR (SIMPLIFIED + CORRECT)
# ======================================================
def horizon_error(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Computes step-wise error across timeline.

    Returns:
    DataFrame with:
    - timestamp
    - absolute error
    """

    y_true, y_pred = y_true.align(y_pred, join="inner")

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()

    return df


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_ferry_data
    from src.features import create_features
    from src.train_test_split import time_split
    from src.baseline_models import (
        naive_forecast,
        moving_average_forecast,
        linear_regression_forecast
    )

    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_features = create_features(df_raw)

    X_train, X_test, y_train, y_test = time_split(df_features)

    naive_pred = naive_forecast(y_train, y_test.index)
    ma_pred = moving_average_forecast(y_train, y_test.index)
    lr_pred = linear_regression_forecast(X_train, y_train, X_test)

    print(evaluate_all(y_test, naive_pred))
    print(evaluate_all(y_test, ma_pred))
    print(evaluate_all(y_test, lr_pred))
