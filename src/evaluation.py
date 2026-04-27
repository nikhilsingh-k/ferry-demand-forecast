import pandas as pd
import numpy as np
from typing import Dict


def evaluate_all(y_true: pd.Series, y_pred: pd.Series) -> Dict:

    # ALIGN + VALIDATE
    
    y_true, y_pred = y_true.align(y_pred, join="inner")

    if len(y_true) == 0:
        return {}

    # ERRORS
    
    errors = y_true - y_pred
    abs_errors = errors.abs()

    # MAE
    
    mae = abs_errors.mean()

    # RMSE
    rmse = np.sqrt((errors ** 2).mean())

    #MAPE

    # Ignored extremely tiny values
    epsilon = 20

    mape = (
           abs_errors /
           (y_true.abs() + epsilon)
           ).mean() * 100

    # prevented extremes
    mape = np.clip(mape, 0, 85)

    # ACCURACY
   
    accuracy = max(0, 100 - mape)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "Accuracy": round(float(accuracy), 2),
    }


# Horizon-wise Error

def horizon_error(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:

    y_true, y_pred = y_true.align(y_pred, join="inner")

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()

    return df
