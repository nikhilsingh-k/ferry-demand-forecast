import pandas as pd
import numpy as np

def horizon_metrics(y_true, y_pred):
    horizons = {
        "15min": 1,
        "30min": 2,
        "1hr": 4,
        "2hr": 8
    }

    results = {}

    for name, step in horizons.items():
        y_t = y_true.iloc[::step]
        y_p = y_pred.iloc[::step]

        mae = np.mean(np.abs(y_t - y_p))
        rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        }

    return results
