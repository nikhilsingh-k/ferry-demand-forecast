import pandas as pd
import numpy as np

def rolling_forecast_validation(
    model_func,
    X,
    y,
    initial_train_size=50000,
    step_size=5000,
    horizon=96
):
    """
    Walk-forward (rolling) validation

    Parameters:
    - model_func: forecasting function
    - X, y: full dataset
    - initial_train_size: starting training window
    - step_size: how much to move forward each iteration
    - horizon: forecast length

    Returns:
    - DataFrame with errors
    """

    results = []

    total_length = len(X)

    for start in range(initial_train_size, total_length - horizon, step_size):

        X_train = X.iloc[:start]
        y_train = y.iloc[:start]

        X_test = X.iloc[start:start + horizon]
        y_test = y.iloc[start:start + horizon]

        try:
            pred = model_func(X_train, y_train, X_test)

            pred = pred.reindex(X_test.index).ffill().bfill().fillna(0)

            error = (y_test - pred).abs().mean()

            results.append({
                "train_end": X_train.index[-1],
                "test_start": X_test.index[0],
                "MAE": error
            })

        except Exception:
            continue

    return pd.DataFrame(results)
