import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# ======================================================
# 1. NAIVE FORECAST
# ======================================================
def naive_forecast(y_train: pd.Series, y_test_index: pd.Index) -> pd.Series:
    """
    Naive forecast: last observed value repeated
    """
    last_value = y_train.iloc[-1]

    predictions = np.full(len(y_test_index), last_value)

    return pd.Series(predictions, index=y_test_index, name="Naive")


# ======================================================
# 2. MOVING AVERAGE FORECAST
# ======================================================
def moving_average_forecast(
    y_train: pd.Series,
    y_test_index: pd.Index,
    window: int = 4
) -> pd.Series:
    """
    Moving average forecast (recursive style)
    """

    if len(y_train) < window:
        raise ValueError("Not enough data for moving average window")

    history = list(y_train.iloc[-window:])
    predictions = []

    for _ in range(len(y_test_index)):
        pred = np.mean(history[-window:])
        predictions.append(pred)

        # recursive update
        history.append(pred)

    return pd.Series(predictions, index=y_test_index, name=f"MA_{window}")


# ======================================================
# 3. LINEAR REGRESSION
# ======================================================
def linear_regression_forecast(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> pd.Series:
    """
    Linear regression model
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return pd.Series(predictions, index=X_test.index, name="LinearRegression")


# ================= TEST =================
if __name__ == "__main__":
    from src.data_loader import load_ferry_data
    from src.features import create_features
    from src.train_test_split import time_split

    df_raw = load_ferry_data("data/Toronto Island Ferry Tickets.csv")
    df_features = create_features(df_raw)

    X_train, X_test, y_train, y_test = time_split(df_features)

    # Predictions
    naive_pred = naive_forecast(y_train, y_test.index)
    ma_pred = moving_average_forecast(y_train, y_test.index)
    lr_pred = linear_regression_forecast(X_train, y_train, X_test)

    print(naive_pred.head())
    print(ma_pred.head())
    print(lr_pred.head())
