import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def _safe_series(values, index, name="pred"):
    values = np.asarray(values, dtype=float).flatten()
    n = len(index)
    if len(values) < n:
        values = np.concatenate([values, np.full(n - len(values), values[-1] if len(values) else 0)])
    return pd.Series(np.clip(values[:n], 0, None), index=index, name=name)

# ALL MODELS FOLLOW: model(X_train, y_train, X_test)
def naive_forecast(X_train, y_train, X_test):

    if len(y_train) == 0:
        return pd.Series([0] * len(X_test), index=X_test.index)

    # Use same time from previous day (96 × 15min = 24h)
    seasonal_lag = 96

    preds = []

    for i in range(len(X_test)):
        if len(y_train) >= seasonal_lag + i:
            preds.append(y_train.iloc[-seasonal_lag + (i % seasonal_lag)])
        else:
            preds.append(y_train.iloc[-1])

    return pd.Series(preds, index=X_test.index)

def moving_average_forecast(X_train, y_train, X_test, window=24):

    if len(y_train) == 0:
        return pd.Series([0] * len(X_test), index=X_test.index)

    window = min(window, len(y_train))

    preds = []

    rolling_values = y_train.tail(window).tolist()

    for _ in range(len(X_test)):

        next_pred = np.mean(rolling_values[-window:])

        preds.append(next_pred)

        rolling_values.append(next_pred)

    return pd.Series(preds, index=X_test.index)

def linear_regression_forecast(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train.fillna(0), y_train.fillna(0))
    return _safe_series(model.predict(X_test.fillna(0)), X_test.index, "Linear Regression")

def random_forest_forecast(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train.fillna(0), y_train.fillna(0))
    return _safe_series(model.predict(X_test.fillna(0)), X_test.index, "Random Forest")

def gradient_boosting_forecast(X_train, y_train, X_test):
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    model.fit(X_train.fillna(0), y_train.fillna(0))
    return _safe_series(model.predict(X_test.fillna(0)), X_test.index, "Gradient Boosting")

def xgboost_forecast(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train.fillna(0), y_train.fillna(0))
    return _safe_series(model.predict(X_test.fillna(0)), X_test.index, "XGBoost")
