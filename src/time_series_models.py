import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(y_train, steps):
    model = ARIMA(y_train, order=(1,1,1))
    fitted = model.fit()

    forecast = fitted.forecast(steps=steps)

    forecast.index = pd.date_range(
        start=y_train.index[-1],
        periods=steps+1,
        freq="15min"
    )[1:]

    return forecast
