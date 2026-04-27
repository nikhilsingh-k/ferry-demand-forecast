import pandas as pd
from prophet import Prophet

def prophet_forecast(df, periods):

    df_prophet = df.copy()

    df_prophet = df_prophet.reset_index()

    df_prophet.columns = ["ds"] + list(df_prophet.columns[1:])
    
    df_prophet = df_prophet[["ds", "Sales Count"]]
    df_prophet.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods, freq="15min")

    forecast = model.predict(future)

    result = forecast.set_index("ds")["yhat"].tail(periods)

    return result
