# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:53:44 2025

@author: Zane Hambly
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def fit_arima_forecast(series: pd.Series, steps: int = 30,
                       order=(2,1,2), seasonal_order=(1,1,1,12)):
    """
    Fit a SARIMAX model and return forecast + confidence intervals.
    """
    # Ensure business-day frequency and fill gaps
    series = series.asfreq("B").ffill()

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    fit = model.fit(disp=False)

    pred = fit.get_forecast(steps=steps)
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    forecast.index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="B"
    )
    conf_int.index = forecast.index

    return forecast, conf_int


def backtest_arima(series: pd.Series, steps: int = 30,
                   order=(2,1,2), seasonal_order=(1,1,1,12)):
    """
    Hold out the last `steps` points, fit SARIMAX on the rest,
    forecast forward, and compute error metrics.
    """
    # Ensure business-day frequency and fill gaps
    series = series.asfreq("B").ffill()

    train, test = series[:-steps], series[-steps:]

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fit = model.fit(disp=False)

    pred = fit.get_forecast(steps=steps)
    forecast = pred.predicted_mean

    # Align forecast index with test
    forecast.index = test.index

    # Error metrics
    mape = mean_absolute_percentage_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    
    r2 = r2_score(test.values, forecast.values)

    return forecast, test, mape, rmse, r2

