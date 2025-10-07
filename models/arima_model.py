# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:53:44 2025

@author: Zane Hambly
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_arima_forecast(series: pd.Series, steps: int = 30):
    """
    Fit a SARIMAX model and return forecast + confidence intervals.
    """
    model = SARIMAX(series, order=(2,1,2), seasonal_order=(1,1,1,12))
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

