# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 14:09:35 2025
Metals predictor V1.3
@author: Zane Hambly
"""

# models/xgboost_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def _make_features(series: pd.Series, lags: int = 10, rolling: int = 5):
    """
    Create lag and rolling features for a univariate series.
    """
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df[f"roll_mean_{rolling}"] = df["y"].shift(1).rolling(rolling).mean()
    df = df.dropna()
    return df

def fit_xgb_forecast(series: pd.Series, steps: int = 30,
                     lags: int = 10, rolling: int = 5,
                     **xgb_params):
    """
    Fit XGBoost on lagged features and forecast forward.
    """
    series = series.asfreq("B").ffill()
    df = _make_features(series, lags=lags, rolling=rolling)

    X, y = df.drop("y", axis=1), df["y"]
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        **xgb_params
    )
    model.fit(X, y)

    # Recursive forecasting
    last_row = df.iloc[-1].drop("y").values.reshape(1, -1)
    preds = []
    history = list(series.values)

    for _ in range(steps):
        # Build features from updated history
        temp = pd.Series(history[-lags:])
        feats = {}
        for lag in range(1, lags+1):
            feats[f"lag_{lag}"] = history[-lag]
        feats[f"roll_mean_{rolling}"] = pd.Series(history[-rolling:]).mean()
        X_next = pd.DataFrame([feats])
        yhat = model.predict(X_next)[0]
        preds.append(yhat)
        history.append(yhat)

    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                                   periods=steps, freq="B")
    forecast = pd.Series(preds, index=forecast_index, name="forecast")

    return forecast, model

def backtest_xgb(series: pd.Series, steps: int = 30,
                 lags: int = 10, rolling: int = 5,
                 **xgb_params):
    """
    Hold out last `steps` points, fit XGB on earlier data, forecast forward,
    and compute error metrics.
    """
    series = series.asfreq("B").ffill()
    train, test = series[:-steps], series[-steps:]

    forecast, model = fit_xgb_forecast(train, steps=steps,
                                       lags=lags, rolling=rolling,
                                       **xgb_params)
    forecast.index = test.index

    mape = mean_absolute_percentage_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test.values, forecast.values)

    return forecast, test, mape, rmse, r2
