# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 19:46:21 2025

@author: GGPC
"""

# test_arima_v1_2.py
import pandas as pd
from src.data_loader import get_price_data
from models.arima_model import fit_arima_forecast, backtest_arima, rolling_backtest

# --- Load data ---
df = get_price_data("Gold", period="5y", interval="1d")

print("Data preview:")
print(df.head())
print("Columns:", df.columns)

# --- Run tests for different trend settings ---
for trend in ["none", "linear", "quadratic"]:
    print(f"\n=== Testing trend = {trend} ===")

    # Forecast
    forecast, conf_int = fit_arima_forecast(df, steps=30, trend=trend)
    print("Forecast head:")
    print(forecast.head())

    # Backtest
    forecast_bt, test, mape, rmse, r2 = backtest_arima(df, steps=30, trend=trend)
    print("Backtest metrics:")
    print(f"MAPE: {mape:.2%}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
    
    df = get_price_data("Gold", period="5y", interval="1d")

    results = rolling_backtest(df, steps=20, n_splits=5, trend="quadratic")
    print(results)
    print("Average MAPE:", results["MAPE"].mean())


