# models/arima_model.py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def _ensure_series(data: pd.DataFrame | pd.Series) -> pd.Series:
    """Ensure we always return a numeric Series from input data."""
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        # Prefer common column names
        for col in ["Price", "Close", "Adj Close"]:
            if col in data.columns:
                return data[col]
        # Otherwise take the first numeric column
        num_cols = data.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            return data[num_cols[0]]
    raise ValueError("Input must be a Series or DataFrame with a numeric column")

def fit_arima_forecast(series: pd.Series, steps: int = 30,
                       order=(1,1,1), seasonal_order=(0,1,1,5),
                       trend: str = "none", log_transform: bool = False):
    """
    v1.2: Fit SARIMAX with optional linear/quadratic time trend regressor.
    """
    series = _ensure_series(series).asfreq("B").ffill()

    # Build exogenous regressors
    t = np.arange(len(series))
    if trend == "linear":
        exog = t.reshape(-1, 1)
    elif trend == "quadratic":
        exog = np.column_stack([t, t**2])
    else:
        exog = None

    model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    # Future exog
    if exog is not None:
        t_future = np.arange(len(series), len(series) + steps)
        if trend == "linear":
            exog_future = t_future.reshape(-1, 1)
        else:
            exog_future = np.column_stack([t_future, t_future**2])
    else:
        exog_future = None

    pred = fit.get_forecast(steps=steps, exog=exog_future)
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()

    forecast.index = pd.date_range(series.index[-1] + pd.Timedelta(days=1),
                                   periods=steps, freq="B")
    conf_int.index = forecast.index
    return forecast, conf_int

def backtest_arima(series: pd.Series, steps: int = 30,
                   order=(1,1,1), seasonal_order=(0,1,1,5),
                   trend: str = "none", log_transform: bool = False):
    """
    v1.2: Hold out the last `steps` points, fit SARIMAX with optional trend regressor,
    forecast forward, and compute error metrics.
    """
    series = _ensure_series(series).asfreq("B").ffill()
    train, test = series[:-steps], series[-steps:]

    t_train = np.arange(len(train))
    t_test = np.arange(len(train), len(train) + steps)

    if trend == "linear":
        exog_train = t_train.reshape(-1, 1)
        exog_test = t_test.reshape(-1, 1)
    elif trend == "quadratic":
        exog_train = np.column_stack([t_train, t_train**2])
        exog_test = np.column_stack([t_test, t_test**2])
    else:
        exog_train = exog_test = None

    model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    pred = fit.get_forecast(steps=steps, exog=exog_test)
    forecast = pred.predicted_mean
    forecast.index = test.index

    mape = mean_absolute_percentage_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test.values, forecast.values)

    return forecast, test, mape, rmse, r2

def rolling_backtest(series: pd.Series, steps: int = 20, n_splits: int = 5,
                     order=(1,1,1), seasonal_order=(0,1,1,5), trend: str = "none", log_transform: bool = False):
    """
    v1.2: Rolling backtest with n_splits folds.
    Each fold trains up to a point and forecasts the next `steps` trading days.
    Returns a DataFrame of metrics for each fold.
    """
    series = _ensure_series(series).asfreq("B").ffill()
    metrics = []

    for i in range(n_splits):
        split_point = len(series) - (n_splits - i) * steps
        train, test = series[:split_point], series[split_point:split_point+steps]

        # Build exog regressors
        t_train = np.arange(len(train))
        t_test = np.arange(len(train), len(train) + steps)

        if trend == "linear":
            exog_train = t_train.reshape(-1, 1)
            exog_test = t_test.reshape(-1, 1)
        elif trend == "quadratic":
            exog_train = np.column_stack([t_train, t_train**2])
            exog_test = np.column_stack([t_test, t_test**2])
        else:
            exog_train = exog_test = None

        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)

        pred = fit.get_forecast(steps=len(test), exog=exog_test)
        forecast = pred.predicted_mean
        forecast.index = test.index

        mape = mean_absolute_percentage_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        r2 = r2_score(test.values, forecast.values)

        metrics.append({"Fold": i+1, "MAPE": mape, "RMSE": rmse, "R²": r2})

    return pd.DataFrame(metrics)

