# -*- coding: utf-8 -*-
"""
Created on Tue Oct 7 15:54:35 2025

@author: Zane Hambly
"""

import streamlit as st
import pandas as pd
from src.data_loader import get_price_data
from models.arima_model import fit_arima_forecast, backtest_arima
import plotly.graph_objects as go

# --- Streamlit UI ---
st.set_page_config(page_title="Metal Price Predictor", layout="wide")

st.title("🪙 Metal Price Predictor")

# Sidebar controls
st.sidebar.header("Options")

metal = st.sidebar.selectbox(
    "Choose a metal",
    ["Gold", "Silver", "Copper", "Platinum", "Aluminum"]
)

model_type = st.sidebar.radio(
    "Model",
    ["Historical Data - No AI", "Long-term (ARIMA)"]
)

horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=30,
    max_value=90,
    value=30,
    step=30
)

# v1.1: Backtest horizon slider
backtest_horizon = st.sidebar.slider(
    "Backtest horizon (days to hold out)",
    min_value=30,
    max_value=90,
    value=30,
    step=30
)

# v1.1: Mode toggle (Forecast vs Backtest)
mode = st.sidebar.radio(
    "View",
    ["Forecast", "Backtest"]
)

# --- Load data ---
data = get_price_data(metal, period="5y", interval="1d")

# Convert Series -> DataFrame for Plotly
df_hist = data.reset_index()
df_hist.columns = ["Date", "Price"]

# --- Raw data chart (always shown) ---
st.subheader(f"{metal} Closing Prices (Last 5 Years)")

fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(
    x=df_hist["Date"],
    y=df_hist["Price"],
    mode="lines",
    name="Price",
    line=dict(color="black")  # solid black
))

# Constrain axes for the top chart
x_min_hist = df_hist["Date"].min()
x_max_hist = df_hist["Date"].max()
y_min_hist = df_hist["Price"].min()
y_max_hist = df_hist["Price"].max()

fig_raw.update_layout(
    title=f"{metal} Closing Prices",
    xaxis=dict(
        range=[x_min_hist, x_max_hist],
        dtick="M1",
        tickformat="%b\n%Y",
        fixedrange=True
    ),
    yaxis=dict(
        range=[y_min_hist, y_max_hist],
        title="Price (USD)",
        fixedrange=True
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
)

st.plotly_chart(fig_raw, use_container_width=True)
st.markdown("---")#v1.1 seperation
st.markdown("## Forecast vs Backtest")
if model_type != "Long-term (ARIMA)":
    st.info("To enable Forecast and Backtest, select **Long‑term (ARIMA)** in the sidebar options.") #v1.1 explainer to user to select option

# --- Forecast / Backtest v1.1 ---
if model_type == "Long-term (ARIMA)":
    # --- Forecast ---
    forecast, conf_int = fit_arima_forecast(data, steps=horizon)

    df_hist_plot = data.reset_index()
    df_hist_plot.columns = ["Date", "Price"]

    df_forecast = forecast.reset_index()
    df_forecast.columns = ["Date", "Price"]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=df_hist_plot["Date"], y=df_hist_plot["Price"],
        mode="lines", name="History", line=dict(color="black")
    ))
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast["Date"], y=df_forecast["Price"],
        mode="lines", name="Forecast", line=dict(color="blue")
    ))
    ci_lower, ci_upper = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast["Date"].tolist() + df_forecast["Date"].tolist()[::-1],
        y=ci_upper.tolist() + ci_lower.tolist()[::-1],
        fill="toself", fillcolor="rgba(0,0,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=True, name="Confidence interval"
    ))
    fig_forecast.update_layout(
        title=f"{metal} Price with {horizon}-Day Forecast",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
    )

    # --- Backtest ---
    forecast_bt, test, mape, rmse, r2 = backtest_arima(data, steps=backtest_horizon)

    df_test = test.reset_index(); df_test.columns = ["Date", "Actual"]
    df_forecast_bt = forecast_bt.reset_index(); df_forecast_bt.columns = ["Date", "Forecast"]

    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(
        x=df_test["Date"], y=df_test["Actual"],
        mode="lines", name="Actual", line=dict(color="black")
    ))
    fig_backtest.add_trace(go.Scatter(
        x=df_forecast_bt["Date"], y=df_forecast_bt["Forecast"],
        mode="lines", name="Forecast", line=dict(color="blue")
    ))
    fig_backtest.update_layout(
        title=f"{metal} Backtest ({backtest_horizon} days)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
    )

    # --- Side-by-side layout ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{horizon}-Day Forecast (ARIMA)")
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.caption("Black = historical prices | Blue = forecast | Shaded = confidence interval")

        st.dataframe(df_forecast)
        if horizon > 30:
            st.info(
                f"ℹ️ A {horizon}-day forecast carries more uncertainty. "
                "The shaded confidence band widens as the horizon increases. "
                "Use 60‑ and 90‑day forecasts as scenario ranges rather than precise predictions."
            )
    with col2:
        st.subheader(f"Backtest (ARIMA) - last {backtest_horizon} days held out")
        st.plotly_chart(fig_backtest, use_container_width=True)
        st.caption("Black = actual held‑out prices | Blue = model prediction")


    # --- Metrics panel ---
    st.markdown("### 📊 Forecast Accuracy Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE", f"{mape:.2%}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R²", f"{r2:.2f}")
    
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
                    **MAPE (Mean Absolute Percentage Error)**  
                    - Shows the average error as a percentage of actual values.  
                    - Lower is better.  
                    - Example: 5% means predictions are off by ~5% on average.  

                    **RMSE (Root Mean Square Error)**  
                    - Measures the typical size of errors in the same units as the data (USD).  
                    - Sensitive to large mistakes.  
                    - Lower is better.  

                    **R² (Coefficient of Determination)**  
                    - Explains how much of the variation in actual prices the model captures.  
                    - 1.0 = perfect fit, 0 = no better than guessing the mean, negative = worse than guessing.  
    """)


   

    # --- Downloadable report ---
    report_df = df_forecast.copy()
    report_df["MAPE"] = mape
    report_df["RMSE"] = rmse
    report_df["R²"] = r2
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast Report (CSV)",
        data=csv,
        file_name=f"{metal}_forecast_report.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    """
    ⚠️ **Friendly Disclaimer**  
    This app is just a prototype and is meant for learning, exploring, and having a bit of fun with data.  
    The forecasts are based on historical patterns and math models — they’re **not financial advice** and definitely not a crystal ball 🔮.  
    Please don’t make trading or investment decisions based solely on what you see here.  
    Always do your own research (and maybe chat with a real financial advisor if money’s on the line).  
    """,
    unsafe_allow_html=True
)


