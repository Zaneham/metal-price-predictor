# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 15:54:35 2025

@author: Zane Hambly
"""

import streamlit as st
import pandas as pd
from src.data_loader import get_price_data
from models.arima_model import fit_arima_forecast
import plotly.express as px
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
    ["Historical Data only - no AI", "Long-term (ARIMA)"]
)

 
horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=30,
    max_value=90,
    value=30,
    step=30
)


# --- Load data ---
data = get_price_data(metal, period="5y", interval="1d")

# Convert Series -> DataFrame for Plotly
df_hist = data.reset_index()
df_hist.columns = ["Date", "Price"]

st.subheader(f"{metal} Closing Prices (Last 5 Years)")
fig = px.line(df_hist, x="Date", y="Price", title=f"{metal} Closing Prices")

# Constrain axes for the top chart
x_min_hist = df_hist["Date"].min()
x_max_hist = df_hist["Date"].max()
y_min_hist = df_hist["Price"].min()
y_max_hist = df_hist["Price"].max()

fig.update_layout(
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
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- Forecast ---
if model_type == "Long-term (ARIMA)":
    st.subheader(f"{horizon}-Day Forecast (ARIMA)")
    forecast, conf_int = fit_arima_forecast(data, steps=horizon)

    # Reset indices for plotting
    df_hist = data.reset_index()
    df_hist.columns = ["Date", "Price"]

    df_forecast = forecast.reset_index()
    df_forecast.columns = ["Date", "Price"]

    # Build figure with explicit traces
    fig_forecast = go.Figure()

    # History in black
    fig_forecast.add_trace(go.Scatter(
        x=df_hist["Date"], y=df_hist["Price"],
        mode="lines",
        name="History",
        line=dict(color="black")
    ))

    # Forecast in blue dashed
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast["Date"], y=df_forecast["Price"],
        mode="lines",
        name="Forecast",
        line=dict(color="blue", dash="dash")
    ))

    # Confidence interval shading
    ci_lower = conf_int.iloc[:, 0]
    ci_upper = conf_int.iloc[:, 1]
    fig_forecast.add_trace(go.Scatter(
        x=df_forecast["Date"].tolist() + df_forecast["Date"].tolist()[::-1],
        y=ci_upper.tolist() + ci_lower.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0,0,255,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval"
    ))

    # Constrain axes to history + forecast
    x_min = df_hist["Date"].min()
    x_max = df_forecast["Date"].max()
    y_min = min(df_hist["Price"].min(), df_forecast["Price"].min(), ci_lower.min())
    y_max = max(df_hist["Price"].max(), df_forecast["Price"].max(), ci_upper.max())

    fig_forecast.update_layout(
        title=f"{metal} Price with {horizon}-Day Forecast",
        xaxis=dict(
            range=[x_min, x_max],
            dtick="M1",
            tickformat="%b\n%Y",
            fixedrange=True
        ),
        yaxis=dict(
            range=[y_min, y_max],
            title="Price (USD)",
            fixedrange=True
        )
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("Forecast values:")
    st.dataframe(df_forecast)

    # --- Add explanatory note only for longer horizons ---
    if horizon > 30:
        st.info(
            f"ℹ️ A {horizon}-day forecast carries more uncertainty. "
            "The shaded confidence band widens as the horizon increases. "
            "Use 60‑ and 90‑day forecasts as scenario ranges rather than precise predictions."
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


