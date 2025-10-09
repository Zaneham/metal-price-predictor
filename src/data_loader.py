# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:57:25 2025

@author: Zane Hambly
"""

import yfinance as yf
import pandas as pd
import streamlit as st

TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Aluminum": "ALI=F",
}

@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def get_price_data(metal: str, period="5y", interval="1d") -> pd.Series:
    ticker = TICKERS[metal]
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    return df["Close"].dropna()
