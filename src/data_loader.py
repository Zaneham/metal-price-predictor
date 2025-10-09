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

@st.cache_data(show_spinner="Downloading data...")
def get_price_data(metal: str, period: str = "5y", interval: str = "1d") -> pd.Series:
    """Fetch and cache closing prices for a given metal."""
    ticker = TICKERS[metal]
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df["Close"].dropna()



#This function brings out the closing frice for various metals