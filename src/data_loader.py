# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:57:25 2025

@author: Zane Hambly
"""

import yfinance as yf
import pandas as pd 

TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Aluminum": "ALI=F"
}

    
def get_price_data(metal: str, period: str = "5y", interval: str = "1d") -> pd.Series:
    ticker = TICKERS[metal]
    df = yf.download(ticker, period=period, interval=interval)
    return df["Close"].dropna()

#This function brings out the closing frice for various metals