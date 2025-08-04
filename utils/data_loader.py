# utils/data_loader.py

import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

from curl_cffi import requests
session = requests.Session(impersonate="chrome")

today = datetime.today()

def fetch_stock_data(symbol, start_date="2023-01-01", end_date=today, interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
        symbol (str): Ticker symbol (e.g., "RELIANCE.NS")
        start_date (str): Start date in "YYYY-MM-DD"
        end_date (str): End date (default is today)
        interval (str): Data interval ("1d", "1h", etc.)

    Returns:
        pd.DataFrame: Historical OHLCV stock data
    """
    try:
        logging.info(f"Fetching data for {symbol} from {start_date} to {end_date or 'today'}")
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False, session=session, auto_adjust=True, threads=True)
        # Flatten the MultiIndex columns
        df.columns = [col[0] for col in df.columns]

        if df.empty:
            logging.warning(f"No data found for {symbol}")
        else:
            logging.info(f"Downloaded {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

