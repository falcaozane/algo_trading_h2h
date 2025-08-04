# indicators/sma.py

import pandas as pd

def ema(df, period=20, column="Close"):
    """
    Calculates Exponential Moving Average (EMA)
    """
    return df[column].ewm(span=period, adjust=False).mean()