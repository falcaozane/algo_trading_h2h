# indicators/sma.py

import pandas as pd

def sma(df, period=20, column="Close"):
    """
    Calculates Simple Moving Average (SMA)
    """
    return df[column].rolling(window=period).mean()



def ema(df, period=20, column="Close"):
    """
    Calculates Exponential Moving Average (EMA)
    """
    return df[column].ewm(span=period, adjust=False).mean()

