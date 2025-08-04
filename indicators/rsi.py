# indicators/rsi.py

import pandas as pd

def rsi(df, period=14, column="Close"):
    """
    Calculates Relative Strength Index (RSI)
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
