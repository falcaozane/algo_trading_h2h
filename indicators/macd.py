
# indicators/macd.py

import pandas as pd

def macd(df, fast_period=12, slow_period=26, signal_period=9, column="Close"):
    """
    Calculates MACD Line, Signal Line, and Histogram
    """
    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
