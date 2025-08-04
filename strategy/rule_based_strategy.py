# strategy/rule_based_strategy.py

"""
    Generates buy/sell signals based on:
    - Buy when RSI < 30 and SMA20 crosses above SMA50
    - Sell when RSI > 70 and SMA20 crosses below SMA50

    Returns:
        pd.DataFrame: DataFrame with new 'Signal' column: 1 = Buy, -1 = Sell, 0 = Hold
"""

import pandas as pd

def generate_signals_sma(df, rsi_col='RSI', sma_short_col='SMA20', sma_long_col='SMA50'):
    df = df.copy()
    df['SMA_Signal'] = 0
    
    # Method 1: Use OR condition (either RSI or SMA crossover)
    # Buy condition: RSI < 30 OR SMA20 crosses above SMA50
    rsi_oversold = df[rsi_col] < 30
    sma_bullish_cross = (df[sma_short_col].shift(1) < df[sma_long_col].shift(1)) & (df[sma_short_col] > df[sma_long_col])
    buy_signal = rsi_oversold | sma_bullish_cross
    df.loc[buy_signal, 'SMA_Signal'] = 1
    
    # Sell condition: RSI > 70 OR SMA20 crosses below SMA50
    rsi_overbought = df[rsi_col] > 70
    sma_bearish_cross = (df[sma_short_col].shift(1) > df[sma_long_col].shift(1)) & (df[sma_short_col] < df[sma_long_col])
    sell_signal = rsi_overbought | sma_bearish_cross
    df.loc[sell_signal, 'SMA_Signal'] = -1
    
    return df


def generate_signals_ema(df, rsi_col='RSI', ema_short_col='EMA20', ema_long_col='EMA50'):
    df = df.copy()
    df['EMA_Signal'] = 0

    # Method 1: Use OR condition (either RSI or EMA crossover)
    # Buy condition: RSI < 30 OR EMA20 crosses above EMA50
    rsi_oversold = df[rsi_col] < 30
    ema_bullish_cross = (df[ema_short_col].shift(1) < df[ema_long_col].shift(1)) & (df[ema_short_col] > df[ema_long_col])
    buy_signal = rsi_oversold | ema_bullish_cross
    df.loc[buy_signal, 'EMA_Signal'] = 1

    # Sell condition: RSI > 70 OR EMA20 crosses below EMA50
    rsi_overbought = df[rsi_col] > 70
    ema_bearish_cross = (df[ema_short_col].shift(1) > df[ema_long_col].shift(1)) & (df[ema_short_col] < df[ema_long_col])
    sell_signal = rsi_overbought | ema_bearish_cross
    df.loc[sell_signal, 'EMA_Signal'] = -1
    
    return df