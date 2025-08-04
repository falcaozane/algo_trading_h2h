# main.py

from utils.logger import setup_logger
from utils.data_loader import fetch_stock_data
from indicators.rsi import rsi
from indicators.sma import sma
from indicators.ema import ema
from indicators.macd import macd
from strategy.rule_based_strategy import generate_signals
from utils.backtester import backtest_signals

import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup logging
setup_logger()

# 2. Define configuration
stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
start_date = '2024-02-01'
rsi_period = 14
sma_short = 20
sma_long = 50
ema_short = 20
ema_long = 50

for symbol in stocks:
    print(f"\n--- Running for: {symbol} ---")

    # 3. Fetch stock data
    df = fetch_stock_data(symbol, start_date=start_date)

    if df.empty:
        print(f"No data for {symbol}, skipping...")
        continue

    # 4. Add indicators
    df['RSI'] = rsi(df, period=rsi_period)
    df['SMA20'] = sma(df, period=sma_short)
    df['SMA50'] = sma(df, period=sma_long)
    df['EMA20'] = ema(df, period=ema_short)
    df['EMA50'] = ema(df, period=ema_long)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df)

    # 5. Generate buy/sell signals
    df = generate_signals(df, rsi_col='RSI', sma_short_col='SMA20', sma_long_col='SMA50')

    # 6. Backtest strategy
    results = backtest_signals(df, signal_col='Signal')

    # 7. Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(results['Total'], label='Equity Curve')
    plt.title(f"{symbol} - Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8. Print final portfolio value
    final_value = results['Total'].iloc[-1]
    print(f"Final portfolio value for {symbol}: ₹{final_value:,.2f}")
