# streamlit_app.py

from turtle import color
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.data_loader import fetch_stock_data
from indicators.rsi import rsi
from indicators.sma import sma
from indicators.ema import ema
from indicators.macd import macd
from strategy.rule_based_strategy import generate_signals_sma, generate_signals_ema
from utils.backtester import backtest_signals
#from utils.google_sheets import log_to_google_sheets,create_or_get_spreadsheet,

# ADD THE FUNCTION HERE - RIGHT AFTER IMPORTS
def display_strategy_results(df, results, metrics, strategy_name, period_short, period_long, initial_cash, selected_stock):
    """
    Display comprehensive strategy results in Streamlit interface
    """
    
    # Performance metrics in columns
    st.subheader("📊 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Final Value", metrics['Final Portfolio Value'])
        st.metric("📈 Total Return", metrics['Total Return'])
    
    with col2:
        st.metric("🎯 Buy & Hold Return", metrics['Buy & Hold Return'])
        st.metric("📊 Total Trades", metrics['Total Trades'])
    
    with col3:
        st.metric("🏆 Win Rate", metrics['Win Rate'])
        st.metric("⚡ Sharpe Ratio", metrics['Sharpe Ratio'])
    
    with col4:
        st.metric("📉 Max Drawdown", metrics['Maximum Drawdown'])
        st.metric("🔥 Volatility", metrics['Volatility (Annual)'])

    # Get signals for plotting
    signal_col = f'{strategy_name}_Signal'
    buy_signals = df[df[signal_col] == 1]
    sell_signals = df[df[signal_col] == -1]
    
    # Main price chart with signals
    st.subheader(f"📉 {selected_stock} Price Chart with {strategy_name} Strategy")
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price', alpha=0.8, linewidth=2.5, color='black')
    ax1.plot(df.index, df[f'{strategy_name}{period_short}'], 
             label=f'{strategy_name}{period_short}', alpha=0.8, linewidth=2, color='blue')
    ax1.plot(df.index, df[f'{strategy_name}{period_long}'], 
             label=f'{strategy_name}{period_long}', alpha=0.8, linewidth=2, color='red')
    
    # Plot buy/sell signals
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', s=120, color='green', label='Buy Signal', zorder=5, alpha=0.8)
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', s=120, color='red', label='Sell Signal', zorder=5, alpha=0.8)
    
    ax1.set_title(f"{selected_stock} - {strategy_name} Strategy Signals", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Portfolio performance comparison
    st.subheader("📈 Portfolio Performance vs Buy & Hold")
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    # Strategy equity curve
    ax2.plot(results.index, results['Total'], label='Strategy Portfolio', 
             color='green', linewidth=3, alpha=0.9)
    
    # Buy & hold comparison
    buy_hold_value = initial_cash * (df['Close'] / df['Close'].iloc[0])
    ax2.plot(df.index, buy_hold_value, label='Buy & Hold', 
             color='blue', linewidth=2.5, alpha=0.8, linestyle='--')
    
    ax2.set_title("Strategy Performance Comparison", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Technical indicators in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💹 RSI Indicator")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(df.index, df['RSI'], color='purple', linewidth=2, label='RSI')
        ax3.axhline(30, color='red', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.axhline(70, color='green', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(50, color='gray', linestyle='-', alpha=0.5, label='Midline (50)')
        
        # Fill overbought/oversold regions
        ax3.fill_between(df.index, 0, 30, alpha=0.1, color='red')
        ax3.fill_between(df.index, 70, 100, alpha=0.1, color='green')
        
        ax3.set_title("RSI with Trading Signals", fontsize=14, fontweight='bold')
        ax3.set_ylabel("RSI Value", fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
    
    with col2:
        st.subheader("📊 MACD Indicator")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        
        # MACD lines
        ax4.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
        ax4.plot(df.index, df['MACD_signal'], label='Signal Line', color='orange', linewidth=2)
        
        # MACD histogram with colors
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
        ax4.bar(df.index, df['MACD_hist'], label='MACD Histogram', 
               color=colors, alpha=0.6, width=1)
        
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title("MACD Indicator", fontsize=14, fontweight='bold')
        ax4.set_ylabel("MACD Value", fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        
    st.subheader("📈 Bollinger Bands")
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
    ax5.plot(df.index, df['SMA20'], label='20-day SMA', color='blue', linewidth=1.5)
    ax5.plot(df.index, df['Upper_Band'], label='Upper Band', color='red', linestyle='--', linewidth=1.5)
    ax5.plot(df.index, df['Lower_Band'], label='Lower Band', color='green', linestyle='--', linewidth=1.5)
    ax5.fill_between(df.index, df['Upper_Band'], df['Lower_Band'],
                     color='lightgray', alpha=0.3, label='Bollinger Bands Area')
    ax5.set_title("Bollinger Bands", fontsize=14, fontweight='bold')
    ax5.set_ylabel("Price (₹)", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    
    # Trade analysis
    if not metrics['Trades DataFrame'].empty:
        st.subheader("📋 Trade Analysis")
        
        trades_df = metrics['Trades DataFrame']
        
        # Trade statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_trade_duration = (pd.to_datetime(trades_df['exit_date']) - 
                                pd.to_datetime(trades_df['entry_date'])).dt.days.mean()
            st.metric("📅 Avg Trade Duration", f"{avg_trade_duration:.1f} days")
            
        with col2:
            best_trade = trades_df['return_pct'].max()
            st.metric("🚀 Best Trade", f"{best_trade:.2%}")
            
        with col3:
            worst_trade = trades_df['return_pct'].min()
            st.metric("💥 Worst Trade", f"{worst_trade:.2%}")
        
        # Trade history table
        st.subheader("📊 Detailed Trade History")
        display_trades = trades_df.copy()
        display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
        display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
        display_trades['Entry Price'] = display_trades['entry_price'].apply(lambda x: f"₹{x:.2f}")
        display_trades['Exit Price'] = display_trades['exit_price'].apply(lambda x: f"₹{x:.2f}")
        display_trades['P&L (₹)'] = display_trades['profit_loss'].apply(lambda x: f"₹{x:,.2f}")
        display_trades['Return %'] = display_trades['return_pct'].apply(lambda x: f"{x:.2%}")
        display_trades['Duration'] = (pd.to_datetime(trades_df['exit_date']) - 
                                    pd.to_datetime(trades_df['entry_date'])).dt.days
        
        trade_display = display_trades[['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 
                                      'P&L (₹)', 'Return %', 'Duration', 'exit_reason']].copy()
        trade_display.columns = ['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 
                               'Profit/Loss', 'Return %', 'Days', 'Exit Reason']
        
        st.dataframe(trade_display, use_container_width=True)
    
    else:
        st.info("📝 No trades were executed during this period with the current parameters.")
    
    # Signal summary table
    st.subheader("📋 Trading Signals Summary")
    signal_summary = df[df[signal_col] != 0].copy()
    
    if not signal_summary.empty:
        signal_summary['Signal Type'] = signal_summary[signal_col].map({1: '🟢 BUY', -1: '🔴 SELL'})
        signal_summary['Price'] = signal_summary['Close'].apply(lambda x: f"₹{x:.2f}")
        signal_summary['RSI'] = signal_summary['RSI'].apply(lambda x: f"{x:.1f}")
        signal_summary[f'{strategy_name}{period_short}'] = signal_summary[f'{strategy_name}{period_short}'].apply(lambda x: f"₹{x:.2f}")
        signal_summary[f'{strategy_name}{period_long}'] = signal_summary[f'{strategy_name}{period_long}'].apply(lambda x: f"₹{x:.2f}")
        
        display_signals = signal_summary[['Signal Type', 'Price', 'RSI', 
                                        f'{strategy_name}{period_short}', 
                                        f'{strategy_name}{period_long}']].copy()
        display_signals.index = display_signals.index.strftime('%Y-%m-%d')
        
        st.dataframe(display_signals, use_container_width=True)
    else:
        st.info("📝 No trading signals were generated during this period with the current parameters.")

# ---------------------------------------
st.set_page_config(layout="wide", page_title="Algo Trading Dashboard", page_icon="📈")
st.title("📈 Algo-Trading Dashboard: Technical Analysis & Backtesting")

# Sidebar config
st.sidebar.header("📊 Configuration")

# Stock selection
stocks = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 
          'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS', 
          'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 
          'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 
          'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 
          'JIOFIN.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 
          'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 
          'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 
          'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 
          'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'ZOMATO.NS']

selected_stock = st.sidebar.selectbox("Select Stock", stocks)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))

# Strategy selection
strategy_type = st.sidebar.selectbox("Strategy Type", ["SMA-based", "EMA-based", "Both"])

st.sidebar.subheader("📈 Technical Indicators")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_short = st.sidebar.slider("Short-term SMA", 5, 30, 20)
sma_long = st.sidebar.slider("Long-term SMA", 30, 100, 50)
ema_short = st.sidebar.slider("Short-term EMA", 5, 30, 20)
ema_long = st.sidebar.slider("Long-term EMA", 30, 100, 50)

st.sidebar.subheader("💰 Backtesting Parameters")
initial_cash = st.sidebar.number_input("Initial Capital (₹)", min_value=10000, value=100000, step=10000)
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 20.0, 5.0, step=1.0) / 100
take_profit = st.sidebar.slider("Take Profit (%)", 0.0, 50.0, 15.0, step=5.0) / 100

# Enable/disable risk management
use_risk_mgmt = st.sidebar.checkbox("Enable Risk Management", value=True)

# Load data with progress bar
with st.spinner(f'Loading data for {selected_stock}...'):
    df = fetch_stock_data(selected_stock, start_date=start_date.strftime("%Y-%m-%d"))

st.subheader(f"📊 Stock Data for {selected_stock}")
st.write(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to Present")
st.write(f"**Total Records:** {len(df)} days")

if df.empty:
    st.error("❌ No data found for the selected stock and date range.")
    st.stop()

# Apply indicators
with st.spinner('Calculating technical indicators...'):
    df['RSI'] = rsi(df, period=rsi_period)
    df['SMA20'] = sma(df, period=sma_short)
    df['SMA50'] = sma(df, period=sma_long)
    df['EMA20'] = ema(df, period=ema_short)
    df['EMA50'] = ema(df, period=ema_long)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df)
    df['Upper_Band'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()

# Apply strategies based on selection
if strategy_type in ["SMA-based", "Both"]:
    df = generate_signals_sma(df, rsi_col='RSI', sma_short_col='SMA20', sma_long_col='SMA50')

if strategy_type in ["EMA-based", "Both"]:
    df = generate_signals_ema(df, rsi_col='RSI', ema_short_col='EMA20', ema_long_col='EMA50')

# Backtesting section
st.header("🔍 Backtesting Results")

# Create tabs for different strategies
if strategy_type == "Both":
    tab1, tab2 = st.tabs(["SMA Strategy", "EMA Strategy"])
    
    with tab1:
        st.subheader("📊 SMA Strategy Results")
        sma_results, sma_metrics = backtest_signals(
            df, 
            signal_col='SMA_Signal', 
            price_col='Close', 
            initial_cash=initial_cash,
            transaction_cost=transaction_cost if use_risk_mgmt else 0,
            stop_loss=stop_loss if use_risk_mgmt else None,
            take_profit=take_profit if use_risk_mgmt else None
        )
        display_strategy_results(df, sma_results, sma_metrics, "SMA", sma_short, sma_long, initial_cash, selected_stock)
    
    with tab2:
        st.subheader("📊 EMA Strategy Results")
        ema_results, ema_metrics = backtest_signals(
            df, 
            signal_col='EMA_Signal', 
            price_col='Close', 
            initial_cash=initial_cash,
            transaction_cost=transaction_cost if use_risk_mgmt else 0,
            stop_loss=stop_loss if use_risk_mgmt else None,
            take_profit=take_profit if use_risk_mgmt else None
        )
        display_strategy_results(df, ema_results, ema_metrics, "EMA", ema_short, ema_long, initial_cash, selected_stock)

else:
    # Single strategy
    signal_col = 'SMA_Signal' if strategy_type == "SMA-based" else 'EMA_Signal'
    strategy_name = strategy_type.split('-')[0]
    
    results, metrics = backtest_signals(
        df, 
        signal_col=signal_col, 
        price_col='Close', 
        initial_cash=initial_cash,
        transaction_cost=transaction_cost if use_risk_mgmt else 0,
        stop_loss=stop_loss if use_risk_mgmt else None,
        take_profit=take_profit if use_risk_mgmt else None
    )
    
    period_short = sma_short if strategy_type == "SMA-based" else ema_short
    period_long = sma_long if strategy_type == "SMA-based" else ema_long
    display_strategy_results(df, results, metrics, strategy_name, period_short, period_long, initial_cash, selected_stock)

# Data download section
st.subheader("💾 Download Data")
col1, col2 = st.columns(2)

with col1:
    csv_data = df.to_csv(index=True)
    st.download_button(
        label="📁 Download Full Dataset (CSV)",
        data=csv_data,
        file_name=f"{selected_stock}_analysis_{start_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    if 'results' in locals():
        results_csv = results.to_csv(index=True)
        st.download_button(
            label="📊 Download Backtest Results (CSV)",
            data=results_csv,
            file_name=f"{selected_stock}_backtest_{start_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("Developed by Zane Vijay Falcao")