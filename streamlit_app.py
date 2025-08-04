# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import fetch_stock_data
from indicators.rsi import rsi
from indicators.sma import sma
from indicators.ema import ema
from indicators.macd import macd
from strategy.rule_based_strategy import generate_signals_sma, generate_signals_ema
from utils.backtester import backtest_signals

# Function to display strategy results with Plotly
def display_strategy_results(df, results, metrics, strategy_name, period_short, period_long, initial_cash, selected_stock):
    """
    Display comprehensive strategy results in Streamlit interface using Plotly
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
    
    # 1. Main price chart with signals and moving averages
    st.subheader(f"📉 {selected_stock} Price Chart with {strategy_name} Strategy")
    
    fig_price = go.Figure()
    
    # Add price line
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='purple', width=2),
        hovertemplate='<b>Price</b>: ₹%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>'
    ))
    
    # Add moving averages
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{strategy_name}{period_short}'],
        mode='lines',
        name=f'{strategy_name}{period_short}',
        line=dict(color='blue', width=2),
        hovertemplate=f'<b>{strategy_name}{period_short}</b>: ₹%{{y:.2f}}<br><b>Date</b>: %{{x}}<extra></extra>'
    ))
    
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{strategy_name}{period_long}'],
        mode='lines',
        name=f'{strategy_name}{period_long}',
        line=dict(color='red', width=2),
        hovertemplate=f'<b>{strategy_name}{period_long}</b>: ₹%{{y:.2f}}<br><b>Date</b>: %{{x}}<extra></extra>'
    ))
    
    # Add buy signals
    if not buy_signals.empty:
        fig_price.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(color='darkgreen', width=1)
            ),
            hovertemplate='<b>BUY</b><br><b>Price</b>: ₹%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>'
        ))
    
    # Add sell signals
    if not sell_signals.empty:
        fig_price.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(color='darkred', width=1)
            ),
            hovertemplate='<b>SELL</b><br><b>Price</b>: ₹%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>'
        ))
    
    # Add trend zones
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{strategy_name}{period_short}'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig_price.add_trace(go.Scatter(
        x=df.index,
        y=df[f'{strategy_name}{period_long}'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        fillcolor='rgba(0,255,0,0.1)',
        name='Bullish Zone',
        showlegend=True
    ))
    
    fig_price.update_layout(
        title=f"{selected_stock} - {strategy_name} Strategy Signals",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # 2. Portfolio performance comparison
    st.subheader("📈 Portfolio Performance vs Buy & Hold")
    
    # Calculate buy & hold
    buy_hold_value = initial_cash * (df['Close'] / df['Close'].iloc[0])
    
    fig_perf = go.Figure()
    
    fig_perf.add_trace(go.Scatter(
        x=results.index,
        y=results['Total'],
        mode='lines',
        name='Strategy Portfolio',
        line=dict(color='green', width=3),
        hovertemplate='<b>Strategy</b>: ₹%{y:,.0f}<br><b>Date</b>: %{x}<extra></extra>'
    ))
    
    fig_perf.add_trace(go.Scatter(
        x=df.index,
        y=buy_hold_value,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='blue', width=2, dash='dash'),
        hovertemplate='<b>Buy & Hold</b>: ₹%{y:,.0f}<br><b>Date</b>: %{x}<extra></extra>'
    ))
    
    fig_perf.update_layout(
        title="Strategy vs Buy & Hold Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # 3. Technical indicators in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💹 RSI Indicator")
        
        fig_rsi = go.Figure()
        
        # RSI line
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            hovertemplate='<b>RSI</b>: %{y:.1f}<br><b>Date</b>: %{x}<extra></extra>'
        ))
        
        # Overbought/Oversold lines
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (30)")
        fig_rsi.add_hline(y=50, line_dash="solid", line_color="gray", 
                         annotation_text="Midline (50)", opacity=0.5)
        
        # Fill zones
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.1, 
                         line_width=0, annotation_text="Oversold Zone")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.1, 
                         line_width=0, annotation_text="Overbought Zone")
        
        # Add buy/sell signals on RSI
        if not buy_signals.empty:
            fig_rsi.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['RSI'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                showlegend=False
            ))
        
        if not sell_signals.empty:
            fig_rsi.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['RSI'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                showlegend=False
            ))
        
        fig_rsi.update_layout(
            title="RSI with Trading Signals",
            xaxis_title="Date",
            yaxis_title="RSI Value",
            height=400,
            yaxis=dict(range=[0, 100]),
            template='plotly_white'
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        st.subheader("📊 MACD Indicator")
        
        fig_macd = make_subplots(rows=2, cols=1, 
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_width=[0.7, 0.3])
        
        # MACD line
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2),
            hovertemplate='<b>MACD</b>: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Signal line
        fig_macd.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_signal'],
            mode='lines',
            name='Signal Line',
            line=dict(color='orange', width=2),
            hovertemplate='<b>Signal</b>: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Zero line
        fig_macd.add_hline(y=0, line_dash="solid", line_color="pink", 
                          opacity=0.5, row=1, col=1)
        
        # MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
        fig_macd.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_hist'],
            name='MACD Histogram',
            marker_color=colors,
            opacity=0.6,
            hovertemplate='<b>Histogram</b>: %{y:.3f}<extra></extra>'
        ), row=2, col=1)
        
        fig_macd.update_layout(
            title="MACD Indicator",
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        fig_macd.update_xaxes(title_text="Date", row=2, col=1)
        fig_macd.update_yaxes(title_text="MACD Value", row=1, col=1)
        fig_macd.update_yaxes(title_text="Histogram", row=2, col=1)
        
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # 4. Bollinger Bands
    st.subheader("📈 Bollinger Bands")
    
    fig_bb = go.Figure()
    
    # Price line
    fig_bb.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='purple', width=2),
        hovertemplate='<b>Price</b>: ₹%{y:.2f}<extra></extra>'
    ))
    
    # 20-day SMA
    fig_bb.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA20'],
        mode='lines',
        name='20-day SMA',
        line=dict(color='blue', width=1.5),
        hovertemplate='<b>SMA20</b>: ₹%{y:.2f}<extra></extra>'
    ))
    
    # Upper Band
    fig_bb.add_trace(go.Scatter(
        x=df.index,
        y=df['Upper_Band'],
        mode='lines',
        name='Upper Band',
        line=dict(color='red', dash='dash', width=1.5),
        hovertemplate='<b>Upper Band</b>: ₹%{y:.2f}<extra></extra>'
    ))
    
    # Lower Band with fill
    fig_bb.add_trace(go.Scatter(
        x=df.index,
        y=df['Lower_Band'],
        mode='lines',
        name='Lower Band',
        line=dict(color='green', dash='dash', width=1.5),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)',
        hovertemplate='<b>Lower Band</b>: ₹%{y:.2f}<extra></extra>'
    ))
    
    fig_bb.update_layout(
        title="Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_bb, use_container_width=True)
    
    # 5. Drawdown Analysis
    st.subheader("📉 Drawdown Analysis")
    
    # Calculate drawdown
    returns = results['Total'].pct_change().fillna(0)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=df.index,
        y=drawdown * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=1),
        hovertemplate='<b>Drawdown</b>: %{y:.1f}%<extra></extra>'
    ))
    
    fig_dd.update_layout(
        title="Portfolio Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # 6. Trade analysis
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
        
        # Trade returns distribution
        st.subheader("📊 Trade Returns Distribution")
        
        returns_pct = trades_df['return_pct'] * 100
        
        fig_hist = px.histogram(
            x=returns_pct,
            nbins=20,
            title="Distribution of Trade Returns",
            labels={'x': 'Return (%)', 'y': 'Number of Trades'},
            color_discrete_sequence=['steelblue']
        )
        
        # Add vertical lines for mean and zero
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                          annotation_text="Break Even")
        fig_hist.add_vline(x=returns_pct.mean(), line_dash="solid", line_color="green", 
                          annotation_text=f"Mean: {returns_pct.mean():.1f}%")
        
        fig_hist.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Trade timeline
        st.subheader("📅 Trade Timeline")
        
        fig_timeline = go.Figure()
        
        for i, trade in trades_df.iterrows():
            color = 'green' if trade['return_pct'] > 0 else 'red'
            fig_timeline.add_trace(go.Scatter(
                x=[trade['entry_date'], trade['exit_date']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines+markers',
                name=f"Trade {i+1}",
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>Trade {i+1}</b><br>' +
                             f'Entry: ₹{trade["entry_price"]:.2f}<br>' +
                             f'Exit: ₹{trade["exit_price"]:.2f}<br>' +
                             f'Return: {trade["return_pct"]:.2%}<br>' +
                             f'Duration: {(pd.to_datetime(trade["exit_date"]) - pd.to_datetime(trade["entry_date"])).days} days<extra></extra>',
                showlegend=False
            ))
        
        fig_timeline.update_layout(
            title="Individual Trade Performance Timeline",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
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
    
    # 7. Signal summary table
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
          'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'ETERNAL.NS']

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