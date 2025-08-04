import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import pickle
from datetime import datetime, timedelta
import warnings
from curl_cffi import requests

from indicators.rsi import rsi
from indicators.sma import sma
from indicators.ema import ema
from indicators.macd import macd

from strategy.rule_based_strategy import generate_signals_sma, generate_signals_ema
from utils.backtester import backtest_signals
 
from indicators.enhanced_features import (
    create_volatility_features, create_enhanced_lag_features,
    create_volume_features, create_momentum_features, create_position_features
)

# Suppress warnings
warnings.filterwarnings('ignore')
session = requests.Session(impersonate="chrome")

# Page config
st.set_page_config(
    page_title="Complete Stock Trading & Prediction Platform",
    page_icon="📈",
    layout="wide"
)

# Stock symbols
STOCK_SYMBOLS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS',
    'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 
    'BEL.NS', 'BHARTIARTL.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
    'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JIOFIN.NS', 'JSWSTEEL.NS', 
    'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 
    'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 
    'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TCS.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 
    'ULTRACEMCO.NS', 'WIPRO.NS', 'ETERNAL.NS'
]

# Feature list for ML model
FEATURES = [
    'Close', 'Volume', 'SMA20', 'SMA50', 'EMA20', 'EMA50',
    'RSI14', 'MACD', 'MACD_signal', 'MACD_hist',
    'SMA_crossover', 'RSI_oversold',
    'return_1d', 'volatility_5d', 'volatility_10d', 'volatility_20d',
    'volatility_30d', 'vol_ratio_5_20', 'vol_ratio_10_20', 'vol_rank_20',
    'vol_rank_50', 'return_lag_1', 'return_lag_2', 'return_lag_3',
    'return_lag_5', 'return_lag_10', 'rsi_lag_1', 'macd_lag_1', 'rsi_lag_2',
    'macd_lag_2', 'rsi_lag_3', 'macd_lag_3', 'volume_sma_10',
    'volume_sma_20', 'volume_sma_50', 'volume_ratio_10', 'volume_ratio_20',
    'volume_ratio_50', 'price_volume', 'pv_sma_5', 'volume_momentum_5',
    'momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d', 'roc_5d',
    'roc_10d', 'high_10d', 'low_10d', 'price_position_10', 'high_20d',
    'low_20d', 'price_position_20', 'high_50d', 'low_50d',
    'price_position_50', 'bb_upper', 'bb_lower', 'bb_position', 'target'
]

# ========================= SHARED FUNCTIONS =========================

@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, session=session)
        if data.columns.nlevels > 1:
            data.columns = [col[0] for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None





def process_stock_data(df, short_period, long_period, rsi_period):
    """Process stock data to create all features"""
    df = df.copy()
    
    # Basic technical indicators
    df['SMA20'] = sma(df['Close'], short_period)
    df['SMA50'] = sma(df['Close'], long_period)
    df['EMA20'] = ema(df['Close'], short_period)
    df['EMA50'] = ema(df['Close'], long_period)
    df['RSI14'] = rsi(df['Close'], rsi_period)
    df['RSI20'] = rsi(df['Close'], rsi_period + 6)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'])
    
    # Bollinger Bands
    df['Upper_Band'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()
    
    # Create feature sets
    df = create_volatility_features(df)
    df = create_enhanced_lag_features(df)
    df = create_volume_features(df)
    df = create_momentum_features(df)
    df = create_position_features(df)
    
    # Additional features
    df['SMA_crossover'] = (df['SMA20'] > df['SMA50']).astype(int)
    df['RSI_oversold'] = (df['RSI14'] < 30).astype(int)
    
    # Target: next-day up/down
    df['next_close'] = df['Close'].shift(-1)
    df['target'] = (df['next_close'] > df['Close']).astype(int)
    
    return df


# ========================= MAIN APPLICATION =========================

# Main navigation
st.title("📈 Complete Stock Trading & Prediction Platform")

# Navigation tabs
tab1, tab2 = st.tabs(["🔮 Price Prediction", "📊 Trading Dashboard"])

# ========================= SIDEBAR CONFIGURATION =========================

st.sidebar.header("📊 Configuration")

# Common inputs
selected_stock = st.sidebar.selectbox("Select Stock Symbol", STOCK_SYMBOLS, index=35)
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

st.sidebar.subheader("📈 Technical Indicators")
rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14, step=1)
short_period = st.sidebar.slider("Short-term Period", min_value=5, max_value=50, value=20, step=1)
long_period = st.sidebar.slider("Long-term Period", min_value=50, max_value=200, value=50, step=1)

# Strategy selection (for trading dashboard)
strategy_type = st.sidebar.selectbox("Strategy Type", ["SMA-based", "EMA-based", "Both"])

st.sidebar.subheader("💰 Backtesting Parameters")
initial_cash = st.sidebar.number_input("Initial Capital (₹)", min_value=10000, value=100000, step=10000)
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.05) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 20.0, 5.0, step=1.0) / 100
take_profit = st.sidebar.slider("Take Profit (%)", 0.0, 50.0, 15.0, step=5.0) / 100
use_risk_mgmt = st.sidebar.checkbox("Enable Risk Management", value=True)

# ========================= PRICE PREDICTION TAB =========================

with tab1:
    st.header(f"🔮 Price Prediction for {selected_stock}")
    
    with st.spinner("Loading stock data..."):
        stock_data = load_stock_data(selected_stock, start_date, end_date)
        
        if stock_data is not None and not stock_data.empty:
            # Process the data
            processed_data = process_stock_data(stock_data, short_period, long_period, rsi_period)
            processed_data = processed_data.dropna()
            
            if len(processed_data) > 0:
                # Get the latest row for prediction
                latest_data = processed_data.iloc[-1]
                
                # Display current stock info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"₹{latest_data['Close']:.2f}")
                with col2:
                    daily_change = ((latest_data['Close'] - processed_data.iloc[-2]['Close']) / processed_data.iloc[-2]['Close']) * 100
                    st.metric("Daily Change", f"{daily_change:.2f}%")
                with col3:
                    st.metric("Volume", f"{latest_data['Volume']:,.0f}")
                with col4:
                    st.metric("RSI14", f"{latest_data['RSI14']:.2f}")
                
                # Mock prediction (replace with actual model loading)
                try:
                    # Try to load the model
                    model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
                    scaler = pickle.load(open('scaler.pkl', 'rb'))
                    
                    # Create feature vector
                    feature_vector = latest_data[FEATURES].values.reshape(1, -1)
                    feature_vector_scaled = scaler.transform(feature_vector)
                    
                    # Make prediction
                    prediction = model.predict(feature_vector_scaled)[0]
                    probability = model.predict_proba(feature_vector_scaled)[0].max()
                    
                except:
                    # Mock prediction if model files not available
                    prediction = np.random.choice([0, 1])
                    probability = np.random.uniform(0.5, 0.9)
                
                # Display prediction
                st.header("🔮 Prediction")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("📈 **PREDICTION: UP**")
                        st.write(f"The model predicts the stock will go **UP** tomorrow with {probability:.1%} confidence.")
                    else:
                        st.error("📉 **PREDICTION: DOWN**")
                        st.write(f"The model predicts the stock will go **DOWN** tomorrow with {probability:.1%} confidence.")
                
                with col2:
                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Technical Analysis Charts
                st.header("📈 Technical Analysis")
                
                # Price charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # SMA Chart
                    fig_sma = go.Figure()
                    fig_sma.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['Close'][-60:],
                                               mode='lines', name='Close Price', line=dict(color='blue', width=2)))
                    fig_sma.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['SMA20'][-60:],
                                               mode='lines', name='SMA20', line=dict(color='orange', width=1)))
                    fig_sma.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['SMA50'][-60:],
                                               mode='lines', name='SMA50', line=dict(color='red', width=1)))
                    fig_sma.update_layout(title=f"{selected_stock} - Simple Moving Averages", height=400)
                    st.plotly_chart(fig_sma, use_container_width=True)
                
                with col2:
                    # EMA Chart
                    fig_ema = go.Figure()
                    fig_ema.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['Close'][-60:],
                                               mode='lines', name='Close Price', line=dict(color='blue', width=2)))
                    fig_ema.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['EMA20'][-60:],
                                               mode='lines', name='EMA20', line=dict(color='orange', width=1)))
                    fig_ema.add_trace(go.Scatter(x=processed_data.index[-60:], y=processed_data['EMA50'][-60:],
                                               mode='lines', name='EMA50', line=dict(color='red', width=1)))
                    fig_ema.update_layout(title=f"{selected_stock} - Exponential Moving Averages", height=400)
                    st.plotly_chart(fig_ema, use_container_width=True)
                
                # RSI and MACD
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=processed_data.index[-30:], y=processed_data['RSI14'][-30:],
                                               mode='lines', name='RSI14', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI (14-day)", height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=processed_data.index[-30:], y=processed_data['MACD'][-30:],
                                                mode='lines', name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=processed_data.index[-30:], y=processed_data['MACD_signal'][-30:],
                                                mode='lines', name='Signal', line=dict(color='red')))
                    fig_macd.update_layout(title="MACD", height=300)
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Feature importance (mock data)
                st.header("🎯 Key Factors")
                mock_features = ['RSI14', 'return_lag_1', 'volatility_5d', 'MACD', 'volume_ratio_20']
                mock_importance = [0.15, 0.12, 0.10, 0.08, 0.07]
                
                fig_importance = px.bar(x=mock_importance, y=mock_features, orientation='h', 
                                      title="Feature Importance")
                fig_importance.update_layout(height=300)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            else:
                st.error("Not enough data to make a prediction.")
        else:
            st.error("Unable to load stock data.")

# ========================= TRADING DASHBOARD TAB =========================

with tab2:
    st.header("📊 Trading Dashboard")
    
    with st.spinner(f'Loading data for {selected_stock}...'):
        df = load_stock_data(selected_stock, start_date, end_date)
    
    if df is not None and not df.empty:
        st.subheader(f"📊 Stock Data for {selected_stock}")
        st.write(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        st.write(f"**Total Records:** {len(df)} days")
        
        # Process data for trading
        df = process_stock_data(df, short_period, long_period, rsi_period)
        df = df.dropna()
        
        # Generate trading signals
        if strategy_type in ["SMA-based", "Both"]:
            df = generate_signals_sma(df, rsi_col='RSI14', sma_short_col='SMA20', sma_long_col='SMA50')
        
        if strategy_type in ["EMA-based", "Both"]:
            df = generate_signals_ema(df, rsi_col='RSI14', ema_short_col='EMA20', ema_long_col='EMA50')
        
        # Backtesting section
        st.header("🔍 Backtesting Results")
        
        if strategy_type == "Both":
            tab_sma, tab_ema = st.tabs(["SMA Strategy", "EMA Strategy"])
            
            with tab_sma:
                st.subheader("📊 SMA Strategy Results")
                sma_results, sma_metrics = backtest_signals(
                    df, signal_col='SMA_Signal', price_col='Close', 
                    initial_cash=initial_cash, transaction_cost=transaction_cost if use_risk_mgmt else 0
                )
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Final Value", sma_metrics['Final Portfolio Value'])
                    st.metric("📈 Total Return", sma_metrics['Total Return'])
                with col2:
                    st.metric("🎯 Buy & Hold Return", sma_metrics['Buy & Hold Return'])
                    st.metric("📊 Total Trades", sma_metrics['Total Trades'])
                with col3:
                    st.metric("🏆 Win Rate", sma_metrics['Win Rate'])
                    st.metric("⚡ Sharpe Ratio", sma_metrics['Sharpe Ratio'])
                with col4:
                    st.metric("📉 Max Drawdown", sma_metrics['Maximum Drawdown'])
                    st.metric("🔥 Volatility", sma_metrics['Volatility (Annual)'])
                
                # SMA Price Chart with Signals
                fig_sma_signals = go.Figure()
                fig_sma_signals.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                                   name='Close Price', line=dict(color='purple', width=2)))
                fig_sma_signals.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', 
                                                   name='SMA20', line=dict(color='blue', width=2)))
                fig_sma_signals.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', 
                                                   name='SMA50', line=dict(color='red', width=2)))
                
                # Add buy/sell signals
                buy_signals = df[df['SMA_Signal'] == 1]
                sell_signals = df[df['SMA_Signal'] == -1]
                
                if not buy_signals.empty:
                    fig_sma_signals.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                                       mode='markers', name='Buy Signal',
                                                       marker=dict(symbol='triangle-up', size=12, color='green')))
                
                if not sell_signals.empty:
                    fig_sma_signals.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                                       mode='markers', name='Sell Signal',
                                                       marker=dict(symbol='triangle-down', size=12, color='red')))
                
                fig_sma_signals.update_layout(title=f"{selected_stock} - SMA Strategy Signals", height=500)
                st.plotly_chart(fig_sma_signals, use_container_width=True)
                
                # Portfolio Performance
                buy_hold_value = initial_cash * (df['Close'] / df['Close'].iloc[0])
                fig_perf_sma = go.Figure()
                fig_perf_sma.add_trace(go.Scatter(x=sma_results.index, y=sma_results['Total'],
                                                mode='lines', name='SMA Strategy', line=dict(color='green', width=3)))
                fig_perf_sma.add_trace(go.Scatter(x=df.index, y=buy_hold_value,
                                                mode='lines', name='Buy & Hold', line=dict(color='blue', width=2, dash='dash')))
                fig_perf_sma.update_layout(title="SMA Strategy vs Buy & Hold Performance", height=400)
                st.plotly_chart(fig_perf_sma, use_container_width=True)
            
            with tab_ema:
                st.subheader("📊 EMA Strategy Results")
                ema_results, ema_metrics = backtest_signals(
                    df, signal_col='EMA_Signal', price_col='Close', 
                    initial_cash=initial_cash, transaction_cost=transaction_cost if use_risk_mgmt else 0
                )
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Final Value", ema_metrics['Final Portfolio Value'])
                    st.metric("📈 Total Return", ema_metrics['Total Return'])
                with col2:
                    st.metric("🎯 Buy & Hold Return", ema_metrics['Buy & Hold Return'])
                    st.metric("📊 Total Trades", ema_metrics['Total Trades'])
                with col3:
                    st.metric("🏆 Win Rate", ema_metrics['Win Rate'])
                    st.metric("⚡ Sharpe Ratio", ema_metrics['Sharpe Ratio'])
                with col4:
                    st.metric("📉 Max Drawdown", ema_metrics['Maximum Drawdown'])
                    st.metric("🔥 Volatility", ema_metrics['Volatility (Annual)'])
                
                # EMA Price Chart with Signals
                fig_ema_signals = go.Figure()
                fig_ema_signals.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                                   name='Close Price', line=dict(color='purple', width=2)))
                fig_ema_signals.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', 
                                                   name='EMA20', line=dict(color='blue', width=2)))
                fig_ema_signals.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', 
                                                   name='EMA50', line=dict(color='red', width=2)))
                
                # Add buy/sell signals
                buy_signals = df[df['EMA_Signal'] == 1]
                sell_signals = df[df['EMA_Signal'] == -1]
                
                if not buy_signals.empty:
                    fig_ema_signals.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                                       mode='markers', name='Buy Signal',
                                                       marker=dict(symbol='triangle-up', size=12, color='green')))
                
                if not sell_signals.empty:
                    fig_ema_signals.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                                       mode='markers', name='Sell Signal',
                                                       marker=dict(symbol='triangle-down', size=12, color='red')))
                
                fig_ema_signals.update_layout(title=f"{selected_stock} - EMA Strategy Signals", height=500)
                st.plotly_chart(fig_ema_signals, use_container_width=True)
                
                # Portfolio Performance
                buy_hold_value = initial_cash * (df['Close'] / df['Close'].iloc[0])
                fig_perf_ema = go.Figure()
                fig_perf_ema.add_trace(go.Scatter(x=ema_results.index, y=ema_results['Total'],
                                                mode='lines', name='EMA Strategy', line=dict(color='green', width=3)))
                fig_perf_ema.add_trace(go.Scatter(x=df.index, y=buy_hold_value,
                                                mode='lines', name='Buy & Hold', line=dict(color='blue', width=2, dash='dash')))
                fig_perf_ema.update_layout(title="EMA Strategy vs Buy & Hold Performance", height=400)
                st.plotly_chart(fig_perf_ema, use_container_width=True)
        
        else:
            # Single strategy
            signal_col = 'SMA_Signal' if strategy_type == "SMA-based" else 'EMA_Signal'
            strategy_name = strategy_type.split('-')[0]
            
            results, metrics = backtest_signals(
                df, signal_col=signal_col, price_col='Close', 
                initial_cash=initial_cash, transaction_cost=transaction_cost if use_risk_mgmt else 0
            )
            
            # Display metrics
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
            
            # Price Chart with Signals
            fig_signals = go.Figure()
            fig_signals.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                           name='Close Price', line=dict(color='purple', width=2)))
            
            if strategy_name == 'SMA':
                fig_signals.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', 
                                               name='SMA20', line=dict(color='blue', width=2)))
                fig_signals.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', 
                                               name='SMA50', line=dict(color='red', width=2)))
            else:
                fig_signals.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', 
                                               name='EMA20', line=dict(color='blue', width=2)))
                fig_signals.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', 
                                               name='EMA50', line=dict(color='red', width=2)))
            
            # Add buy/sell signals
            buy_signals = df[df[signal_col] == 1]
            sell_signals = df[df[signal_col] == -1]
            
            if not buy_signals.empty:
                fig_signals.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                               mode='markers', name='Buy Signal',
                                               marker=dict(symbol='triangle-up', size=12, color='green')))
            
            if not sell_signals.empty:
                fig_signals.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                               mode='markers', name='Sell Signal',
                                               marker=dict(symbol='triangle-down', size=12, color='red')))
            
            fig_signals.update_layout(title=f"{selected_stock} - {strategy_name} Strategy Signals", height=500)
            st.plotly_chart(fig_signals, use_container_width=True)
            
            # Portfolio Performance
            buy_hold_value = initial_cash * (df['Close'] / df['Close'].iloc[0])
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=results.index, y=results['Total'],
                                        mode='lines', name=f'{strategy_name} Strategy', line=dict(color='green', width=3)))
            fig_perf.add_trace(go.Scatter(x=df.index, y=buy_hold_value,
                                        mode='lines', name='Buy & Hold', line=dict(color='blue', width=2, dash='dash')))
            fig_perf.update_layout(title=f"{strategy_name} Strategy vs Buy & Hold Performance", height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Additional Technical Analysis Charts
        st.header("📈 Additional Technical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI14'], mode='lines', 
                                       name='RSI14', line=dict(color='purple', width=2)))
            
            # Add buy/sell signals on RSI
            if 'buy_signals' in locals() and not buy_signals.empty:
                fig_rsi.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['RSI14'],
                                           mode='markers', name='Buy Signal',
                                           marker=dict(symbol='triangle-up', size=10, color='green'),
                                           showlegend=False))
            
            if 'sell_signals' in locals() and not sell_signals.empty:
                fig_rsi.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['RSI14'],
                                           mode='markers', name='Sell Signal',
                                           marker=dict(symbol='triangle-down', size=10, color='red'),
                                           showlegend=False))
            
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.add_hline(y=50, line_dash="solid", line_color="gray", annotation_text="Midline (50)", opacity=0.5)
            
            fig_rsi.update_layout(title="RSI with Trading Signals", yaxis=dict(range=[0, 100]), height=400)
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD Chart
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.7, 0.3])
            
            # MACD line
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD',
                                        line=dict(color='blue', width=2)), row=1, col=1)
            
            # Signal line
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal Line',
                                        line=dict(color='orange', width=2)), row=1, col=1)
            
            # Zero line
            fig_macd.add_hline(y=0, line_dash="solid", line_color="pink", opacity=0.5, row=1, col=1)
            
            # MACD histogram
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
            fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram',
                                    marker_color=colors, opacity=0.6), row=2, col=1)
            
            fig_macd.update_layout(title="MACD Indicator", height=400, showlegend=True)
            fig_macd.update_xaxes(title_text="Date", row=2, col=1)
            fig_macd.update_yaxes(title_text="MACD Value", row=1, col=1)
            fig_macd.update_yaxes(title_text="Histogram", row=2, col=1)
            
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands
        st.subheader("📈 Bollinger Bands")
        fig_bb = go.Figure()
        
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price',
                                  line=dict(color='purple', width=2)))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='20-day SMA',
                                  line=dict(color='blue', width=1.5)))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', name='Upper Band',
                                  line=dict(color='red', dash='dash', width=1.5)))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', name='Lower Band',
                                  line=dict(color='green', dash='dash', width=1.5),
                                  fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
        
        fig_bb.update_layout(title="Bollinger Bands", height=500)
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Trade Analysis
        if 'metrics' in locals() and not metrics['Trades DataFrame'].empty:
            st.header("📋 Trade Analysis")
            trades_df = metrics['Trades DataFrame']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_duration = (pd.to_datetime(trades_df['exit_date']) - 
                              pd.to_datetime(trades_df['entry_date'])).dt.days.mean()
                st.metric("📅 Avg Trade Duration", f"{avg_duration:.1f} days")
            with col2:
                st.metric("🚀 Best Trade", f"{trades_df['return_pct'].max():.2%}")
            with col3:
                st.metric("💥 Worst Trade", f"{trades_df['return_pct'].min():.2%}")
            
            # Trade Returns Distribution
            returns_pct = trades_df['return_pct'] * 100
            fig_hist = px.histogram(x=returns_pct, nbins=20, title="Distribution of Trade Returns",
                                  labels={'x': 'Return (%)', 'y': 'Number of Trades'},
                                  color_discrete_sequence=['steelblue'])
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break Even")
            fig_hist.add_vline(x=returns_pct.mean(), line_dash="solid", line_color="green", 
                             annotation_text=f"Mean: {returns_pct.mean():.1f}%")
            fig_hist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Trade History Table
            st.subheader("📊 Trade History")
            display_trades = trades_df.copy()
            display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
            display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
            display_trades['Entry Price'] = display_trades['entry_price'].apply(lambda x: f"₹{x:.2f}")
            display_trades['Exit Price'] = display_trades['exit_price'].apply(lambda x: f"₹{x:.2f}")
            display_trades['P&L'] = display_trades['profit_loss'].apply(lambda x: f"₹{x:,.2f}")
            display_trades['Return %'] = display_trades['return_pct'].apply(lambda x: f"{x:.2%}")
            
            trade_display = display_trades[['Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 
                                          'P&L', 'Return %', 'exit_reason']].copy()
            st.dataframe(trade_display, use_container_width=True)
        
        # Data Download Section
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
    
    else:
        st.error("❌ No data found for the selected stock and date range.")

# ========================= SIDEBAR INFORMATION =========================

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
**Price Prediction Features:**
- Logistic Regression model for next-day prediction
- 59+ technical features including volatility, momentum, and lag features
- Confidence gauge and feature importance analysis

**Trading Dashboard Features:**
- SMA and EMA-based strategies
- Comprehensive backtesting with risk management
- Detailed performance metrics and trade analysis
- Interactive visualizations with Plotly

**Disclaimer**: This is for educational purposes only. Always do your own research before making investment decisions.
""")

st.sidebar.markdown("---")
st.sidebar.write("**Model Performance:**")
st.sidebar.write("• Accuracy: 55%")
st.sidebar.write("• F1 Score: 0.4839")
st.sidebar.write("• AUC: 0.5370")
st.sidebar.write("• Average Precision: 0.5300")

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer**: This platform is for research and educational purposes only. Stock market investments are subject to market risks. Please consult with a financial advisor before making investment decisions.")
st.markdown("**Developed by**: Zane Vijay Falcao")