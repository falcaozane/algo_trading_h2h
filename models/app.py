import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import plotly.graph_objects as go
import plotly.express as px 
from datetime import datetime, timedelta
import warnings
from curl_cffi import requests
session = requests.Session(impersonate="chrome")
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction App")
st.markdown("This app uses a trained Logistic Regression model to predict whether a stock will go **UP** ⬆️ or **DOWN** ⬇️ the next day.")

# Sidebar for user inputs
st.sidebar.header("🔧 Configuration")

# Stock symbols from your model
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

# User inputs
selected_stock = st.sidebar.selectbox("Select Stock Symbol", STOCK_SYMBOLS, index=35)  # Default to RELIANCE.NS
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
prediction_mode = st.sidebar.button("Start Analysis")
rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14, step=1)
short_period = st.sidebar.slider("Short-term", min_value=5, max_value=50, value=20, step=1)
long_period = st.sidebar.slider("Long-term", min_value=50, max_value=200, value=50, step=1)

# Helper functions (same as in your original code)
def SMA(series, period):
    return series.rolling(window=period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def create_volatility_features(df):
    if 'return_1d' not in df.columns:
        df['return_1d'] = df['Close'].pct_change()
    
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()
    
    df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']
    df['vol_ratio_10_20'] = df['volatility_10d'] / df['volatility_20d']
    df['vol_rank_20'] = df['volatility_5d'].rolling(20).rank(pct=True)
    df['vol_rank_50'] = df['volatility_5d'].rolling(50).rank(pct=True)
    
    return df

def create_enhanced_lag_features(df):
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
    
    for lag in [1, 2, 3]:
        if 'RSI14' in df.columns:
            df[f'rsi_lag_{lag}'] = df['RSI14'].shift(lag)
        if 'MACD' in df.columns:
            df[f'macd_lag_{lag}'] = df['MACD'].shift(lag)
    
    if 'volume_ratio_20' in df.columns:
        for lag in [1, 2]:
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio_20'].shift(lag)
    
    return df

def create_volume_features(df):
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_sma_50'] = df['Volume'].rolling(50).mean()
    
    df['volume_ratio_10'] = df['Volume'] / df['volume_sma_10']
    df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']
    df['volume_ratio_50'] = df['Volume'] / df['volume_sma_50']
    
    df['price_volume'] = df['Close'] * df['Volume']
    df['pv_sma_5'] = df['price_volume'].rolling(5).mean()
    df['volume_momentum_5'] = df['Volume'] / df['Volume'].shift(5)
    
    return df

def create_momentum_features(df):
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}d'] = df['Close'] / df['Close'].shift(period) - 1
    
    for period in [5, 10]:
        df[f'roc_{period}d'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    return df

def create_position_features(df):
    for period in [10, 20, 50]:
        df[f'high_{period}d'] = df['High'].rolling(period).max()
        df[f'low_{period}d'] = df['Low'].rolling(period).min()
        df[f'price_position_{period}'] = (df['Close'] - df[f'low_{period}d']) / (df[f'high_{period}d'] - df[f'low_{period}d'])
    
    if 'SMA20' in df.columns:
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['SMA20'] + (bb_std * 2)
        df['bb_lower'] = df['SMA20'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def process_stock_data(df):
    """Process stock data to create all features"""
    df = df.copy()
    
    # Basic technical indicators
    df['SMA20'] = SMA(df['Close'], short_period)
    df['SMA50'] = SMA(df['Close'], long_period)
    df['EMA20'] = EMA(df['Close'], short_period)
    df['EMA50'] = EMA(df['Close'], long_period)
    df['RSI14'] = RSI(df['Close'], rsi_period)
    df['RSI20'] = RSI(df['Close'], rsi_period + 6)  # Example for another RSI period
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = MACD(df['Close'])
    
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

@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date,session=session)
        # Flatten the MultiIndex columns
        data.columns = [col[0] for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Feature list (same as in your model)
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
    'price_position_50', 'bb_upper', 'bb_lower', 'bb_position','target'
]

# Main app logic
st.header(f"📊 Latest Data Prediction for {selected_stock}")
    
    
with st.spinner("Loading stock data..."):
    stock_data = load_stock_data(selected_stock, start_date, end_date)
    
    if stock_data is not None and not stock_data.empty:
        # Process the data
        processed_data = process_stock_data(stock_data)
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
            
            # Create feature vector
            feature_vector = latest_data[FEATURES].values.reshape(1, -1)
            
            # For demo purposes, create a mock prediction (since we don't have the actual model file)
            # In real implementation, you would load your saved model:
            model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
            scaler = pickle.load(open('scaler.pkl', 'rb'))  # You'd need to save this too
            
            
            # Scale the features
            feature_vector_scaled = scaler.transform(feature_vector)

            # Make prediction
            prediction = model.predict(feature_vector_scaled)[0]
            probability = model.predict_proba(feature_vector_scaled)[0].max()
            
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
            
            # Technical indicators chart
            st.header("📈 Technical Analysis")
            
            # Price and  Simple moving averages
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-60:], 
                y=processed_data['Close'][-60:],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-60:], 
                y=processed_data['SMA20'][-60:],
                mode='lines',
                name='SMA20',
                line=dict(color='orange', width=1)
            ))
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-60:], 
                y=processed_data['SMA50'][-60:],
                mode='lines',
                name='SMA50',
                line=dict(color='red', width=1)
            ))
            
            fig_price.update_layout(
                title=f"{selected_stock} - Price and Simple Moving Averages (Last 60 Days)",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Price and Exponential moving averages
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-30:], 
                y=processed_data['Close'][-30:],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-30:], 
                y=processed_data['EMA20'][-30:],
                mode='lines',
                name='EMA20',
                line=dict(color='orange', width=1)
            ))
            fig_price.add_trace(go.Scatter(
                x=processed_data.index[-30:], 
                y=processed_data['EMA50'][-30:],
                mode='lines',
                name='EMA50',
                line=dict(color='red', width=1)
            ))
            
            fig_price.update_layout(
                title=f"{selected_stock} - Price and Exponential Moving Averages (Last 60 Days)",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # RSI chart
            col1, col2 = st.columns(2)
            with col1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=processed_data.index[-30:], 
                    y=processed_data['RSI14'][-30:],
                    mode='lines',
                    name='RSI14',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(
                    title="RSI (14-day)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    height=300
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD chart
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=processed_data.index[-30:], 
                    y=processed_data['MACD'][-30:],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=processed_data.index[-30:], 
                    y=processed_data['MACD_signal'][-30:],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ))
                fig_macd.update_layout(
                    title="MACD",
                    xaxis_title="Date",
                    yaxis_title="MACD",
                    height=300
                )
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Feature importance (mock data for demo)
            st.header("🎯 Key Factors")
            st.write("Most important features affecting the prediction:")
            
            mock_features = ['RSI14', 'return_lag_1', 'volatility_5d', 'MACD', 'volume_ratio_20']
            mock_importance = [0.15, 0.12, 0.10, 0.08, 0.07]
            
            fig_importance = px.bar(
                x=mock_importance, 
                y=mock_features,
                orientation='h',
                title="Feature Importance"
            )
            fig_importance.update_layout(height=300)
            st.plotly_chart(fig_importance, use_container_width=True)
            
        else:
            st.error("Not enough data to make a prediction. Please try a different stock or date range.")
    else:
        st.error("Unable to load stock data. Please check the symbol and try again.")



# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app uses a Logistic Regression model trained on:
- **50 Indian stocks** from NSE
- **59 technical features** including RSI, MACD, moving averages, volatility measures, and lag features
- **Historical data** for pattern recognition

**Disclaimer**: This is for educational purposes only. Always do your own research before making investment decisions.
""")

st.sidebar.markdown("---")
st.sidebar.write("**Model Performance:**")
st.sidebar.write("• Accuracy: 55%")
st.sidebar.write("• F1 Score: 0.4839")
st.sidebar.write("• AUC: 0.5370")
st.sidebar.write("Average Precision (AP): 0.5300")

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer**: This prediction model is for research purposes only. Stock market investments are subject to market risks. Please consult with a financial advisor before making investment decisions.")