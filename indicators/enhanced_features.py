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