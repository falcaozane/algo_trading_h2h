# utils/backtester.py

import pandas as pd
import numpy as np

def backtest_signals(df, signal_col='Signal', price_col='Close', initial_cash=100000, 
                    transaction_cost=0.001, stop_loss=None, take_profit=None):
    """
    Enhanced backtest strategy using buy/sell signals.

    Parameters:
        df (pd.DataFrame): DataFrame with signal and price columns
        signal_col (str): Name of the signal column (1 = Buy, -1 = Sell)
        price_col (str): Name of the price column to use for trading
        initial_cash (float): Starting cash for the backtest
        transaction_cost (float): Transaction cost as percentage (0.001 = 0.1%)
        stop_loss (float): Stop loss percentage (0.05 = 5%)
        take_profit (float): Take profit percentage (0.10 = 10%)

    Returns:
        tuple: (results_df, performance_metrics)
    """
    df = df.copy()
    df['Position'] = 0  # 1 if holding, 0 otherwise
    df['Cash'] = initial_cash
    df['Holdings_Value'] = 0
    df['Total'] = initial_cash
    df['Returns'] = 0
    df['Trade_Action'] = ''

    position = 0  # Whether we hold a stock
    cash = initial_cash
    shares = 0
    entry_price = 0
    trades = []
    
    for i in range(len(df)):
        current_price = df[price_col].iloc[i]
        signal = df[signal_col].iloc[i]
        
        # Check stop loss and take profit if holding position
        if position == 1 and shares > 0:
            price_change = (current_price - entry_price) / entry_price
            
            # Stop loss check
            if stop_loss and price_change <= -stop_loss:
                # Force sell due to stop loss
                cash = shares * current_price * (1 - transaction_cost)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'profit_loss': cash - (shares * entry_price),
                    'return_pct': price_change,
                    'exit_reason': 'Stop Loss'
                })
                shares = 0
                position = 0
                df.at[df.index[i], 'Trade_Action'] = 'STOP_LOSS'
                
            # Take profit check
            elif take_profit and price_change >= take_profit:
                # Force sell due to take profit
                cash = shares * current_price * (1 - transaction_cost)
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'profit_loss': cash - (shares * entry_price),
                    'return_pct': price_change,
                    'exit_reason': 'Take Profit'
                })
                shares = 0
                position = 0
                df.at[df.index[i], 'Trade_Action'] = 'TAKE_PROFIT'

        # Process regular buy/sell signals
        if signal == 1 and position == 0 and cash > 0:
            # Buy signal
            cost_with_fees = cash * (1 + transaction_cost)
            if cost_with_fees <= cash:
                shares = cash / (current_price * (1 + transaction_cost))
                cash = 0
                position = 1
                entry_price = current_price
                entry_date = df.index[i]
                df.at[df.index[i], 'Trade_Action'] = 'BUY'

        elif signal == -1 and position == 1 and shares > 0:
            # Sell signal
            cash = shares * current_price * (1 - transaction_cost)
            
            # Record trade
            price_change = (current_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': current_price,
                'shares': shares,
                'profit_loss': cash - (shares * entry_price),
                'return_pct': price_change,
                'exit_reason': 'Signal'
            })
            
            shares = 0
            position = 0
            df.at[df.index[i], 'Trade_Action'] = 'SELL'

        # Update portfolio values
        holdings_value = shares * current_price if shares > 0 else 0
        total_value = cash + holdings_value
        
        df.at[df.index[i], 'Position'] = position
        df.at[df.index[i], 'Cash'] = cash
        df.at[df.index[i], 'Holdings_Value'] = holdings_value
        df.at[df.index[i], 'Total'] = total_value
        
        # Calculate daily returns
        if i > 0:
            prev_total = df['Total'].iloc[i-1]
            df.at[df.index[i], 'Returns'] = (total_value - prev_total) / prev_total

    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(df, trades, initial_cash)
    
    return df[['Close', signal_col, 'Position', 'Cash', 'Holdings_Value', 'Total', 
              'Returns', 'Trade_Action']], performance_metrics


def calculate_performance_metrics(df, trades, initial_cash):
    """Calculate comprehensive performance metrics"""
    final_value = df['Total'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash
    
    # Calculate buy and hold return for comparison
    buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
    
    # Risk metrics
    returns = df['Returns'].dropna()
    if len(returns) > 0:
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        volatility = 0
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Trade statistics
    if trades:
        trades_df = pd.DataFrame(trades)
        win_rate = len(trades_df[trades_df['return_pct'] > 0]) / len(trades_df)
        avg_win = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if len(trades_df[trades_df['return_pct'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['return_pct'] < 0]['return_pct'].mean() if len(trades_df[trades_df['return_pct'] < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Buy & Hold Return': f"{buy_hold_return:.2%}",
        'Final Portfolio Value': f"₹{final_value:,.2f}",
        'Total Trades': len(trades),
        'Win Rate': f"{win_rate:.2%}",
        'Average Win': f"{avg_win:.2%}",
        'Average Loss': f"{avg_loss:.2%}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Volatility (Annual)': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Trades DataFrame': pd.DataFrame(trades) if trades else pd.DataFrame()
    }
    
