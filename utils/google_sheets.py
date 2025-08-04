# utils/google_sheets.py

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import streamlit as st
import json

class TradingGoogleSheets:
    def __init__(self, credentials_json_path=None, credentials_dict=None, sheet_name="Trading_Log"):
        """
        Initialize Google Sheets connection
        
        Parameters:
            credentials_json_path (str): Path to service account JSON file
            credentials_dict (dict): Service account credentials as dictionary (for Streamlit secrets)
            sheet_name (str): Name of the Google Sheet to create/use
        """
        self.sheet_name = sheet_name
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Initialize credentials
        if credentials_dict:
            # For Streamlit deployment with secrets
            self.creds = Credentials.from_service_account_info(credentials_dict, scopes=self.scope)
        elif credentials_json_path:
            # For local development with JSON file
            self.creds = Credentials.from_service_account_file(credentials_json_path, scopes=self.scope)
        else:
            raise ValueError("Either credentials_json_path or credentials_dict must be provided")
        
        self.client = gspread.authorize(self.creds)
        self.spreadsheet = None
        
    def create_or_get_spreadsheet(self):
        """Create a new spreadsheet or get existing one"""
        try:
            # Try to open existing spreadsheet
            self.spreadsheet = self.client.open(self.sheet_name)
            print(f"Opened existing spreadsheet: {self.sheet_name}")
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            self.spreadsheet = self.client.create(self.sheet_name)
            print(f"Created new spreadsheet: {self.sheet_name}")
            
            # Share with your email (replace with your email)
            # self.spreadsheet.share('your-email@gmail.com', perm_type='user', role='writer')
        
        # Create the three required worksheets
        self.setup_worksheets()
        return self.spreadsheet
    
    def setup_worksheets(self):
        """Setup the required worksheets with headers"""
        worksheets_config = {
            "Trade_Log": [
                "Timestamp", "Stock", "Strategy", "Signal_Type", "Price", "RSI", 
                "MA_Short", "MA_Long", "Entry_Date", "Exit_Date", "Entry_Price", 
                "Exit_Price", "Shares", "Profit_Loss", "Return_Pct", "Exit_Reason", "Duration_Days"
            ],
            "Summary_PL": [
                "Date", "Stock", "Strategy", "Total_Trades", "Winning_Trades", 
                "Losing_Trades", "Win_Rate", "Total_PL", "Best_Trade", "Worst_Trade", 
                "Avg_Win", "Avg_Loss", "Profit_Factor", "Max_Drawdown", "Sharpe_Ratio", 
                "Final_Portfolio_Value", "Total_Return"
            ],
            "Performance_Metrics": [
                "Date", "Stock", "Strategy", "Initial_Capital", "Final_Value", 
                "Total_Return", "Buy_Hold_Return", "Alpha", "Volatility", 
                "Sharpe_Ratio", "Max_Drawdown", "Total_Trades", "Win_Rate", 
                "Avg_Trade_Duration", "Transaction_Cost", "Notes"
            ]
        }
        
        existing_sheets = [ws.title for ws in self.spreadsheet.worksheets()]
        
        for sheet_name, headers in worksheets_config.items():
            if sheet_name not in existing_sheets:
                # Create new worksheet
                worksheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
                worksheet.append_row(headers)
                print(f"Created worksheet: {sheet_name}")
            else:
                print(f"Worksheet already exists: {sheet_name}")
    
    def log_trade_signals(self, df, strategy_name, stock_symbol):
        """Log all trade signals to Trade_Log worksheet"""
        try:
            worksheet = self.spreadsheet.worksheet("Trade_Log")
            
            # Get signals from dataframe
            signal_col = f'{strategy_name}_Signal'
            signals_df = df[df[signal_col] != 0].copy()
            
            if signals_df.empty:
                print("No signals to log")
                return
            
            # Prepare data for logging
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            rows_to_add = []
            for idx, row in signals_df.iterrows():
                signal_type = "BUY" if row[signal_col] == 1 else "SELL"
                
                row_data = [
                    current_time,  # Timestamp
                    stock_symbol,  # Stock
                    strategy_name,  # Strategy
                    signal_type,  # Signal_Type
                    round(row['Close'], 2),  # Price
                    round(row['RSI'], 2),  # RSI
                    round(row[f'{strategy_name}20'], 2),  # MA_Short
                    round(row[f'{strategy_name}50'], 2),  # MA_Long
                    "",  # Entry_Date (filled when trade completes)
                    "",  # Exit_Date
                    "",  # Entry_Price
                    "",  # Exit_Price
                    "",  # Shares
                    "",  # Profit_Loss
                    "",  # Return_Pct
                    "",  # Exit_Reason
                    ""   # Duration_Days
                ]
                rows_to_add.append(row_data)
            
            # Add all rows at once
            if rows_to_add:
                worksheet.append_rows(rows_to_add)
                print(f"Logged {len(rows_to_add)} signals to Trade_Log")
                
        except Exception as e:
            print(f"Error logging trade signals: {e}")
    
    def log_completed_trades(self, trades_df, strategy_name, stock_symbol):
        """Log completed trades to Trade_Log worksheet"""
        try:
            worksheet = self.spreadsheet.worksheet("Trade_Log")
            
            if trades_df.empty:
                print("No completed trades to log")
                return
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            rows_to_add = []
            for _, trade in trades_df.iterrows():
                row_data = [
                    current_time,  # Timestamp
                    stock_symbol,  # Stock
                    strategy_name,  # Strategy
                    "COMPLETED_TRADE",  # Signal_Type
                    round(trade['exit_price'], 2),  # Price (exit price)
                    "",  # RSI
                    "",  # MA_Short
                    "",  # MA_Long
                    trade['entry_date'],  # Entry_Date
                    trade['exit_date'],  # Exit_Date
                    round(trade['entry_price'], 2),  # Entry_Price
                    round(trade['exit_price'], 2),  # Exit_Price
                    round(trade['shares'], 4),  # Shares
                    round(trade['profit_loss'], 2),  # Profit_Loss
                    round(trade['return_pct'] * 100, 2),  # Return_Pct
                    trade['exit_reason'],  # Exit_Reason
                    (pd.to_datetime(trade['exit_date']) - pd.to_datetime(trade['entry_date'])).days  # Duration_Days
                ]
                rows_to_add.append(row_data)
            
            if rows_to_add:
                worksheet.append_rows(rows_to_add)
                print(f"Logged {len(rows_to_add)} completed trades to Trade_Log")
                
        except Exception as e:
            print(f"Error logging completed trades: {e}")
    
    def log_summary_pl(self, metrics, strategy_name, stock_symbol, trades_df):
        """Log summary P&L to Summary_PL worksheet"""
        try:
            worksheet = self.spreadsheet.worksheet("Summary_PL")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Calculate additional metrics
            total_trades = len(trades_df) if not trades_df.empty else 0
            winning_trades = len(trades_df[trades_df['return_pct'] > 0]) if not trades_df.empty else 0
            losing_trades = len(trades_df[trades_df['return_pct'] < 0]) if not trades_df.empty else 0
            
            # Extract numeric values from metrics
            win_rate = float(metrics['Win Rate'].strip('%')) if metrics['Win Rate'] != '0.00%' else 0
            total_pl = float(metrics['Final Portfolio Value'].replace('₹', '').replace(',', '')) - 100000  # Assuming 100k initial
            best_trade = trades_df['return_pct'].max() * 100 if not trades_df.empty else 0
            worst_trade = trades_df['return_pct'].min() * 100 if not trades_df.empty else 0
            avg_win = float(metrics['Average Win'].strip('%')) if metrics['Average Win'] != '0.00%' else 0
            avg_loss = float(metrics['Average Loss'].strip('%')) if metrics['Average Loss'] != '0.00%' else 0
            profit_factor = float(metrics['Profit Factor']) if metrics['Profit Factor'] != '0.00' else 0
            max_drawdown = float(metrics['Maximum Drawdown'].strip('%'))
            sharpe_ratio = float(metrics['Sharpe Ratio'])
            final_value = float(metrics['Final Portfolio Value'].replace('₹', '').replace(',', ''))
            total_return = float(metrics['Total Return'].strip('%'))
            
            row_data = [
                current_date,  # Date
                stock_symbol,  # Stock
                strategy_name,  # Strategy
                total_trades,  # Total_Trades
                winning_trades,  # Winning_Trades
                losing_trades,  # Losing_Trades
                round(win_rate, 2),  # Win_Rate
                round(total_pl, 2),  # Total_PL
                round(best_trade, 2),  # Best_Trade
                round(worst_trade, 2),  # Worst_Trade
                round(avg_win, 2),  # Avg_Win
                round(avg_loss, 2),  # Avg_Loss
                round(profit_factor, 2),  # Profit_Factor
                round(max_drawdown, 2),  # Max_Drawdown
                round(sharpe_ratio, 2),  # Sharpe_Ratio
                round(final_value, 2),  # Final_Portfolio_Value
                round(total_return, 2)  # Total_Return
            ]
            
            worksheet.append_row(row_data)
            print("Logged summary P&L to Summary_PL")
            
        except Exception as e:
            print(f"Error logging summary P&L: {e}")
    
    def log_performance_metrics(self, metrics, strategy_name, stock_symbol, initial_capital, 
                              transaction_cost, notes=""):
        """Log performance metrics to Performance_Metrics worksheet"""
        try:
            worksheet = self.spreadsheet.worksheet("Performance_Metrics")
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Extract and clean numeric values
            final_value = float(metrics['Final Portfolio Value'].replace('₹', '').replace(',', ''))
            total_return = float(metrics['Total Return'].strip('%'))
            buy_hold_return = float(metrics['Buy & Hold Return'].strip('%'))
            alpha = total_return - buy_hold_return
            volatility = float(metrics['Volatility (Annual)'].strip('%'))
            sharpe_ratio = float(metrics['Sharpe Ratio'])
            max_drawdown = float(metrics['Maximum Drawdown'].strip('%'))
            total_trades = metrics['Total Trades']
            win_rate = float(metrics['Win Rate'].strip('%'))
            
            # Calculate average trade duration (you might need to pass this from trades_df)
            avg_trade_duration = 0  # You can calculate this from trades_df if needed
            
            row_data = [
                current_date,  # Date
                stock_symbol,  # Stock
                strategy_name,  # Strategy
                initial_capital,  # Initial_Capital
                round(final_value, 2),  # Final_Value
                round(total_return, 2),  # Total_Return
                round(buy_hold_return, 2),  # Buy_Hold_Return
                round(alpha, 2),  # Alpha
                round(volatility, 2),  # Volatility
                round(sharpe_ratio, 2),  # Sharpe_Ratio
                round(max_drawdown, 2),  # Max_Drawdown
                total_trades,  # Total_Trades
                round(win_rate, 2),  # Win_Rate
                avg_trade_duration,  # Avg_Trade_Duration
                transaction_cost * 100,  # Transaction_Cost (as percentage)
                notes  # Notes
            ]
            
            worksheet.append_row(row_data)
            print("Logged performance metrics to Performance_Metrics")
            
        except Exception as e:
            print(f"Error logging performance metrics: {e}")
    
    def get_sheet_url(self):
        """Get the URL of the Google Sheet"""
        if self.spreadsheet:
            return self.spreadsheet.url
        return None
    
    def clear_worksheet(self, worksheet_name):
        """Clear all data from a worksheet (except headers)"""
        try:
            worksheet = self.spreadsheet.worksheet(worksheet_name)
            worksheet.clear()
            # Re-add headers based on the worksheet
            if worksheet_name == "Trade_Log":
                headers = ["Timestamp", "Stock", "Strategy", "Signal_Type", "Price", "RSI", 
                          "MA_Short", "MA_Long", "Entry_Date", "Exit_Date", "Entry_Price", 
                          "Exit_Price", "Shares", "Profit_Loss", "Return_Pct", "Exit_Reason", "Duration_Days"]
            elif worksheet_name == "Summary_PL":
                headers = ["Date", "Stock", "Strategy", "Total_Trades", "Winning_Trades", 
                          "Losing_Trades", "Win_Rate", "Total_PL", "Best_Trade", "Worst_Trade", 
                          "Avg_Win", "Avg_Loss", "Profit_Factor", "Max_Drawdown", "Sharpe_Ratio", 
                          "Final_Portfolio_Value", "Total_Return"]
            elif worksheet_name == "Performance_Metrics":
                headers = ["Date", "Stock", "Strategy", "Initial_Capital", "Final_Value", 
                          "Total_Return", "Buy_Hold_Return", "Alpha", "Volatility", 
                          "Sharpe_Ratio", "Max_Drawdown", "Total_Trades", "Win_Rate", 
                          "Avg_Trade_Duration", "Transaction_Cost", "Notes"]
            
            worksheet.append_row(headers)
            print(f"Cleared and reset worksheet: {worksheet_name}")
            
        except Exception as e:
            print(f"Error clearing worksheet {worksheet_name}: {e}")


# Integration function for your Streamlit app
def log_to_google_sheets(df, results, metrics, strategy_name, stock_symbol, 
                        initial_cash, transaction_cost, credentials_dict=None, 
                        credentials_json_path=None):
    """
    Main function to log all data to Google Sheets
    
    Parameters:
        df: DataFrame with signals and indicators
        results: Backtest results DataFrame
        metrics: Performance metrics dictionary
        strategy_name: Name of the strategy (SMA/EMA)
        stock_symbol: Stock symbol
        initial_cash: Initial capital
        transaction_cost: Transaction cost percentage
        credentials_dict: Google service account credentials (for Streamlit)
        credentials_json_path: Path to JSON credentials file (for local)
    """
    try:
        # Initialize Google Sheets connection
        sheets_logger = TradingGoogleSheets(
            credentials_dict=credentials_dict,
            credentials_json_path=credentials_json_path,
            sheet_name=f"Trading_Log_{stock_symbol}"
        )
        
        # Create or get spreadsheet
        spreadsheet = sheets_logger.create_or_get_spreadsheet()
        
        # Log trade signals
        sheets_logger.log_trade_signals(df, strategy_name, stock_symbol)
        
        # Log completed trades if available
        if not metrics['Trades DataFrame'].empty:
            sheets_logger.log_completed_trades(metrics['Trades DataFrame'], strategy_name, stock_symbol)
        
        # Log summary P&L
        trades_df = metrics['Trades DataFrame'] if not metrics['Trades DataFrame'].empty else pd.DataFrame()
        sheets_logger.log_summary_pl(metrics, strategy_name, stock_symbol, trades_df)
        
        # Log performance metrics
        sheets_logger.log_performance_metrics(
            metrics, strategy_name, stock_symbol, initial_cash, transaction_cost
        )
        
        return sheets_logger.get_sheet_url()
        
    except Exception as e:
        print(f"Error in log_to_google_sheets: {e}")
        return None


# # Streamlit integration function
# def add_google_sheets_to_streamlit(df, results, metrics, strategy_name, stock_symbol, 
#                                  initial_cash, transaction_cost):
#     """Add Google Sheets logging functionality to your Streamlit app"""
    
#     st.subheader("📊 Google Sheets Integration")
    
#     # Check if credentials are configured
#     if 'google_sheets_credentials' in st.secrets:
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             if st.button("📤 Log to Google Sheets", type="primary"):
#                 with st.spinner("Logging data to Google Sheets..."):
#                     sheet_url = log_to_google_sheets(
#                         df=df,
#                         results=results, 
#                         metrics=metrics,
#                         strategy_name=strategy_name,
#                         stock_symbol=stock_symbol,
#                         initial_cash=initial_cash,
#                         transaction_cost=transaction_cost,
#                         credentials_dict=dict(st.secrets.google_sheets_credentials)
#                     )
                    
#                     if sheet_url:
#                         st.success("✅ Data logged successfully!")
#                         st.markdown(f"🔗 [View Google Sheet]({sheet_url})")
#                     else:
#                         st.error("❌ Failed to log data to Google Sheets")
        
#         with col2:
#             st.info("💡 **Auto-logging enabled**\n\nData will be saved to:\n- Trade signals\n- P&L summary\n- Performance metrics")
    
#     else:
#         st.warning("⚠️ Google Sheets credentials not configured. Add your service account credentials to Streamlit secrets to enable logging.")
        
#         with st.expander("📋 Setup Instructions"):
#             st.markdown("""
#             **To enable Google Sheets integration:**
            
#             1. **Create a Google Cloud Project**
#             2. **Enable Google Sheets & Drive APIs**
#             3. **Create a Service Account**
#             4. **Download the JSON credentials**
#             5. **Add credentials to Streamlit secrets**
            
#             **In your `.streamlit/secrets.toml` file:**
#             ```toml
#             [google_sheets_credentials]
#             type = "service_account"
#             project_id = "your-project-id"
#             private_key_id = "your-private-key-id"
#             private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
#             client_email = "your-service-account@your-project.iam.gserviceaccount.com"
#             client_id = "your-client-id"
#             auth_uri = "https://accounts.google.com/o/oauth2/auth"
#             token_uri = "https://oauth2.googleapis.com/token"
#             auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
#             client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
#             universe_domain = "googleapis.com"
#             ```
#             """)