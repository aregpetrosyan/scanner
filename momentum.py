"""
Momentum/Trend Following Trading Bot
Uses Moving Average Crossover Strategy with Telegram notifications
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Get from @BotFather
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"      # Your chat ID

SYMBOL = "SPY"              # Stock/ETF to trade
FAST_MA = 10                # Fast moving average period
SLOW_MA = 50                # Slow moving average period
CHECK_INTERVAL = 3600       # Check every hour (in seconds)

# ==================== TELEGRAM FUNCTIONS ====================
def send_telegram_message(message):
    """Send notification to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data)
        return response.json()
    except Exception as e:
        print(f"Telegram error: {e}")
        return None

# ==================== TRADING LOGIC ====================
def get_data(symbol, period="3mo"):
    """Fetch historical data from yfinance"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d")
    return df

def calculate_signals(df):
    """Calculate moving averages and generate signals"""
    # Calculate moving averages
    df['MA_Fast'] = df['Close'].rolling(window=FAST_MA).mean()
    df['MA_Slow'] = df['Close'].rolling(window=SLOW_MA).mean()
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MA_Fast'] > df['MA_Slow'], 'Signal'] = 1   # Bullish (BUY)
    df.loc[df['MA_Fast'] < df['MA_Slow'], 'Signal'] = -1  # Bearish (SELL)
    
    # Detect crossovers
    df['Position'] = df['Signal'].diff()
    # Position = 2: Bullish crossover (buy signal)
    # Position = -2: Bearish crossover (sell signal)
    
    return df

def get_current_signal(symbol):
    """Get current trading signal"""
    df = get_data(symbol)
    df = calculate_signals(df)
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    result = {
        'symbol': symbol,
        'price': current['Close'],
        'ma_fast': current['MA_Fast'],
        'ma_slow': current['MA_Slow'],
        'signal': current['Signal'],
        'crossover': current['Position']
    }
    
    return result, df

def check_for_signals():
    """Main function to check for trading signals"""
    try:
        result, df = get_current_signal(SYMBOL)
        
        # Check for crossovers
        if result['crossover'] == 2:
            # Bullish crossover - BUY SIGNAL
            message = f"""
🟢 <b>BUY SIGNAL - {result['symbol']}</b>

Price: ${result['price']:.2f}
Fast MA ({FAST_MA}): ${result['ma_fast']:.2f}
Slow MA ({SLOW_MA}): ${result['ma_slow']:.2f}

⬆️ Fast MA crossed ABOVE Slow MA
Trend: BULLISH

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            send_telegram_message(message)
            print(f"BUY signal sent for {SYMBOL}")
            
        elif result['crossover'] == -2:
            # Bearish crossover - SELL SIGNAL
            message = f"""
🔴 <b>SELL SIGNAL - {result['symbol']}</b>

Price: ${result['price']:.2f}
Fast MA ({FAST_MA}): ${result['ma_fast']:.2f}
Slow MA ({SLOW_MA}): ${result['ma_slow']:.2f}

⬇️ Fast MA crossed BELOW Slow MA
Trend: BEARISH

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            send_telegram_message(message)
            print(f"SELL signal sent for {SYMBOL}")
            
        else:
            # No crossover, just status update
            trend = "BULLISH" if result['signal'] == 1 else "BEARISH"
            print(f"{SYMBOL}: ${result['price']:.2f} | Trend: {trend} | No crossover")
            
    except Exception as e:
        error_msg = f"❌ Error checking {SYMBOL}: {str(e)}"
        send_telegram_message(error_msg)
        print(error_msg)

# ==================== BACKTESTING FUNCTIONS ====================
def backtest_strategy(symbol, start_date=None, end_date=None):
    """Backtest the momentum strategy"""
    # Fetch data
    if start_date and end_date:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
    else:
        df = get_data(symbol, period="2y")
    
    # Calculate signals
    df = calculate_signals(df)
    
    # Simulate trading
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    
    # Calculate cumulative returns
    df['Cumulative_Market'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    
    # Performance metrics
    total_return = (df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    market_return = (df['Cumulative_Market'].iloc[-1] - 1) * 100
    
    # Count trades
    trades = len(df[df['Position'].abs() == 2])
    
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS - {symbol}")
    print("="*50)
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Strategy Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {market_return:.2f}%")
    print(f"Outperformance: {total_return - market_return:.2f}%")
    print(f"Total Trades: {trades}")
    print("="*50 + "\n")
    
    return df

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("Momentum Trading Bot Started")
    print(f"Symbol: {SYMBOL}")
    print(f"Fast MA: {FAST_MA}, Slow MA: {SLOW_MA}")
    print(f"Check interval: {CHECK_INTERVAL} seconds\n")
    
    # Run backtest first
    print("Running backtest...")
    backtest_strategy(SYMBOL)
    
    # Send startup message
    startup_msg = f"""
🤖 <b>Momentum Bot Started</b>

Symbol: {SYMBOL}
Strategy: MA Crossover ({FAST_MA}/{SLOW_MA})
Interval: {CHECK_INTERVAL}s

Bot is now monitoring for signals...
"""
    send_telegram_message(startup_msg)
    
    # Main loop
    while True:
        try:
            check_for_signals()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nBot stopped by user")
            send_telegram_message("🛑 Momentum Bot Stopped")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying
