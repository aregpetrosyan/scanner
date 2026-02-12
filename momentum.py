"""
Momentum Trading Bot - GitHub Actions Version
Designed to run on schedule (not continuous loop)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import json

# ==================== CONFIGURATION FROM ENV ====================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

SYMBOL = os.environ.get('SYMBOL', 'SPY')
FAST_MA = int(os.environ.get('FAST_MA', 10))
SLOW_MA = int(os.environ.get('SLOW_MA', 50))

# File to track state between runs
STATE_FILE = 'bot_state.json'

# ==================== STATE MANAGEMENT ====================
def load_state():
    """Load previous state from file"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state):
    """Save state to file"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ==================== TELEGRAM FUNCTIONS ====================
def send_telegram_message(message):
    """Send notification to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured")
        return None
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
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
    df['MA_Fast'] = df['Close'].rolling(window=FAST_MA).mean()
    df['MA_Slow'] = df['Close'].rolling(window=SLOW_MA).mean()
    
    df['Signal'] = 0
    df.loc[df['MA_Fast'] > df['MA_Slow'], 'Signal'] = 1
    df.loc[df['MA_Fast'] < df['MA_Slow'], 'Signal'] = -1
    
    df['Position'] = df['Signal'].diff()
    
    return df

def check_for_signals():
    """Check for trading signals and send notifications"""
    try:
        # Get data and calculate signals
        df = get_data(SYMBOL)
        df = calculate_signals(df)
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Load previous state
        state = load_state()
        last_signal = state.get(SYMBOL, {}).get('last_signal')
        last_crossover_date = state.get(SYMBOL, {}).get('last_crossover_date')
        
        # Current signal info
        current_signal = int(current['Signal'])
        crossover = int(current['Position']) if not pd.isna(current['Position']) else 0
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if we have a NEW crossover (not already reported today)
        new_crossover = False
        
        if crossover == 2 and last_crossover_date != today:  # Bullish crossover
            new_crossover = True
            signal_type = "BUY"
            emoji = "🟢"
            trend = "BULLISH"
            direction = "⬆️ Fast MA crossed ABOVE Slow MA"
            
        elif crossover == -2 and last_crossover_date != today:  # Bearish crossover
            new_crossover = True
            signal_type = "SELL"
            emoji = "🔴"
            trend = "BEARISH"
            direction = "⬇️ Fast MA crossed BELOW Slow MA"
        
        # Send notification if new crossover detected
        if new_crossover:
            message = f"""
{emoji} <b>{signal_type} SIGNAL - {SYMBOL}</b>

Price: ${current['Close']:.2f}
Fast MA ({FAST_MA}): ${current['MA_Fast']:.2f}
Slow MA ({SLOW_MA}): ${current['MA_Slow']:.2f}

{direction}
Trend: {trend}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            send_telegram_message(message)
            print(f"✅ {signal_type} signal sent for {SYMBOL}")
            
            # Update state
            state[SYMBOL] = {
                'last_signal': signal_type,
                'last_crossover_date': today,
                'price': float(current['Close']),
                'ma_fast': float(current['MA_Fast']),
                'ma_slow': float(current['MA_Slow'])
            }
            save_state(state)
            
            # Log to file
            log_trade(signal_type, current)
            
        else:
            # No new crossover, just status
            trend = "BULLISH" if current_signal == 1 else "BEARISH"
            print(f"📊 {SYMBOL}: ${current['Close']:.2f} | Trend: {trend} | No new crossover")
            
            # Send daily status (optional - only at market close)
            hour = datetime.now().hour
            if hour == 16 and os.environ.get('DAILY_STATUS', 'false') == 'true':
                status_msg = f"""
📊 <b>Daily Status - {SYMBOL}</b>

Price: ${current['Close']:.2f}
Fast MA ({FAST_MA}): ${current['MA_Fast']:.2f}
Slow MA ({SLOW_MA}): ${current['MA_Slow']:.2f}

Current Trend: {trend}
Last Signal: {last_signal or 'None'}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                send_telegram_message(status_msg)
        
        return {
            'success': True,
            'symbol': SYMBOL,
            'price': float(current['Close']),
            'signal': current_signal,
            'crossover': new_crossover
        }
        
    except Exception as e:
        error_msg = f"❌ <b>Error</b>\n\nSymbol: {SYMBOL}\nError: {str(e)}"
        send_telegram_message(error_msg)
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

def log_trade(signal_type, current_data):
    """Log trades to CSV file"""
    log_file = f'trades_{SYMBOL}.csv'
    
    trade_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': SYMBOL,
        'signal': signal_type,
        'price': current_data['Close'],
        'ma_fast': current_data['MA_Fast'],
        'ma_slow': current_data['MA_Slow']
    }
    
    df = pd.DataFrame([trade_data])
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)
    
    print(f"Trade logged to {log_file}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("="*60)
    print("MOMENTUM TRADING BOT - Scheduled Run")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbol: {SYMBOL}")
    print(f"Strategy: MA Crossover ({FAST_MA}/{SLOW_MA})")
    print("="*60)
    
    # Check if market is open (optional)
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    
    # Only run during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
    # Note: This is simplified; adjust for your timezone and holidays
    if day < 5:  # Monday = 0, Friday = 4
        result = check_for_signals()
        
        if result['success']:
            print(f"\n✅ Run completed successfully")
            print(f"Symbol: {result['symbol']}")
            print(f"Price: ${result['price']:.2f}")
            print(f"New crossover: {result['crossover']}")
        else:
            print(f"\n❌ Run failed: {result.get('error')}")
    else:
        print("Weekend - market closed, skipping check")
    
    print("="*60)
