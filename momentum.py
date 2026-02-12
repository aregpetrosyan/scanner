import yfinance as yf
import pandas as pd
import requests
import os

# --- Configuration ---
SYMBOL = "BTC-USD"
FAST_WINDOW = 20
SLOW_WINDOW = 50

# Secrets from GitHub Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Skipping notification.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def get_signals(symbol):
    data = yf.download(symbol, period="60d", interval="1d")
    data['EMA_Fast'] = data['Close'].ewm(span=FAST_WINDOW, adjust=False).mean()
    data['EMA_Slow'] = data['Close'].ewm(span=SLOW_WINDOW, adjust=False).mean()
    
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    price = round(latest['Close'], 2)
    
    if previous['EMA_Fast'] <= previous['EMA_Slow'] and latest['EMA_Fast'] > latest['EMA_Slow']:
        return "🚀 BUY (Golden Cross)", price
    elif previous['EMA_Fast'] >= previous['EMA_Slow'] and latest['EMA_Fast'] < latest['EMA_Slow']:
        return "⚠️ SELL (Death Cross)", price
    else:
        return "HOLD", price

if __name__ == "__main__":
    signal, price = get_signals(SYMBOL)
    
    status_msg = f"*{SYMBOL} Update*\nPrice: ${price}\nSignal: {signal}"
    print(status_msg)
    
    # Only notify if there's an action to take
    if "HOLD" not in signal:
        send_telegram_msg(status_msg)
