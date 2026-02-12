import yfinance as yf
import pandas as pd
import requests
import os

# --- Configuration ---
FAST_WINDOW = 20
SLOW_WINDOW = 50

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload).raise_for_status()
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_signals(symbol):
    try:
        # Fetching 100 days to ensure EMA calculation has enough data points
        data = yf.download(symbol, period="100d", interval="1d", progress=False)
        
        if data.empty or len(data) < SLOW_WINDOW:
            return None, None

        data['EMA_Fast'] = data['Close'].ewm(span=FAST_WINDOW, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=SLOW_WINDOW, adjust=False).mean()
        
        latest_fast = float(data['EMA_Fast'].iloc[-1])
        latest_slow = float(data['EMA_Slow'].iloc[-1])
        prev_fast = float(data['EMA_Fast'].iloc[-2])
        prev_slow = float(data['EMA_Slow'].iloc[-2])
        price = round(float(data['Close'].iloc[-1]), 2)
        
        if prev_fast <= prev_slow and latest_fast > latest_slow:
            return "🚀 BUY", price
        elif prev_fast >= prev_slow and latest_fast < latest_slow:
            return "⚠️ SELL", price
        return "HOLD", price
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None, None

if __name__ == "__main__":
    # Check if tickers.txt exists
    if not os.path.exists("tickers.txt"):
        print("Error: tickers.txt not found!")
        exit(1)

    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    signals_found = []

    for ticker in tickers:
        print(f"Scanning {ticker}...")
        signal, price = get_signals(ticker)
        
        if signal and signal != "HOLD":
            msg = f"{signal} *{ticker}* at ${price}"
            signals_found.append(msg)
    
    if signals_found:
        final_report = "📊 *Trend Alert*\n\n" + "\n".join(signals_found)
        send_telegram_msg(final_report)
    else:
        print("No new trend crossovers detected today.")
