import yfinance as yf
import requests
import os

# --- SETTINGS ---
TICKER_FILE = "tickers.txt"
# Get secrets from GitHub Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload)

def scan():
    with open(TICKER_FILE, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1mo")
            if df.empty: continue
            
            # Simple "Dip" Logic: Price < 5% below 20-day Average
            current_price = df['Close'].iloc[-1]
            avg_20 = df['Close'].tail(20).mean()
            
            if current_price < (avg_20 * 0.95):
                msg = f"🟢 BUY THE DIP: {symbol} at ${current_price:.2f} (5% below average)"
                print(msg)
                send_telegram(msg)
        except Exception as e:
            print(f"Error {symbol}: {e}")

if __name__ == "__main__":
    scan()
