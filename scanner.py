import yfinance as yf
import requests
import os
import pandas as pd
import time

# --- CONFIGURATION ---
TICKER_FILE = "tickers.txt"
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY") # Get from Finnhub.io
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def get_sentiment(symbol):
    """Checks if recent news is bullish using Finnhub's pre-calculated scores"""
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_KEY}"
        res = requests.get(url).json()
        # bullishPercent: 0.0 (Bearish) to 1.0 (Bullish)
        return res.get('sentiment', {}).get('bullishPercent', 0.5) 
    except:
        return 0.5 # Neutral fallback

def scan():
    with open(TICKER_FILE, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print(f"📡 Scanning {len(tickers)} stocks for aggressive dips...")

    for symbol in tickers:
        try:
            # 1. Pull 1 month of data (enough for RSI)
            df = yf.Ticker(symbol).history(period="1mo")
            if len(df) < 14: continue

            # 2. Manual RSI Calculation (standard 14-period)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            curr = df.iloc[-1]
            
            # 3. AGGRESSIVE FILTER: Just RSI and Sentiment
            # You can change 30 to 35 if you want even more alerts
            if curr['RSI'] < 30:
                sentiment = get_sentiment(symbol)
                
                # We only block if sentiment is EXTREMELY negative (< 0.3)
                if sentiment > 0.3:
                    msg = (f"🔥 DIP DETECTED: {symbol}\n"
                           f"Price: ${curr['Close']:.2f}\n"
                           f"RSI: {curr['RSI']:.1f}\n"
                           f"Sentiment: {sentiment*100:.0f}% Bullish\n"
                           f"Status: Oversold + News looks okay.")
                    send_telegram(msg)
            
            time.sleep(1) # Be kind to the API

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

if __name__ == "__main__":
    scan()
