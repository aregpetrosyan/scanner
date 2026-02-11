import yfinance as yf
import requests
import os
import time

# --- SETUP ---
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def get_sentiment(symbol):
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_KEY}"
        res = requests.get(url).json()
        return res.get('sentiment', {}).get('bullishPercent', 0.5)
    except: return 0.5

def scan():
    results = []
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    for symbol in tickers:
        try:
            df = yf.Ticker(symbol).history(period="1mo")
            if len(df) < 14: continue
            
            # RSI Calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

            # TARGET: RSI < 35 (Slightly looser to catch more deals)
            if rsi < 35:
                sentiment = get_sentiment(symbol)
                deal_score = (100 - rsi) * sentiment
                
                # Reasoning Logic
                if sentiment > 0.7: reason = "💎 Irrational Dip: Market panic vs Strong News."
                elif sentiment > 0.5: reason = "📈 Healthy Pullback: Uptrend remains intact."
                else: reason = "⚠️ Risky Dip: News sentiment is weak."

                results.append({
                    "symbol": symbol, "score": deal_score, "rsi": rsi, 
                    "sentiment": sentiment, "reason": reason, "price": df['Close'].iloc[-1]
                })
            time.sleep(0.2) # Faster scan
        except: continue

    results.sort(key=lambda x: x['score'], reverse=True)

    if results:
        msg = "🏆 TOP 5 BEST DEALS 🏆\n\n"
        for i, res in enumerate(results[:5]):
            msg += f"{i+1}. {res['symbol']} (Score: {res['score']:.1f})\n"
            msg += f"{res['reason']}\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f}\n\n"
        send_telegram(msg)
    else:
        send_telegram("😴 No high-quality dips found in the last scan.")

if __name__ == "__main__":
    scan()
