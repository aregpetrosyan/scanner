import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta

# --- SETUP ---
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        # Using parse_mode 'Markdown' for better formatting
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def get_stock_context(symbol):
    """Fetches the latest news headline and sentiment score"""
    try:
        # 1. Get Sentiment Score
        sent_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_KEY}"
        sentiment = requests.get(sent_url).json().get('sentiment', {}).get('bullishPercent', 0.5)

        # 2. Get Latest Headline
        # Finnhub requires a date range for company news
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={FINNHUB_KEY}"
        news_data = requests.get(news_url).json()
        
        headline = "No recent news found."
        if news_data and len(news_data) > 0:
            headline = news_data[0].get('headline', 'No headline available.')
            
        return sentiment, headline
    except:
        return 0.5, "Error fetching news context."

def scan():
    results = []
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    send_telegram("🔍 *Scan Started:* Analyzing Top 100 for the best dips...")

    for symbol in tickers:
        try:
            df = yf.Ticker(symbol).history(period="1mo")
            if len(df) < 14: continue
            
            # RSI Calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

            # Aggressive Target: RSI < 38
            if rsi < 38:
                sentiment, headline = get_stock_context(symbol)
                deal_score = (100 - rsi) * sentiment
                
                results.append({
                    "symbol": symbol, "score": deal_score, "rsi": rsi, 
                    "sentiment": sentiment, "headline": headline, "price": df['Close'].iloc[-1]
                })
            time.sleep(0.3) # Respect API limits
        except: continue

    # Sort by the best "Spring Effect"
    results.sort(key=lambda x: x['score'], reverse=True)

    if results:
        msg = "🏆 *TOP 10 BEST DIP DEALS* 🏆\n\n"
        for i, res in enumerate(results[:10]):
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"📰 `{res['headline'][:100]}...`\n" # Truncate long headlines
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f} | Bullish: {res['sentiment']*100:.0f}%\n\n"
        send_telegram(msg)
    else:
        send_telegram("😴 No high-quality dips found. Market may be overbought.")

if __name__ == "__main__":
    scan()
