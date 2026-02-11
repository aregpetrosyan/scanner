import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Sentiment Engine
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# --- CONFIG ---
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def get_context(symbol):
    """Calculates news sentiment and fetches the latest headline"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={FINNHUB_KEY}"
        news = requests.get(url).json()
        if news and isinstance(news, list) and len(news) > 0:
            headline = news[0].get('headline', 'No news.')
            score = (sia.polarity_scores(headline)['compound'] + 1) / 2
            return score, headline
    except: pass
    return 0.5, "Sentiment unavailable."

def scan():
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    results = []
    send_telegram(f"🚀 *Scanning {len(tickers)} Titans...* Identifying the Top 5 Deals.")

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            if len(hist) < 20: continue

            # 1. RSI Calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            # 2. Volume Surge (Relative Volume)
            avg_vol = hist['Volume'].iloc[-21:-1].mean()
            curr_vol = hist['Volume'].iloc[-1]
            vol_ratio = curr_vol / avg_vol

            # 3. Analyst Upside
            info = stock.info
            curr_price = hist['Close'].iloc[-1]
            # Safety check for missing targets
            target = info.get('targetMeanPrice')
            if target is None:
                target = curr_price 
            upside = (target - curr_price) / curr_price

            # CRITERIA: RSI under 45 (Starting to get cheap)
            if rsi < 45:
                sentiment, news = get_context(symbol)
                
                # DEAL SCORE FORMULA
                score = ((50 - rsi) * 1.2) + (vol_ratio * 10) + (upside * 100) + (sentiment * 5)
                
                results.append({
                    "symbol": symbol, "score": score, "rsi": rsi, 
                    "vol": vol_ratio, "upside": upside * 100, 
                    "news": news, "price": curr_price
                })
            
            time.sleep(1.2) # API Rate Limit Safety
        except Exception as e:
            print(f"Error on {symbol}: {e}")

    # Rank and take Top 5
    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]

    if top_5:
        msg = "🎯 *TOP 5 HIGH-CONVICTION DEALS* 🎯\n\n"
        for i, res in enumerate(top_5):
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f}\n"
            # FIXED FORMATTING HERE:
            msg += f"📊 Vol: {res['vol']:.1f}x | Upside: {res['upside']:.1f}%\n"
            msg += f"📰 `{res['news'][:75]}...`\n\n"
        send_telegram(msg)
    else:
        send_telegram("📭 No high-conviction dips found right now.")

if __name__ == "__main__":
    scan()
