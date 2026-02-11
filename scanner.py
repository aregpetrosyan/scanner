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
VIX_THRESHOLD = 35 

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def get_market_sentiment():
    try:
        vix = yf.Ticker("^VIX")
        return vix.history(period="1d")['Close'].iloc[-1]
    except: return 20.0

def get_context(symbol):
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
    vix_now = get_market_sentiment()
    if vix_now > VIX_THRESHOLD:
        send_telegram(f"⚠️ *VIX High ({vix_now:.2f}):* Market panic detected. Standing by.")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    results = []
    send_telegram(f"🚀 *Scan Start* (VIX: {vix_now:.2f}) | Monitoring {len(tickers)} stocks...")

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            if len(hist) < 20: continue

            # --- VOLUME NORMALIZATION ---
            # Compare current volume to YESTERDAY'S total volume
            # This makes the ratio 1.0 if we match yesterday's pace
            yesterday_vol = hist['Volume'].iloc[-2]
            current_vol = hist['Volume'].iloc[-1]
            vol_ratio = current_vol / yesterday_vol
            
            # --- RSI & UPSIDE ---
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            info = stock.info
            curr_price = hist['Close'].iloc[-1]
            target = info.get('targetMeanPrice', curr_price)
            upside = (target - curr_price) / curr_price

            # CRITERIA: RSI under 42 (Oversold territory)
            if rsi < 42:
                sentiment, news = get_context(symbol)
                
                # New Score: Higher weight to RSI and Upside since SMA is gone
                score = ((45 - rsi) * 2.5) + (vol_ratio * 15) + (upside * 120)
                
                results.append({
                    "symbol": symbol, "score": score, "rsi": rsi, 
                    "vol": vol_ratio, "upside": upside * 100, 
                    "news": news, "price": curr_price
                })
            time.sleep(1.1)
        except: continue

    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]

    if top_5:
        msg = f"📉 *TOP 5 REVERSAL OPPORTUNITIES*\n\n"
        for i, res in enumerate(top_5):
            # Volume Emoji: 🔥 for high interest, ❄️ for low interest
            v_emoji = "🔥" if res['vol'] > 1.0 else "❄️"
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f}\n"
            msg += f"📊 Vol: {res['vol']:.2f}x {v_emoji} | Upside: {res['upside']:.1f}%\n"
            msg += f"📰 `{res['news'][:70]}...`\n\n"
        send_telegram(msg)

if __name__ == "__main__":
    scan()
