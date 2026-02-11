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
VIX_THRESHOLD = 30  # Don't buy if market fear is extreme

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def get_market_sentiment():
    """Returns the current VIX level as a fear gauge"""
    try:
        vix = yf.Ticker("^VIX")
        current_vix = vix.history(period="1d")['Close'].iloc[-1]
        return current_vix
    except:
        return 20.0 # Default to 'normal' if API fails

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
    
    # Global Filter: If VIX is too high, the risk is too great
    if vix_now > VIX_THRESHOLD:
        send_telegram(f"⚠️ *Market Alert:* VIX is at `{vix_now:.2f}` (High Fear). Scanning paused to avoid falling knives.")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    results = []
    send_telegram(f"🔍 *Market Calm (VIX: {vix_now:.2f})* | Scanning {len(tickers)} Titans...")

    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y") # 1y for SMA checks
            if len(hist) < 200: continue

            # 1. Trend Filter: Price must be above 200-day SMA
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            curr_price = hist['Close'].iloc[-1]
            if curr_price < sma_200: continue 

            # 2. RSI Calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            # 3. Volume Surge
            avg_vol = hist['Volume'].iloc[-21:-1].mean()
            curr_vol = hist['Volume'].iloc[-1]
            vol_ratio = curr_vol / avg_vol

            # 4. Upside
            info = stock.info
            target = info.get('targetMeanPrice', curr_price)
            upside = (target - curr_price) / curr_price

            if rsi < 45:
                sentiment, news = get_context(symbol)
                score = ((50 - rsi) * 1.2) + (vol_ratio * 10) + (upside * 100) + (sentiment * 5)
                
                results.append({
                    "symbol": symbol, "score": score, "rsi": rsi, 
                    "vol": vol_ratio, "upside": upside * 100, 
                    "news": news, "price": curr_price
                })
            time.sleep(1.2)
        except: continue

    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]

    if top_5:
        msg = f"🎯 *TOP DEALS (VIX: {vix_now:.2f})* 🎯\n\n"
        for i, res in enumerate(top_5):
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f}\n"
            msg += f"📊 Vol: {res['vol']:.1f}x | Upside: {res['upside']:.1f}%\n"
            msg += f"📰 `{res['news'][:70]}...`\n\n"
        send_telegram(msg)

if __name__ == "__main__":
    scan()
