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

def get_time_multiplier():
    """Calculates what % of the market day has passed to normalize volume."""
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < market_open: return 0.05
    if now > market_close: return 1.0
    total_seconds = (market_close - market_open).total_seconds()
    elapsed_seconds = (now - market_open).total_seconds()
    return max(0.1, elapsed_seconds / total_seconds)

def get_news_sentiment(symbol):
    """Fetches news from Finnhub and returns a sentiment score (0 to 1)."""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={FINNHUB_KEY}"
        news = requests.get(url).json()
        
        if news and isinstance(news, list) and len(news) > 0:
            headlines = [n.get('headline', '') for n in news[:10]] # Check top 10
            combined_text = " ".join(headlines)
            # VADER compound score is -1 to 1; we normalize it to 0 to 1
            score = (sia.polarity_scores(combined_text)['compound'] + 1) / 2
            latest_headline = headlines[0]
            return score, latest_headline
    except Exception as e:
        print(f"News error for {symbol}: {e}")
    return 0.5, "No recent news found."

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def scan():
    vix_now = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
    time_mult = get_time_multiplier()
    
    if vix_now > VIX_THRESHOLD:
        send_telegram(f"⚠️ *VIX High ({vix_now:.2f}):* Skipping scan to avoid volatility.")
        return

    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    results = []
    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")
            if len(hist) < 20: continue

            # Volume & RSI
            vol_ratio = (hist['Volume'].iloc[-1] / time_mult) / hist['Volume'].iloc[-2]
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            # Upside & Sentiment
            info = stock.info
            curr_price = hist['Close'].iloc[-1]
            upside = (info.get('targetMeanPrice', curr_price) - curr_price) / curr_price
            
            if rsi < 45: # Broadened filter for 2026 volatility
                sentiment_score, headline = get_news_sentiment(symbol)
                
                # FINAL DEAL SCORE FORMULA
                # High score = Low RSI + High Vol + High Upside + Positive News
                score = ((50 - rsi) * 1.5) + (vol_ratio * 10) + (upside * 100) + (sentiment_score * 20)
                
                results.append({
                    "symbol": symbol, "score": score, "rsi": rsi, 
                    "vol": vol_ratio, "upside": upside * 100, 
                    "sentiment": sentiment_score, "headline": headline, "price": curr_price
                })
            time.sleep(1.2) # Finnhub rate limit safety
        except: continue

    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]

    if top_5:
        msg = f"🎯 *TOP 5 REVERSAL DEALS* (VIX: {vix_now:.2f})\n\n"
        for i, res in enumerate(top_5):
            s_emoji = "💎" if res['sentiment'] > 0.7 else "⚖️" if res['sentiment'] > 0.4 else "⚠️"
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f} | {s_emoji}\n"
            msg += f"📊 Vol: {res['vol']:.2f}x | Upside: {res['upside']:.1f}%\n"
            msg += f"📰 `{res['headline'][:70]}...`\n\n"
        send_telegram(msg)

if __name__ == "__main__":
    scan()
