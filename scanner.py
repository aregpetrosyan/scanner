import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
import pytz
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize Sentiment
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# --- CONFIG ---
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_market_context():
    """Calculates how much of the trading day has elapsed (EST)."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Define Market Hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now < market_open: return 0.05
    if now > market_close: return 1.0
    
    elapsed = (now - market_open).total_seconds()
    total = (market_close - market_open).total_seconds()
    return max(0.1, elapsed / total)

def get_news_sentiment(symbol):
    """Fetches news and ensures the symbol is clean for the API."""
    clean_symbol = symbol.replace("-", ".") # Finnhub uses DOT, YFinance uses HYPHEN
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={start}&to={end}&token={FINNHUB_KEY}"
        r = requests.get(url)
        news = r.json()
        
        if news and isinstance(news, list) and len(news) > 0:
            headline = news[0].get('headline', '')
            score = (sia.polarity_scores(headline)['compound'] + 1) / 2
            return score, headline
    except Exception as e:
        print(f"Sentiment error for {symbol}: {e}")
    return 0.5, "No news context found."

def scan():
    time_pct = get_market_context()
    
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    results = []
    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            # Use '1d' interval to get the most recent data point
            df = stock.history(period="5d") 
            if df.empty: continue

            # VOLUME FIX: Compare today to yesterday's TOTAL, but adjust for time
            yesterday_vol = df['Volume'].iloc[-2]
            today_vol = df['Volume'].iloc[-1]
            
            # If today_vol is tiny, YFinance is lagging. 
            # We use 'time_pct' to project what today's volume WILL be.
            projected_vol = today_vol / time_pct
            vol_ratio = projected_vol / yesterday_vol

            # RSI Calculation
            hist_6m = stock.history(period="6mo")
            delta = hist_6m['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            if rsi < 45:
                sentiment, headline = get_news_sentiment(symbol)
                info = stock.info
                price = df['Close'].iloc[-1]
                target = info.get('targetMeanPrice', price)
                upside = (target - price) / price
                
                # RE-WEIGHTED SCORE: If Vol is lagging, RSI and Upside carry the weight
                score = ((50 - rsi) * 2.0) + (vol_ratio * 5) + (upside * 100) + (sentiment * 10)
                
                results.append({
                    "symbol": symbol, "score": score, "rsi": rsi, 
                    "vol": vol_ratio, "upside": upside * 100, 
                    "sentiment": sentiment, "headline": headline, "price": price
                })
            time.sleep(1) # Prevent API Rate Limits
        except: continue

    results.sort(key=lambda x: x['score'], reverse=True)
    top_5 = results[:5]

    if top_5:
        msg = f"🔍 *Intraday Scanner* (Market Day: {time_pct*100:.0f}% complete)\n\n"
        for i, res in enumerate(top_5):
            s_icon = "🟢" if res['sentiment'] > 0.6 else "🔴" if res['sentiment'] < 0.4 else "⚪"
            msg += f"*{i+1}. {res['symbol']}* | Score: {res['score']:.1f}\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f} {s_icon}\n"
            msg += f"📊 RVOL: {res['vol']:.2f}x | Upside: {res['upside']:.1f}%\n"
            msg += f"📰 `{res['headline'][:60]}...`\n\n"
        
        # Send to Telegram
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    scan()
