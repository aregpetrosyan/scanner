import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
# New library for free sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the sentiment dictionary (only happens once)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

def get_real_sentiment_and_news(symbol):
    """Fetches real headlines and calculates custom sentiment score"""
    try:
        # Use a 3-day window to ensure we get news
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={FINNHUB_KEY}"
        news_items = requests.get(url).json()
        
        if not news_items or not isinstance(news_items, list):
            return 0.0, "No recent news found."

        # Get top 3 headlines
        top_news = news_items[:3]
        combined_text = " ".join([item.get('headline', '') for item in top_news])
        
        # Calculate our own sentiment score (-1 to 1)
        # We convert it to a 0.0 to 1.0 scale to keep your "Best Deal" math working
        raw_score = sia.polarity_scores(combined_text)['compound']
        normalized_sentiment = (raw_score + 1) / 2 
        
        latest_headline = top_news[0].get('headline', 'No headline.')
        return normalized_sentiment, latest_headline
    except Exception as e:
        return 0.5, f"News search failed: {str(e)[:30]}"

def scan():
    results = []
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    send_telegram("🔍 *Deep Scan Started:* Calculating real sentiment for top dips...")

    for symbol in tickers:
        try:
            # yf.Ticker can be slow, adding a small delay
            stock = yf.Ticker(symbol)
            df = stock.history(period="1mo")
            if len(df) < 14: continue
            
            # RSI Calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            # TARGET: Slightly wider net (RSI < 40)
            if rsi < 40:
                sentiment, headline = get_real_sentiment_and_news(symbol)
                # "Best Deal" uses our NEW calculated sentiment
                deal_score = (100 - rsi) * sentiment
                
                results.append({
                    "symbol": symbol, "score": deal_score, "rsi": rsi, 
                    "sentiment": sentiment, "headline": headline, "price": df['Close'].iloc[-1]
                })
            time.sleep(1.1) # CRITICAL: Stay under Finnhub's free limit (60 calls/min)
        except: continue

    # Rank by our new Deal Score
    results.sort(key=lambda x: x['score'], reverse=True)

    if results:
        msg = "🏆 *TOP 10 BEST DEALS* 🏆\n\n"
        for i, res in enumerate(results[:10]):
            msg += f"*{i+1}. {res['symbol']}* (Score: {res['score']:.1f})\n"
            msg += f"📰 `{res['headline'][:80]}...`\n"
            msg += f"💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f} | Sent: {res['sentiment']*100:.0f}%\n\n"
        send_telegram(msg)
    else:
        send_telegram("😴 No dips found. Market is too strong right now.")

if __name__ == "__main__":
    scan()
