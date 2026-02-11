import yfinance as yf
import requests
import os
import time

# --- CONFIG ---
TICKER_FILE = "tickers.txt"
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

def get_sentiment(symbol):
    try:
        url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol}&token={FINNHUB_KEY}"
        res = requests.get(url).json()
        return res.get('sentiment', {}).get('bullishPercent', 0.5)
    except:
        return 0.5

def scan():
    results = []
    with open(TICKER_FILE, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    for symbol in tickers:
        try:
            df = yf.Ticker(symbol).history(period="1mo")
            if len(df) < 14: continue

            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]

            if rsi < 40: # Look at anything "Cheap-ish"
                sentiment = get_sentiment(symbol)
                # BEST DEAL FORMULA
                deal_score = (100 - rsi) * sentiment
                
                results.append({
                    "symbol": symbol,
                    "rsi": rsi,
                    "sentiment": sentiment,
                    "score": deal_score,
                    "price": df['Close'].iloc[-1]
                })
            time.sleep(0.5)
        except: continue

    # SORT BY BEST DEAL SCORE
    results.sort(key=lambda x: x['score'], reverse=True)

    # FORMAT NOTIFICATION
    if results:
        msg = "🏆 TOP DIP DEALS TODAY 🏆\n\n"
        for i, res in enumerate(results[:5]): # Top 5
            msg += f"{i+1}. {res['symbol']}: Score {res['score']:.1f}\n"
            msg += f"   💰 ${res['price']:.2f} | RSI: {res['rsi']:.1f} | Bullish: {res['sentiment']*100:.0f}%\n\n"
        
        # Send to Telegram (Logic as before)
        print(msg) 

if __name__ == "__main__":
    scan()
