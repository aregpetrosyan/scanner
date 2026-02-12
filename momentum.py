import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

# --- Configuration ---
FAST_WINDOW = 20
SLOW_WINDOW = 50
RSI_PERIOD = 14
VOLUME_PERIOD = 20
MIN_VOLUME = 100000  # Minimum average volume to consider

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload).raise_for_status()
    except Exception as e:
        print(f"Telegram Error: {e}")

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def get_momentum_analysis(symbol):
    """Comprehensive momentum analysis"""
    try:
        # Fetch more historical data for better indicator calculation
        data = yf.download(symbol, period="6mo", interval="1d", progress=False)
        
        if data.empty or len(data) < SLOW_WINDOW:
            return None
        
        # Calculate EMAs
        data['EMA_Fast'] = data['Close'].ewm(span=FAST_WINDOW, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=SLOW_WINDOW, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        data['RSI'] = calculate_rsi(data, RSI_PERIOD)
        
        # Calculate MACD
        data['MACD'], data['Signal'], data['MACD_Hist'] = calculate_macd(data)
        
        # Volume analysis
        data['Avg_Volume'] = data['Volume'].rolling(window=VOLUME_PERIOD).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Avg_Volume']
        
        # Price momentum (rate of change)
        data['ROC_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
        data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
        data['ROC_20'] = ((data['Close'] - data['Close'].shift(20)) / data['Close'].shift(20)) * 100
        
        # ATR for volatility (simplified)
        data['High-Low'] = data['High'] - data['Low']
        data['ATR'] = data['High-Low'].rolling(window=14).mean()
        
        # Consecutive green/red days
        data['Daily_Change'] = data['Close'].diff()
        
        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Extract key metrics
        metrics = {
            'symbol': symbol,
            'price': round(float(latest['Close']), 2),
            'change_1d': round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            'change_5d': round(float(latest['ROC_5']), 2),
            'change_10d': round(float(latest['ROC_10']), 2),
            'change_20d': round(float(latest['ROC_20']), 2),
            'volume': int(latest['Volume']),
            'avg_volume': int(latest['Avg_Volume']),
            'volume_ratio': round(float(latest['Volume_Ratio']), 2),
            'rsi': round(float(latest['RSI']), 2),
            'macd': round(float(latest['MACD']), 4),
            'macd_signal': round(float(latest['Signal']), 4),
            'macd_hist': round(float(latest['MACD_Hist']), 4),
            'ema_fast': round(float(latest['EMA_Fast']), 2),
            'ema_slow': round(float(latest['EMA_Slow']), 2),
            'ema_200': round(float(latest['EMA_200']), 2) if not pd.isna(latest['EMA_200']) else None,
            'atr': round(float(latest['ATR']), 2),
            'prev_ema_fast': round(float(prev['EMA_Fast']), 2),
            'prev_ema_slow': round(float(prev['EMA_Slow']), 2),
        }
        
        # Calculate consecutive days
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(data) - 1, 0, -1):
            if data.iloc[i]['Daily_Change'] > 0:
                if consecutive_down == 0:
                    consecutive_up += 1
                else:
                    break
            elif data.iloc[i]['Daily_Change'] < 0:
                if consecutive_up == 0:
                    consecutive_down += 1
                else:
                    break
            else:
                break
        
        metrics['consecutive_up'] = consecutive_up
        metrics['consecutive_down'] = consecutive_down
        
        # Determine signals and momentum status
        metrics['signal'] = determine_signal(metrics)
        metrics['momentum_score'] = calculate_momentum_score(metrics)
        metrics['trend_strength'] = determine_trend_strength(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def determine_signal(metrics):
    """Determine trading signal based on multiple indicators"""
    signals = []
    
    # EMA Crossover
    if metrics['prev_ema_fast'] <= metrics['prev_ema_slow'] and metrics['ema_fast'] > metrics['ema_slow']:
        signals.append("EMA_BULLISH_CROSS")
    elif metrics['prev_ema_fast'] >= metrics['prev_ema_slow'] and metrics['ema_fast'] < metrics['ema_slow']:
        signals.append("EMA_BEARISH_CROSS")
    
    # MACD Crossover
    prev_macd_hist = metrics['macd'] - metrics['macd_signal']
    if prev_macd_hist <= 0 and metrics['macd_hist'] > 0:
        signals.append("MACD_BULLISH_CROSS")
    elif prev_macd_hist >= 0 and metrics['macd_hist'] < 0:
        signals.append("MACD_BEARISH_CROSS")
    
    # Momentum signals
    if metrics['ema_fast'] > metrics['ema_slow'] and metrics['price'] > metrics['ema_fast']:
        signals.append("UPTREND")
    elif metrics['ema_fast'] < metrics['ema_slow'] and metrics['price'] < metrics['ema_fast']:
        signals.append("DOWNTREND")
    
    # Strong momentum
    if metrics['change_5d'] > 5 and metrics['change_10d'] > 8 and metrics['rsi'] < 70:
        signals.append("STRONG_MOMENTUM_UP")
    
    # Volume confirmation
    if metrics['volume_ratio'] > 1.5:
        signals.append("HIGH_VOLUME")
    
    # Breakout above 200 EMA
    if metrics['ema_200'] and metrics['price'] > metrics['ema_200'] and metrics['ema_fast'] > metrics['ema_200']:
        signals.append("ABOVE_200EMA")
    
    return signals

def calculate_momentum_score(metrics):
    """Calculate overall momentum score (0-100)"""
    score = 50  # Base score
    
    # Trend alignment (max 25 points)
    if metrics['ema_fast'] > metrics['ema_slow']:
        score += 10
        if metrics['price'] > metrics['ema_fast']:
            score += 10
        if metrics['ema_200'] and metrics['price'] > metrics['ema_200']:
            score += 5
    else:
        score -= 10
        if metrics['price'] < metrics['ema_fast']:
            score -= 10
        if metrics['ema_200'] and metrics['price'] < metrics['ema_200']:
            score -= 5
    
    # Price momentum (max 20 points)
    if metrics['change_5d'] > 3:
        score += 10
    elif metrics['change_5d'] < -3:
        score -= 10
    
    if metrics['change_10d'] > 5:
        score += 10
    elif metrics['change_10d'] < -5:
        score -= 10
    
    # RSI (max 15 points)
    if 40 < metrics['rsi'] < 70:
        score += 15  # Sweet spot
    elif metrics['rsi'] > 80 or metrics['rsi'] < 20:
        score -= 10  # Extreme conditions
    
    # MACD (max 10 points)
    if metrics['macd_hist'] > 0:
        score += 10
    else:
        score -= 10
    
    # Volume (max 10 points)
    if metrics['volume_ratio'] > 1.5:
        score += 10
    elif metrics['volume_ratio'] < 0.7:
        score -= 5
    
    # Consecutive days (max 10 points)
    if metrics['consecutive_up'] >= 3:
        score += 10
    elif metrics['consecutive_down'] >= 3:
        score -= 10
    
    return max(0, min(100, score))  # Clamp between 0-100

def determine_trend_strength(metrics):
    """Determine trend strength"""
    score = metrics['momentum_score']
    
    if score >= 75:
        return "🔥 VERY_STRONG"
    elif score >= 60:
        return "💪 STRONG"
    elif score >= 50:
        return "✅ MODERATE"
    elif score >= 40:
        return "⚠️ WEAK"
    else:
        return "❌ BEARISH"

def format_stock_report(stock):
    """Format individual stock report"""
    signals_str = ", ".join(stock['signal']) if stock['signal'] else "HOLD"
    
    report = f"""
*{stock['symbol']}* - ${stock['price']} ({stock['change_1d']:+.2f}%)
{stock['trend_strength']} | Score: {stock['momentum_score']}/100

📈 *Performance:*
  5D: {stock['change_5d']:+.2f}% | 10D: {stock['change_10d']:+.2f}% | 20D: {stock['change_20d']:+.2f}%

📊 *Indicators:*
  RSI: {stock['rsi']:.1f} | MACD: {stock['macd_hist']:+.4f}
  EMA20: ${stock['ema_fast']} | EMA50: ${stock['ema_slow']}

📢 *Volume:*
  Current: {stock['volume']:,} ({stock['volume_ratio']:.1f}x avg)
  Avg: {stock['avg_volume']:,}

🎯 *Signals:* {signals_str}
"""
    return report

if __name__ == "__main__":
    # Check if tickers.txt exists
    if not os.path.exists("tickers.txt"):
        print("Error: tickers.txt not found!")
        exit(1)
    
    with open("tickers.txt", "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    
    print(f"\n{'='*60}")
    print(f"🔍 MOMENTUM SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    all_stocks = []
    
    for ticker in tickers:
        print(f"Scanning {ticker}...", end=" ")
        metrics = get_momentum_analysis(ticker)
        
        if metrics and metrics['avg_volume'] >= MIN_VOLUME:
            all_stocks.append(metrics)
            print(f"✓ Score: {metrics['momentum_score']}")
        else:
            print("✗ Skipped (low volume or error)")
    
    if not all_stocks:
        print("\nNo stocks met the criteria.")
        exit(0)
    
    # Sort by momentum score
    all_stocks.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    # Categorize stocks
    strong_momentum = [s for s in all_stocks if s['momentum_score'] >= 60]
    crossover_signals = [s for s in all_stocks if any('CROSS' in sig for sig in s['signal'])]
    high_volume_stocks = [s for s in all_stocks if s['volume_ratio'] >= 2.0]
    
    # Print console summary
    print(f"\n{'='*60}")
    print(f"📊 SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Total Analyzed: {len(all_stocks)}")
    print(f"Strong Momentum (≥60): {len(strong_momentum)}")
    print(f"Crossover Signals: {len(crossover_signals)}")
    print(f"High Volume (≥2x): {len(high_volume_stocks)}")
    print(f"{'='*60}\n")
    
    # Prepare Telegram report
    telegram_sections = []
    
    if strong_momentum:
        telegram_sections.append("🔥 *TOP MOMENTUM STOCKS*\n")
        for stock in strong_momentum[:5]:  # Top 5
            telegram_sections.append(format_stock_report(stock))
    
    if crossover_signals:
        telegram_sections.append("\n⚡ *CROSSOVER SIGNALS*\n")
        for stock in crossover_signals[:3]:  # Top 3
            telegram_sections.append(format_stock_report(stock))
    
    if high_volume_stocks:
        telegram_sections.append("\n📢 *HIGH VOLUME ALERTS*\n")
        for stock in high_volume_stocks[:3]:  # Top 3
            telegram_sections.append(format_stock_report(stock))
    
    if telegram_sections:
        header = f"📈 *MOMENTUM SCAN REPORT*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Scanned: {len(all_stocks)} stocks\n"
        header += "=" * 30
        
        final_report = header + "\n" + "\n".join(telegram_sections)
        send_telegram_msg(final_report)
        print("✓ Report sent to Telegram")
    else:
        print("No significant momentum or signals detected.")
    
    # Save detailed CSV report
    df = pd.DataFrame(all_stocks)
    df = df.sort_values('momentum_score', ascending=False)
    output_file = f"momentum_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Detailed report saved to {output_file}")
