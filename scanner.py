"""
Production-Grade Buy-the-Dip Scanner
Identifies oversold stocks with recovery potential using statistical methods
"""

import yfinance as yf
import requests
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz
import pandas as pd
import numpy as np
from dataclasses import dataclass
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
try:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logger.warning(f"Sentiment analyzer init failed: {e}")
    sia = None

# === CONFIGURATION ===
@dataclass
class Config:
    """Scanner configuration"""
    # API Keys
    FINNHUB_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Scanning Parameters
    RSI_OVERSOLD: float = 35  # More aggressive oversold threshold
    RSI_PERIOD: int = 14
    MIN_PRICE: float = 5.0  # Avoid penny stocks
    MAX_PRICE: float = 1000.0
    MIN_AVG_VOLUME: float = 500000  # Liquidity filter
    
    # Scoring Weights (normalized to sum to 1.0)
    WEIGHT_TECHNICAL: float = 0.35  # RSI, Price action
    WEIGHT_VOLUME: float = 0.25     # Volume surge
    WEIGHT_FUNDAMENTAL: float = 0.25  # Analyst targets, financials
    WEIGHT_SENTIMENT: float = 0.15   # News sentiment
    
    # Risk Parameters
    MAX_DRAWDOWN_THRESHOLD: float = -0.15  # -15% from recent high
    MIN_SUPPORT_DISTANCE: float = 0.02     # 2% above support
    
    # Market Context
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16
    MARKET_CLOSE_MINUTE: int = 0
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.TELEGRAM_TOKEN or not self.TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials missing - notifications disabled")
        return True

config = Config()
config.validate()


# === MARKET CONTEXT ===
class MarketContext:
    """Handles market hours and context"""
    
    @staticmethod
    def get_market_session() -> Tuple[float, str]:
        """
        Returns (progress_pct, session_name)
        progress_pct: 0.0 to 1.0 representing market day completion
        session_name: 'premarket', 'open', 'midday', 'close', 'afterhours'
        """
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        
        market_open = now.replace(
            hour=config.MARKET_OPEN_HOUR,
            minute=config.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = now.replace(
            hour=config.MARKET_CLOSE_HOUR,
            minute=config.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )
        
        if now < market_open:
            return 0.0, "premarket"
        elif now > market_close:
            return 1.0, "afterhours"
        
        elapsed = (now - market_open).total_seconds()
        total = (market_close - market_open).total_seconds()
        progress = elapsed / total
        
        if progress < 0.25:
            return progress, "open"
        elif progress < 0.75:
            return progress, "midday"
        else:
            return progress, "close"
    
    @staticmethod
    def is_trading_day() -> bool:
        """Check if today is a trading day"""
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        return now.weekday() < 5  # Monday = 0, Friday = 4


# === DATA FETCHING ===
class DataFetcher:
    """Handles all external data fetching with error handling"""
    
    @staticmethod
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock price data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            # Get more data for better calculations
            df = stock.history(period="6mo", interval="1d")
            
            if df.empty:
                logger.debug(f"No data for {symbol}")
                return None
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_stock_info(symbol: str) -> Dict:
        """Fetch stock info with error handling"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return info if info else {}
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    @staticmethod
    def get_news_sentiment(symbol: str) -> Tuple[float, str, int]:
        """
        Fetch news and calculate sentiment
        Returns: (sentiment_score, headline, news_count)
        """
        if not config.FINNHUB_KEY or not sia:
            return 0.5, "No news data available", 0
        
        # Convert ticker format for Finnhub
        clean_symbol = symbol.replace("-", ".")
        
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': clean_symbol,
                'from': start,
                'to': end,
                'token': config.FINNHUB_KEY
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            news = response.json()
            
            if not news or not isinstance(news, list):
                return 0.5, "No recent news", 0
            
            # Analyze multiple headlines for better accuracy
            sentiments = []
            headlines = []
            
            for article in news[:5]:  # Top 5 articles
                headline = article.get('headline', '')
                if headline:
                    score = sia.polarity_scores(headline)['compound']
                    sentiments.append(score)
                    headlines.append(headline)
            
            if not sentiments:
                return 0.5, "No valid headlines", 0
            
            # Average sentiment, normalized to 0-1
            avg_sentiment = (np.mean(sentiments) + 1) / 2
            top_headline = headlines[0] if headlines else "News available"
            
            return avg_sentiment, top_headline, len(news)
            
        except Exception as e:
            logger.error(f"Sentiment error for {symbol}: {e}")
            return 0.5, "Sentiment fetch failed", 0


# === TECHNICAL ANALYSIS ===
class TechnicalAnalysis:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI with proper error handling"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate support and resistance levels using recent price action
        Returns: (support_level, resistance_level)
        """
        try:
            recent = df.tail(20)  # Last 20 days
            
            # Support: Recent low
            support = recent['Low'].min()
            
            # Resistance: Recent high
            resistance = recent['High'].max()
            
            return float(support), float(resistance)
        except Exception as e:
            logger.error(f"Support/Resistance error: {e}")
            return 0.0, 0.0
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns
        Returns: Dict with volume metrics
        """
        try:
            recent_20d = df.tail(20)
            recent_5d = df.tail(5)
            
            avg_vol_20d = recent_20d['Volume'].mean()
            avg_vol_5d = recent_5d['Volume'].mean()
            current_vol = df['Volume'].iloc[-1]
            
            # Volume surge detection
            vol_surge = current_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0
            vol_trend = avg_vol_5d / avg_vol_20d if avg_vol_20d > 0 else 1.0
            
            return {
                'current': float(current_vol),
                'avg_20d': float(avg_vol_20d),
                'surge_ratio': float(vol_surge),
                'trend_ratio': float(vol_trend)
            }
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return {
                'current': 0,
                'avg_20d': 0,
                'surge_ratio': 1.0,
                'trend_ratio': 1.0
            }
    
    @staticmethod
    def calculate_price_action(df: pd.DataFrame) -> Dict:
        """
        Calculate price action metrics
        Returns: Dict with price metrics
        """
        try:
            recent_20d = df.tail(20)
            
            current_price = df['Close'].iloc[-1]
            high_20d = recent_20d['High'].max()
            low_20d = recent_20d['Low'].min()
            
            # Distance from high (drawdown)
            drawdown = (current_price - high_20d) / high_20d
            
            # Distance from low (recovery potential)
            from_low = (current_price - low_20d) / low_20d
            
            # Price volatility (standard deviation)
            volatility = recent_20d['Close'].pct_change().std()
            
            return {
                'current': float(current_price),
                'high_20d': float(high_20d),
                'low_20d': float(low_20d),
                'drawdown': float(drawdown),
                'from_low': float(from_low),
                'volatility': float(volatility)
            }
        except Exception as e:
            logger.error(f"Price action error: {e}")
            return {
                'current': 0,
                'high_20d': 0,
                'low_20d': 0,
                'drawdown': 0,
                'from_low': 0,
                'volatility': 0
            }


# === FUNDAMENTAL ANALYSIS ===
class FundamentalAnalysis:
    """Fundamental metrics analysis"""
    
    @staticmethod
    def calculate_upside_potential(info: Dict, current_price: float) -> Dict:
        """
        Calculate upside potential from analyst targets
        Returns: Dict with target metrics
        """
        try:
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            
            if not target_mean or current_price <= 0:
                return {
                    'mean_upside': 0.0,
                    'high_upside': 0.0,
                    'target_mean': 0.0,
                    'analyst_count': 0
                }
            
            mean_upside = (target_mean - current_price) / current_price
            high_upside = ((target_high - current_price) / current_price) if target_high else mean_upside
            
            analyst_count = info.get('numberOfAnalystOpinions', 0)
            
            return {
                'mean_upside': float(mean_upside),
                'high_upside': float(high_upside),
                'target_mean': float(target_mean),
                'analyst_count': int(analyst_count)
            }
        except Exception as e:
            logger.error(f"Upside calculation error: {e}")
            return {
                'mean_upside': 0.0,
                'high_upside': 0.0,
                'target_mean': 0.0,
                'analyst_count': 0
            }
    
    @staticmethod
    def get_quality_metrics(info: Dict) -> Dict:
        """
        Extract quality metrics from stock info
        Returns: Dict with quality indicators
        """
        try:
            return {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'recommendation': info.get('recommendationKey', 'none'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Quality metrics error: {e}")
            return {}


# === SCORING ENGINE ===
@dataclass
class StockScore:
    """Container for stock scoring data"""
    symbol: str
    total_score: float
    
    # Component scores (0-100 each)
    technical_score: float
    volume_score: float
    fundamental_score: float
    sentiment_score: float
    
    # Raw metrics
    rsi: float
    price: float
    drawdown: float
    volume_surge: float
    analyst_upside: float
    sentiment: float
    
    # Additional data
    headline: str
    analyst_count: int
    market_cap: float
    sector: str
    
    # Risk flags
    risk_flags: List[str]


class ScoringEngine:
    """Statistical scoring engine for buy-the-dip signals"""
    
    @staticmethod
    def score_technical(rsi: float, price_action: Dict) -> Tuple[float, List[str]]:
        """
        Score technical indicators
        Returns: (score 0-100, risk_flags)
        """
        flags = []
        
        # RSI score (inverse - lower is better for oversold)
        rsi_score = max(0, 100 - rsi) if rsi < 50 else 0
        
        # Drawdown score (negative is good for dip buying)
        drawdown = price_action['drawdown']
        if drawdown < config.MAX_DRAWDOWN_THRESHOLD:
            drawdown_score = 0  # Too much drawdown
            flags.append(f"Heavy drawdown: {drawdown:.1%}")
        else:
            # -5% to 0% drawdown is ideal
            drawdown_score = abs(drawdown) * 100 * 2  # Scale to 0-100
            drawdown_score = min(100, drawdown_score)
        
        # Combine
        technical_score = (rsi_score * 0.6) + (drawdown_score * 0.4)
        
        return technical_score, flags
    
    @staticmethod
    def score_volume(volume_metrics: Dict, session: str) -> Tuple[float, List[str]]:
        """
        Score volume patterns
        Returns: (score 0-100, risk_flags)
        
        Scoring Logic:
        - 0.0-0.5x volume = 0 points (very low/concerning)
        - 0.5-1.0x volume = 0-25 points (below average)
        - 1.0-1.5x volume = 25-50 points (average)
        - 1.5-2.5x volume = 50-85 points (good)
        - 2.5x+ volume = 85-100 points (excellent surge)
        """
        flags = []
        
        surge = volume_metrics['surge_ratio']
        trend = volume_metrics['trend_ratio']
        
        # Volume surge scoring with realistic thresholds
        if surge < 0.5:
            surge_score = 0
            flags.append(f"Very low volume: {surge:.2f}x")
        elif surge < 1.0:
            # 0.5-1.0x = 0-25 points (below average)
            surge_score = ((surge - 0.5) / 0.5) * 25
            flags.append(f"Below avg volume: {surge:.2f}x")
        elif surge < 1.5:
            # 1.0-1.5x = 25-50 points (average)
            surge_score = 25 + ((surge - 1.0) / 0.5) * 25
        elif surge < 2.5:
            # 1.5-2.5x = 50-85 points (good)
            surge_score = 50 + ((surge - 1.5) / 1.0) * 35
        else:
            # 2.5x+ = 85-100 points (excellent)
            surge_score = min(100, 85 + ((surge - 2.5) / 1.5) * 15)
        
        # Score volume trend (5-day vs 20-day average)
        if trend < 0.8:
            trend_score = 0
            flags.append(f"Declining vol trend: {trend:.2f}x")
        elif trend < 1.0:
            trend_score = ((trend - 0.8) / 0.2) * 50
        else:
            trend_score = min(100, 50 + (trend - 1.0) * 50)
        
        # Combine (surge is more important for dip buying)
        volume_score = (surge_score * 0.8) + (trend_score * 0.2)
        
        return volume_score, flags
    
    @staticmethod
    def score_fundamental(upside_metrics: Dict, quality_metrics: Dict) -> Tuple[float, List[str]]:
        """
        Score fundamental indicators
        Returns: (score 0-100, risk_flags)
        """
        flags = []
        
        mean_upside = upside_metrics['mean_upside']
        analyst_count = upside_metrics['analyst_count']
        
        # Upside score
        if mean_upside < 0:
            upside_score = 0
            flags.append("Negative analyst target")
        elif mean_upside > 0.5:  # 50%+ upside
            upside_score = 100
        else:
            upside_score = mean_upside * 200  # Scale to 0-100
        
        # Analyst coverage score
        if analyst_count < 3:
            coverage_score = 30
            flags.append(f"Low analyst coverage: {analyst_count}")
        else:
            coverage_score = min(100, analyst_count * 10)
        
        # Quality adjustment
        recommendation = quality_metrics.get('recommendation', 'none')
        if recommendation in ['strong_buy', 'buy']:
            quality_multiplier = 1.2
        elif recommendation in ['strong_sell', 'sell']:
            quality_multiplier = 0.5
            flags.append(f"Sell rating: {recommendation}")
        else:
            quality_multiplier = 1.0
        
        # Combine
        fundamental_score = ((upside_score * 0.7) + (coverage_score * 0.3)) * quality_multiplier
        fundamental_score = min(100, fundamental_score)
        
        return fundamental_score, flags
    
    @staticmethod
    def score_sentiment(sentiment: float, news_count: int) -> Tuple[float, List[str]]:
        """
        Score news sentiment
        Returns: (score 0-100, risk_flags)
        """
        flags = []
        
        # Convert 0-1 sentiment to 0-100 score
        # 0.5 is neutral, <0.4 is negative, >0.6 is positive
        if sentiment < 0.3:
            sentiment_score = 0
            flags.append("Very negative news")
        elif sentiment < 0.4:
            sentiment_score = 25
            flags.append("Negative news")
        elif sentiment > 0.6:
            sentiment_score = 100
        else:
            # Neutral is okay for dip buying
            sentiment_score = 50
        
        # Adjust for news volume
        if news_count == 0:
            sentiment_score = 50  # Neutral if no news
        elif news_count < 3:
            flags.append("Limited news coverage")
        
        return sentiment_score, flags
    
    @classmethod
    def calculate_composite_score(
        cls,
        symbol: str,
        df: pd.DataFrame,
        info: Dict,
        session: str
    ) -> Optional[StockScore]:
        """
        Calculate composite score for a stock
        Returns: StockScore object or None if not viable
        """
        try:
            # Calculate all metrics
            rsi = TechnicalAnalysis.calculate_rsi(df['Close'], config.RSI_PERIOD)
            price_action = TechnicalAnalysis.calculate_price_action(df)
            volume_metrics = TechnicalAnalysis.calculate_volume_profile(df)
            upside_metrics = FundamentalAnalysis.calculate_upside_potential(info, price_action['current'])
            quality_metrics = FundamentalAnalysis.get_quality_metrics(info)
            
            # Get sentiment
            sentiment, headline, news_count = DataFetcher.get_news_sentiment(symbol)
            
            # Initial filters
            if price_action['current'] < config.MIN_PRICE:
                return None
            if price_action['current'] > config.MAX_PRICE:
                return None
            if volume_metrics['avg_20d'] < config.MIN_AVG_VOLUME:
                return None
            if rsi > config.RSI_OVERSOLD:
                return None  # Not oversold enough
            
            # Calculate component scores
            technical_score, tech_flags = cls.score_technical(rsi, price_action)
            volume_score, vol_flags = cls.score_volume(volume_metrics, session)
            fundamental_score, fund_flags = cls.score_fundamental(upside_metrics, quality_metrics)
            sentiment_score, sent_flags = cls.score_sentiment(sentiment, news_count)
            
            # Combine risk flags
            risk_flags = tech_flags + vol_flags + fund_flags + sent_flags
            
            # Calculate weighted total score
            total_score = (
                technical_score * config.WEIGHT_TECHNICAL +
                volume_score * config.WEIGHT_VOLUME +
                fundamental_score * config.WEIGHT_FUNDAMENTAL +
                sentiment_score * config.WEIGHT_SENTIMENT
            )
            
            # CRITICAL: Apply volume penalty for extremely low volume
            # Low volume stocks are hard to enter/exit and unreliable for dip buying
            vol_surge = volume_metrics['surge_ratio']
            if vol_surge < 0.5:
                total_score *= 0.3  # Reduce score by 70% if volume is very low
                risk_flags.append("CRITICAL: Very low volume")
            elif vol_surge < 0.8:
                total_score *= 0.6  # Reduce score by 40% if volume is low
            
            return StockScore(
                symbol=symbol,
                total_score=total_score,
                technical_score=technical_score,
                volume_score=volume_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                rsi=rsi,
                price=price_action['current'],
                drawdown=price_action['drawdown'],
                volume_surge=volume_metrics['surge_ratio'],
                analyst_upside=upside_metrics['mean_upside'],
                sentiment=sentiment,
                headline=headline,
                analyst_count=upside_metrics['analyst_count'],
                market_cap=quality_metrics.get('market_cap', 0),
                sector=quality_metrics.get('sector', 'Unknown'),
                risk_flags=risk_flags
            )
            
        except Exception as e:
            logger.error(f"Scoring error for {symbol}: {e}")
            return None


# === NOTIFICATION ===
class Notifier:
    """Handles notifications to Telegram"""
    
    @staticmethod
    def format_message(results: List[StockScore], session: str, progress: float) -> str:
        """Format results as Telegram message"""
        if not results:
            return "📊 *Buy-the-Dip Scanner*\n\nNo opportunities found in current scan."
        
        # Session emoji
        session_emoji = {
            'premarket': '🌅',
            'open': '🔔',
            'midday': '☀️',
            'close': '🌆',
            'afterhours': '🌙'
        }
        
        msg = f"{session_emoji.get(session, '📊')} *Buy-the-Dip Scanner*\n"
        msg += f"Session: {session.title()} ({progress*100:.0f}% complete)\n"
        msg += f"Found {len(results)} opportunities\n\n"
        
        for i, stock in enumerate(results, 1):
            # Sentiment emoji
            if stock.sentiment > 0.6:
                sent_emoji = "🟢"
            elif stock.sentiment < 0.4:
                sent_emoji = "🔴"
            else:
                sent_emoji = "⚪"
            
            msg += f"*{i}. {stock.symbol}* — Score: {stock.total_score:.0f}/100\n"
            msg += f"💰 ${stock.price:.2f} | RSI: {stock.rsi:.0f} {sent_emoji}\n"
            msg += f"📊 Vol: {stock.volume_surge:.1f}x | Drawdown: {stock.drawdown:.1%}\n"
            msg += f"🎯 Analyst Upside: {stock.analyst_upside:.1%} ({stock.analyst_count} analysts)\n"
            
            # Component scores as progress bars
            msg += f"📈 T:{stock.technical_score:.0f} V:{stock.volume_score:.0f} "
            msg += f"F:{stock.fundamental_score:.0f} S:{stock.sentiment_score:.0f}\n"
            
            if stock.risk_flags:
                msg += f"⚠️ {', '.join(stock.risk_flags[:2])}\n"
            
            msg += f"📰 `{stock.headline[:55]}...`\n\n"
        
        msg += f"_Scanned at {datetime.now().strftime('%H:%M ET')}_"
        return msg
    
    @staticmethod
    def send_telegram(message: str) -> bool:
        """Send message to Telegram"""
        if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.info("Telegram not configured - message not sent")
            logger.info(f"\n{message}")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


# === MAIN SCANNER ===
class DipScanner:
    """Main scanning orchestration"""
    
    @staticmethod
    def load_tickers(filename: str = "tickers.txt") -> List[str]:
        """Load ticker symbols from file"""
        try:
            with open(filename, "r") as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers from {filename}")
            return tickers
        except FileNotFoundError:
            logger.error(f"Ticker file {filename} not found")
            return []
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []
    
    @staticmethod
    def scan_all(tickers: List[str], top_n: int = 5) -> List[StockScore]:
        """
        Scan all tickers and return top opportunities
        """
        if not MarketContext.is_trading_day():
            logger.warning("Not a trading day")
        
        progress, session = MarketContext.get_market_session()
        logger.info(f"Market session: {session} ({progress*100:.0f}% complete)")
        
        results = []
        
        for i, symbol in enumerate(tickers, 1):
            try:
                logger.info(f"[{i}/{len(tickers)}] Scanning {symbol}...")
                
                # Fetch data
                df = DataFetcher.get_stock_data(symbol)
                if df is None:
                    continue
                
                info = DataFetcher.get_stock_info(symbol)
                
                # Score the stock
                score = ScoringEngine.calculate_composite_score(symbol, df, info, session)
                
                if score and score.total_score > 30:  # Minimum viable score
                    results.append(score)
                    logger.info(f"  ✓ {symbol}: Score {score.total_score:.0f}")
                
                # Rate limiting
                time.sleep(1.5)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"\nScan complete: {len(results)} opportunities found")
        
        return results[:top_n]
    
    @staticmethod
    def run(ticker_file: str = "tickers.txt", top_n: int = 5) -> None:
        """
        Main execution function
        """
        logger.info("=" * 60)
        logger.info("Buy-the-Dip Scanner - Production Version")
        logger.info("=" * 60)
        
        # Load tickers
        tickers = DipScanner.load_tickers(ticker_file)
        if not tickers:
            logger.error("No tickers to scan")
            return
        
        # Scan
        results = DipScanner.scan_all(tickers, top_n)
        
        # Notify
        if results:
            progress, session = MarketContext.get_market_session()
            message = Notifier.format_message(results, session, progress)
            Notifier.send_telegram(message)
        else:
            logger.info("No opportunities found")
        
        logger.info("=" * 60)
        logger.info("Scan complete")
        logger.info("=" * 60)


# === ENTRY POINT ===
if __name__ == "__main__":
    DipScanner.run()
