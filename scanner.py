"""
Enhanced Buy-the-Dip Scanner with Decision Support
Analyzes signals and provides complete trade recommendations
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
    RSI_OVERSOLD: float = 35
    RSI_PERIOD: int = 14
    MIN_PRICE: float = 5.0
    MAX_PRICE: float = 1000.0
    MIN_AVG_VOLUME: float = 500000
    
    # Scoring Weights
    WEIGHT_TECHNICAL: float = 0.35
    WEIGHT_VOLUME: float = 0.25
    WEIGHT_FUNDAMENTAL: float = 0.25
    WEIGHT_SENTIMENT: float = 0.15
    
    # Risk Parameters
    MAX_DRAWDOWN_THRESHOLD: float = -0.15
    DEFAULT_STOP_LOSS_PCT: float = 0.03  # 3% stop loss
    DEFAULT_TARGET_PCT: float = 0.07     # 7% target
    RISK_PER_TRADE_PCT: float = 0.01     # 1% account risk
    
    # Market Context
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16
    MARKET_CLOSE_MINUTE: int = 0

config = Config()


# === MARKET CONTEXT ===
class MarketContext:
    """Enhanced market context analysis"""
    
    @staticmethod
    def get_market_session() -> Tuple[float, str]:
        """Returns (progress_pct, session_name)"""
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
    def get_market_performance() -> Dict:
        """Get SPY and QQQ performance for context"""
        try:
            spy = yf.Ticker("SPY")
            qqq = yf.Ticker("QQQ")
            
            spy_data = spy.history(period="5d")
            qqq_data = qqq.history(period="5d")
            
            if len(spy_data) < 2 or len(qqq_data) < 2:
                return {"spy_change": 0, "qqq_change": 0, "market_trend": "unknown"}
            
            spy_change = ((spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[-2]) / 
                         spy_data['Close'].iloc[-2])
            qqq_change = ((qqq_data['Close'].iloc[-1] - qqq_data['Close'].iloc[-2]) / 
                         qqq_data['Close'].iloc[-2])
            
            # Determine market trend
            if spy_change < -0.015:
                trend = "bearish"
            elif spy_change > 0.015:
                trend = "bullish"
            else:
                trend = "neutral"
            
            return {
                "spy_change": float(spy_change),
                "qqq_change": float(qqq_change),
                "market_trend": trend
            }
        except Exception as e:
            logger.error(f"Market performance error: {e}")
            return {"spy_change": 0, "qqq_change": 0, "market_trend": "unknown"}
    
    @staticmethod
    def get_vix() -> float:
        """Get VIX (fear index) level"""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return 20.0  # Default neutral value


# === DATA FETCHING ===
class DataFetcher:
    """Enhanced data fetching with news categorization"""
    
    @staticmethod
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock price data"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="6mo", interval="1d")
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_stock_info(symbol: str) -> Dict:
        """Fetch stock info"""
        try:
            stock = yf.Ticker(symbol)
            return stock.info if stock.info else {}
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    @staticmethod
    def get_news_analysis(symbol: str) -> Dict:
        """
        Enhanced news analysis with categorization
        Returns: Dict with sentiment, headlines, and dip reason
        """
        if not config.FINNHUB_KEY or not sia:
            return {
                "sentiment": 0.5,
                "top_headline": "No news data available",
                "news_count": 0,
                "dip_reason": "unknown",
                "reason_confidence": "low"
            }
        
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
                return {
                    "sentiment": 0.5,
                    "top_headline": "No recent news",
                    "news_count": 0,
                    "dip_reason": "unknown",
                    "reason_confidence": "low"
                }
            
            # Analyze headlines
            sentiments = []
            headlines = []
            
            for article in news[:5]:
                headline = article.get('headline', '')
                if headline:
                    score = sia.polarity_scores(headline)['compound']
                    sentiments.append(score)
                    headlines.append(headline)
            
            if not sentiments:
                return {
                    "sentiment": 0.5,
                    "top_headline": "No valid headlines",
                    "news_count": 0,
                    "dip_reason": "unknown",
                    "reason_confidence": "low"
                }
            
            avg_sentiment = (np.mean(sentiments) + 1) / 2
            top_headline = headlines[0] if headlines else "News available"
            
            # Categorize reason for dip
            dip_reason, confidence = DataFetcher._categorize_dip_reason(headlines)
            
            return {
                "sentiment": float(avg_sentiment),
                "top_headline": top_headline,
                "news_count": len(news),
                "dip_reason": dip_reason,
                "reason_confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"News analysis error for {symbol}: {e}")
            return {
                "sentiment": 0.5,
                "top_headline": "Sentiment fetch failed",
                "news_count": 0,
                "dip_reason": "unknown",
                "reason_confidence": "low"
            }
    
    @staticmethod
    def _categorize_dip_reason(headlines: List[str]) -> Tuple[str, str]:
        """
        Categorize why stock is down based on headlines
        Returns: (reason, confidence)
        """
        all_text = " ".join(headlines).lower()
        
        # Negative company-specific events
        if any(word in all_text for word in ['earnings miss', 'guidance cut', 'downgrade', 'lawsuit', 'investigation']):
            return "company_specific_negative", "high"
        
        # Positive news but still down (overreaction?)
        if any(word in all_text for word in ['beats expectations', 'upgrade', 'partnership', 'deal']):
            return "profit_taking_after_news", "medium"
        
        # Market-wide events
        if any(word in all_text for word in ['market', 'fed', 'rate', 'inflation', 'selloff']):
            return "market_wide_selloff", "high"
        
        # Sector issues
        if any(word in all_text for word in ['sector', 'industry', 'peers']):
            return "sector_weakness", "medium"
        
        # No specific reason found
        return "technical_correction", "low"


# === TECHNICAL ANALYSIS (Same as before) ===
class TechnicalAnalysis:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame) -> Dict:
        """
        Find key support and resistance levels
        Returns: Dict with support/resistance and distance
        """
        try:
            recent_60d = df.tail(60)
            current_price = df['Close'].iloc[-1]
            
            # Find local lows (support) and highs (resistance)
            lows = recent_60d['Low'].nsmallest(5).values
            highs = recent_60d['High'].nlargest(5).values
            
            # Nearest support below current price
            support_candidates = lows[lows < current_price]
            support = support_candidates.max() if len(support_candidates) > 0 else lows.min()
            
            # Nearest resistance above current price
            resistance_candidates = highs[highs > current_price]
            resistance = resistance_candidates.min() if len(resistance_candidates) > 0 else highs.max()
            
            # Calculate distances
            support_distance = (current_price - support) / current_price
            resistance_distance = (resistance - current_price) / current_price
            
            return {
                "support": float(support),
                "resistance": float(resistance),
                "current": float(current_price),
                "support_distance_pct": float(support_distance),
                "resistance_distance_pct": float(resistance_distance),
                "risk_reward_ratio": float(resistance_distance / support_distance) if support_distance > 0 else 0
            }
        except Exception as e:
            logger.error(f"Support/resistance error: {e}")
            return {}
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            recent_20d = df.tail(20)
            recent_5d = df.tail(5)
            
            avg_vol_20d = recent_20d['Volume'].mean()
            avg_vol_5d = recent_5d['Volume'].mean()
            current_vol = df['Volume'].iloc[-1]
            
            vol_surge = current_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0
            vol_trend = avg_vol_5d / avg_vol_20d if avg_vol_20d > 0 else 1.0
            
            return {
                'current': float(current_vol),
                'avg_20d': float(avg_vol_20d),
                'surge_ratio': float(vol_surge),
                'trend_ratio': float(vol_trend)
            }
        except:
            return {'current': 0, 'avg_20d': 0, 'surge_ratio': 1.0, 'trend_ratio': 1.0}
    
    @staticmethod
    def calculate_price_action(df: pd.DataFrame) -> Dict:
        """Calculate price action metrics"""
        try:
            recent_20d = df.tail(20)
            
            current_price = df['Close'].iloc[-1]
            high_20d = recent_20d['High'].max()
            low_20d = recent_20d['Low'].min()
            
            drawdown = (current_price - high_20d) / high_20d
            from_low = (current_price - low_20d) / low_20d
            volatility = recent_20d['Close'].pct_change().std()
            
            return {
                'current': float(current_price),
                'high_20d': float(high_20d),
                'low_20d': float(low_20d),
                'drawdown': float(drawdown),
                'from_low': float(from_low),
                'volatility': float(volatility)
            }
        except:
            return {'current': 0, 'high_20d': 0, 'low_20d': 0, 'drawdown': 0, 'from_low': 0, 'volatility': 0}


# === FUNDAMENTAL ANALYSIS ===
class FundamentalAnalysis:
    """Fundamental metrics analysis"""
    
    @staticmethod
    def calculate_upside_potential(info: Dict, current_price: float) -> Dict:
        """Calculate upside from analyst targets"""
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
            
            return {
                'mean_upside': float(mean_upside),
                'high_upside': float(high_upside),
                'target_mean': float(target_mean),
                'analyst_count': int(info.get('numberOfAnalystOpinions', 0))
            }
        except:
            return {'mean_upside': 0.0, 'high_upside': 0.0, 'target_mean': 0.0, 'analyst_count': 0}
    
    @staticmethod
    def get_quality_metrics(info: Dict) -> Dict:
        """Extract quality indicators"""
        try:
            return {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'recommendation': info.get('recommendationKey', 'none'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except:
            return {}


# === SCORING ENGINE (Enhanced) ===
@dataclass
class StockScore:
    """Container for stock scoring data"""
    symbol: str
    total_score: float
    
    # Component scores
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
    
    # NEW: Trade recommendation data
    support_level: float
    resistance_level: float
    risk_reward_ratio: float
    dip_reason: str
    reason_confidence: str


class ScoringEngine:
    """Enhanced scoring with trade recommendations"""
    
    @staticmethod
    def score_technical(rsi: float, price_action: Dict) -> Tuple[float, List[str]]:
        """Score technical indicators"""
        flags = []
        
        rsi_score = max(0, 100 - rsi) if rsi < 50 else 0
        
        drawdown = price_action['drawdown']
        if drawdown < config.MAX_DRAWDOWN_THRESHOLD:
            drawdown_score = 0
            flags.append(f"Heavy drawdown: {drawdown:.1%}")
        else:
            drawdown_score = abs(drawdown) * 100 * 2
            drawdown_score = min(100, drawdown_score)
        
        technical_score = (rsi_score * 0.6) + (drawdown_score * 0.4)
        return technical_score, flags
    
    @staticmethod
    def score_volume(volume_metrics: Dict) -> Tuple[float, List[str]]:
        """Score volume with realistic thresholds"""
        flags = []
        surge = volume_metrics['surge_ratio']
        trend = volume_metrics['trend_ratio']
        
        if surge < 0.5:
            surge_score = 0
            flags.append(f"Very low volume: {surge:.2f}x")
        elif surge < 1.0:
            surge_score = ((surge - 0.5) / 0.5) * 25
            flags.append(f"Below avg volume: {surge:.2f}x")
        elif surge < 1.5:
            surge_score = 25 + ((surge - 1.0) / 0.5) * 25
        elif surge < 2.5:
            surge_score = 50 + ((surge - 1.5) / 1.0) * 35
        else:
            surge_score = min(100, 85 + ((surge - 2.5) / 1.5) * 15)
        
        if trend < 0.8:
            trend_score = 0
            flags.append(f"Declining vol trend: {trend:.2f}x")
        elif trend < 1.0:
            trend_score = ((trend - 0.8) / 0.2) * 50
        else:
            trend_score = min(100, 50 + (trend - 1.0) * 50)
        
        volume_score = (surge_score * 0.8) + (trend_score * 0.2)
        return volume_score, flags
    
    @staticmethod
    def score_fundamental(upside_metrics: Dict, quality_metrics: Dict) -> Tuple[float, List[str]]:
        """Score fundamental indicators"""
        flags = []
        
        mean_upside = upside_metrics['mean_upside']
        analyst_count = upside_metrics['analyst_count']
        
        if mean_upside < 0:
            upside_score = 0
            flags.append("Negative analyst target")
        elif mean_upside > 0.5:
            upside_score = 100
        else:
            upside_score = mean_upside * 200
        
        if analyst_count < 3:
            coverage_score = 30
            flags.append(f"Low analyst coverage: {analyst_count}")
        else:
            coverage_score = min(100, analyst_count * 10)
        
        recommendation = quality_metrics.get('recommendation', 'none')
        if recommendation in ['strong_buy', 'buy']:
            quality_multiplier = 1.2
        elif recommendation in ['strong_sell', 'sell']:
            quality_multiplier = 0.5
            flags.append(f"Sell rating: {recommendation}")
        else:
            quality_multiplier = 1.0
        
        fundamental_score = ((upside_score * 0.7) + (coverage_score * 0.3)) * quality_multiplier
        fundamental_score = min(100, fundamental_score)
        
        return fundamental_score, flags
    
    @staticmethod
    def score_sentiment(sentiment: float, news_count: int) -> Tuple[float, List[str]]:
        """Score news sentiment"""
        flags = []
        
        if sentiment < 0.3:
            sentiment_score = 0
            flags.append("Very negative news")
        elif sentiment < 0.4:
            sentiment_score = 25
            flags.append("Negative news")
        elif sentiment > 0.6:
            sentiment_score = 100
        else:
            sentiment_score = 50
        
        if news_count == 0:
            sentiment_score = 50
        elif news_count < 3:
            flags.append("Limited news coverage")
        
        return sentiment_score, flags
    
    @classmethod
    def calculate_composite_score(
        cls,
        symbol: str,
        df: pd.DataFrame,
        info: Dict
    ) -> Optional[StockScore]:
        """Calculate composite score with trade recommendation data"""
        try:
            # Calculate all metrics
            rsi = TechnicalAnalysis.calculate_rsi(df['Close'], config.RSI_PERIOD)
            price_action = TechnicalAnalysis.calculate_price_action(df)
            volume_metrics = TechnicalAnalysis.calculate_volume_profile(df)
            support_resistance = TechnicalAnalysis.find_support_resistance(df)
            upside_metrics = FundamentalAnalysis.calculate_upside_potential(info, price_action['current'])
            quality_metrics = FundamentalAnalysis.get_quality_metrics(info)
            
            # Get news analysis
            news_data = DataFetcher.get_news_analysis(symbol)
            
            # Initial filters
            if price_action['current'] < config.MIN_PRICE:
                return None
            if price_action['current'] > config.MAX_PRICE:
                return None
            if volume_metrics['avg_20d'] < config.MIN_AVG_VOLUME:
                return None
            if rsi > config.RSI_OVERSOLD:
                return None
            
            # Calculate component scores
            technical_score, tech_flags = cls.score_technical(rsi, price_action)
            volume_score, vol_flags = cls.score_volume(volume_metrics)
            fundamental_score, fund_flags = cls.score_fundamental(upside_metrics, quality_metrics)
            sentiment_score, sent_flags = cls.score_sentiment(news_data['sentiment'], news_data['news_count'])
            
            # Combine risk flags
            risk_flags = tech_flags + vol_flags + fund_flags + sent_flags
            
            # Calculate weighted total score
            total_score = (
                technical_score * config.WEIGHT_TECHNICAL +
                volume_score * config.WEIGHT_VOLUME +
                fundamental_score * config.WEIGHT_FUNDAMENTAL +
                sentiment_score * config.WEIGHT_SENTIMENT
            )
            
            # Apply volume penalty
            vol_surge = volume_metrics['surge_ratio']
            if vol_surge < 0.5:
                total_score *= 0.3
                risk_flags.append("CRITICAL: Very low volume")
            elif vol_surge < 0.8:
                total_score *= 0.6
            
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
                sentiment=news_data['sentiment'],
                headline=news_data['top_headline'],
                analyst_count=upside_metrics['analyst_count'],
                market_cap=quality_metrics.get('market_cap', 0),
                sector=quality_metrics.get('sector', 'Unknown'),
                risk_flags=risk_flags,
                support_level=support_resistance.get('support', 0),
                resistance_level=support_resistance.get('resistance', 0),
                risk_reward_ratio=support_resistance.get('risk_reward_ratio', 0),
                dip_reason=news_data['dip_reason'],
                reason_confidence=news_data['reason_confidence']
            )
            
        except Exception as e:
            logger.error(f"Scoring error for {symbol}: {e}")
            return None


# === TRADE RECOMMENDATION ENGINE ===
@dataclass
class TradeRecommendation:
    """Complete trade recommendation with entry/exit plan"""
    symbol: str
    score: StockScore
    
    # Market context
    market_trend: str
    spy_change: float
    qqq_change: float
    vix_level: float
    
    # Trade plan
    confidence: str  # HIGH, MEDIUM, LOW
    suggested_action: str  # BUY_NOW, WAIT_FOR_CONFIRMATION, SKIP
    
    # Entry strategies
    conservative_entry: float
    aggressive_entry: float
    stop_loss: float
    target_price: float
    
    # Position sizing
    position_size_shares: int
    risk_amount: float
    potential_profit: float
    
    # Risk assessment
    risk_pct: float
    reward_pct: float
    risk_reward_ratio: float
    
    # Contextual info
    why_its_down: str
    key_levels: str
    red_flags: List[str]
    green_flags: List[str]


class TradeRecommendationEngine:
    """Generates complete trade recommendations"""
    
    def __init__(self, account_size: float = 10000):
        self.account_size = account_size
    
    def generate_recommendation(self, score: StockScore) -> TradeRecommendation:
        """Generate complete trade recommendation"""
        
        # Get market context
        market_data = MarketContext.get_market_performance()
        vix = MarketContext.get_vix()
        
        # Determine confidence level
        confidence = self._assess_confidence(score, market_data, vix)
        
        # Determine suggested action
        action = self._determine_action(score, confidence, market_data)
        
        # Calculate entry points
        conservative_entry, aggressive_entry = self._calculate_entries(score)
        
        # Calculate stop loss and target
        stop_loss = self._calculate_stop_loss(score)
        target_price = self._calculate_target(score)
        
        # Calculate position sizing
        position_size, risk_amount = self._calculate_position_size(
            score.price, stop_loss
        )
        
        # Calculate risk/reward
        risk_pct = abs((score.price - stop_loss) / score.price)
        reward_pct = abs((target_price - score.price) / score.price)
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        potential_profit = (target_price - score.price) * position_size
        
        # Contextual information
        why_down = self._explain_dip_reason(score)
        key_levels = self._format_key_levels(score)
        red_flags, green_flags = self._categorize_flags(score, market_data)
        
        return TradeRecommendation(
            symbol=score.symbol,
            score=score,
            market_trend=market_data['market_trend'],
            spy_change=market_data['spy_change'],
            qqq_change=market_data['qqq_change'],
            vix_level=vix,
            confidence=confidence,
            suggested_action=action,
            conservative_entry=conservative_entry,
            aggressive_entry=aggressive_entry,
            stop_loss=stop_loss,
            target_price=target_price,
            position_size_shares=position_size,
            risk_amount=risk_amount,
            potential_profit=potential_profit,
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            risk_reward_ratio=rr_ratio,
            why_its_down=why_down,
            key_levels=key_levels,
            red_flags=red_flags,
            green_flags=green_flags
        )
    
    def _assess_confidence(self, score: StockScore, market_data: Dict, vix: float) -> str:
        """Assess overall confidence in trade"""
        
        # Start with score-based confidence
        if score.total_score >= 70:
            confidence = "HIGH"
        elif score.total_score >= 55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Downgrade if critical issues
        if "CRITICAL" in " ".join(score.risk_flags):
            confidence = "LOW"
        
        # Downgrade if market is very bearish
        if market_data['market_trend'] == 'bearish' and market_data['spy_change'] < -0.02:
            if confidence == "HIGH":
                confidence = "MEDIUM"
        
        # Downgrade if VIX is very high (market fear)
        if vix > 30:
            if confidence == "HIGH":
                confidence = "MEDIUM"
        
        # Downgrade if company-specific bad news
        if score.dip_reason == "company_specific_negative":
            if confidence == "HIGH":
                confidence = "MEDIUM"
            elif confidence == "MEDIUM":
                confidence = "LOW"
        
        return confidence
    
    def _determine_action(self, score: StockScore, confidence: str, market_data: Dict) -> str:
        """Determine recommended action"""
        
        # Skip if confidence is low
        if confidence == "LOW":
            return "SKIP"
        
        # Skip if company-specific bad news
        if score.dip_reason == "company_specific_negative":
            return "SKIP"
        
        # Skip if very low volume
        if score.volume_surge < 0.5:
            return "SKIP"
        
        # Wait for confirmation if medium confidence
        if confidence == "MEDIUM":
            # But allow aggressive traders to enter
            if score.total_score >= 60 and score.volume_surge >= 1.2:
                return "BUY_NOW"
            else:
                return "WAIT_FOR_CONFIRMATION"
        
        # High confidence - ready to buy
        return "BUY_NOW"
    
    def _calculate_entries(self, score: StockScore) -> Tuple[float, float]:
        """Calculate conservative and aggressive entry points"""
        
        current_price = score.price
        support = score.support_level
        
        # Aggressive entry: current price
        aggressive_entry = current_price
        
        # Conservative entry: wait for bounce confirmation
        # Typically 1-2% above current or at minor resistance
        conservative_entry = current_price * 1.015  # 1.5% bounce
        
        return conservative_entry, aggressive_entry
    
    def _calculate_stop_loss(self, score: StockScore) -> float:
        """Calculate stop loss level"""
        
        current_price = score.price
        support = score.support_level
        
        if support > 0 and support < current_price:
            # Set stop just below support with buffer
            support_distance = (current_price - support) / current_price
            if support_distance < 0.05:  # Support within 5%
                stop = support * 0.98  # 2% below support
            else:
                # Support too far, use default stop
                stop = current_price * (1 - config.DEFAULT_STOP_LOSS_PCT)
        else:
            # No clear support, use default
            stop = current_price * (1 - config.DEFAULT_STOP_LOSS_PCT)
        
        return stop
    
    def _calculate_target(self, score: StockScore) -> float:
        """Calculate target price"""
        
        current_price = score.price
        resistance = score.resistance_level
        
        # Use resistance if it's reasonable
        if resistance > current_price:
            resistance_distance = (resistance - current_price) / current_price
            if 0.05 <= resistance_distance <= 0.15:  # 5-15% upside
                return resistance
        
        # Otherwise use default target
        return current_price * (1 + config.DEFAULT_TARGET_PCT)
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> Tuple[int, float]:
        """
        Calculate position size based on account risk
        Returns: (shares, risk_amount)
        """
        
        risk_per_trade = self.account_size * config.RISK_PER_TRADE_PCT
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return 0, 0
        
        shares = int(risk_per_trade / risk_per_share)
        actual_risk = shares * risk_per_share
        
        return shares, actual_risk
    
    def _explain_dip_reason(self, score: StockScore) -> str:
        """Generate human-readable explanation of why stock is down"""
        
        reason_map = {
            "company_specific_negative": "❌ Bad company news (earnings miss, downgrade, or negative event)",
            "profit_taking_after_news": "✓ Profit-taking after good news (potential overreaction)",
            "market_wide_selloff": "✓ Market-wide selloff (not company-specific)",
            "sector_weakness": "⚠️ Sector rotation or industry weakness",
            "technical_correction": "✓ Technical correction (no clear negative catalyst)",
            "unknown": "? Unclear reason (check news manually)"
        }
        
        return reason_map.get(score.dip_reason, "Unknown")
    
    def _format_key_levels(self, score: StockScore) -> str:
        """Format support/resistance info"""
        
        if score.support_level == 0:
            return "No clear support/resistance identified"
        
        support_dist = ((score.price - score.support_level) / score.price) * 100
        resistance_dist = ((score.resistance_level - score.price) / score.price) * 100
        
        return (
            f"Support: ${score.support_level:.2f} ({support_dist:.1f}% below) | "
            f"Resistance: ${score.resistance_level:.2f} ({resistance_dist:.1f}% above)"
        )
    
    def _categorize_flags(self, score: StockScore, market_data: Dict) -> Tuple[List[str], List[str]]:
        """Separate red flags from green flags"""
        
        red_flags = []
        green_flags = []
        
        # From existing risk flags
        for flag in score.risk_flags:
            if any(word in flag.lower() for word in ['critical', 'very low', 'heavy', 'negative', 'sell']):
                red_flags.append(flag)
            else:
                red_flags.append(flag)
        
        # Add green flags
        if score.volume_surge >= 1.5:
            green_flags.append(f"Strong volume: {score.volume_surge:.2f}x average")
        
        if score.rsi < 30:
            green_flags.append(f"Deeply oversold: RSI {score.rsi:.0f}")
        
        if score.analyst_upside > 0.15:
            green_flags.append(f"High analyst upside: {score.analyst_upside:.1%}")
        
        if market_data['market_trend'] == 'bullish':
            green_flags.append("Bullish market environment")
        
        if score.dip_reason in ['market_wide_selloff', 'technical_correction']:
            green_flags.append("Dip appears to be overreaction")
        
        if score.risk_reward_ratio >= 2.0:
            green_flags.append(f"Favorable risk/reward: {score.risk_reward_ratio:.1f}:1")
        
        return red_flags, green_flags


# === ENHANCED NOTIFICATION ===
class EnhancedNotifier:
    """Enhanced notifications with full trade recommendations"""
    
    @staticmethod
    def format_detailed_recommendation(rec: TradeRecommendation) -> str:
        """Format detailed trade recommendation message"""
        
        score = rec.score
        
        # Header
        msg = f"{'═' * 50}\n"
        msg += f"📊 TRADE SIGNAL: {rec.symbol}\n"
        msg += f"Score: {score.total_score:.0f}/100 | Confidence: {rec.confidence}\n"
        msg += f"{'═' * 50}\n\n"
        
        # Current status
        msg += "📈 CURRENT STATUS:\n"
        msg += f"Price: ${score.price:.2f} | RSI: {score.rsi:.0f} "
        msg += f"{'🟢' if score.sentiment > 0.6 else '🔴' if score.sentiment < 0.4 else '⚪'}\n"
        msg += f"Volume: {score.volume_surge:.2f}x avg | Drawdown: {score.drawdown:.1%}\n"
        msg += f"Component: T:{score.technical_score:.0f} V:{score.volume_score:.0f} "
        msg += f"F:{score.fundamental_score:.0f} S:{score.sentiment_score:.0f}\n\n"
        
        # Why it's down
        msg += "📰 WHY IT'S DOWN:\n"
        msg += f"{rec.why_its_down}\n"
        msg += f"Market: SPY {rec.spy_change:+.1%}, QQQ {rec.qqq_change:+.1%} "
        msg += f"({rec.market_trend.upper()})\n"
        msg += f"VIX: {rec.vix_level:.1f} {'⚠️ HIGH' if rec.vix_level > 25 else '✓'}\n\n"
        
        # Technical setup
        msg += "🎯 TECHNICAL SETUP:\n"
        msg += f"{rec.key_levels}\n"
        msg += f"Analyst Target: ${score.analyst_upside * score.price + score.price:.2f} "
        msg += f"({score.analyst_upside:+.1%}, {score.analyst_count} analysts)\n\n"
        
        # Recommended action
        action_emoji = {
            "BUY_NOW": "✅",
            "WAIT_FOR_CONFIRMATION": "⏸️",
            "SKIP": "❌"
        }
        
        msg += f"{'─' * 50}\n"
        msg += f"{action_emoji.get(rec.suggested_action, '?')} RECOMMENDATION: {rec.suggested_action.replace('_', ' ')}\n"
        msg += f"{'─' * 50}\n\n"
        
        if rec.suggested_action != "SKIP":
            # Trade plan
            msg += "💼 TRADE PLAN:\n\n"
            
            msg += "Conservative Entry:\n"
            msg += f"  → Wait for bounce to ${rec.conservative_entry:.2f}\n"
            msg += f"  → Stop: ${rec.stop_loss:.2f}\n"
            msg += f"  → Target: ${rec.target_price:.2f}\n"
            msg += f"  → Risk: {rec.risk_pct:.1%} | Reward: {rec.reward_pct:.1%}\n\n"
            
            msg += "Aggressive Entry:\n"
            msg += f"  → Enter NOW at ${rec.aggressive_entry:.2f}\n"
            msg += f"  → Stop: ${rec.stop_loss:.2f}\n"
            msg += f"  → Target: ${rec.target_price:.2f}\n"
            msg += f"  → Risk: {rec.risk_pct:.1%} | Reward: {rec.reward_pct:.1%}\n\n"
            
            msg += f"Risk/Reward Ratio: {rec.risk_reward_ratio:.2f}:1 "
            msg += f"{'✅' if rec.risk_reward_ratio >= 2 else '⚠️'}\n\n"
            
            # Position sizing
            msg += "💰 POSITION SIZING (1% risk on $20k account):\n"
            msg += f"Shares: {rec.position_size_shares}\n"
            msg += f"Risk Amount: ${rec.risk_amount:.2f}\n"
            msg += f"Potential Profit: ${rec.potential_profit:.2f}\n\n"
        
        # Green flags
        if rec.green_flags:
            msg += "✅ GREEN FLAGS:\n"
            for flag in rec.green_flags:
                msg += f"  • {flag}\n"
            msg += "\n"
        
        # Red flags
        if rec.red_flags:
            msg += "⚠️ RED FLAGS:\n"
            for flag in rec.red_flags:
                msg += f"  • {flag}\n"
            msg += "\n"
        
        # Latest headline
        msg += f"📰 Latest: `{score.headline[:60]}...`\n\n"
        
        msg += f"{'═' * 50}\n"
        msg += f"Scanned at {datetime.now().strftime('%H:%M ET')}\n"
        msg += f"{'═' * 50}"
        
        return msg
    
    @staticmethod
    def send_telegram(message: str) -> bool:
        """Send message to Telegram"""
        if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.info("Telegram not configured")
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
            
            logger.info("Telegram notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


# === MAIN SCANNER ===
class EnhancedDipScanner:
    """Enhanced scanner with full decision support"""
    
    def __init__(self, account_size: float = 10000):
        self.recommendation_engine = TradeRecommendationEngine(account_size)
    
    @staticmethod
    def load_tickers(filename: str = "tickers.txt") -> List[str]:
        """Load ticker symbols"""
        try:
            with open(filename, "r") as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers")
            return tickers
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
            return []
    
    def scan_and_recommend(self, tickers: List[str], top_n: int = 5) -> List[TradeRecommendation]:
        """
        Scan tickers and generate trade recommendations
        """
        logger.info("=" * 60)
        logger.info("Enhanced Buy-the-Dip Scanner with Decision Support")
        logger.info("=" * 60)
        
        progress, session = MarketContext.get_market_session()
        logger.info(f"Market session: {session} ({progress*100:.0f}% complete)")
        
        scores = []
        
        for i, symbol in enumerate(tickers, 1):
            try:
                logger.info(f"[{i}/{len(tickers)}] Analyzing {symbol}...")
                
                # Fetch data
                df = DataFetcher.get_stock_data(symbol)
                if df is None:
                    continue
                
                info = DataFetcher.get_stock_info(symbol)
                
                # Score the stock
                score = ScoringEngine.calculate_composite_score(symbol, df, info)
                
                if score and score.total_score > 30:
                    scores.append(score)
                    logger.info(f"  ✓ {symbol}: Score {score.total_score:.0f}")
                
                time.sleep(1.5)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        top_scores = scores[:top_n]
        
        # Generate recommendations
        recommendations = []
        for score in top_scores:
            rec = self.recommendation_engine.generate_recommendation(score)
            recommendations.append(rec)
        
        logger.info(f"\nAnalysis complete: {len(recommendations)} recommendations generated")
        
        return recommendations
    
    def run(self, ticker_file: str = "tickers.txt", top_n: int = 5):
        """Main execution"""
        
        # Load tickers
        tickers = self.load_tickers(ticker_file)
        if not tickers:
            logger.error("No tickers to scan")
            return
        
        # Scan and generate recommendations
        recommendations = self.scan_and_recommend(tickers, top_n)
        
        # Send notifications
        for rec in recommendations:
            message = EnhancedNotifier.format_detailed_recommendation(rec)
            EnhancedNotifier.send_telegram(message)
            print("\n" + message + "\n")  # Also print to console
            time.sleep(2)  # Avoid rate limits
        
        logger.info("=" * 60)
        logger.info("Scan complete")
        logger.info("=" * 60)


# === ENTRY POINT ===
if __name__ == "__main__":
    # Fixed account size - no configuration needed
    ACCOUNT_SIZE = 20000
    
    scanner = EnhancedDipScanner(account_size=ACCOUNT_SIZE)
    scanner.run()
