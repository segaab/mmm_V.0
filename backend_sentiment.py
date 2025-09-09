import pandas as pd
import numpy as np
import datetime
import time
import os
import re
import requests
from yahooquery import Ticker
from newsapi import NewsApiClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import logging
import threading
import json
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('market_monitor')

# Constants
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'your_news_api_key_here')
RISK_COUNTRIES = ['Russia', 'Ukraine', 'Israel', 'Gaza', 'Lebanon', 'Iran', 'North Korea', 
                 'Venezuela', 'Nigeria', 'Libya', 'Syria', 'Yemen', 'Iraq', 'Afghanistan']
ENERGY_KEYWORDS = ['oil', 'gas', 'energy', 'crude', 'petroleum', 'OPEC', 'barrel', 'WTI', 'Brent']
ECONOMIC_KEYWORDS = ['GDP', 'inflation', 'interest rate', 'Fed', 'ECB', 'economy', 'recession', 
                    'unemployment', 'jobs', 'economic growth', 'consumer price']
GEOPOLITICAL_KEYWORDS = ['war', 'conflict', 'sanctions', 'tariff', 'trade war', 'geopolitical', 
                        'military', 'attack', 'invasion', 'treaty', 'agreement']

# Asset mappings
ASSET_SYMBOLS = {
    "Oil (WTI)": "CL=F",
    "Oil (Brent)": "BZ=F",
    "Gold": "GC=F",
    "USD/EUR": "EURUSD=X",
    "USD/CNY": "CNY=X"
}

REGION_SOURCES = {
    "USA": ["bloomberg.com", "cnbc.com", "wsj.com", "reuters.com", "nytimes.com"],
    "European Union": ["ft.com", "reuters.com", "bloomberg.com", "euronews.com", "politico.eu"],
    "Australia": ["afr.com", "abc.net.au", "smh.com.au", "news.com.au"],
    "China": ["scmp.com", "globaltimes.cn", "chinadaily.com.cn", "xinhuanet.com"]
}

class DataFetcher:
    """Class to handle all data fetching operations"""
    
    def __init__(self, news_api_key=NEWS_API_KEY):
        """Initialize the data fetcher with API keys"""
        self.news_api = NewsApiClient(api_key=news_api_key)
        logger.info("DataFetcher initialized")
        
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def fetch_price_data(self, asset_name, interval='1m', period='1d'):
        """
        Fetch price and volume data from Yahoo Finance
        
        Parameters:
        - asset_name: Asset name as displayed in the UI
        - interval: Time interval (1m, 5m, 15m, 30m, 1h, 1d)
        - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        
        Returns:
        - DataFrame with OHLCV data
        """
        try:
            symbol = ASSET_SYMBOLS.get(asset_name)
            if not symbol:
                logger.error(f"Unknown asset name: {asset_name}")
                return pd.DataFrame()
            
            logger.info(f"Fetching price data for {asset_name} ({symbol})")
            ticker = Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Reset index if it's MultiIndex
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index()
                # If we have a 'symbol' column, drop it
                if 'symbol' in data.columns:
                    data = data.drop('symbol', axis=1)
            
            # Rename columns to match our standard format
            data = data.rename(columns={
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Ensure timestamp is datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            logger.info(f"Successfully fetched {len(data)} records for {asset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_news(self, keywords, sources=None, from_date=None, to_date=None, language='en', page_size=100):
        """
        Fetch news articles from NewsAPI
        
        Parameters:
        - keywords: List of keywords to search for
        - sources: List of news sources
        - from_date: Start date (default: 7 days ago)
        - to_date: End date (default: now)
        - language: Article language (default: 'en')
        - page_size: Number of articles to fetch (default: 100)
        
        Returns:
        - List of news articles as dictionaries
        """
        try:
            if from_date is None:
                from_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            if to_date is None:
                to_date = datetime.datetime.now().strftime('%Y-%m-%d')
                
            # Convert keywords list to query string
            query = ' OR '.join(keywords)
            
            logger.info(f"Fetching news with query: {query}")
            response = self.news_api.get_everything(
                q=query,
                sources=','.join(sources) if sources else None,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by='relevancy',
                page_size=page_size
            )
            
            articles = response.get('articles', [])
            logger.info(f"Fetched {len(articles)} news articles")
            
            # Process each article to add sentiment and categorization
            processed_articles = []
            for article in articles:
                # Calculate sentiment
                sentiment = self.calculate_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                
                # Determine category
                category = self.categorize_news(article.get('title', '') + ' ' + article.get('description', ''))
                
                # Determine region
                source_domain = article.get('source', {}).get('name', '').lower()
                region = self.determine_region(source_domain)
                
                # Add processed fields
                article['sentiment'] = sentiment
                article['category'] = category
                article['region'] = region
                article['published_at'] = pd.to_datetime(article.get('publishedAt'))
                
                processed_articles.append(article)
                
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def calculate_sentiment(self, text):
        """
        Calculate sentiment score for a text using VADER
        
        Parameters:
        - text: Text to analyze
        
        Returns:
        - Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            # Return compound score (normalized between -1 and 1)
            return scores['compound']
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return 0.0
    
    def categorize_news(self, text):
        """
        Categorize news article based on keywords
        
        Parameters:
        - text: Article text (title + description)
        
        Returns:
        - Category (Energy, Economic, Geopolitical, or Other)
        """
        text = text.lower()
        
        # Check for energy-related keywords
        if any(keyword.lower() in text for keyword in ENERGY_KEYWORDS):
            return "Energy"
        
        # Check for economic keywords
        if any(keyword.lower() in text for keyword in ECONOMIC_KEYWORDS):
            return "Economic"
        
        # Check for geopolitical keywords
        if any(keyword.lower() in text for keyword in GEOPOLITICAL_KEYWORDS):
            return "Geopolitical"
        
        # Default category
        return "Other"
    
    def determine_region(self, source_domain):
        """
        Determine region based on news source domain
        
        Parameters:
        - source_domain: News source domain
        
        Returns:
        - Region (USA, European Union, Australia, China, or Global)
        """
        for region, domains in REGION_SOURCES.items():
            if any(domain in source_domain for domain in domains):
                return region
        
        return "Global"
    
    def fetch_region_risk_data(self):
        """
        Simulate fetching country risk data
        In a production environment, this would connect to a risk data provider
        
        Returns:
        - DataFrame with country risk data
        """
        # In a real implementation, this would fetch from a data provider
        # For now, we'll simulate risk data
        
        countries = RISK_COUNTRIES + ['United States', 'China', 'Germany', 'Japan', 'United Kingdom', 'France']
        
        risk_data = []
        now = datetime.datetime.now()
        
        for country in countries:
            # Generate random risk scores with higher values for known risk countries
            base_risk = 0.7 if country in RISK_COUNTRIES else 0.2
            variation = np.random.uniform(-0.2, 0.2)
            
            risk_data.append({
                'country': country,
                'political_risk': min(1.0, max(0.1, base_risk + variation)),
                'economic_risk': min(1.0, max(0.1, base_risk + np.random.uniform(-0.3, 0.3))),
                'financial_risk': min(1.0, max(0.1, base_risk + np.random.uniform(-0.25, 0.25))),
                'composite_risk': min(1.0, max(0.1, base_risk + variation/2)),
                'timestamp': now
            })
        
        return pd.DataFrame(risk_data)


class DataCleaner:
    """Class to handle data cleaning operations"""
    
    def clean_price_data(self, df):
        """
        Clean price data by handling missing values and outliers
        
        Parameters:
        - df: DataFrame with price data
        
        Returns:
        - Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        if df_clean.isnull().any().any():
            logger.info("Fixing missing values in price data")
            
            # Forward fill for OHLC data
            for col in ['open', 'high', 'low', 'close']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(method='ffill')
            
            # For volume, use 0 or the median
            if 'volume' in df_clean.columns:
                df_clean['volume'] = df_clean['volume'].fillna(df_clean['volume'].median())
        
        # Handle outliers in price data (using IQR method)
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Identify outliers
                outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
                if outliers.any():
                    logger.info(f"Found {outliers.sum()} outliers in {col}")
                    
                    # Replace outliers with nearest valid value
                    df_clean.loc[outliers, col] = df_clean.loc[~outliers, col].median()
        
        # Ensure sorted by timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
        
        return df_clean
    
    def clean_news_data(self, news_list):
        """
        Clean and normalize news data
        
        Parameters:
        - news_list: List of news article dictionaries
        
        Returns:
        - Cleaned list of news articles
        """
        if not news_list:
            logger.warning("Empty news list provided for cleaning")
            return []
        
        cleaned_news = []
        
        for article in news_list:
            # Skip articles with missing essential fields
            if not article.get('title') or not article.get('url'):
                continue
            
            # Create a standardized article object
            clean_article = {
                'date': pd.to_datetime(article.get('publishedAt', article.get('published_at'))),
                'title': article.get('title', '').strip(),
                'content': article.get('description', '').strip(),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'sentiment': article.get('sentiment', 0),
                'category': article.get('category', 'Other'),
                'region': article.get('region', 'Global')
            }
            
            # Add event type based on category
            if clean_article['category'] == 'Energy':
                clean_article['event_type'] = 'Supply'
            elif clean_article['category'] == 'Economic':
                clean_article['event_type'] = 'Economic'
            elif clean_article['category'] == 'Geopolitical':
                clean_article['event_type'] = 'Geopolitical'
            else:
                clean_article['event_type'] = 'Other'
            
            cleaned_news.append(clean_article)
        
        # Sort by date (most recent first)
        cleaned_news.sort(key=lambda x: x['date'], reverse=True)
        
        return cleaned_news


class MarketAnalyzer:
    """Class to handle market analysis operations"""
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for price data
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with added technical indicators
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for technical analysis")
            return df
        
        # Make a copy to avoid modifying the original
        df_ta = df.copy()
        
        # Calculate moving averages
        df_ta['sma_20'] = ta.trend.sma_indicator(df_ta['close'], window=20)
        df_ta['sma_50'] = ta.trend.sma_indicator(df_ta['close'], window=50)
        df_ta['ema_10'] = ta.trend.ema_indicator(df_ta['close'], window=10)
        
        # Calculate RSI
        df_ta['rsi_14'] = ta.momentum.rsi(df_ta['close'], window=14)
        
        # Calculate MACD
        macd = ta.trend.MACD(df_ta['close'])
        df_ta['macd'] = macd.macd()
        df_ta['macd_signal'] = macd.macd_signal()
        df_ta['macd_diff'] = macd.macd_diff()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_ta['close'])
        df_ta['bb_upper'] = bollinger.bollinger_hband()
        df_ta['bb_middle'] = bollinger.bollinger_mavg()
        df_ta['bb_lower'] = bollinger.bollinger_lband()
        
        # Volume indicators
        if 'volume' in df_ta.columns:
            # On-Balance Volume
            df_ta['obv'] = ta.volume.on_balance_volume(df_ta['close'], df_ta['volume'])
            
            # Volume Weighted Average Price
            df_ta['vwap'] = self.calculate_vwap(df_ta)
        
        # Momentum and Trend
        df_ta['pct_change'] = df_ta['close'].pct_change() * 100
        df_ta['momentum'] = df_ta['close'].diff(5)
        
        # Calculate volatility
        df_ta['volatility_5'] = df_ta['pct_change'].rolling(5).std()
        df_ta['volatility_20'] = df_ta['pct_change'].rolling(20).std()
        
        return df_ta
    
    def calculate_vwap(self, df):
        """
        Calculate Volume Weighted Average Price
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - Series with VWAP values
        """
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        df['vwap_numerator'] = df['typical_price'] * df['volume']
        df['cumulative_vwap_numerator'] = df['vwap_numerator'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        
        vwap = df['cumulative_vwap_numerator'] / df['cumulative_volume']
        
        # Clean up temporary columns
        df.drop(['typical_price', 'vwap_numerator', 'cumulative_vwap_numerator', 'cumulative_volume'], 
                axis=1, inplace=True)
        
        return vwap
    
    def calculate_price_volume_metrics(self, df):
        """
        Calculate price-volume based microstructure metrics
        (alternative to bid-ask metrics when that data is unavailable)
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with added microstructure metrics
        """
        


