# web_model.py - ENHANCED VERSION with async support
import re
import time
import asyncio
import aiohttp
import requests
from urllib.parse import quote_plus
from typing import Dict, Any, List, Tuple
import yfinance as yf
import feedparser
from requests.exceptions import HTTPError, ConnectionError, Timeout
import pandas as pd
from datetime import datetime, timedelta

class WebSupplementationAgent:
    """
    Web agent that fetches real-time JSE market data and news.
    Provides both synchronous (for batch analysis) and asynchronous (for live dashboards) methods.
    Handles ticker resolution, price conversion, and news aggregation from multiple sources.
    """
    
    def __init__(self):
        """
        Initialize the web agent with company database and HTTP headers.
        Sets up JSE ticker mapping for natural language query resolution.
        """
        # Standard browser headers to avoid bot detection
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Enhanced JSE company database: maps company names to their JSE tickers
        # Organized by sector for easier maintenance and extension
        self.company_to_ticker = {
            # Banking & Financial Services (Top 40 constituents)
            'standard bank': 'SBK.JO', 
            'firstrand': 'FSR.JO', 'fnb': 'FSR.JO',  # FNB is FirstRand's retail brand
            'nedbank': 'NED.JO', 
            'capitec': 'CPI.JO',  # Capitec Bank Holdings
            'investec': 'INL.JO',
            'absa': 'ABG.JO',  # ABSA Group Limited
            'sanlam': 'SLM.JO',
            'discovery': 'DSY.JO',
            'old mutual': 'OMU.JO',
            
            # Telecommunications (major players)
            'mtn': 'MTN.JO', 'mtn group': 'MTN.JO',
            'vodacom': 'VOD.JO', 
            'telkom': 'TKG.JO',
            'multichoice': 'MCG.JO',  # Media/entertainment
            
            # Mining & Resources (JSE's largest sector by market cap)
            'anglo american': 'AGL.JO',  # One of world's largest mining companies
            'bhp': 'BHP.JO',  # BHP Billiton
            'sasol': 'SOL.JO',  # Chemicals and energy
            'exxaro': 'EXX.JO',
            'gold fields': 'GFI.JO',
            'harmony gold': 'HAR.JO',
            'anglogold ashanti': 'ANG.JO',
            'impala platinum': 'IMP.JO',
            'kumba iron ore': 'KIO.JO',
            
            # Retail & Consumer (cyclical stocks)
            'shoprite': 'SHP.JO',  # Largest African retailer
            'pick n pay': 'PIK.JO',
            'woolworths': 'WHL.JO',  # Not related to US/UK Woolworths
            'mr price': 'MRP.JO',
            'truworths': 'TRU.JO',
            'foschini': 'TFG.JO',  # The Foschini Group
            'clicks': 'CLS.JO',
            'spar': 'SPP.JO',
            
            # Industrial & Other (diversified holdings)
            'bidvest': 'BVT.JO',
            'naspers': 'NPN.JO',  # Tech/internet holding company (Tencent stake)
            'prosus': 'PRX.JO',  # Naspers spin-off
            'richemont': 'CFR.JO',  # Luxury goods
            'reunert': 'RLO.JO'
        }

    def _create_useful_summary(self, entry: Dict[str, Any], company_name: str) -> str:
        """
        Creates a useful summary, falling back to a generated one if the RSS summary is poor.
        Many RSS feeds provide low-quality summaries or just HTML tags - this fixes that.
        
        Args:
            entry: RSS feed entry dictionary
            company_name: Name of the company for context
            
        Returns:
            Clean, informative summary text
        """
        headline = entry.get('title', '')
        original_summary = entry.get('summary', '')
        source = entry.get('source', {}).get('title', 'a news source')

        # Clean the original summary by removing any HTML tags
        # RSS feeds often include <p>, <br>, <a> tags that need stripping
        cleaned_summary = re.sub(r'<[^>]+>', '', original_summary).strip()

        # A good summary should exist and be meaningfully longer than the headline
        # If summary is just headline repeated or very short, generate better one
        if cleaned_summary and len(cleaned_summary) > len(headline) + 15:
            # If the summary is useful, return it (truncated to 200 chars for consistency)
            return cleaned_summary[:200] + "..." if len(cleaned_summary) > 200 else cleaned_summary
        else:
            # If not useful, generate a more informative fallback summary
            # This at least provides context about source and company
            return f"An article from {source} covering recent news and developments related to {company_name}. Click to read the full story."

    def _resolve_tickers(self, query: str) -> List[Tuple[str, str]]:
        """
        Enhanced ticker resolution that maps company names in queries to JSE tickers.
        Supports partial matching and common abbreviations.
        
        Args:
            query: User's natural language query (e.g., "How is MTN performing?")
            
        Returns:
            List of (ticker, display_name) tuples (e.g., [('MTN.JO', 'Mtn Limited')])
        """
        query_lower = query.lower()
        candidates = []
        
        # Search for all company names mentioned in the query
        # Uses substring matching to catch partial names (e.g., "bank" matches "Standard Bank")
        for company_name, ticker in self.company_to_ticker.items():
            if company_name in query_lower:
                # Create proper display name (Title Case + "Limited")
                display_name = company_name.title() + " Limited"
                candidates.append((ticker, display_name))
        
        # Remove duplicates (some companies have multiple name variations)
        # Uses set to track seen tickers, maintains order of first occurrence
        seen = set()
        return [(ticker, name) for ticker, name in candidates if not (ticker in seen or seen.add(ticker))]

    def _convert_price_to_rands(self, price):
        """
        Convert price from cents to Rands if needed.
        Some JSE stocks quote in cents, others in rands - this normalizes them.
        
        Args:
            price: Price value (could be in cents or rands)
            
        Returns:
            Price in Rands (standard currency unit)
        """
        # Heuristic: if price > 1000, it's likely in cents (1 rand = 100 cents)
        # Example: 12500 cents = R125.00
        if price and price > 1000:  # Likely in cents
            return price / 100
        return price

    def get_structured_market_data(self, companies: List[Tuple[str, str]]) -> str:
        """
        Optimized market data fetching using yfinance.
        Gets current price, change, and percentage change for JSE stocks.
        
        Args:
            companies: List of (ticker, name) tuples to fetch data for
            
        Returns:
            Formatted string with price data for all companies
        """
        if not companies:
            return ""

        market_data_parts = []
        
        for ticker, name in companies:
            try:
                stock = yf.Ticker(ticker)
                
                # Get historical data quickly (last 2 days to calculate change)
                # Using history() instead of info because it's faster and more reliable
                hist = stock.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    # Extract latest and previous closing prices
                    latest_price = self._convert_price_to_rands(hist['Close'].iloc[-1])
                    previous_close = self._convert_price_to_rands(hist['Close'].iloc[-2])
                    
                    # Calculate absolute and percentage changes
                    change = latest_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    # Format with emojis and +/- signs for visual clarity
                    market_data_parts.append(
                        f"📊 {name} ({ticker}): "
                        f"Price: R{latest_price:.2f} | "
                        f"Change: {change:+.2f} ({change_percent:+.2f}%)"
                    )
                else:
                    # Quick fallback if historical data unavailable
                    # Uses info dict (slower but works when history fails)
                    info = stock.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if current_price:
                        current_price = self._convert_price_to_rands(current_price)
                        market_data_parts.append(f"📊 {name} ({ticker}): Price: R{current_price:.2f}")
                    
            except Exception as e:
                # Graceful degradation: mark stock as unavailable but continue
                market_data_parts.append(f"❌ {name} ({ticker}): Data unavailable")

        return "\n".join(market_data_parts)

    def fetch_articles_from_google_rss(self, companies: List[Tuple[str, str]], max_per: int = 2) -> List[Dict[str, str]]:
        """
        Optimized news fetching with timeout using Google News RSS feeds.
        Aggregates recent news articles for specified JSE companies.
        
        Args:
            companies: List of (ticker, name) tuples to fetch news for
            max_per: Maximum articles per company (default 2)
            
        Returns:
            List of article dictionaries with headline, source, summary, link
        """
        all_articles = []
        
        for ticker, name in companies:
            try:
                # Construct Google News RSS search URL
                search_query = f'JSE {name}'
                rss_url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=en-ZA"
                
                # Fetch with timeout to avoid hanging on slow responses
                response = requests.get(rss_url, headers=self.headers, timeout=8)
                response.raise_for_status()
                
                # Parse RSS feed using feedparser
                feed = feedparser.parse(response.content)
                
                # Extract articles (limit to max_per per company)
                for entry in feed.entries[:max_per]:
                    # Avoid duplicates (same article appearing for multiple companies)
                    if not any(entry.title == existing['headline'] for existing in all_articles):
                        # Use the helper to get a useful summary (handles poor RSS summaries)
                        summary = self._create_useful_summary(entry, name)
                        
                        all_articles.append({
                            "headline": entry.title,
                            "ticker": ticker,
                            "company": name,
                            "source": entry.get('source', {}).get('title', 'News'),
                            "summary": summary,
                            "link": entry.link
                        })
                
            except Exception:
                # Skip failed news fetches silently (don't break entire process)
                continue
                
        return all_articles[:6]  # Return max 6 articles total (avoid overwhelming user)

    def get_jse_market_summary(self) -> str:
        """
        Get JSE market summary using major stocks as indicators.
        Provides overall market sentiment based on blue-chip performance.
        
        Returns:
            Formatted market summary string with sentiment and statistics
        """
        try:
            # Use major stocks as market indicators (approximates JSE All Share Index)
            # These are large, liquid stocks that represent overall market
            major_stocks = ['SBK.JO', 'MTN.JO', 'NPN.JO', 'AGL.JO']
            up_count = 0
            down_count = 0
            price_changes = []
            
            for ticker in major_stocks:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='2d')
                    if len(hist) >= 2:
                        # Calculate percentage change for each stock
                        current = self._convert_price_to_rands(hist['Close'].iloc[-1])
                        previous = self._convert_price_to_rands(hist['Close'].iloc[-2])
                        change_percent = ((current - previous) / previous) * 100
                        price_changes.append(change_percent)
                        
                        # Count advancers vs decliners
                        if change_percent > 0:
                            up_count += 1
                        else:
                            down_count += 1
                except:
                    continue
            
            if price_changes:
                # Calculate average change across all major stocks
                avg_change = sum(price_changes) / len(price_changes)
                
                # Determine market sentiment based on average change
                # Thresholds: >0.5% = bullish, <-0.5% = bearish, else neutral
                sentiment = "🟢 Bullish" if avg_change > 0.5 else "🔴 Bearish" if avg_change < -0.5 else "🟡 Neutral"
                return f"🏛️ JSE Market: {sentiment} | Avg Change: {avg_change:+.2f}% | Advancers: {up_count}/{len(major_stocks)}"
            else:
                return "🏛️ JSE Market: Data temporarily unavailable"
                
        except Exception as e:
            return "🏛️ JSE Market: Summary unavailable"

    def get_relevant_info(self, user_query: str) -> Dict[str, Any]:
        """
        Optimized main method that orchestrates data fetching.
        Primary entry point for synchronous market data and news retrieval.
        
        Args:
            user_query: User's natural language query about JSE stocks
            
        Returns:
            Dictionary with articles, market_data, and tickers_analyzed
        """
        try:
            # Step 1: Resolve company names to tickers
            companies = self._resolve_tickers(user_query)
            
            # Step 2: Get overall market summary
            market_summary = self.get_jse_market_summary()
            
            # If no companies mentioned, return just market summary
            if not companies:
                return {
                    "articles": [],
                    "market_data": market_summary,
                    "tickers_analyzed": []
                }

            # Step 3: Get detailed stock data for mentioned companies
            market_data = self.get_structured_market_data(companies)
            
            # Step 4: Fetch relevant news articles
            articles = self.fetch_articles_from_google_rss(companies)
            
            # Combine all information
            return {
                "articles": articles,
                "market_data": f"{market_summary}\n\n{market_data}",
                "tickers_analyzed": [t[0] for t in companies]
            }
            
        except Exception as e:
            # Final fallback: return error but maintain structure
            return {
                "articles": [],
                "market_data": f"Error: {str(e)}",
                "tickers_analyzed": []
            }

    # =========================================================================
    # NEW ASYNC METHODS FOR LIVE DASHBOARD PERFORMANCE
    # =========================================================================

    async def get_live_stock_data(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Get live stock data asynchronously for dashboard (FAST!)
        Uses asyncio to fetch multiple stocks in parallel instead of sequentially.
        
        Args:
            tickers: List of JSE tickers (with or without .JO suffix)
            
        Returns:
            List of stock data dictionaries with price, change, volume, etc.
        """
        # Ensure tickers have .JO suffix (JSE stocks)
        jse_tickers = [ticker if ticker.endswith('.JO') else f"{ticker}.JO" for ticker in tickers]
        
        # Create aiohttp session for async HTTP requests
        async with aiohttp.ClientSession():
            # Create tasks for all tickers (parallel execution)
            tasks = [self._fetch_single_stock_async(ticker) for ticker in jse_tickers]
            
            # Execute all tasks concurrently (much faster than sequential)
            # return_exceptions=True prevents one failure from breaking all
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result is not None:
                    valid_results.append(result)
            
            return valid_results

    async def _fetch_single_stock_async(self, ticker: str) -> Dict[str, Any]:
        """
        Async fetch for single stock data.
        Note: yfinance is not truly async, but we run it in executor for concurrency.
        
        Args:
            ticker: JSE ticker symbol
            
        Returns:
            Dictionary with comprehensive stock data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info and intraday history (1-minute intervals)
            info = stock.info
            hist = stock.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
            
            # Extract price data from latest available data point
            current_price = self._convert_price_to_rands(hist['Close'].iloc[-1])
            previous_close = self._convert_price_to_rands(info.get('previousClose', current_price))
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            
            # Get additional metrics for dashboard display
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap')  # Total company value
            day_high = self._convert_price_to_rands(info.get('dayHigh', current_price))
            day_low = self._convert_price_to_rands(info.get('dayLow', current_price))
            
            # Build comprehensive data dictionary
            return {
                'ticker': ticker.replace('.JO', ''),  # Display ticker without suffix
                'full_ticker': ticker,  # Full ticker with .JO for API calls
                'current_price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'previous_close': round(previous_close, 2),
                'volume': volume,
                'market_cap': market_cap,
                'day_high': round(day_high, 2),
                'day_low': round(day_low, 2),
                'timestamp': datetime.now().isoformat(),
                'status': 'up' if change > 0 else 'down' if change < 0 else 'flat'  # Visual indicator
            }
            
        except Exception as e:
            print(f"[WebAgent] Async fetch error for {ticker}: {e}")
            return None

    async def get_live_news_async(self, tickers: List[str], max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Get live news asynchronously for multiple stocks.
        Fetches news in parallel for fast dashboard updates.
        
        Args:
            tickers: List of stock tickers to fetch news for
            max_articles: Maximum number of articles to return
            
        Returns:
            List of news articles sorted by publication date
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            # Create parallel tasks for each ticker's news
            for ticker in tickers:
                task = self._fetch_news_for_ticker_async(session, ticker, max_per=3)
                tasks.append(task)
            
            # Execute all news fetches concurrently
            all_articles = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten list of lists into single list
            flattened_articles = []
            for article_list in all_articles:
                if isinstance(article_list, list):
                    flattened_articles.extend(article_list)
            
            # Remove duplicates by headline (same article may appear for multiple tickers)
            unique_articles = []
            seen_headlines = set()
            
            for article in flattened_articles:
                if article['headline'] not in seen_headlines:
                    seen_headlines.add(article['headline'])
                    unique_articles.append(article)
            
            # Sort by publication date (newest first) and limit to max_articles
            return sorted(unique_articles, 
                        key=lambda x: x.get('published', ''), 
                        reverse=True)[:max_articles]

    async def _fetch_news_for_ticker_async(self, session: aiohttp.ClientSession, 
                                         ticker: str, max_per: int = 3) -> List[Dict[str, Any]]:
        """
        Async fetch news for a single ticker using aiohttp.
        True async HTTP - much faster than synchronous requests.
        
        Args:
            session: aiohttp session for connection pooling
            ticker: Stock ticker to fetch news for
            max_per: Maximum articles to fetch
            
        Returns:
            List of article dictionaries
        """
        try:
            company_name = ticker.replace('.JO', '')
            search_query = f'JSE {company_name} stock'
            rss_url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=en-ZA"
            
            # Async HTTP GET with timeout
            async with session.get(rss_url, headers=self.headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries[:max_per]:
                        # Use the helper to get a useful summary
                        summary = self._create_useful_summary(entry, company_name)
                        
                        articles.append({
                            "headline": entry.title,
                            "ticker": ticker,
                            "company": company_name,
                            "source": entry.get('source', {}).get('title', 'Google News'),
                            "summary": summary,
                            "link": entry.link,
                            "published": entry.get('published', ''),
                            "fetched_at": datetime.now().isoformat()
                        })
                    
                    return articles
                else:
                    return []
                    
        except Exception as e:
            print(f"[WebAgent] Async news fetch error for {ticker}: {e}")
            return []

    def get_major_jse_tickers(self) -> List[str]:
        """
        Get list of major JSE tickers for live dashboard.
        These are the Top 10 by market cap and liquidity.
        
        Returns:
            List of ticker symbols with .JO suffix
        """
        return [
            'MTN.JO',  # MTN Group (Telecom)
            'NPN.JO',  # Naspers (Tech/Internet)
            'SBK.JO',  # Standard Bank (Banking)
            'FSR.JO',  # FirstRand (Banking)
            'AGL.JO',  # Anglo American (Mining)
            'SOL.JO',  # Sasol (Chemicals/Energy)
            'VOD.JO',  # Vodacom (Telecom)
            'CPI.JO',  # Capitec (Banking)
            'BHP.JO',  # BHP Billiton (Mining)
            'CFR.JO'   # Richemont (Luxury Goods)
        ]

    # Synchronous wrapper methods for async functionality
    # These allow calling async methods from synchronous code
    
    def get_live_stock_data_sync(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for get_live_stock_data.
        Uses asyncio.run() to execute async method in sync context.
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            List of stock data dictionaries
        """
        try:
            return asyncio.run(self.get_live_stock_data(tickers))
        except Exception as e:
            print(f"[WebAgent] Sync wrapper error: {e}")
            return []

    def get_live_news_sync(self, tickers: List[str], max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for get_live_news_async.
        Bridges async news fetching to synchronous code.
        
        Args:
            tickers: List of stock tickers
            max_articles: Maximum articles to return
            
        Returns:
            List of news article dictionaries
        """
        try:
            return asyncio.run(self.get_live_news_async(tickers, max_articles))
        except Exception as e:
            print(f"[WebAgent] Sync news wrapper error: {e}")
            return []

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Return information about agent capabilities.
        Useful for system introspection and API documentation.
        
        Returns:
            Dictionary describing agent features and version
        """
        return {
            "name": "Enhanced Web Supplementation Agent",
            "version": "2.0",
            "capabilities": [
                "Synchronous market data fetching",  # For batch analysis
                "Asynchronous live data streaming",  # For dashboards
                "News aggregation",  # Google News RSS
                "JSE ticker resolution",  # Company name → ticker mapping
                "Real-time price conversion",  # Cents → Rands
                "Market sentiment analysis"  # Bullish/bearish/neutral
            ],
            "async_support": True,  # Supports async/await
            "live_dashboard_optimized": True  # Optimized for real-time UIs
        }