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
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Enhanced JSE company database
        self.company_to_ticker = {
            # Banking & Financial Services
            'standard bank': 'SBK.JO', 
            'firstrand': 'FSR.JO', 'fnb': 'FSR.JO',
            'nedbank': 'NED.JO', 
            'capitec': 'CPI.JO', 
            'investec': 'INL.JO',
            'absa': 'ABG.JO',
            'sanlam': 'SLM.JO',
            'discovery': 'DSY.JO',
            'old mutual': 'OMU.JO',
            
            # Telecommunications
            'mtn': 'MTN.JO', 'mtn group': 'MTN.JO',
            'vodacom': 'VOD.JO', 
            'telkom': 'TKG.JO',
            'multichoice': 'MCG.JO',
            
            # Mining & Resources
            'anglo american': 'AGL.JO', 
            'bhp': 'BHP.JO',
            'sasol': 'SOL.JO', 
            'exxaro': 'EXX.JO',
            'gold fields': 'GFI.JO',
            'harmony gold': 'HAR.JO',
            'anglogold ashanti': 'ANG.JO',
            'impala platinum': 'IMP.JO',
            'kumba iron ore': 'KIO.JO',
            
            # Retail & Consumer
            'shoprite': 'SHP.JO',
            'pick n pay': 'PIK.JO',
            'woolworths': 'WHL.JO',
            'mr price': 'MRP.JO',
            'truworths': 'TRU.JO',
            'foschini': 'TFG.JO',
            'clicks': 'CLS.JO',
            'spar': 'SPP.JO',
            
            # Industrial & Other
            'bidvest': 'BVT.JO',
            'naspers': 'NPN.JO',
            'prosus': 'PRX.JO',
            'richemont': 'CFR.JO',
            'reunert': 'RLO.JO'
        }

    def _resolve_tickers(self, query: str) -> List[Tuple[str, str]]:
        """Enhanced ticker resolution"""
        query_lower = query.lower()
        candidates = []
        
        for company_name, ticker in self.company_to_ticker.items():
            if company_name in query_lower:
                display_name = company_name.title() + " Limited"
                candidates.append((ticker, display_name))
        
        # Remove duplicates
        seen = set()
        return [(ticker, name) for ticker, name in candidates if not (ticker in seen or seen.add(ticker))]

    def _convert_price_to_rands(self, price):
        """Convert price from cents to Rands if needed"""
        if price > 1000:  # Likely in cents
            return price / 100
        return price

    def get_structured_market_data(self, companies: List[Tuple[str, str]]) -> str:
        """Optimized market data fetching"""
        if not companies:
            return ""

        market_data_parts = []
        
        for ticker, name in companies:
            try:
                stock = yf.Ticker(ticker)
                
                # Get historical data quickly
                hist = stock.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    latest_price = self._convert_price_to_rands(hist['Close'].iloc[-1])
                    previous_close = self._convert_price_to_rands(hist['Close'].iloc[-2])
                    change = latest_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    market_data_parts.append(
                        f"📊 {name} ({ticker}): "
                        f"Price: R{latest_price:.2f} | "
                        f"Change: {change:+.2f} ({change_percent:+.2f}%)"
                    )
                else:
                    # Quick fallback
                    info = stock.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if current_price:
                        current_price = self._convert_price_to_rands(current_price)
                        market_data_parts.append(f"📊 {name} ({ticker}): Price: R{current_price:.2f}")
                    
            except Exception as e:
                market_data_parts.append(f"❌ {name} ({ticker}): Data unavailable")

        return "\n".join(market_data_parts)

    def fetch_articles_from_google_rss(self, companies: List[Tuple[str, str]], max_per: int = 2) -> List[Dict[str, str]]:
        """Optimized news fetching with timeout"""
        all_articles = []
        
        for ticker, name in companies:
            try:
                search_query = f'JSE {name}'
                rss_url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=en-ZA"
                
                response = requests.get(rss_url, headers=self.headers, timeout=8)
                response.raise_for_status()
                
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries[:max_per]:
                    # Avoid duplicates
                    if not any(entry.title == existing['headline'] for existing in all_articles):
                        summary = re.sub(r'<[^>]+>', '', entry.get('summary', entry.title)).strip()
                        
                        all_articles.append({
                            "headline": entry.title,
                            "ticker": ticker,
                            "company": name,
                            "source": entry.get('source', {}).get('title', 'News'),
                            "summary": summary[:150] + "..." if len(summary) > 150 else summary,
                        })
                
            except Exception:
                continue  # Skip failed news fetches
                
        return all_articles[:6]  # Return max 6 articles

    def get_jse_market_summary(self) -> str:
        """Get JSE market summary"""
        try:
            # Use major stocks as market indicators
            major_stocks = ['SBK.JO', 'MTN.JO', 'NPN.JO', 'AGL.JO']
            up_count = 0
            down_count = 0
            price_changes = []
            
            for ticker in major_stocks:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='2d')
                    if len(hist) >= 2:
                        current = self._convert_price_to_rands(hist['Close'].iloc[-1])
                        previous = self._convert_price_to_rands(hist['Close'].iloc[-2])
                        change_percent = ((current - previous) / previous) * 100
                        price_changes.append(change_percent)
                        
                        if change_percent > 0:
                            up_count += 1
                        else:
                            down_count += 1
                except:
                    continue
            
            if price_changes:
                avg_change = sum(price_changes) / len(price_changes)
                sentiment = "🟢 Bullish" if avg_change > 0.5 else "🔴 Bearish" if avg_change < -0.5 else "🟡 Neutral"
                return f"🏛️ JSE Market: {sentiment} | Avg Change: {avg_change:+.2f}% | Advancers: {up_count}/{len(major_stocks)}"
            else:
                return "🏛️ JSE Market: Data temporarily unavailable"
                
        except Exception as e:
            return "🏛️ JSE Market: Summary unavailable"

    def get_relevant_info(self, user_query: str) -> Dict[str, Any]:
        """Optimized main method"""
        try:
            companies = self._resolve_tickers(user_query)
            market_summary = self.get_jse_market_summary()
            
            if not companies:
                return {
                    "articles": [],
                    "market_data": market_summary,
                    "tickers_analyzed": []
                }

            market_data = self.get_structured_market_data(companies)
            articles = self.fetch_articles_from_google_rss(companies)
            
            return {
                "articles": articles,
                "market_data": f"{market_summary}\n\n{market_data}",
                "tickers_analyzed": [t[0] for t in companies]
            }
            
        except Exception as e:
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
        
        Args:
            tickers: List of JSE tickers (with or without .JO suffix)
            
        Returns:
            List of stock data dictionaries
        """
        # Ensure tickers have .JO suffix
        jse_tickers = [ticker if ticker.endswith('.JO') else f"{ticker}.JO" for ticker in tickers]
        
        async with aiohttp.ClientSession():
            tasks = [self._fetch_single_stock_async(ticker) for ticker in jse_tickers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and None results
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result is not None:
                    valid_results.append(result)
            
            return valid_results

    async def _fetch_single_stock_async(self, ticker: str) -> Dict[str, Any]:
        """
        Async fetch for single stock data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get basic info and history
            info = stock.info
            hist = stock.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
            
            # Extract price data
            current_price = self._convert_price_to_rands(hist['Close'].iloc[-1])
            previous_close = self._convert_price_to_rands(info.get('previousClose', current_price))
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
            
            # Get additional metrics
            volume = info.get('volume', 0)
            market_cap = info.get('marketCap')
            day_high = self._convert_price_to_rands(info.get('dayHigh', current_price))
            day_low = self._convert_price_to_rands(info.get('dayLow', current_price))
            
            return {
                'ticker': ticker.replace('.JO', ''),
                'full_ticker': ticker,
                'current_price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'previous_close': round(previous_close, 2),
                'volume': volume,
                'market_cap': market_cap,
                'day_high': round(day_high, 2),
                'day_low': round(day_low, 2),
                'timestamp': datetime.now().isoformat(),
                'status': 'up' if change > 0 else 'down' if change < 0 else 'flat'
            }
            
        except Exception as e:
            print(f"[WebAgent] Async fetch error for {ticker}: {e}")
            return None

    async def get_live_news_async(self, tickers: List[str], max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Get live news asynchronously for multiple stocks
        
        Args:
            tickers: List of stock tickers
            max_articles: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for ticker in tickers:
                task = self._fetch_news_for_ticker_async(session, ticker, max_per=3)
                tasks.append(task)
            
            all_articles = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten and deduplicate
            flattened_articles = []
            for article_list in all_articles:
                if isinstance(article_list, list):
                    flattened_articles.extend(article_list)
            
            # Remove duplicates by headline
            unique_articles = []
            seen_headlines = set()
            
            for article in flattened_articles:
                if article['headline'] not in seen_headlines:
                    seen_headlines.add(article['headline'])
                    unique_articles.append(article)
            
            return sorted(unique_articles, 
                        key=lambda x: x.get('published', ''), 
                        reverse=True)[:max_articles]

    async def _fetch_news_for_ticker_async(self, session: aiohttp.ClientSession, 
                                         ticker: str, max_per: int = 3) -> List[Dict[str, Any]]:
        """
        Async fetch news for a single ticker
        """
        try:
            company_name = ticker.replace('.JO', '')
            search_query = f'JSE {company_name} stock'
            rss_url = f"https://news.google.com/rss/search?q={quote_plus(search_query)}&hl=en-ZA"
            
            async with session.get(rss_url, headers=self.headers, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries[:max_per]:
                        summary = re.sub(r'<[^>]+>', '', entry.get('summary', entry.title)).strip()
                        
                        articles.append({
                            "headline": entry.title,
                            "ticker": ticker,
                            "company": company_name,
                            "source": entry.get('source', {}).get('title', 'Google News'),
                            "summary": summary[:200] + "..." if len(summary) > 200 else summary,
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
        Get list of major JSE tickers for live dashboard
        """
        return [
            'MTN.JO', 'NPN.JO', 'SBK.JO', 'FSR.JO', 'AGL.JO', 
            'SOL.JO', 'VOD.JO', 'CPI.JO', 'BHP.JO', 'CFR.JO'
        ]

    # Synchronous wrapper for async methods (for compatibility)
    def get_live_stock_data_sync(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for get_live_stock_data
        """
        try:
            return asyncio.run(self.get_live_stock_data(tickers))
        except Exception as e:
            print(f"[WebAgent] Sync wrapper error: {e}")
            return []

    def get_live_news_sync(self, tickers: List[str], max_articles: int = 10) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for get_live_news_async
        """
        try:
            return asyncio.run(self.get_live_news_async(tickers, max_articles))
        except Exception as e:
            print(f"[WebAgent] Sync news wrapper error: {e}")
            return []

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """
        Return information about agent capabilities
        """
        return {
            "name": "Enhanced Web Supplementation Agent",
            "version": "2.0",
            "capabilities": [
                "Synchronous market data fetching",
                "Asynchronous live data streaming", 
                "News aggregation",
                "JSE ticker resolution",
                "Real-time price conversion",
                "Market sentiment analysis"
            ],
            "async_support": True,
            "live_dashboard_optimized": True
        }