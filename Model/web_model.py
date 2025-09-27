# Model/web_model.py - ENHANCED WITH ANTI-HALLUCINATION MEASURES

import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import investpy  # For dynamic symbol lookup


class WebSupplementationAgent:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Cache configuration
        self.cache_dir = "cache"
        self.cache_ttl_hours = 6
        self._ensure_cache_dir()

        # Dynamic JSE symbols cache
        self.jse_symbols_cache = None
        self.symbols_last_updated = None
        self.symbols_cache_ttl = 24  # hours

        # Initialize summarization model with fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None
            ).to(self.device)
            print(f"[WebSupplementationAgent] Loaded summarization model: {model_name} on {self.device}")
        except Exception as e:
            print(f"[WebSupplementationAgent] Error loading summarization model {model_name}: {e}")
            print("[WebSupplementationAgent] Operating without summarization model")
            self.model = None
            self.tokenizer = None

    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_key(self, symbol: str, data_type: str = "stock") -> str:
        today = datetime.now().date().isoformat()
        content = f"{data_type}_{symbol}_{today}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - mod_time < timedelta(hours=self.cache_ttl_hours):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[WebSupplementationAgent] Cache read error: {e}")
                    return None
        return None

    def _cache_set(self, cache_key: str, data: Dict[str, Any]):
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[WebSupplementationAgent] Cache write error: {e}")

    # ========== DYNAMIC SYMBOL MANAGEMENT ==========
    def get_jse_symbols(self, force_refresh: bool = False) -> List[str]:
        """Get JSE symbols dynamically from investpy with fallback"""
        # Check if cache is still valid
        if (not force_refresh and self.jse_symbols_cache and 
            self.symbols_last_updated and 
            (datetime.now() - self.symbols_last_updated) < timedelta(hours=self.symbols_cache_ttl)):
            return self.jse_symbols_cache

        try:
            print("[WebSupplementationAgent] Fetching JSE symbols from investpy...")
            # Try to get symbols from investpy
            stocks = investpy.get_stocks(country='south africa')
            jse_symbols = stocks['symbol'].tolist()
            
            # Filter and clean symbols
            valid_symbols = [sym for sym in jse_symbols if isinstance(sym, str) and len(sym) <= 5 and sym.isalpha()]
            
            # Add major JSE stocks as fallback if investpy fails
            if not valid_symbols:
                valid_symbols = ['AGL', 'SBK', 'NPN', 'FSR', 'BIL', 'GFI', 'NED', 'MTN', 'VOD', 'SHP', 'WHL', 'TRU']
            
            self.jse_symbols_cache = valid_symbols[:50]  # Limit to top 50
            self.symbols_last_updated = datetime.now()
            
            print(f"[WebSupplementationAgent] Loaded {len(self.jse_symbols_cache)} JSE symbols")
            return self.jse_symbols_cache
            
        except Exception as e:
            print(f"[WebSupplementationAgent] Error fetching symbols from investpy: {e}")
            # Fallback to known symbols
            fallback_symbols = ['AGL', 'SBK', 'NPN', 'FSR', 'BIL', 'GFI', 'NED', 'MTN', 'VOD', 'SHP']
            self.jse_symbols_cache = fallback_symbols
            self.symbols_last_updated = datetime.now()
            return fallback_symbols

    def is_valid_jse_symbol(self, symbol: str) -> bool:
        """Check if symbol is a valid JSE symbol"""
        valid_symbols = self.get_jse_symbols()
        return symbol.upper() in [s.upper() for s in valid_symbols]

    def extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract valid JSE symbols from query text"""
        valid_symbols = self.get_jse_symbols()
        found_symbols = []
        query_upper = query.upper()
        
        for symbol in valid_symbols:
            if symbol.upper() in query_upper:
                found_symbols.append(symbol)
        
        return found_symbols[:3]  # Limit to 3 symbols

    # ========== ENHANCED STOCK DATA WITH DATE VALIDATION ==========
    def fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock data with date validation and fallback"""
        if not self.is_valid_jse_symbol(symbol):
            return self._create_error_stock_data(symbol, f"Invalid JSE symbol: {symbol}")

        jse_symbol = f"{symbol}.JO"
        cache_key = self._get_cache_key(jse_symbol, "stock")

        # Check cache first with date validation
        cached = self._cache_get(cache_key)
        if cached and self._is_data_current(cached.get('last_updated')):
            print(f"[WebSupplementationAgent] Using cached data for {jse_symbol}")
            return cached

        try:
            print(f"[WebSupplementationAgent] Fetching fresh data for {jse_symbol}")
            ticker = yf.Ticker(jse_symbol)
            
            # Try different periods to get current data
            for period in ["1d", "5d", "1mo"]:
                hist = ticker.history(period=period)
                if not hist.empty and len(hist) > 1:
                    break
            
            if hist.empty:
                # Try to use cached data even if stale
                if cached:
                    print(f"[WebSupplementationAgent] Using stale cached data for {jse_symbol}")
                    cached['data_status'] = 'stale'
                    return cached
                raise ValueError("No historical data available")

            # Validate data is current (within 2 days for market data)
            latest_date = hist.index[-1].date()
            current_date = datetime.now().date()
            days_diff = (current_date - latest_date).days
            
            if days_diff > 2:
                print(f"[WebSupplementationAgent] Data for {symbol} is {days_diff} days old")
            
            # Calculate metrics
            current_price = float(hist["Close"][-1])
            price_1d_ago = float(hist["Close"][-2]) if len(hist) >= 2 else current_price
            price_1w_ago = float(hist["Close"][-6]) if len(hist) >= 6 else price_1d_ago
            price_1m_ago = float(hist["Close"][0]) if len(hist) >= 20 else price_1w_ago
            
            daily_change = ((current_price / price_1d_ago) - 1) * 100 if price_1d_ago != current_price else 0.0
            weekly_change = ((current_price / price_1w_ago) - 1) * 100 if price_1w_ago != current_price else 0.0
            monthly_change = ((current_price / price_1m_ago) - 1) * 100 if price_1m_ago != current_price else 0.0

            # Get company info
            info = ticker.info
            pe_ratio = info.get('trailingPE', None)
            
            stock_data = {
                "symbol": symbol.upper(),  # Force uppercase for consistency
                "jse_code": jse_symbol,
                "company_name": info.get('longName', symbol),
                "sector": info.get('sector', 'Unknown'),
                "current_price": round(current_price, 2),
                "daily_change": round(daily_change, 2),
                "weekly_change": round(weekly_change, 2),
                "monthly_change": round(monthly_change, 2),
                "currency": "ZAR",
                "source": "Yahoo Finance",
                "data_date": latest_date.isoformat(),
                "data_freshness": "current" if days_diff <= 2 else f"{days_diff} days old",
                "last_updated": datetime.now().isoformat(),
                "data_status": "fresh",
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else "N/A"
            }

            self._cache_set(cache_key, stock_data)
            return stock_data

        except Exception as e:
            print(f"[WebSupplementationAgent] Error fetching {jse_symbol}: {e}")
            # Return cached data if available, even if stale
            if cached:
                print(f"[WebSupplementationAgent] Using cached fallback for {jse_symbol}")
                cached['data_status'] = 'fallback'
                return cached
            return self._create_error_stock_data(symbol, str(e))

    def _is_data_current(self, last_updated: str) -> bool:
        """Check if data is current enough to use"""
        if not last_updated:
            return False
        
        try:
            last_update = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            return (datetime.now() - last_update) < timedelta(hours=self.cache_ttl_hours)
        except:
            return False

    def _create_error_stock_data(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create error stock data"""
        return {
            "symbol": symbol.upper(),
            "jse_code": f"{symbol}.JO",
            "company_name": f"{symbol} (Error)",
            "current_price": "N/A",
            "daily_change": "N/A",
            "weekly_change": "N/A",
            "monthly_change": "N/A",
            "currency": "ZAR",
            "source": "Error",
            "error": error,
            "data_status": "error",
            "last_updated": datetime.now().isoformat()
        }

    def _get_overall_data_status(self, stock_data: List[Dict]) -> str:
        """Determine overall data freshness status"""
        if not stock_data:
            return "no_data"
        
        statuses = [s.get('data_status', 'unknown') for s in stock_data]
        
        if any(s == 'error' for s in statuses):
            return "partial_error"
        elif any(s == 'stale' for s in statuses):
            return "partially_stale"
        elif all(s == 'fresh' for s in statuses):
            return "fresh"
        else:
            return "mixed"

    # ========== ENHANCED CONTEXT METHODS ==========
    def get_jse_companies_financials(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Fetch data for multiple tickers with validation"""
        results = []
        for ticker in tickers:
            if ticker and self.is_valid_jse_symbol(ticker):
                data = self.fetch_stock_data(ticker)
                results.append(data)
            else:
                print(f"[WebSupplementationAgent] Skipping invalid ticker: {ticker}")
        return results

    def get_api_context(self, query: str, tickers: List[str] = None) -> Dict[str, Any]:
        """CRITICAL METHOD - Get comprehensive API context with symbol validation"""
        if tickers is None:
            tickers = self.extract_symbols_from_query(query)
        else:
            # Validate provided tickers against investpy list
            tickers = [t for t in tickers if self.is_valid_jse_symbol(t)]
        
        stock_data = self.get_jse_companies_financials(tickers)
        
        # Add current time and date for the model to use
        current_time = datetime.now()
        
        return {
            "stock_data": stock_data,
            "query_tickers": tickers,
            "symbols_used": [s['symbol'] for s in stock_data if s.get('symbol') and not s.get('error')],
            "current_date": current_time.strftime("%Y-%m-%d"),
            "current_time": current_time.strftime("%H:%M"),
            "market_day": current_time.strftime("%A"),
            "data_timestamp": current_time.isoformat(),
            "data_status": self._get_overall_data_status(stock_data)
        }

    def get_relevant_info(self, user_query: str, tickers: List[str]) -> List[Dict[str, Any]]:
        """Get relevant info with symbol validation"""
        if tickers:
            # Validate tickers
            tickers = [t for t in tickers if self.is_valid_jse_symbol(t)]
        
        financial_data = self.get_jse_companies_financials(tickers) if tickers else []
        
        # If no specific tickers, provide some general JSE market context
        if not tickers or not financial_data:
            print("[WebSupplementationAgent] No specific tickers, using default JSE stocks")
            default_tickers = ['SBK', 'AGL', 'NPN']  # Major JSE stocks for context
            financial_data = self.get_jse_companies_financials(default_tickers)
        
        enhanced_articles = []
        for data in financial_data:
            if data.get('error'):
                continue  # Skip error responses
                
            # Create article-like structure from stock data
            article = {
                "headline": f"{data.get('symbol')} - {data.get('company_name', 'JSE Stock')}",
                "summary": f"Price: ZAR {data.get('current_price', 'N/A')} ({data.get('daily_change', 'N/A')}% daily)",
                "source": data.get('source', 'Yahoo Finance'),
                "symbol": data.get('symbol'),
                "current_price": data.get('current_price'),
                "data_status": data.get('data_status', 'unknown'),
                "data_date": data.get('data_date', 'Unknown')
            }
            enhanced_articles.append(article)
        
        return enhanced_articles

    def _summarize_content(self, text: str, max_length: int = 150) -> str:
        """Use the summarization model to summarize content"""
        if not self.model or not self.tokenizer:
            return text[:max_length] + "..." if len(text) > max_length else text
            
        try:
            # Prepare input for summarization
            input_text = f"summarize: {text}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"[WebSupplementationAgent] Summarization error: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text

    def get_api_context(self, query: str, tickers: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive API context with symbol validation"""
        print(f"[WebSupplementationAgent] get_api_context called with query: {query}")
    
        if tickers is None:
            # Extract tickers from query
            tickers = []
            query_upper = query.upper()
            common_jse = ['MTN', 'AGL', 'SBK', 'NPN', 'FSR', 'BIL', 'GFI', 'NED', 'VOD', 'SHP']
            for symbol in common_jse:
                if symbol in query_upper:
                    tickers.append(symbol)
    
        print(f"[WebSupplementationAgent] Using tickers: {tickers}")
    
        # Get stock data
        stock_data = []
        for ticker in tickers[:3]:  # Limit to 3
            try:
                data = self.fetch_stock_data(ticker)
                if not data.get('error'):
                    stock_data.append(data)
                    print(f"[WebSupplementationAgent] Got data for {ticker}: R{data.get('current_price', 'N/A')}")
            except Exception as e:
                print(f"[WebSupplementationAgent] Error fetching {ticker}: {e}")
    
        # Add current time
        current_time = datetime.now()
    
        result = {
            "stock_data": stock_data,
            "query_tickers": tickers,
            "symbols_used": [s['symbol'] for s in stock_data if s.get('symbol')],
            "current_date": current_time.strftime("%Y-%m-%d"),
            "current_time": current_time.strftime("%H:%M"),
            "market_day": current_time.strftime("%A"),
            "data_timestamp": current_time.isoformat(),
            "data_status": "fresh" if stock_data else "no_data"
        }
    
        print(f"[WebSupplementationAgent] Returning context with {len(stock_data)} stocks")
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model info with symbol data"""
        symbol_count = len(self.get_jse_symbols()) if self.jse_symbols_cache else 0
        
        return {
            "model_name": self.model_name,
            "jse_symbols_available": symbol_count,
            "symbols_last_updated": self.symbols_last_updated.isoformat() if self.symbols_last_updated else "Never",
            "cache_status": "Active",
            "features": [
                "Dynamic JSE symbol lookup via investpy",
                "Date-validated stock data",
                "Intelligent caching with freshness tracking",
                "Symbol validation and correction",
                "Anti-hallucination measures"
            ],
            "status": "Ready"
        }
