# Model/web_model.py
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from urllib.parse import urljoin, urlparse, quote_plus
import time
import yfinance as yf
from typing import List, Dict, Any, Tuple
import feedparser
from datetime import datetime, timedelta

class WebSupplementationAgent:
    def __init__(self, model_name="google/flan-t5-small"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.data_source = "Yahoo Finance"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            print(f"[WebSupplementationAgent] Loaded summarization model: {model_name}")
        except Exception as e:
            print(f"[WebSupplementationAgent] Error loading model {model_name}: {e}")
            self.model = None
            self.tokenizer = None

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        self.jse_ticker_list = [
            'AGL', 'AMS', 'ANG', 'APN', 'ARI', 'BHP', 'BID', 'BTI', 'BVT', 'CFR', 
            'CLS', 'CPI', 'DSY', 'EXX', 'FSR', 'GFI', 'GLN', 'GRT', 'HAR', 'IMP', 
            'INL', 'INP', 'MCG', 'MNP', 'MRP', 'MTN', 'NED', 'NPH', 'NPN', 'NRP', 
            'OMU', 'OUT', 'PIK', 'PPH', 'PRX', 'REM', 'RLO', 'SBK', 'SHP', 'SLM', 
            'SOL', 'SSW', 'TFG', 'TRU', 'VOD', 'WHL'
        ]
        
        # *** FINAL REFINEMENT: Standardized to official JSE names ***
        self.ticker_to_company = {
            'AGL': 'Anglo American plc', 'SBK': 'Standard Bank Group Limited', 'NPN': 'Naspers Limited',
            'MTN': 'MTN Group Limited', 'BHP': 'BHP Group Limited', 'BTI': 'British American Tobacco p.l.c.',
            'GFI': 'Gold Fields Limited', 'SOL': 'Sasol Limited', 'VOD': 'Vodacom Group Limited', 'SHP': 'Shoprite Holdings Limited',
            'TFG': 'The Foschini Group Limited', 'WHL': 'Woolworths Holdings Limited', 'FSR': 'FirstRand Limited',
            'NED': 'Nedbank Group Limited', 'IMP': 'Impala Platinum Holdings Limited', 'AMS': 'Anglo American Platinum Limited',
            'CPI': 'Capitec Bank Holdings Limited', 'MNP': 'Mondi plc', 'NRP': 'NEPI Rockcastle plc',
            'REM': 'Remgro Limited', 'SSW': 'Sibanye Stillwater Limited', 'ANG': 'AngloGold Ashanti plc',
            'BID': 'Bid Corporation Limited', 'BVT': 'The Bidvest Group Limited', 'CFR': 'Compagnie Financière Richemont SA',
            'CLS': 'Clicks Group Limited', 'DSY': 'Discovery Limited', 'EXX': 'Exxaro Resources Limited',
            'GLN': 'Glencore plc', 'GRT': 'Growthpoint Properties Limited', 'HAR': 'Harmony Gold Mining Company Limited',
            'INL': 'Investec Limited', 'INP': 'Investec Property Fund Limited', 'MCG': 'MultiChoice Group Limited',
            'MRP': 'Mr Price Group Limited', 'NPH': 'Northam Platinum Holdings Limited', 'OMU': 'Old Mutual Limited',
            'OUT': 'OUTsurance Group Limited', 'PIK': "Pick n Pay Stores Limited", 'PPH': 'Pepkor Holdings Limited',
            'PRX': 'Prosus N.V.', 'RLO': 'Reunert Limited', 'SLM': 'Sanlam Limited', 'TRU': 'Truworths International Limited',
            'APN': 'Aspen Pharmacare Holdings Limited', 'ARI': 'African Rainbow Minerals Limited'
        }
        
        self.company_to_ticker = {name.lower(): ticker for ticker, name in self.ticker_to_company.items()}

    def _extract_stock_symbols(self, text: str) -> Tuple[List[str], List[str]]:
        text_upper = text.upper()
        potential_tickers = set(re.findall(r'\b[A-Z]{3,5}\b', text_upper))
        valid_jse_tickers = {t for t in potential_tickers if t in self.jse_ticker_list}
        
        for name, ticker in self.company_to_ticker.items():
            if re.search(r'\b' + re.escape(name) + r'\b', text.lower()):
                valid_jse_tickers.add(ticker)

        invalid_tickers = list(potential_tickers - valid_jse_tickers)
        print(f"[WebAgent] Found Tickers: {list(valid_jse_tickers)}")
        return list(valid_jse_tickers), invalid_tickers

    def get_structured_market_data(self, tickers: List[str]) -> str:
        if not tickers: return ""
        
        jse_tickers = [ticker.upper() + ".JO" for ticker in tickers]
        data_parts = []
        try:
            data = yf.download(jse_tickers, period="1mo", progress=False, group_by='ticker', threads=True)
            if data.empty: return ""
            
            for ticker_jo in jse_tickers:
                stock_code = ticker_jo.replace(".JO", "")
                try:
                    stock_data = data[ticker_jo].dropna() if len(jse_tickers) > 1 else data.dropna()
                    if len(stock_data) < 2: continue
                    
                    latest, previous = stock_data.iloc[-1], stock_data.iloc[-2]
                    
                    close_price_rand = latest['Close'] / 100.0
                    absolute_change_rand = (latest['Close'] - previous['Close']) / 100.0
                    change_percent = (absolute_change_rand / (previous['Close'] / 100.0)) * 100
                    change_str = f"R{absolute_change_rand:+.2f} ({change_percent:+.2f}%)"
                    
                    long_name = self.ticker_to_company.get(stock_code, stock_code)
                    
                    data_parts.append(
                        f"- {long_name} ({stock_code}): Price R{close_price_rand:.2f}, Change {change_str} (Source: Yahoo Finance)"
                    )
                except Exception as e:
                    print(f"[WebAgent] Error processing {ticker_jo}: {e}")
            
            return "\n".join(data_parts)
        except Exception as e:
            print(f"[WebAgent] yfinance download error: {e}")
            return ""

    def fetch_articles_from_google_rss(self, tickers: List[str], max_per_ticker: int = 2) -> List[Dict[str, Any]]:
        all_articles = []
        for ticker in tickers:
            company_name = self.ticker_to_company.get(ticker, ticker)
            search_query = f'"{company_name}" OR "{ticker}" stock JSE'
            try:
                encoded_query = quote_plus(search_query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-ZA&gl=ZA&ceid=ZA:en"
                
                response = requests.get(rss_url, timeout=10, headers=self.headers)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                if not feed.entries:
                    time.sleep(1)
                    continue

                for entry in feed.entries[:max_per_ticker]:
                    title = re.sub(r'\s+', ' ', entry.title.strip())
                    source = entry.get('source', {}).get('title', 'News Source')
                    summary = entry.get('summary', title)
                    summary = re.sub(r'<[^>]+>', '', summary).strip()
                    
                    all_articles.append({'headline': title, 'ticker': ticker, 'source': source, 'summary': summary})
                
                time.sleep(1)
            except Exception as e:
                print(f"[WebAgent] Error fetching RSS for {ticker}: {e}")
        
        print(f"[WebAgent] Total articles fetched: {len(all_articles)}")
        return all_articles

    def get_relevant_info(self, user_query: str, max_articles: int = 4) -> Dict[str, Any]:
        valid_tickers, _ = self._extract_stock_symbols(user_query)
        if not valid_tickers:
            return {"articles": [], "market_data": "", "tickers_analyzed": []}
        
        market_data_str = self.get_structured_market_data(valid_tickers)
        articles = self.fetch_articles_from_google_rss(valid_tickers, max_per_ticker=2)
        
        return {"articles": articles, "market_data": market_data_str, "tickers_analyzed": valid_tickers}

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name if self.model else "RSS-based"}