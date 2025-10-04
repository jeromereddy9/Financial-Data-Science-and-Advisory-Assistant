# web_model/web_supplementation_agent.py
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from urllib.parse import urljoin, urlparse
import time
import yfinance as yf
from typing import List, Dict, Any, Tuple

class WebSupplementationAgent:
    def __init__(self, model_name="google/flan-t5-small"):
        # Using FLAN-T5 for summarization - it's not gated and good at text summarization
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # **FIX**: Define the data source here so it's easy to change later.
        self.data_source = "Yahoo Finance"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
        except Exception as e:
            print(f"[WebSupplementationAgent] Error loading model {model_name}: {e}")
            self.model = None
            self.tokenizer = None

        self.sources = {
            "fin24": "https://www.fin24.com/search?query={query}",
            "moneyweb": "https://www.moneyweb.co.za/search/{query}/"
        }

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

    def _extract_stock_symbols(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extracts potential stock symbols and partitions them into valid JSE tickers and invalid/non-JSE tickers.
        """
        potential_tickers = set(re.findall(r'\b[A-Z]{3,5}\b', text.upper()))
        
        valid_jse_tickers = [ticker for ticker in potential_tickers if ticker in self.jse_ticker_list]
        invalid_tickers = [ticker for ticker in potential_tickers if ticker not in self.jse_ticker_list]
        
        print(f"[WebSupplementationAgent] Found valid JSE tickers: {valid_jse_tickers}")
        if invalid_tickers:
            print(f"[WebSupplementationAgent] Found invalid/non-JSE tickers: {invalid_tickers}")
            
        return valid_jse_tickers, invalid_tickers

    def get_structured_market_data(self, tickers: List[str], invalid_tickers: List[str] = None) -> str:
        """
        **MODIFIED**: Fetches data including absolute and percentage change, corrects for
        JSE prices in cents, and provides a clean, LLM-ready string.
        """
        data_str = ""
        
        if tickers:
            jse_tickers = [ticker.upper() + ".JO" for ticker in tickers]
            data_str += "LATEST JSE MARKET DATA:\n"
            try:
                # Use a 5-day period to ensure we can calculate a percentage change.
                data = yf.download(jse_tickers, period="5d", progress=False, auto_adjust=True, group_by='ticker')
                
                if not data.empty:
                    for ticker_jo in jse_tickers:
                        stock_code = ticker_jo.replace(".JO", "")
                        try:
                            stock_data = data[ticker_jo].dropna()
                            if len(stock_data) >= 2:
                                latest = stock_data.iloc[-1]
                                previous = stock_data.iloc[-2]
                                date_updated = latest.name.strftime('%Y-%m-%d')
                                
                                # **FIX**: Calculate both percentage and absolute change.
                                change_percent = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
                                absolute_change = latest['Close'] - previous['Close']
                                
                                info = yf.Ticker(ticker_jo).info
                                long_name = info.get('longName', stock_code)

                                # **FIX**: JSE prices are in cents. Convert to Rands for clarity.
                                close_price_rand = latest['Close'] / 100
                                change_rand = absolute_change / 100
                                
                                # **FIX**: Modified the output string to include all necessary data points.
                                data_str += (
                                    f"- {stock_code} ({long_name}): Last Price: R{close_price_rand:.2f}, "
                                    f"Day's Change: {change_rand:+.2f} ({change_percent:+.2f}%) "
                                    f"(Source: {self.data_source} as of {date_updated})\n"
                                )
                            else:
                                data_str += f"- {stock_code}: Not enough data to calculate performance.\n"
                        except Exception as e:
                            print(f"[WebSupplementationAgent] Error processing data for {ticker_jo}: {e}")
                            data_str += f"- {stock_code}: Could not process data for this ticker.\n"
            except Exception as e:
                print(f"[WebSupplementationAgent] yfinance download error: {e}")
                data_str += "An error occurred while fetching JSE market data.\n"
        
        if invalid_tickers:
            data_str += f"\nNOTICE: The following tickers were ignored as they are not recognized as supported JSE stocks: {', '.join(invalid_tickers)}."
        
        if not data_str.strip():
            return "No market data could be retrieved for the query."

        return data_str

    def get_relevant_info(self, user_query: str, max_articles: int = 3) -> Dict[str, Any]:
        """
        Main entry point now handles both valid and invalid tickers gracefully.
        """
        print(f"[WebSupplementationAgent] Fetching external context for query: '{user_query}'")
        
        articles = self.fetch_articles(user_query, max_articles)
        summarized_articles = self.summarize_articles(articles)
        
        valid_tickers, invalid_tickers = self._extract_stock_symbols(user_query)
        
        market_data_str = self.get_structured_market_data(valid_tickers, invalid_tickers)
        
        return {
            "articles": summarized_articles,
            "market_data": market_data_str
        }

    def _clean_url(self, url, base_url):
        """Clean and make URLs absolute"""
        if not url:
            return ""
        
        url = url.strip()
        
        if url.startswith('/'):
            parsed_base = urlparse(base_url)
            url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        elif not url.startswith('http'):
            url = urljoin(base_url, url)
        
        return url

    def _extract_article_content(self, url, source):
        """Extract main article content from a URL"""
        try:
            time.sleep(1)
            resp = requests.get(url, timeout=10, headers=self.headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            content = ""
            
            article_selectors = [
                "article", ".article-content", ".post-content", ".content",
                "[class*='article']", "[class*='story']", "main"
            ]
            
            for selector in article_selectors:
                article_elem = soup.select_one(selector)
                if article_elem:
                    paragraphs = article_elem.find_all(['p', 'div'], recursive=True)
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    break
            
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            content = re.sub(r'\s+', ' ', content)
            content = content[:2000]
            
            return content if len(content) > 100 else ""
            
        except Exception as e:
            print(f"[WebSupplementationAgent] Error extracting content from {url}: {e}")
            return ""

    def fetch_articles(self, query, max_articles=5):
        """
        Pull articles from predefined sources based on user query.
        """
        results = []
        query_formatted = query.replace(" ", "+")
        
        for source, url_template in self.sources.items():
            search_url = url_template.format(query=query_formatted)
            try:
                resp = requests.get(search_url, timeout=10, headers=self.headers)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                articles = []

                if source == "fin24":
                    article_links = soup.find_all('a', href=True)
                    for link in article_links[:max_articles]:
                        href = link.get('href', '')
                        if '/news/' in href or '/markets/' in href or '/economy/' in href:
                            headline = link.get_text(strip=True)
                            if headline and len(headline) > 10:
                                full_url = self._clean_url(href, "https://www.fin24.com")
                                articles.append({"headline": headline, "url": full_url, "source": source})
                                if len(articles) >= max_articles: break

                elif source == "moneyweb":
                    article_links = soup.find_all('a', href=True)
                    for link in article_links[:max_articles]:
                        href = link.get('href', '')
                        if 'moneyweb.co.za' in href and ('/news/' in href or '/investing/' in href or '/markets/' in href):
                            headline = link.get_text(strip=True)
                            if headline and len(headline) > 10:
                                full_url = self._clean_url(href, "https://www.moneyweb.co.za")
                                articles.append({"headline": headline, "url": full_url, "source": source})
                                if len(articles) >= max_articles: break

                for article in articles:
                    if article['url']:
                        content = self._extract_article_content(article['url'], source)
                        article['content'] = content
                    else:
                        article['content'] = ""
                
                articles = [a for a in articles if a.get('content', '') and len(a['content']) > 100]
                results.extend(articles)
                
            except requests.exceptions.HTTPError as e:
                print(f"[WebSupplementationAgent] HTTP Error {e.response.status_code} from {source}: {e}")
            except Exception as e:
                print(f"[WebSupplementationAgent] Error fetching from {source}: {e}")

        return results[:max_articles]

    def summarize_articles(self, articles, max_length=150):
        """
        Summarize a list of articles using the LM or fallback method.
        """
        summaries = []
        
        for article in articles:
            content = article.get("content", "")
            headline = article.get("headline", "")
            
            if not content and not headline:
                continue
                
            try:
                text_to_summarize = content if content else headline
                
                if self.model and self.tokenizer and len(text_to_summarize) > 100:
                    prompt = f"Summarize this financial news article: {text_to_summarize[:1500]}"
                    
                    inputs = self.tokenizer(
                        prompt, return_tensors="pt", truncation=True,
                        max_length=512, padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        summary_ids = self.model.generate(
                            inputs.input_ids, max_length=max_length, min_length=30,
                            length_penalty=2.0, num_beams=3, early_stopping=True, do_sample=False
                        )

                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    summary = summary.replace(prompt, "").strip()
                    if not summary:
                        summary = self._extractive_summary(text_to_summarize)
                        
                else:
                    summary = self._extractive_summary(text_to_summarize)
                
                article["summary"] = summary
                
            except Exception as e:
                print(f"[WebSupplementationAgent] Error summarizing article: {e}")
                article["summary"] = self._extractive_summary(content or headline)
            
            summaries.append(article)
            
        return summaries

    def _extractive_summary(self, text, max_sentences=3):
        """
        Create an extractive summary by taking the first few sentences.
        """
        if not text:
            return "No content available."
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        summary_sentences = []
        for sentence in sentences[:max_sentences]:
            if len(sentence) > 30:
                summary_sentences.append(sentence)
        
        summary = '. '.join(summary_sentences)
        return summary[:500] + "..." if len(summary) > 500 else summary