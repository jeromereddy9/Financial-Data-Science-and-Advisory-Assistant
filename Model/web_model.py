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
        **MODIFIED**: Extracts potential stock symbols and partitions them into valid JSE tickers and invalid/non-JSE tickers.
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
        **MODIFIED**: Fetches data including the company's full name, adds a timestamp, and includes a warning for invalid tickers.
        """
        data_str = ""
        
        if tickers:
            jse_tickers = [ticker.upper() + ".JO" for ticker in tickers]
            data_str += "LATEST JSE MARKET DATA:\n"
            try:
                # Use a short period to get the latest data efficiently
                data = yf.download(jse_tickers, period="5d", progress=False, auto_adjust=True, group_by='ticker')
                
                if not data.empty:
                    for ticker_jo in jse_tickers:
                        stock_code = ticker_jo.replace(".JO", "")
                        try:
                            # Check if data for the ticker exists and is not all NaN
                            if ticker_jo in data.columns and not data[ticker_jo]['Close'].isnull().all():
                                latest = data[ticker_jo].dropna().iloc[-1]
                                date_updated = latest.name.strftime('%Y-%m-%d')
                                
                                # Fetch Ticker info for the long name
                                info = yf.Ticker(ticker_jo).info
                                long_name = info.get('longName', stock_code)
                                
                                # **MODIFIED**: Added the company's long name and timestamp to the output string.
                                data_str += (
                                    f"- {stock_code} ({long_name}): Last Price: R{latest['Close']:.2f} (as of {date_updated}), "
                                    f"Day's Range: R{latest['Low']:.2f} - R{latest['High']:.2f}, "
                                    f"Volume: {latest['Volume']:,.0f}\n"
                                )
                            else:
                                data_str += f"- {stock_code}: No recent data found. The ticker may be invalid or delisted.\n"
                        except Exception as e:
                            print(f"[WebSupplementationAgent] Error processing data for {ticker_jo}: {e}")
                            data_str += f"- {stock_code}: Could not process data for this ticker.\n"
            except Exception as e:
                print(f"[WebSupplementationAgent] yfinance download error: {e}")
                data_str += "An error occurred while fetching JSE market data.\n"
        
        # **NEW**: Add the safety net warning if any invalid tickers were found.
        if invalid_tickers:
            data_str += f"\nNOTICE: The following tickers were ignored as they are not recognized as supported JSE stocks: {', '.join(invalid_tickers)}."
        
        if not data_str.strip():
            return "No market data could be retrieved for the query."

        return data_str

    def get_relevant_info(self, user_query: str, max_articles: int = 3) -> Dict[str, Any]:
        """
        **MODIFIED**: Main entry point now handles both valid and invalid tickers gracefully.
        """
        print(f"[WebSupplementationAgent] Fetching external context for query: '{user_query}'")
        
        articles = self.fetch_articles(user_query, max_articles)
        summarized_articles = self.summarize_articles(articles)
        
        # **MODIFIED**: Capture both valid and invalid tickers.
        valid_tickers, invalid_tickers = self._extract_stock_symbols(user_query)
        
        # **MODIFIED**: Pass both lists to the data fetching method.
        market_data_str = self.get_structured_market_data(valid_tickers, invalid_tickers)
        
        return {
            "articles": summarized_articles,
            "market_data": market_data_str
        }

    def _clean_url(self, url, base_url):
        """Clean and make URLs absolute"""
        if not url:
            return ""
        
        # Remove any leading/trailing whitespace
        url = url.strip()
        
        # Make relative URLs absolute
        if url.startswith('/'):
            parsed_base = urlparse(base_url)
            url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        elif not url.startswith('http'):
            url = urljoin(base_url, url)
        
        return url

    def _extract_article_content(self, url, source):
        """Extract main article content from a URL"""
        try:
            time.sleep(1)  # Be respectful to servers
            resp = requests.get(url, timeout=10, headers=self.headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            content = ""
            
            # Try common article content selectors
            article_selectors = [
                "article", 
                ".article-content", 
                ".post-content", 
                ".content",
                "[class*='article']",
                "[class*='story']",
                "main"
            ]
            
            for selector in article_selectors:
                article_elem = soup.select_one(selector)
                if article_elem:
                    # Get all paragraph text
                    paragraphs = article_elem.find_all(['p', 'div'], recursive=True)
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    break
            
            # Fallback: get all paragraph text from body
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
            content = content[:2000]  # Limit content length
            
            return content if len(content) > 100 else ""  # Only return if substantial content
            
        except Exception as e:
            print(f"[WebSupplementationAgent] Error extracting content from {url}: {e}")
            return ""

    def fetch_articles(self, query, max_articles=5):
        """
        Pull articles from predefined sources based on user query.
        Returns a list of dicts: {headline, url, source, content}.
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

                # Source-specific article extraction
                if source == "fin24":
                    # Look for article links in search results
                    article_links = soup.find_all('a', href=True)
                    for link in article_links[:max_articles]:
                        href = link.get('href', '')
                        if '/news/' in href or '/markets/' in href or '/economy/' in href:
                            headline = link.get_text(strip=True)
                            if headline and len(headline) > 10:  # Filter out short/empty headlines
                                full_url = self._clean_url(href, "https://www.fin24.com")
                                articles.append({
                                    "headline": headline,
                                    "url": full_url,
                                    "source": source
                                })
                                if len(articles) >= max_articles:
                                    break

                elif source == "moneyweb":
                    # Look for article links in search results  
                    article_links = soup.find_all('a', href=True)
                    for link in article_links[:max_articles]:
                        href = link.get('href', '')
                        if 'moneyweb.co.za' in href and ('/news/' in href or '/investing/' in href or '/markets/' in href):
                            headline = link.get_text(strip=True)
                            if headline and len(headline) > 10:
                                full_url = self._clean_url(href, "https://www.moneyweb.co.za")
                                articles.append({
                                    "headline": headline,
                                    "url": full_url,
                                    "source": source
                                })
                                if len(articles) >= max_articles:
                                    break

                # Extract content for each article
                for article in articles:
                    if article['url']:
                        content = self._extract_article_content(article['url'], source)
                        article['content'] = content
                    else:
                        article['content'] = ""
                
                # Only keep articles with substantial content
                articles = [a for a in articles if a.get('content', '') and len(a['content']) > 100]
                results.extend(articles)
                
            except requests.exceptions.HTTPError as e:
                print(f"[WebSupplementationAgent] HTTP Error {e.response.status_code} from {source}: {e}")
            except Exception as e:
                print(f"[WebSupplementationAgent] Error fetching from {source}: {e}")

        return results[:max_articles]  # Limit total results

    def summarize_articles(self, articles, max_length=150):
        """
        Summarize a list of articles using the LM or fallback method.
        Always returns list of dicts with 'summary'.
        """
        summaries = []
        
        for article in articles:
            content = article.get("content", "")
            headline = article.get("headline", "")
            
            if not content and not headline:
                continue
                
            try:
                # Use the full content for summarization, fallback to headline
                text_to_summarize = content if content else headline
                
                if self.model and self.tokenizer and len(text_to_summarize) > 100:
                    # Use ML model for summarization
                    prompt = f"Summarize this financial news article: {text_to_summarize[:1500]}"
                    
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        summary_ids = self.model.generate(
                            inputs.input_ids,
                            max_length=max_length,
                            min_length=30,
                            length_penalty=2.0,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False
                        )

                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    # Clean up the summary
                    summary = summary.replace(prompt, "").strip()
                    if not summary:
                        summary = self._extractive_summary(text_to_summarize)
                        
                else:
                    # Fallback: extractive summary
                    summary = self._extractive_summary(text_to_summarize)
                
                article["summary"] = summary
                
            except Exception as e:
                print(f"[WebSupplementationAgent] Error summarizing article: {e}")
                # Fallback to extractive summary
                article["summary"] = self._extractive_summary(content or headline)
            
            summaries.append(article)
            
        return summaries

    def _extractive_summary(self, text, max_sentences=3):
        """
        Create an extractive summary by taking the first few sentences.
        Fallback method when ML summarization fails.
        """
        if not text:
            return "No content available."
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Take first few sentences that seem substantial
        summary_sentences = []
        for sentence in sentences[:max_sentences]:
            if len(sentence) > 30:  # Only substantial sentences
                summary_sentences.append(sentence)
        
        summary = '. '.join(summary_sentences)
        return summary[:500] + "..." if len(summary) > 500 else summary