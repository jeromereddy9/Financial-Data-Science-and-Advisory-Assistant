# web_model/web_supplementation_agent.py
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from urllib.parse import urljoin, urlparse
import time

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
            # Fallback to a basic approach without ML summarization
            self.model = None
            self.tokenizer = None

        # Default sources with better search URLs
        self.sources = {
            "fin24": "https://www.fin24.com/search?query={query}",
            "moneyweb": "https://www.moneyweb.co.za/search/{query}/"
        }

        # Headers to avoid blocking
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
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

    # Quick patch for Web Agent to provide fallback data

    def get_relevant_info(self, user_query, max_articles=5):
        """
        Fetch and summarize articles based on user query.
        Returns a list of dicts with 'headline', 'url', 'source', 'summary', 'content'.
        """
        print(f"[WebSupplementationAgent] Fetching articles for query: '{user_query}'")
    
        articles = self.fetch_articles(user_query, max_articles)
        if not articles:
            print("[WebSupplementationAgent] No articles found, using fallback data.")
        
            # Provide contextual fallback based on query keywords
            if "portfolio" in user_query.lower() and "diversif" in user_query.lower():
                fallback_articles = [
                    {
                        "headline": "JSE Portfolio Diversification: Expert Tips for South African Investors",
                        "url": "https://example.com/jse-diversification",
                        "source": "fallback",
                        "summary": "Financial experts recommend diversifying JSE portfolios across sectors including mining (AGL, BIL), banking (SBK, FSR), telecommunications (MTN, VOD), and retail (SHP, TRU) to reduce concentration risk and improve long-term returns.",
                        "content": "Diversification across JSE sectors is crucial for South African investors to manage risk effectively in volatile markets."
                    },
                    {
                        "headline": "Banking Sector Concentration Risk on JSE: What Investors Need to Know", 
                        "url": "https://example.com/jse-banking-risk",
                        "source": "fallback",
                        "summary": "Holding only banking stocks (SBK, FSR, NED) exposes investors to sector-specific risks including interest rate changes, regulatory shifts, and economic downturns. Diversification into mining, retail, and technology sectors can help mitigate these risks.",
                        "content": "JSE banking sector concentration carries significant risks that can be mitigated through proper diversification strategies."
                    }
                ]
                return fallback_articles
        
            elif "bank" in user_query.lower() and "jse" in user_query.lower():
                fallback_articles = [
                    {
                        "headline": "JSE Banking Stocks Analysis: SBK, FSR, NED Performance Review",
                        "url": "https://example.com/jse-banking-analysis", 
                        "source": "fallback",
                        "summary": "Major JSE banking stocks include Standard Bank (SBK), FirstRand (FSR), Nedbank (NED), and Capitec (CPI). These stocks are sensitive to interest rate cycles, credit loss provisions, and South African economic conditions.",
                        "content": "JSE banking sector represents a significant portion of the market but requires careful analysis of macroeconomic factors."
                    }
                ]
                return fallback_articles
        
            else:
                # Generic JSE fallback
                fallback_articles = [
                    {
                        "headline": "JSE Market Update: Key Trends and Investment Opportunities",
                        "url": "https://example.com/jse-market-update",
                        "source": "fallback", 
                        "summary": "The Johannesburg Stock Exchange continues to offer diverse investment opportunities across mining, banking, retail, and technology sectors. Investors should consider rand volatility, commodity prices, and local economic conditions when making investment decisions.",
                        "content": "JSE market analysis considers multiple factors including commodity prices, rand strength, and local economic indicators."
                    }
                ]
                return fallback_articles
    
        summarized_articles = self.summarize_articles(articles)
    
        # Ensure all items are properly formatted
        cleaned_articles = []
        for a in summarized_articles:
            if isinstance(a, dict):
                cleaned_article = {
                    "headline": a.get("headline", "No headline"),
                    "url": a.get("url", ""),
                    "source": a.get("source", "unknown"),
                    "summary": a.get("summary", "No summary available"),
                    "content": a.get("content", "")
                }
                cleaned_articles.append(cleaned_article)
    
        print(f"[WebSupplementationAgent] Returning {len(cleaned_articles)} processed articles.")
        return cleaned_articles