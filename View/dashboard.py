# app.py - FIXED VERSION with independent tab refresh
import streamlit as st
import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json

# Add the project root to Python path for importing controller and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# CRITICAL: Initialize all session state variables at the top
# Session state persists across Streamlit reruns (button clicks, interactions)
# Must be initialized before ANY Streamlit commands
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False  # System initialization flag
if 'controller' not in st.session_state:
    st.session_state.controller = None  # Main controller instance
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # Chat history
if 'last_market_refresh' not in st.session_state:
    st.session_state.last_market_refresh = None  # Track market data refresh time
if 'last_news_refresh' not in st.session_state:
    st.session_state.last_news_refresh = None  # Track news refresh time
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()  # Cached stock data
if 'news_data' not in st.session_state:
    st.session_state.news_data = []  # Cached news articles
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""  # Current user input
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "🎯 Advisor Chat"  # Active tab tracker

# Set page config FIRST - must be the first Streamlit command after imports
# Configures browser tab, layout, and sidebar behavior
st.set_page_config(
    page_title="JSE Financial Advisor",
    page_icon="📈",
    layout="wide",  # Use full browser width
    initial_sidebar_state="expanded"  # Show sidebar on load
)

class JSEAdvisoryApp:
    """
    Main Streamlit application class for JSE Financial Advisory System.
    Provides four main interfaces:
    1. Advisor Chat - LLM-powered financial advice
    2. Live Markets - Real-time JSE stock data with independent refresh
    3. News Feed - Aggregated financial news with independent refresh
    4. Analysis Tools - Data visualization and technical analysis
    """
    
    def __init__(self):
        """Initialize the application and load the financial advisory system."""
        self.controller = None
        self.initialize_system()
        
    def initialize_system(self):
        """
        Initialize the financial advisory system.
        Loads all agents (advisor, data, web, memory, summarizer) on first run.
        Uses session state to avoid reloading on every rerun (expensive operation).
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Check if already initialized to avoid redundant loading
        if st.session_state.initialized and st.session_state.controller is not None:
            self.controller = st.session_state.controller
            return True
            
        try:
            with st.spinner("🚀 Initializing JSE Financial Advisory System..."):
                # Import and initialize the main controller
                # This loads all LLM models, takes 10-30 seconds on first run
                from Controller.autogen_controller import FinancialAdvisoryController
                self.controller = FinancialAdvisoryController()
                
                # Store in session state for persistence across reruns
                st.session_state.controller = self.controller
                st.session_state.initialized = True
                
                # Initialize conversation history from controller
                if hasattr(self.controller, 'conversation_history'):
                    st.session_state.conversation_history = self.controller.conversation_history
                
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
            return False

    @st.cache_data(ttl=30)  # Cache for 30 seconds to reduce API calls
    def get_live_stock_data_cached(_self, tickers: List[str]) -> pd.DataFrame:
        """
        Cached version of stock data fetching.
        Decorator ensures same data is returned for 30 seconds without re-fetching.
        Note: _self instead of self because caching doesn't work with instance methods.
        
        Args:
            tickers: List of JSE stock tickers (e.g., ['MTN.JO', 'NPN.JO'])
            
        Returns:
            DataFrame with columns: Ticker, Price, Change, Volume, Status
        """
        return _self.get_live_stock_data_sync(tickers)

    def get_live_stock_data_sync(self, tickers: List[str]) -> pd.DataFrame:
        """
        Synchronous stock data fetching using yfinance.
        Fetches real-time price, change, and volume for JSE stocks.
        
        Args:
            tickers: List of JSE tickers to fetch
            
        Returns:
            Formatted DataFrame ready for display in Streamlit
        """
        try:
            import yfinance as yf
            stock_data = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    # Get intraday data (1-minute intervals for most recent price)
                    hist = stock.history(period='1d', interval='1m')
                    
                    if not hist.empty:
                        # Extract current and previous prices
                        current_price = hist['Close'].iloc[-1]
                        previous_close = info.get('previousClose', current_price)
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        
                        # Convert from cents to Rands if necessary (heuristic)
                        if current_price > 1000:
                            current_price = current_price / 100
                            previous_close = previous_close / 100
                            change = change / 100
                        
                        # Build stock data row for display
                        stock_data.append({
                            'Ticker': ticker.replace('.JO', ''),  # Remove JSE suffix for display
                            'Price (R)': f"{current_price:.2f}",
                            'Change (R)': f"{change:+.2f}",  # +/- sign for visual clarity
                            'Change (%)': f"{change_percent:+.2f}%",
                            'Volume': f"{info.get('volume', 0):,}",  # Comma-separated thousands
                            'Status': '🟢 Up' if change > 0 else '🔴 Down' if change < 0 else '⚪ Flat'
                        })
                except Exception:
                    # Skip failed tickers silently (don't break entire fetch)
                    continue
                    
            return pd.DataFrame(stock_data)
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)  # Cache for 60 seconds (news changes less frequently)
    def get_live_news_cached(_self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Cached version of news fetching.
        News is less time-sensitive than stock prices, so 60s cache is appropriate.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of article dictionaries
        """
        return _self.get_live_news_sync(limit)

    def get_live_news_sync(self, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Synchronous news fetching from Google News RSS feeds.
        Aggregates news for major JSE stocks and removes duplicates.
        
        Args:
            limit: Maximum articles to return
            
        Returns:
            List of article dicts with headline, source, summary, link
        """
        try:
            from Model.web_model import WebSupplementationAgent
            web_agent = WebSupplementationAgent()
            
            # Get news for major JSE stocks (Top 5 by market cap/liquidity)
            major_stocks = ['MTN.JO', 'NPN.JO', 'SBK.JO', 'AGL.JO', 'VOD.JO']
            all_articles = []
            
            # Fetch 2 articles per stock (10 total max)
            for ticker in major_stocks:
                try:
                    articles = web_agent.fetch_articles_from_google_rss(
                        [(ticker, ticker.replace('.JO', ''))], 
                        max_per=2
                    )
                    all_articles.extend(articles)
                except:
                    continue
                
            # Remove duplicates (same article can appear for multiple stocks)
            unique_articles = []
            seen_headlines = set()
            
            for article in all_articles:
                if article['headline'] not in seen_headlines:
                    seen_headlines.add(article['headline'])
                    unique_articles.append(article)
            
            return unique_articles[:limit]
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            return []

    def create_stock_price_chart(self, ticker: str, period: str = '1mo') -> go.Figure:
        """
        Create interactive candlestick chart using Plotly.
        Candlestick charts show OHLC data for technical analysis.
        
        Args:
            ticker: JSE stock ticker (e.g., 'MTN.JO')
            period: Time period (1mo, 3mo, 6mo, 1y, 2y)
            
        Returns:
            Plotly figure object or None if data unavailable
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
                
            fig = go.Figure()
            
            # Add candlestick trace (OHLC bars)
            # Green = close > open (bullish), Red = close < open (bearish)
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            
            # Customize layout for professional appearance
            fig.update_layout(
                title=f"{ticker.replace('.JO', '')} Stock Price - {period.upper()}",
                xaxis_title="Date",
                yaxis_title="Price (ZAR)",
                template="plotly_white",  # Clean white background
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            return None

    def run(self):
        """
        Main application loop - entry point for Streamlit rendering.
        Defines the layout, tabs, and coordinates all UI components.
        """
        # ====================================================================
        # Custom CSS for styling
        # ====================================================================
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stock-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .news-card {
            background-color: #ffffff;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🏛️ JSE Financial Advisor</h1>', unsafe_allow_html=True)
        
        # Initialize system if not already done
        if self.controller is None:
            if not self.initialize_system():
                st.error("Failed to initialize the system. Please refresh the page.")
                return
        
        # ====================================================================
        # Sidebar - Controls and Settings
        # ====================================================================
        st.sidebar.title("📊 Dashboard Controls")
        
        # Separate refresh toggles for each tab
        # KEY FEATURE: Independent refresh rates prevent unwanted data fetching
        st.sidebar.subheader("🔄 Refresh Settings")
        
        # Market refresh settings (faster updates for price-sensitive data)
        market_auto_refresh = st.sidebar.checkbox(
            "Auto-refresh Markets", 
            value=True,  # Default ON for markets (prices change frequently)
            key="market_auto_refresh"
        )
        market_refresh_interval = st.sidebar.slider(
            "Market Refresh (seconds)", 
            10, 300, 30,  # Min, Max, Default
            key="market_refresh_interval"
        )
        
        # News refresh settings (slower updates for less time-sensitive content)
        news_auto_refresh = st.sidebar.checkbox(
            "Auto-refresh News", 
            value=False,  # Default OFF (news changes slowly, avoid API spam)
            key="news_auto_refresh"
        )
        news_refresh_interval = st.sidebar.slider(
            "News Refresh (seconds)", 
            30, 600, 60,  # News can refresh less frequently
            key="news_refresh_interval"
        )
        
        # Display recent conversation history in sidebar
        self.display_conversation_history()
        
        # System info expander (collapsed by default to save space)
        with st.sidebar.expander("🔧 System Info", expanded=False):
            if self.controller is not None:
                health = self.controller.health_check()
                st.write(f"**Status:** {health['overall_status'].upper()}")
                
                # Show memory statistics if available
                if self.controller.agents.get('memory'):
                    memory_stats = self.controller.agents['memory'].get_memory_stats()
                    st.write(f"**Memory Sessions:** {memory_stats.get('total_sessions', 0)}")
        
        # ====================================================================
        # Main Content Area - Tabs
        # ====================================================================
        # Create four main tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Advisor Chat",      # LLM-powered Q&A
            "📈 Live Markets",      # Real-time stock data
            "📰 News Feed",         # Aggregated news
            "📊 Analysis Tools"     # Technical analysis tools
        ])
        
        # Track current tab for independent refresh logic
        # This prevents auto-refresh in inactive tabs (performance optimization)
        if tab1:
            st.session_state.current_tab = "🎯 Advisor Chat"
        if tab2:
            st.session_state.current_tab = "📈 Live Markets"
        if tab3:
            st.session_state.current_tab = "📰 News Feed"
        if tab4:
            st.session_state.current_tab = "📊 Analysis Tools"
        
        # Render each tab's content
        with tab1:
            self.render_advisor_chat()
        
        with tab2:
            self.render_live_markets(market_auto_refresh, market_refresh_interval)
        
        with tab3:
            self.render_news_feed(news_auto_refresh, news_refresh_interval)
        
        with tab4:
            self.render_analysis_tools()

    def display_conversation_history(self):
        """
        Display recent conversation history in sidebar.
        Shows last 5 Q&A pairs in expandable sections for context.
        """
        if st.session_state.conversation_history:
            st.sidebar.subheader("💬 Conversation History")
            # Reverse to show newest first, limit to last 5
            for i, (query, response) in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.sidebar.expander(f"Q: {query[:50]}...", expanded=False):
                    st.write(f"**Q:** {query}")
                    # Show truncated response (full response in main area)
                    st.write(f"**A:** {response[:200]}...")

    def render_advisor_chat(self):
        """
        Render the main advisor chat interface - NO AUTO REFRESH.
        This is a conversational interface, not a live dashboard.
        Updates only on explicit user action (analyze button).
        """
        st.header("💬 JSE Financial Advisor")
        
        # Chat input - use session state to preserve input across reruns
        user_query = st.text_area(
            "Ask about JSE stocks, market analysis, or financial advice:",
            value=st.session_state.user_query,  # Preserve user's text
            placeholder="e.g., Compare MTN and Vodacom performance, What's the outlook for banking stocks?, Explain P/E ratio...",
            height=100,
            key="user_query_input"
        )
        
        # Action buttons in columns for horizontal layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, key="analyze_btn")
        with col2:
            clear_btn = st.button("🗑️ Clear History", use_container_width=True, key="clear_btn")
        with col3:
            example_btn = st.button("💡 Show Examples", use_container_width=True, key="example_btn")
        
        # Show example queries for guidance
        if example_btn:
            st.info("""
            **Example Queries:**
            - Compare MTN and Vodacom stock performance
            - What's happening with Standard Bank shares?
            - Show me mining sector analysis
            - Explain dividend yield for JSE investors
            - Create portfolio allocation for MTN 40%, NPN 30%, SBK 30%
            - Show me candlestick chart for Anglo American
            """)
        
        # Clear conversation history
        if clear_btn:
            if self.controller is not None:
                self.controller.reset_conversation()
                st.session_state.conversation_history = []
                st.session_state.user_query = ""
            st.rerun()  # Force UI refresh
        
        # Process user query (main analysis logic)
        if analyze_btn and user_query and self.controller is not None:
            with st.spinner("🔍 Analyzing your query..."):
                try:
                    # Send query to controller for processing
                    # This triggers the full multi-agent pipeline:
                    # 1. Ticker resolution
                    # 2. Web data fetching (news + prices)
                    # 3. Memory search (similar past queries)
                    # 4. LLM analysis
                    # 5. Data visualization code generation
                    # 6. Summarization
                    response = self.controller.process_user_query(user_query)
                    
                    # Update conversation history in session state
                    st.session_state.conversation_history.append(
                        (user_query, response.get('advisor_full_response', ''))
                    )
                    
                    # ========================================================
                    # Display Response Components
                    # ========================================================
                    st.subheader("💡 Advisor Analysis")
                    
                    # Main LLM response (primary output)
                    st.write(response.get('advisor_full_response', 'No response generated'))
                    
                    # Market data context (price, change, volume)
                    market_data = response.get('market_data_context')
                    if market_data and "unavailable" not in str(market_data).lower():
                        with st.expander("📊 Market Data", expanded=True):
                            st.text(market_data)
                    
                    # News context (recent articles)
                    articles = response.get('web_context', [])
                    if articles:
                        with st.expander(f"📰 Relevant News ({len(articles)} articles)", expanded=False):
                            for article in articles:
                                headline = article.get('headline', 'No headline')
                                link = article.get('link')
                                source = article.get('source', 'Unknown')
                                summary = article.get('summary', 'No summary available')

                                # Clickable headline (opens in new tab)
                                if link:
                                    st.markdown(f"**<a href='{link}' target='_blank'>{headline}</a>**", unsafe_allow_html=True)
                                else:
                                    st.write(f"**{headline}**")
                                
                                st.write(f"*Source: {source}*")
                                st.write(summary)
                                st.divider()
                    
                    # Analysis results (data insights + generated code)
                    analysis = response.get('analysis_results')
                    if analysis:
                        with st.expander("🔬 Data Analysis", expanded=False):
                            st.write(f"**Analysis Type:** {analysis.get('analysis_type', 'N/A')}")
                            st.write(f"**Stocks Analyzed:** {analysis.get('stocks_analyzed', [])}")
                            
                            # Natural language insights
                            insights = analysis.get('insights')
                            if insights:
                                st.write("**Insights:**")
                                st.write(insights)
                            
                            # Show generated Python code for visualization
                            code = analysis.get('code')
                            if code and len(code) > 100:
                                with st.expander("🐍 Generated Analysis Code", expanded=False):
                                    st.code(code, language='python')
                    
                    # Clear the input after successful analysis
                    st.session_state.user_query = ""
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")

    def render_live_markets(self, auto_refresh: bool, refresh_interval: int):
        """
        Render live markets dashboard with independent refresh.
        KEY FEATURE: Only refreshes when this tab is active AND timer expires.
        Prevents unnecessary API calls when user is in other tabs.
        
        Args:
            auto_refresh: Whether to auto-refresh (from sidebar toggle)
            refresh_interval: Seconds between refreshes (from sidebar slider)
        """
        st.header("📈 Live JSE Market Data")
        
        # Major JSE stocks to monitor (Top 8 by market cap/liquidity)
        major_tickers = ['MTN.JO', 'NPN.JO', 'SBK.JO', 'FSR.JO', 'AGL.JO', 'SOL.JO', 'VOD.JO', 'CPI.JO']
        
        # ====================================================================
        # Smart Refresh Logic - Only refresh if:
        # 1. Auto-refresh is enabled AND
        # 2. User is on this tab AND
        # 3. Refresh interval has elapsed
        # ====================================================================
        refresh_data = False
        if auto_refresh and st.session_state.current_tab == "📈 Live Markets":
            current_time = datetime.now()
            if (st.session_state.last_market_refresh is None or 
                (current_time - st.session_state.last_market_refresh).total_seconds() >= refresh_interval):
                refresh_data = True
                st.session_state.last_market_refresh = current_time
        else:
            # Manual refresh button (always available when auto-refresh off)
            if st.button("🔄 Refresh Market Data", key="refresh_market_data"):
                refresh_data = True
                st.session_state.last_market_refresh = datetime.now()
        
        # Fetch data only if refresh is needed (avoid redundant API calls)
        if refresh_data or st.session_state.stock_data.empty:
            with st.spinner("📊 Fetching live market data..."):
                # Uses cached method with 30s TTL for efficiency
                st.session_state.stock_data = self.get_live_stock_data_cached(major_tickers)
        
        if not st.session_state.stock_data.empty:
            # Display last refresh time for transparency
            if st.session_state.last_market_refresh:
                st.caption(f"Last updated: {st.session_state.last_market_refresh.strftime('%H:%M:%S')}")
            
            # ================================================================
            # Stock Table Display
            # ================================================================
            st.subheader("💰 Live Stock Prices")
            st.dataframe(
                st.session_state.stock_data,
                use_container_width=True,  # Full width utilization
                hide_index=True  # No row numbers needed
            )
            
            # ================================================================
            # Interactive Stock Price Charts
            # ================================================================
            st.subheader("📊 Stock Price Charts")
            selected_ticker = st.selectbox(
                "Select stock for detailed chart:",
                [ticker.replace('.JO', '') for ticker in major_tickers],
                key="chart_ticker_select"
            )
            
            if selected_ticker:
                chart_period = st.selectbox(
                    "Chart Period:",
                    ['1mo', '3mo', '6mo', '1y', '2y'],
                    index=2,  # Default to 6 months
                    key="chart_period_select"
                )
                
                # Create and display candlestick chart
                fig = self.create_stock_price_chart(selected_ticker + '.JO', chart_period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Could not load chart data for {selected_ticker}")
            
            # ================================================================
            # Market Summary Metrics
            # ================================================================
            st.subheader("📈 Market Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate market statistics
            up_stocks = len(st.session_state.stock_data[st.session_state.stock_data['Change (R)'].str.contains('\+')])
            down_stocks = len(st.session_state.stock_data[st.session_state.stock_data['Change (R)'].str.contains('-')])
            total_stocks = len(st.session_state.stock_data)
            
            # Display as metrics (large numbers with context)
            with col1:
                st.metric("Total Stocks", total_stocks)
            with col2:
                st.metric("Advancing", up_stocks, delta=f"+{up_stocks}")
            with col3:
                st.metric("Declining", down_stocks, delta=f"-{down_stocks}", delta_color="inverse")
            with col4:
                if total_stocks > 0:
                    # Advance/Decline ratio (>1 = bullish, <1 = bearish)
                    adv_dec_ratio = up_stocks / total_stocks
                    st.metric("Advance/Decline Ratio", f"{adv_dec_ratio:.2f}")
        else:
            st.warning("Unable to fetch live market data. Please check your internet connection.")

    def render_news_feed(self, auto_refresh: bool, refresh_interval: int):
        """
        Render live news feed with independent refresh.
        Similar to markets tab but with separate refresh control.
        News changes less frequently, so refresh is often disabled by default.
        
        Args:
            auto_refresh: Whether to auto-refresh news
            refresh_interval: Seconds between news refreshes
        """
        st.header("📰 JSE Market News")
        
        # Same smart refresh logic as markets tab
        refresh_data = False
        if auto_refresh and st.session_state.current_tab == "📰 News Feed":
            current_time = datetime.now()
            if (st.session_state.last_news_refresh is None or 
                (current_time - st.session_state.last_news_refresh).total_seconds() >= refresh_interval):
                refresh_data = True
                st.session_state.last_news_refresh = current_time
        else:
            # Manual refresh button
            if st.button("🔄 Refresh News", key="refresh_news_data"):
                refresh_data = True
                st.session_state.last_news_refresh = datetime.now()
        
        # Fetch news only if refresh needed
        if refresh_data or not st.session_state.news_data:
            with st.spinner("📰 Fetching latest news..."):
                # Uses cached method with 60s TTL (longer than markets)
                st.session_state.news_data = self.get_live_news_cached(limit=15)
        
        if st.session_state.news_data:
            # Display last refresh time
            if st.session_state.last_news_refresh:
                st.caption(f"Last updated: {st.session_state.last_news_refresh.strftime('%H:%M:%S')}")
            
            st.subheader(f"📢 Latest Market News ({len(st.session_state.news_data)} articles)")
            
            # Display each article in expandable section
            for i, article in enumerate(st.session_state.news_data):
                headline = article.get('headline', 'No headline')
                link = article.get('link')
                
                with st.expander(f"📰 {headline[:80]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Clickable headline
                        if link:
                            st.markdown(f"**<a href='{link}' target='_blank'>{headline}</a>**", unsafe_allow_html=True)
                        else:
                            st.write(f"**{headline}**")
                            
                        st.write(f"**Stock:** {article.get('ticker', 'N/A').replace('.JO', '')}")
                        st.write(article.get('summary', 'No summary available'))
                    
                    with col2:
                        # Metadata
                        st.write(f"*Source: {article.get('source', 'Unknown source')}*")
                        if article.get('published'):
                            st.write(f"*Published: {article.get('published')}*")
        else:
            st.warning("Unable to fetch news articles. Please check your internet connection.")
        
        # ====================================================================
        # News Search Feature - Search specific stock news
        # ====================================================================
        st.subheader("🔍 Search Specific Stock News")
        search_ticker = st.text_input("Enter stock ticker (e.g., MTN, SBK, NPN):", placeholder="MTN", key="news_search_input")
        
        if st.button("Search News", key="search_news_btn") and search_ticker:
            with st.spinner(f"🔍 Searching news for {search_ticker}..."):
                try:
                    from Model.web_model import WebSupplementationAgent
                    web_agent = WebSupplementationAgent()
                    
                    # Resolve ticker from company name
                    companies = web_agent._resolve_tickers(search_ticker)
                    if companies:
                        ticker, name = companies[0]
                        # Fetch articles specifically for this stock
                        articles = web_agent.fetch_articles_from_google_rss([(ticker, name)], max_per=5)
                        
                        if articles:
                            st.write(f"### 📰 News for {name}")
                            for article in articles:
                                headline = article.get('headline', 'No headline')
                                link = article.get('link')
                                source = article.get('source', 'Unknown')
                                summary = article.get('summary', 'No summary')

                                with st.expander(f"**{headline}**", expanded=False):
                                    st.write(f"*Source: {source}*")
                                    st.write(summary)
                                    if link:
                                        st.markdown(f"<a href='{link}' target='_blank'>Read full article</a>", unsafe_allow_html=True)
                        else:
                            st.warning(f"No recent news found for {search_ticker}")
                    else:
                        st.warning(f"Could not find stock: {search_ticker}")
                except Exception as e:
                    st.error(f"Error searching news: {e}")

    def render_analysis_tools(self):
        """
        Render analysis tools and utilities - NO AUTO REFRESH.
        Provides pre-configured analysis templates and quick actions.
        User-triggered only (no background refreshing).
        """
        st.header("📊 JSE Analysis Tools")
        
        col1, col2 = st.columns(2)
        
        # ====================================================================
        # Left Column: Quick Analysis Templates
        # ====================================================================
        with col1:
            st.subheader("📈 Quick Analysis")
            
            # Analysis type selector
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                [
                    "Stock Price Trends",      # Line chart over time
                    "Moving Averages",         # MA20, MA50, MA200
                    "Volume Analysis",         # Price + volume dual chart
                    "Correlation Matrix",      # Stock correlation heatmap
                    "Portfolio Allocation"     # Pie chart of holdings
                ],
                key="analysis_type_select"
            )
            
            # Multi-select for stocks to analyze
            selected_stocks = st.multiselect(
                "Select Stocks:",
                ["MTN", "NPN", "SBK", "FSR", "AGL", "SOL", "VOD", "CPI", "ANG", "GFI"],
                default=["MTN", "NPN", "SBK"],  # Default to Top 3
                key="selected_stocks_multiselect"
            )
            
            # Time period selector
            time_period = st.selectbox(
                "Time Period:",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,  # Default to 6 months
                key="time_period_select"
            )
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary", key="run_analysis_btn"):
                if selected_stocks and self.controller is not None:
                    # Map analysis type to natural language query
                    # This allows reusing the main controller pipeline
                    query_map = {
                        "Stock Price Trends": f"Show me price trends for {', '.join(selected_stocks)} over {time_period}",
                        "Moving Averages": f"Plot moving averages for {', '.join(selected_stocks)}",
                        "Volume Analysis": f"Analyze trading volume for {', '.join(selected_stocks)}",
                        "Correlation Matrix": f"Create correlation matrix for {', '.join(selected_stocks)}",
                        "Portfolio Allocation": f"Show portfolio allocation for {', '.join(selected_stocks)}"
                    }
                    
                    query = query_map.get(analysis_type, f"Analyze {', '.join(selected_stocks)}")
                    
                    with st.spinner("🔍 Running analysis..."):
                        # Process through main controller pipeline
                        response = self.controller.process_user_query(query)
                        
                        # Display results
                        st.subheader("💡 Analysis Results")
                        st.write(response.get('advisor_full_response', 'No response'))
                        
                        # Show generated code if available
                        # Users can copy this code to run locally with more customization
                        analysis = response.get('analysis_results')
                        if analysis and analysis.get('code'):
                            with st.expander("🐍 View Generated Analysis Code"):
                                st.code(analysis['code'], language='python')
                else:
                    st.warning("Please select at least one stock.")
        
        # ====================================================================
        # Right Column: System Utilities
        # ====================================================================
        with col2:
            st.subheader("🔧 System Utilities")
            
            if self.controller is not None:
                # System information display
                st.write("### 🖥️ System Information")
                health = self.controller.health_check()
                
                # Display metrics in columns
                col_a, col_b = st.columns(2)
                with col_a:
                    # Overall system status
                    st.metric("Overall Status", health['overall_status'].upper())
                    # Count active agents
                    st.metric("Active Agents", sum(1 for s in health['agent_statuses'].values() if 'Active' in s or 'Template' in s))
                with col_b:
                    # Conversation history count
                    st.metric("Conversation History", f"{len(st.session_state.conversation_history)}")
                    # Memory sessions count
                    if self.controller.agents.get('memory'):
                        memory_stats = self.controller.agents['memory'].get_memory_stats()
                        st.metric("Memory Sessions", memory_stats.get('total_sessions', 0))
                
                st.divider()
                
                # Quick actions section
                st.write("### ⚡ Quick Actions")
                
                # Refresh all data button
                # Clears all caches and forces fresh data fetch
                if st.button("🔄 Refresh All Data", use_container_width=True, key="refresh_all_btn"):
                    # Clear Streamlit's cache
                    st.cache_data.clear()
                    # Reset session state data
                    st.session_state.stock_data = pd.DataFrame()
                    st.session_state.news_data = []
                    st.session_state.last_market_refresh = None
                    st.session_state.last_news_refresh = None
                    # Force UI refresh
                    st.rerun()

def main():
    """
    Main application entry point.
    Initializes the app in session state and runs the main loop.
    
    Pattern used: Store app instance in session state to maintain state
    across Streamlit reruns (which happen on every user interaction).
    """
    # Initialize the app only once
    if 'app' not in st.session_state:
        st.session_state.app = JSEAdvisoryApp()
    
    # Run the app (renders UI)
    st.session_state.app.run()

if __name__ == "__main__":
    main()