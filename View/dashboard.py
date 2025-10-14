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

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize all session state variables at the top
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'last_market_refresh' not in st.session_state:
    st.session_state.last_market_refresh = None
if 'last_news_refresh' not in st.session_state:
    st.session_state.last_news_refresh = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = pd.DataFrame()
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "🎯 Advisor Chat"

# Set page config FIRST - must be the first Streamlit command
st.set_page_config(
    page_title="JSE Financial Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class JSEAdvisoryApp:
    def __init__(self):
        self.controller = None
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize the financial advisory system"""
        if st.session_state.initialized and st.session_state.controller is not None:
            self.controller = st.session_state.controller
            return True
            
        try:
            with st.spinner("🚀 Initializing JSE Financial Advisory System..."):
                from Controller.autogen_controller import FinancialAdvisoryController
                self.controller = FinancialAdvisoryController()
                st.session_state.controller = self.controller
                st.session_state.initialized = True
                
                # Initialize conversation history from controller
                if hasattr(self.controller, 'conversation_history'):
                    st.session_state.conversation_history = self.controller.conversation_history
                
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
            return False

    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def get_live_stock_data_cached(_self, tickers: List[str]) -> pd.DataFrame:
        """Cached version of stock data fetching"""
        return _self.get_live_stock_data_sync(tickers)

    def get_live_stock_data_sync(self, tickers: List[str]) -> pd.DataFrame:
        """Synchronous stock data fetching"""
        try:
            import yfinance as yf
            stock_data = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period='1d', interval='1m')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        previous_close = info.get('previousClose', current_price)
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                        
                        # Convert to Rands if needed
                        if current_price > 1000:
                            current_price = current_price / 100
                            previous_close = previous_close / 100
                            change = change / 100
                        
                        stock_data.append({
                            'Ticker': ticker.replace('.JO', ''),
                            'Price (R)': f"{current_price:.2f}",
                            'Change (R)': f"{change:+.2f}",
                            'Change (%)': f"{change_percent:+.2f}%",
                            'Volume': f"{info.get('volume', 0):,}",
                            'Status': '🟢 Up' if change > 0 else '🔴 Down' if change < 0 else '⚪ Flat'
                        })
                except Exception:
                    continue
                    
            return pd.DataFrame(stock_data)
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_live_news_cached(_self, limit: int = 15) -> List[Dict[str, Any]]:
        """Cached version of news fetching"""
        return _self.get_live_news_sync(limit)

    def get_live_news_sync(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Synchronous news fetching"""
        try:
            from Model.web_model import WebSupplementationAgent
            web_agent = WebSupplementationAgent()
            
            # Get news for major JSE stocks
            major_stocks = ['MTN.JO', 'NPN.JO', 'SBK.JO', 'AGL.JO', 'VOD.JO']
            all_articles = []
            
            for ticker in major_stocks:
                try:
                    articles = web_agent.fetch_articles_from_google_rss([(ticker, ticker.replace('.JO', ''))], max_per=2)
                    all_articles.extend(articles)
                except:
                    continue
                
            # Remove duplicates and sort by relevance
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
        """Create interactive stock price chart"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
                
            fig = go.Figure()
            
            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            
            # Customize layout
            fig.update_layout(
                title=f"{ticker.replace('.JO', '')} Stock Price - {period.upper()}",
                xaxis_title="Date",
                yaxis_title="Price (ZAR)",
                template="plotly_white",
                height=400
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating chart: {e}")
            return None

    def run(self):
        """Main application loop"""
        # Custom CSS
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
        
        # Sidebar
        st.sidebar.title("📊 Dashboard Controls")
        
        # Separate refresh toggles for each tab
        st.sidebar.subheader("🔄 Refresh Settings")
        
        market_auto_refresh = st.sidebar.checkbox("Auto-refresh Markets", value=True, key="market_auto_refresh")
        market_refresh_interval = st.sidebar.slider("Market Refresh (seconds)", 10, 300, 30, key="market_refresh_interval")
        
        news_auto_refresh = st.sidebar.checkbox("Auto-refresh News", value=False, key="news_auto_refresh")
        news_refresh_interval = st.sidebar.slider("News Refresh (seconds)", 30, 600, 60, key="news_refresh_interval")
        
        # Display conversation history
        self.display_conversation_history()
        
        # System info
        with st.sidebar.expander("🔧 System Info", expanded=False):
            if self.controller is not None:
                health = self.controller.health_check()
                st.write(f"**Status:** {health['overall_status'].upper()}")
                
                if self.controller.agents.get('memory'):
                    memory_stats = self.controller.agents['memory'].get_memory_stats()
                    st.write(f"**Memory Sessions:** {memory_stats.get('total_sessions', 0)}")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Advisor Chat", 
            "📈 Live Markets", 
            "📰 News Feed", 
            "📊 Analysis Tools"
        ])
        
        # Store current tab
        if tab1:
            st.session_state.current_tab = "🎯 Advisor Chat"
        if tab2:
            st.session_state.current_tab = "📈 Live Markets"
        if tab3:
            st.session_state.current_tab = "📰 News Feed"
        if tab4:
            st.session_state.current_tab = "📊 Analysis Tools"
        
        # Render each tab
        with tab1:
            self.render_advisor_chat()
        
        with tab2:
            self.render_live_markets(market_auto_refresh, market_refresh_interval)
        
        with tab3:
            self.render_news_feed(news_auto_refresh, news_refresh_interval)
        
        with tab4:
            self.render_analysis_tools()

    def display_conversation_history(self):
        """Display conversation history"""
        if st.session_state.conversation_history:
            st.sidebar.subheader("💬 Conversation History")
            for i, (query, response) in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.sidebar.expander(f"Q: {query[:50]}...", expanded=False):
                    st.write(f"**Q:** {query}")
                    st.write(f"**A:** {response[:200]}...")

    def render_advisor_chat(self):
        """Render the main advisor chat interface - NO AUTO REFRESH"""
        st.header("💬 JSE Financial Advisor")
        
        # Chat input - use session state to preserve input
        user_query = st.text_area(
            "Ask about JSE stocks, market analysis, or financial advice:",
            value=st.session_state.user_query,
            placeholder="e.g., Compare MTN and Vodacom performance, What's the outlook for banking stocks?, Explain P/E ratio...",
            height=100,
            key="user_query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, key="analyze_btn")
        with col2:
            clear_btn = st.button("🗑️ Clear History", use_container_width=True, key="clear_btn")
        with col3:
            example_btn = st.button("💡 Show Examples", use_container_width=True, key="example_btn")
        
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
        
        if clear_btn:
            if self.controller is not None:
                self.controller.reset_conversation()
                st.session_state.conversation_history = []
                st.session_state.user_query = ""
            st.rerun()
        
        if analyze_btn and user_query and self.controller is not None:
            with st.spinner("🔍 Analyzing your query..."):
                try:
                    # Process the query
                    response = self.controller.process_user_query(user_query)
                    
                    # Update conversation history in session state
                    st.session_state.conversation_history.append((user_query, response.get('advisor_full_response', '')))
                    
                    # Display response
                    st.subheader("💡 Advisor Analysis")
                    
                    # Main response
                    st.write(response.get('advisor_full_response', 'No response generated'))
                    
                    # Market data context
                    market_data = response.get('market_data_context')
                    if market_data and "unavailable" not in str(market_data).lower():
                        with st.expander("📊 Market Data", expanded=True):
                            st.text(market_data)
                    
                    # News context
                    articles = response.get('web_context', [])
                    if articles:
                        with st.expander(f"📰 Relevant News ({len(articles)} articles)", expanded=False):
                            for article in articles:
                                st.write(f"**{article.get('headline', 'No headline')}**")
                                st.write(f"*Source: {article.get('source', 'Unknown')}*")
                                st.write(f"{article.get('summary', 'No summary available')}")
                                st.divider()
                    
                    # Analysis results
                    analysis = response.get('analysis_results')
                    if analysis:
                        with st.expander("🔬 Data Analysis", expanded=False):
                            st.write(f"**Analysis Type:** {analysis.get('analysis_type', 'N/A')}")
                            st.write(f"**Stocks Analyzed:** {analysis.get('stocks_analyzed', [])}")
                            
                            insights = analysis.get('insights')
                            if insights:
                                st.write("**Insights:**")
                                st.write(insights)
                            
                            # Show code for visualization if available
                            code = analysis.get('code')
                            if code and len(code) > 100:
                                with st.expander("🐍 Generated Analysis Code", expanded=False):
                                    st.code(code, language='python')
                    
                    # Clear the input after successful analysis
                    st.session_state.user_query = ""
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")

    def render_live_markets(self, auto_refresh: bool, refresh_interval: int):
        """Render live markets dashboard with independent refresh"""
        st.header("📈 Live JSE Market Data")
        
        # Major JSE stocks to monitor
        major_tickers = ['MTN.JO', 'NPN.JO', 'SBK.JO', 'FSR.JO', 'AGL.JO', 'SOL.JO', 'VOD.JO', 'CPI.JO']
        
        # Check if we need to refresh data - ONLY for this tab
        refresh_data = False
        if auto_refresh and st.session_state.current_tab == "📈 Live Markets":
            current_time = datetime.now()
            if (st.session_state.last_market_refresh is None or 
                (current_time - st.session_state.last_market_refresh).total_seconds() >= refresh_interval):
                refresh_data = True
                st.session_state.last_market_refresh = current_time
        else:
            # Manual refresh button
            if st.button("🔄 Refresh Market Data", key="refresh_market_data"):
                refresh_data = True
                st.session_state.last_market_refresh = datetime.now()
        
        # Get live data only if refresh is needed
        if refresh_data or st.session_state.stock_data.empty:
            with st.spinner("📊 Fetching live market data..."):
                st.session_state.stock_data = self.get_live_stock_data_cached(major_tickers)
        
        if not st.session_state.stock_data.empty:
            # Display last refresh time
            if st.session_state.last_market_refresh:
                st.caption(f"Last updated: {st.session_state.last_market_refresh.strftime('%H:%M:%S')}")
            
            # Display stock table
            st.subheader("💰 Live Stock Prices")
            st.dataframe(
                st.session_state.stock_data,
                use_container_width=True,
                hide_index=True
            )
            
            # Stock price charts
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
                    index=2,
                    key="chart_period_select"
                )
                
                fig = self.create_stock_price_chart(selected_ticker + '.JO', chart_period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Could not load chart data for {selected_ticker}")
            
            # Market summary
            st.subheader("📈 Market Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            up_stocks = len(st.session_state.stock_data[st.session_state.stock_data['Change (R)'].str.contains('\+')])
            down_stocks = len(st.session_state.stock_data[st.session_state.stock_data['Change (R)'].str.contains('-')])
            total_stocks = len(st.session_state.stock_data)
            
            with col1:
                st.metric("Total Stocks", total_stocks)
            with col2:
                st.metric("Advancing", up_stocks, delta=f"+{up_stocks}")
            with col3:
                st.metric("Declining", down_stocks, delta=f"-{down_stocks}", delta_color="inverse")
            with col4:
                if total_stocks > 0:
                    adv_dec_ratio = up_stocks / total_stocks
                    st.metric("Advance/Decline Ratio", f"{adv_dec_ratio:.2f}")
        else:
            st.warning("Unable to fetch live market data. Please check your internet connection.")

    def render_news_feed(self, auto_refresh: bool, refresh_interval: int):
        """Render live news feed with independent refresh"""
        st.header("📰 JSE Market News")
        
        # Check if we need to refresh data - ONLY for this tab
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
        
        # Get live news only if refresh is needed
        if refresh_data or not st.session_state.news_data:
            with st.spinner("📰 Fetching latest news..."):
                st.session_state.news_data = self.get_live_news_cached(limit=15)
        
        if st.session_state.news_data:
            # Display last refresh time
            if st.session_state.last_news_refresh:
                st.caption(f"Last updated: {st.session_state.last_news_refresh.strftime('%H:%M:%S')}")
            
            st.subheader(f"📢 Latest Market News ({len(st.session_state.news_data)} articles)")
            
            for i, article in enumerate(st.session_state.news_data):
                # Create a nice news card
                with st.expander(f"📰 {article.get('headline', 'No headline')[:80]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{article.get('headline', 'No headline')}**")
                        st.write(f"**Stock:** {article.get('ticker', 'N/A').replace('.JO', '')}")
                        st.write(article.get('summary', 'No summary available'))
                        
                        if article.get('link'):
                            st.markdown(f"[Read full article]({article.get('link')})")
                    
                    with col2:
                        st.write(f"*Source: {article.get('source', 'Unknown source')}*")
                        if article.get('published'):
                            st.write(f"*Published: {article.get('published')}*")
        else:
            st.warning("Unable to fetch news articles. Please check your internet connection.")
        
        # News search
        st.subheader("🔍 Search Specific Stock News")
        search_ticker = st.text_input("Enter stock ticker (e.g., MTN, SBK, NPN):", placeholder="MTN", key="news_search_input")
        
        if st.button("Search News", key="search_news_btn") and search_ticker:
            with st.spinner(f"🔍 Searching news for {search_ticker}..."):
                try:
                    from Model.web_model import WebSupplementationAgent
                    web_agent = WebSupplementationAgent()
                    
                    # Resolve ticker
                    companies = web_agent._resolve_tickers(search_ticker)
                    if companies:
                        ticker, name = companies[0]
                        articles = web_agent.fetch_articles_from_google_rss([(ticker, name)], max_per=5)
                        
                        if articles:
                            st.write(f"### 📰 News for {name}")
                            for article in articles:
                                with st.expander(f"**{article.get('headline')}**", expanded=False):
                                    st.write(f"*Source: {article.get('source', 'Unknown')}*")
                                    st.write(article.get('summary', 'No summary'))
                        else:
                            st.warning(f"No recent news found for {search_ticker}")
                    else:
                        st.warning(f"Could not find stock: {search_ticker}")
                except Exception as e:
                    st.error(f"Error searching news: {e}")

    def render_analysis_tools(self):
        """Render analysis tools and utilities - NO AUTO REFRESH"""
        st.header("📊 JSE Analysis Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Quick Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                [
                    "Stock Price Trends",
                    "Moving Averages", 
                    "Volume Analysis",
                    "Correlation Matrix",
                    "Portfolio Allocation"
                ],
                key="analysis_type_select"
            )
            
            selected_stocks = st.multiselect(
                "Select Stocks:",
                ["MTN", "NPN", "SBK", "FSR", "AGL", "SOL", "VOD", "CPI", "ANG", "GFI"],
                default=["MTN", "NPN", "SBK"],
                key="selected_stocks_multiselect"
            )
            
            time_period = st.selectbox(
                "Time Period:",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
                key="time_period_select"
            )
            
            if st.button("🚀 Run Analysis", type="primary", key="run_analysis_btn"):
                if selected_stocks and self.controller is not None:
                    query_map = {
                        "Stock Price Trends": f"Show me price trends for {', '.join(selected_stocks)} over {time_period}",
                        "Moving Averages": f"Plot moving averages for {', '.join(selected_stocks)}",
                        "Volume Analysis": f"Analyze trading volume for {', '.join(selected_stocks)}",
                        "Correlation Matrix": f"Create correlation matrix for {', '.join(selected_stocks)}",
                        "Portfolio Allocation": f"Show portfolio allocation for {', '.join(selected_stocks)}"
                    }
                    
                    query = query_map.get(analysis_type, f"Analyze {', '.join(selected_stocks)}")
                    
                    with st.spinner("🔍 Running analysis..."):
                        response = self.controller.process_user_query(query)
                        
                        # Display results
                        st.subheader("💡 Analysis Results")
                        st.write(response.get('advisor_full_response', 'No response'))
                        
                        # Show generated code if available
                        analysis = response.get('analysis_results')
                        if analysis and analysis.get('code'):
                            with st.expander("🐍 View Generated Analysis Code"):
                                st.code(analysis['code'], language='python')
                else:
                    st.warning("Please select at least one stock.")
        
        with col2:
            st.subheader("🔧 System Utilities")
            
            if self.controller is not None:
                # System info
                st.write("### 🖥️ System Information")
                health = self.controller.health_check()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Overall Status", health['overall_status'].upper())
                    st.metric("Active Agents", sum(1 for s in health['agent_statuses'].values() if 'Active' in s or 'Template' in s))
                with col_b:
                    st.metric("Conversation History", f"{len(st.session_state.conversation_history)}")
                    if self.controller.agents.get('memory'):
                        memory_stats = self.controller.agents['memory'].get_memory_stats()
                        st.metric("Memory Sessions", memory_stats.get('total_sessions', 0))
                
                st.divider()
                
                # Quick actions
                st.write("### ⚡ Quick Actions")
                
                if st.button("🔄 Refresh All Data", use_container_width=True, key="refresh_all_btn"):
                    # Clear cached data to force refresh
                    st.cache_data.clear()
                    st.session_state.stock_data = pd.DataFrame()
                    st.session_state.news_data = []
                    st.session_state.last_market_refresh = None
                    st.session_state.last_news_refresh = None
                    st.rerun()

def main():
    """Main application entry point"""
    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = JSEAdvisoryApp()
    
    # Run the app
    st.session_state.app.run()

if __name__ == "__main__":
    main()