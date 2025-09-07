import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Vibe Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "section" not in st.session_state:
    st.session_state["section"] = "dashboard"

def apply_theme_styles():
    """Apply theme-specific styles"""
    if st.session_state["theme"] == "dark":
        colors = {
            'bg': '#0b0d12',
            'sidebar_bg': '#1a1f2b',
            'panel': '#1e2530',
            'panel_light': '#252c3a',
            'muted': '#2a3441',
            'text': '#e8ebf1',
            'subtext': '#a7b0c0',
            'primary': '#4f8cff',
            'success': '#2ec27e',
            'warning': '#f0b429',
            'danger': '#ef476f',
            'border': '#2a3441'
        }
    else:
        colors = {
            'bg': '#f6f7fb',
            'sidebar_bg': '#ffffff',
            'panel': '#ffffff',
            'panel_light': '#f9fafc',
            'muted': '#eef1f7',
            'text': '#1f2430',
            'subtext': '#667085',
            'primary': '#2e6df6',
            'success': '#2ec27e',
            'warning': '#f0b429',
            'danger': '#ef476f',
            'border': '#e4e7ec'
        }

    st.markdown(f"""
    <style>
    /* Main app styling */
    .stApp {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {{
        background-color: {colors['sidebar_bg']};
    }}
    
    /* Custom card styles */
    .metric-card {{
        background: {colors['panel']};
        padding: 24px;
        border-radius: 16px;
        border: 1px solid {colors['border']};
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(79,140,255,0.15);
    }}
    
    .metric-title {{
        font-size: 14px;
        font-weight: 600;
        color: {colors['subtext']};
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-value {{
        font-size: 32px;
        font-weight: 700;
        color: {colors['text']};
        margin-bottom: 12px;
        line-height: 1;
    }}
    
    /* Status pills */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 600;
        margin: 4px 8px 4px 0;
    }}
    
    .pill-success {{
        background: rgba(46,194,126,0.15);
        color: {colors['success']};
        border: 1px solid rgba(46,194,126,0.3);
    }}
    
    .pill-warning {{
        background: rgba(240,180,41,0.15);
        color: {colors['warning']};
        border: 1px solid rgba(240,180,41,0.3);
    }}
    
    .pill-neutral {{
        background: {colors['muted']};
        color: {colors['subtext']};
        border: 1px solid {colors['border']};
    }}
    
    /* News card styling */
    .news-card {{
        background: {colors['panel']};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {colors['border']};
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }}
    
    .news-card:hover {{
        border-color: {colors['primary']};
        transform: translateY(-1px);
    }}
    
    .news-title {{
        font-size: 16px;
        font-weight: 600;
        color: {colors['text']};
        margin-bottom: 8px;
        line-height: 1.4;
    }}
    
    .news-meta {{
        font-size: 12px;
        color: {colors['subtext']};
        margin-bottom: 12px;
    }}
    
    .news-actions {{
        display: flex;
        gap: 8px;
    }}
    
    .btn-primary {{
        background: {colors['primary']};
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    .btn-secondary {{
        background: transparent;
        color: {colors['subtext']};
        border: 1px solid {colors['border']};
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    
    /* Chart container */
    .chart-container {{
        background: {colors['panel']};
        padding: 24px;
        border-radius: 16px;
        border: 1px solid {colors['border']};
        margin-bottom: 20px;
    }}
    
    .chart-title {{
        font-size: 18px;
        font-weight: 600;
        color: {colors['text']};
        margin-bottom: 16px;
    }}
    
    /* Sector tags */
    .sector-tag {{
        display: inline-block;
        background: {colors['muted']};
        color: {colors['subtext']};
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        border: 1px solid {colors['border']};
    }}
    
    /* Hide streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Custom sidebar styling */
    .sidebar-section {{
        padding: 16px 0;
    }}
    
    .sidebar-title {{
        font-size: 20px;
        font-weight: 700;
        color: {colors['text']};
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .theme-toggle {{
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid {colors['border']};
    }}
    
    /* Search bar */
    .search-container {{
        position: relative;
        margin-bottom: 24px;
    }}
    
    .search-input {{
        width: 100%;
        padding: 12px 16px 12px 40px;
        background: {colors['panel']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        color: {colors['text']};
        font-size: 14px;
    }}
    
    .search-icon {{
        position: absolute;
        left: 12px;
        top: 50%;
        transform: translateY(-50%);
        color: {colors['subtext']};
    }}
    </style>
    """, unsafe_allow_html=True)

def create_sample_chart():
    """Create a sample chart for the dashboard"""
    # Generate sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate JSE Top 40 data
    base_price = 69000
    returns = np.random.normal(0.0002, 0.015, len(dates))
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices[1:]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines',
        name='FTSE/JSE Top 40',
        line=dict(color='#4f8cff', width=2),
        fill='tonexty' if st.session_state["theme"] == "dark" else None,
        fillcolor='rgba(79,140,255,0.1)' if st.session_state["theme"] == "dark" else None
    ))
    
    # Styling based on theme
    bg_color = '#1e2530' if st.session_state["theme"] == "dark" else '#ffffff'
    text_color = '#e8ebf1' if st.session_state["theme"] == "dark" else '#1f2430'
    grid_color = '#2a3441' if st.session_state["theme"] == "dark" else '#eef1f7'
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            showline=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=grid_color,
            showline=False,
            zeroline=False
        ),
        showlegend=False
    )
    
    return fig

def run_ui():
    apply_theme_styles()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📈 Vibe Finance</div>', unsafe_allow_html=True)
        
        # Navigation
        sections = ["dashboard", "datasci", "news", "advisor", "logs", "settings"]
        labels = ["🏠 Dashboard", "📊 Data Explorer", "📰 Market News", "💡 Advisor", "🧪 Validator & Logs", "⚙️ Settings"]
        
        for sec, label in zip(sections, labels):
            if st.button(label, key=f"nav_{sec}", use_container_width=True):
                st.session_state["section"] = sec
                st.rerun()
        
        # Theme toggle
        st.markdown('<div class="theme-toggle"></div>', unsafe_allow_html=True)
        if st.button("🌗 Toggle Theme", key="theme_toggle", use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
        
        st.markdown('<p style="color: #a7b0c0; font-size: 12px; margin-top: 16px;">JSE-focused multi-agent hub</p>', unsafe_allow_html=True)

    # Main content area
    if st.session_state["section"] == "dashboard":
        # Header with search
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("# Dashboard")
        with col2:
            st.button("📤 Export", key="export_btn")
            st.button("➕ New Analysis", key="new_analysis_btn", type="primary")
        
        # Search bar
        st.markdown("""
        <div class="search-container">
            <div class="search-icon">🔍</div>
        </div>
        """, unsafe_allow_html=True)
        search_query = st.text_input("", placeholder="Search tickers, sectors, or news...", label_visibility="collapsed")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">FTSE/JSE Top 40</div>
                <div class="metric-value">69,842</div>
                <div class="status-pill pill-success">▲ 0.76% Today</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">All Share (ALSI)</div>
                <div class="metric-value">80,115</div>
                <div class="status-pill pill-warning">● Flat</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Most Active</div>
                <div class="metric-value">AGL, NPN, SOL</div>
                <div class="sector-tag">Banking</div>
                <div class="sector-tag">Mining</div>
                <div class="sector-tag">Tech</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">Market Overview</div></div>', unsafe_allow_html=True)
            fig = create_sample_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Latest Headlines</div>
                <div class="news-card">
                    <div class="news-title">Rand firms on commodity rally</div>
                    <div class="news-meta">Fin24 • 10:31</div>
                    <div class="news-actions">
                        <button class="btn-secondary">Read</button>
                        <button class="btn-primary">Summarize</button>
                    </div>
                </div>
                <div class="news-card">
                    <div class="news-title">JSE banks outperform after rate outlook</div>
                    <div class="news-meta">Moneyweb • 09:15</div>
                    <div class="news-actions">
                        <button class="btn-secondary">Read</button>
                        <button class="btn-primary">Summarize</button>
                    </div>
                </div>
                <div class="news-card">
                    <div class="news-title">Gold miners lift Top 40</div>
                    <div class="news-meta">NewsAPI • 08:02</div>
                    <div class="news-actions">
                        <button class="btn-secondary">Read</button>
                        <button class="btn-primary">Summarize</button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif st.session_state["section"] == "datasci":
        st.markdown("# Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker", placeholder="e.g., NPN.JO, AGL.JO")
        with col2:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col3:
            end_date = st.date_input("End Date", value=datetime.now())
        
        analysis_type = st.selectbox("Analysis Type", [
            "Price Trends", "Moving Averages", "Volume Analysis", 
            "Volatility (ATR)", "Returns & Drawdowns"
        ])
        
        st.markdown('<div class="chart-container"><div class="chart-title">Visualization</div></div>', unsafe_allow_html=True)
        if ticker:
            fig = create_sample_chart()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a ticker symbol to see analysis")
        
        st.markdown('<div class="chart-container"><div class="chart-title">Generated Code</div></div>', unsafe_allow_html=True)
        if ticker and analysis_type:
            st.code(f"""
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Fetch data for {ticker}
data = yf.download('{ticker}', start='{start_date}', end='{end_date}')

# Perform {analysis_type.lower()} analysis
# ... analysis code here ...
            """, language="python")

    elif st.session_state["section"] == "news":
        st.markdown("# Market News")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            filter_query = st.text_input("Filter", placeholder="Filter by company, sector, or keyword")
        with col2:
            st.selectbox("Source", ["All Sources", "Fin24", "Moneyweb", "NewsAPI", "Business Day"])
        
        # Sample news items
        news_items = [
            {"title": "Rand firms on commodity rally", "source": "Fin24", "time": "10:31", "summary": "The rand strengthened against major currencies..."},
            {"title": "JSE banks outperform after rate outlook", "source": "Moneyweb", "time": "09:15", "summary": "Banking stocks led gains on the JSE..."},
            {"title": "Gold miners lift Top 40", "source": "NewsAPI", "time": "08:02", "summary": "Gold mining companies boosted the benchmark..."},
            {"title": "Tech stocks under pressure", "source": "Business Day", "time": "07:45", "summary": "Technology shares faced selling pressure..."}
        ]
        
        for item in news_items:
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{item['title']}</div>
                <div class="news-meta">{item['source']} • {item['time']}</div>
                <p style="color: #a7b0c0; font-size: 14px; margin: 8px 0;">{item['summary']}</p>
                <div class="news-actions">
                    <button class="btn-secondary">Read Full</button>
                    <button class="btn-primary">AI Summary</button>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif st.session_state["section"] == "advisor":
        st.markdown("# AI Advisor")
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Current Tasks</div>
            <ul style="color: #e8ebf1; margin: 16px 0;">
                <li style="margin: 8px 0;">Analyzing JSE banking sector performance</li>
                <li style="margin: 8px 0;">Monitoring commodity price movements</li>
                <li style="margin: 8px 0;">Tracking rand volatility patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Confidence Level</div>
            <div style="width: 100%; background: #2a3441; border-radius: 12px; height: 16px; margin-top: 8px;">
                <div style="height: 100%; width: 72%; background: linear-gradient(90deg, #f0b429, #2ec27e); border-radius: 12px;"></div>
            </div>
            <div style="color: #a7b0c0; font-size: 12px; margin-top: 4px;">72% - High confidence in current analysis</div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state["section"] == "logs":
        st.markdown("# Validator & Logs")
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Recent Events</div>
            <div style="font-family: monospace; font-size: 12px; color: #a7b0c0;">
                <div style="margin: 8px 0;">[09:31] ✅ Market data synchronized</div>
                <div style="margin: 8px 0;">[09:30] 📊 JSE opening prices updated</div>
                <div style="margin: 8px 0;">[09:15] 📰 News feed refreshed (4 new articles)</div>
                <div style="margin: 8px 0;">[09:00] 🔄 Daily validation completed</div>
                <div style="margin: 8px 0;">[08:45] ⚠️  Minor data discrepancy resolved</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state["section"] == "settings":
        st.markdown("# Settings")
        
        st.markdown("### Data Sources")
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Yahoo Finance", value=True, key="yahoo_finance")
            st.checkbox("Alpha Vantage", value=True, key="alpha_vantage")
            st.checkbox("Investpy", value=True, key="investpy")
        
        with col2:
            st.checkbox("Fin24 RSS", value=True, key="fin24_rss")
            st.checkbox("Moneyweb", value=True, key="moneyweb")
            st.checkbox("NewsAPI", value=True, key="newsapi")
        
        st.markdown("### Refresh Intervals")
        st.slider("Market Data (seconds)", min_value=30, max_value=300, value=60)
        st.slider("News Feed (minutes)", min_value=5, max_value=60, value=15)

if __name__ == "__main__":
    run_ui()