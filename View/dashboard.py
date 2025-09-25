import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
import time
import traceback

# Add the controller path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Financial Advisory System
try:
    from Controller.autogen_controller import FinancialAdvisoryController
    CONTROLLER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import Financial Advisory Controller: {e}")
    CONTROLLER_AVAILABLE = False

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="JSE Financial Advisory System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    if "section" not in st.session_state:
        st.session_state["section"] = "dashboard"
    if "controller" not in st.session_state:
        st.session_state["controller"] = None
    if "controller_status" not in st.session_state:
        st.session_state["controller_status"] = "not_initialized"
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "current_analysis" not in st.session_state:
        st.session_state["current_analysis"] = None
    if "example_query" not in st.session_state:
        st.session_state["example_query"] = None

@st.cache_resource
def initialize_controller():
    """Initialize the Financial Advisory Controller (cached)"""
    if not CONTROLLER_AVAILABLE:
        return None, "import_error"
    
    try:
        with st.spinner("🚀 Initializing AI Financial Advisory System..."):
            controller = FinancialAdvisoryController()
            health = controller.health_check()
            return controller, health['overall_status']
    except Exception as e:
        st.error(f"Failed to initialize controller: {e}")
        return None, "initialization_error"

def apply_theme_styles():
    """Apply theme-specific styles"""
    # Safety check - ensure theme is initialized
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
        
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
    .stApp {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}
    
    .css-1d391kg, .css-12oz5g7, .stSidebar > div {{
        background-color: {colors['sidebar_bg']};
    }}
    
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
    
    .pill-danger {{
        background: rgba(239,71,111,0.15);
        color: {colors['danger']};
        border: 1px solid rgba(239,71,111,0.3);
    }}
    
    .pill-neutral {{
        background: {colors['muted']};
        color: {colors['subtext']};
        border: 1px solid {colors['border']};
    }}
    
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
    
    .query-response {{
        background: {colors['panel']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }}
    
    .code-block {{
        background: {colors['muted']};
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        font-size: 12px;
        color: {colors['text']};
        overflow-x: auto;
        margin: 12px 0;
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
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status in the sidebar"""
    if st.session_state.controller is None:
        st.sidebar.error("⚠️ AI System Offline")
        return
    
    try:
        health = st.session_state.controller.health_check()
        system_info = st.session_state.controller.get_system_info()
        
        if health['overall_status'] == 'healthy':
            st.sidebar.success("✅ AI System Healthy")
        elif health['overall_status'] == 'degraded':
            st.sidebar.warning("⚠️ AI System Degraded")
        else:
            st.sidebar.error("❌ AI System Critical")
        
        # Agent status
        agent_status = system_info.get('system_status', {})
        active_agents = sum(1 for status in agent_status.values() if status)
        total_agents = len(agent_status)
        
        st.sidebar.metric("Active Agents", f"{active_agents}/{total_agents}")
        
        # Memory stats
        if 'memory_stats' in system_info:
            memory_stats = system_info['memory_stats']
            st.sidebar.metric("Session Memory", 
                            f"{memory_stats.get('total_sessions', 0)} sessions")
        
    except Exception as e:
        st.sidebar.error(f"Status Error: {str(e)[:50]}...")

def create_sample_chart():
    """Create a sample chart for the dashboard"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
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
    
    bg_color = '#1e2530' if st.session_state["theme"] == "dark" else '#ffffff'
    text_color = '#e8ebf1' if st.session_state["theme"] == "dark" else '#1f2430'
    grid_color = '#2a3441' if st.session_state["theme"] == "dark" else '#eef1f7'
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(showgrid=True, gridcolor=grid_color, showline=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, showline=False, zeroline=False),
        showlegend=False
    )
    
    return fig

def process_financial_query(query):
    """Process a financial query using the AI system"""
    if st.session_state.controller is None:
        return {"error": "AI system not available"}
    
    try:
        with st.spinner("🤖 AI analyzing your query..."):
            start_time = time.time()
            result = st.session_state.controller.process_user_query(query)
            processing_time = time.time() - start_time
            
            # Add to history
            st.session_state.query_history.insert(0, {
                "query": query,
                "timestamp": datetime.now(),
                "result": result,
                "processing_time": processing_time
            })
            
            # Keep only last 10 queries
            if len(st.session_state.query_history) > 10:
                st.session_state.query_history = st.session_state.query_history[:10]
            
            return result
            
    except Exception as e:
        error_result = {
            "error": True,
            "advisor_full_response": f"Error processing query: {str(e)}",
            "summary": "Query processing failed",
            "session_id": "error"
        }
        return error_result

def display_query_result(result):
    """Display the AI query result in a formatted way"""
    if result.get("error"):
        st.error(f"❌ {result.get('advisor_full_response', 'Unknown error')}")
        return
    
    # Main advisor response
    st.markdown("### 🎯 Financial Advice")
    advisor_response = result.get("advisor_full_response", "No response available")
    st.markdown(f'<div class="query-response">{advisor_response}</div>', unsafe_allow_html=True)
    
    # Summary and insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Quick Summary")
        summary = result.get("summary", "No summary available")
        st.info(summary)
        
    with col2:
        st.markdown("#### 💡 Key Insights")
        insights = result.get("detailed_insights", "No insights available")
        st.info(insights)
    
    # Executive summary
    exec_summary = result.get("executive_summary", {})
    if exec_summary and not exec_summary.get("error"):
        st.markdown("#### 📊 Executive Summary")
        
        tabs = st.tabs(["Key Takeaways", "Recommendations", "Risks", "Actions"])
        
        with tabs[0]:
            st.write(exec_summary.get("key_takeaways", "Not available"))
        
        with tabs[1]:
            st.write(exec_summary.get("investment_recommendation", "Not available"))
            
        with tabs[2]:
            st.write(exec_summary.get("risk_assessment", "Not available"))
            
        with tabs[3]:
            st.write(exec_summary.get("action_items", "Not available"))
    
    # Data analysis results
    analysis_results = result.get("analysis_results")
    if analysis_results and not analysis_results.get("error"):
        st.markdown("#### 📈 Data Analysis")
        
        # Data insights
        data_insights = analysis_results.get("data_insights", "")
        if data_insights:
            st.write("**Market Insights:**")
            st.write(data_insights)
        
        # Generated code
        analysis_code = analysis_results.get("analysis_code", "")
        if analysis_code:
            with st.expander("📄 View Generated Analysis Code"):
                st.code(analysis_code, language="python")
    
    # Web context
    web_context = result.get("web_context", [])
    if web_context:
        st.markdown("#### 📰 Related Market News")
        for i, article in enumerate(web_context[:3], 1):
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{article.get('headline', 'No headline')}</div>
                <div class="news-meta">{article.get('source', 'Unknown')} • JSE Market News</div>
                <p style="color: #a7b0c0; font-size: 14px; margin: 8px 0;">{article.get('summary', 'No summary')[:150]}...</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Session info
    with st.expander("ℹ️ Session Information"):
        st.json({
            "Session ID": result.get("session_id", "unknown"),
            "Request Type": result.get("request_type", "unknown"),
            "Memory Used": result.get("memory_used", False),
            "System Status": result.get("system_status", {})
        })

def main():
    """Main application function"""
    # Initialize session state FIRST
    initialize_session_state()
    
    # Apply theme styles AFTER session state is initialized
    apply_theme_styles()
    
    # Initialize controller if not done
    if st.session_state.controller is None and CONTROLLER_AVAILABLE:
        controller, status = initialize_controller()
        st.session_state.controller = controller
        st.session_state.controller_status = status
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📈 JSE AI Advisory</div>', unsafe_allow_html=True)
        
        # System status
        display_system_status()
        
        st.markdown("---")
        
        # Navigation
        sections = ["dashboard", "advisor", "datasci", "news", "logs", "settings"]
        labels = ["🏠 Dashboard", "💡 AI Advisor", "📊 Data Explorer", "📰 Market News", "🧪 System Logs", "⚙️ Settings"]
        
        for sec, label in zip(sections, labels):
            if st.button(label, key=f"nav_{sec}", use_container_width=True):
                st.session_state["section"] = sec
                st.rerun()
        
        # Theme toggle
        st.markdown("---")
        if st.button("🌗 Toggle Theme", key="theme_toggle", use_container_width=True):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()
        
        st.markdown('<p style="color: #a7b0c0; font-size: 12px; margin-top: 16px;">JSE-focused AI financial advisor</p>', unsafe_allow_html=True)

    # Main content area
    if st.session_state["section"] == "dashboard":
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("# JSE Market Dashboard")
        with col2:
            if st.button("🔄 Refresh Data", key="refresh_btn"):
                st.rerun()
        
        # Quick query bar
        st.markdown("### 🤖 Quick AI Query")
        query_col1, query_col2 = st.columns([4, 1])
        
        with query_col1:
            quick_query = st.text_input("", 
                placeholder="Ask about JSE stocks, sectors, or get investment advice...", 
                label_visibility="collapsed",
                key="quick_query")
        
        with query_col2:
            if st.button("Ask AI", type="primary", disabled=(not CONTROLLER_AVAILABLE or st.session_state.controller is None)):
                if quick_query:
                    result = process_financial_query(quick_query)
                    st.session_state.current_analysis = result
                    st.rerun()
        
        # Display current analysis if available
        if st.session_state.current_analysis:
            st.markdown("---")
            display_query_result(st.session_state.current_analysis)
        
        st.markdown("---")
        
        # Dashboard metrics and charts
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
            ai_status = "Online" if st.session_state.controller else "Offline"
            ai_pill_class = "pill-success" if st.session_state.controller else "pill-danger"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">AI Advisor Status</div>
                <div class="metric-value">{ai_status}</div>
                <div class="status-pill {ai_pill_class}">{'🤖 Ready' if st.session_state.controller else '⚠️ Unavailable'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts row
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container"><div class="chart-title">Market Overview</div></div>', unsafe_allow_html=True)
            fig = create_sample_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="chart-container"><div class="chart-title">Recent Queries</div></div>', unsafe_allow_html=True)
            
            if st.session_state.query_history:
                for i, query_item in enumerate(st.session_state.query_history[:3]):
                    timestamp = query_item["timestamp"].strftime("%H:%M")
                    query_text = query_item["query"][:50] + "..." if len(query_item["query"]) > 50 else query_item["query"]
                    processing_time = f"{query_item['processing_time']:.1f}s"
                    
                    st.markdown(f"""
                    <div class="news-card">
                        <div class="news-title">{query_text}</div>
                        <div class="news-meta">{timestamp} • {processing_time}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent queries")

    elif st.session_state["section"] == "advisor":
        st.markdown("# 🤖 AI Financial Advisor")
        
        if not CONTROLLER_AVAILABLE or st.session_state.controller is None:
            st.error("AI Advisory System is not available. Please check system configuration.")
            return
        
        # Query input
        st.markdown("### Ask Your Financial Question")
        user_query = st.text_area("", 
            placeholder="Example: Should I diversify my portfolio if I only hold JSE banking stocks?",
            height=100,
            label_visibility="collapsed")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🎯 Get Advice", type="primary", disabled=not user_query.strip()):
                if user_query.strip():
                    result = process_financial_query(user_query.strip())
                    display_query_result(result)
        
        with col2:
            if st.button("📋 Example Query"):
                example_queries = [
                    "Should I diversify my JSE banking portfolio?",
                    "What are the trends for AGL and NPN stocks?",
                    "How is the JSE mining sector performing?",
                    "What should I consider when investing in MTN shares?"
                ]
                st.session_state.example_query = np.random.choice(example_queries)
                st.rerun()
        
        # Show example query if set
        if st.session_state.example_query:
            st.info(f"💡 Example: {st.session_state.example_query}")
            if st.button("Use This Example"):
                result = process_financial_query(st.session_state.example_query)
                display_query_result(result)
        
        # Query history
        if st.session_state.query_history:
            st.markdown("### 📚 Query History")
            
            for i, query_item in enumerate(st.session_state.query_history[:5]):
                with st.expander(f"Query {i+1}: {query_item['query'][:60]}..."):
                    st.markdown(f"**Timestamp:** {query_item['timestamp']}")
                    st.markdown(f"**Processing Time:** {query_item['processing_time']:.1f}s")
                    
                    if st.button(f"View Results", key=f"view_result_{i}"):
                        display_query_result(query_item['result'])

    elif st.session_state["section"] == "datasci":
        st.markdown("# 📊 Data Explorer")
        
        st.info("💡 Tip: Ask the AI Advisor to generate data visualizations by including requests like 'show me a chart' or 'visualize the data'")
        
        # Data explorer interface
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
        
        if st.button("🤖 Generate AI Analysis", type="primary"):
            if ticker:
                ai_query = f"Analyze {ticker} stock for {analysis_type.lower()} from {start_date} to {end_date}. Show me a chart and provide insights."
                result = process_financial_query(ai_query)
                display_query_result(result)
            else:
                st.warning("Please enter a ticker symbol")

    elif st.session_state["section"] == "logs":
        st.markdown("# 🧪 System Logs & Validation")
        
        if st.session_state.controller:
            # System health check
            health = st.session_state.controller.health_check()
            system_info = st.session_state.controller.get_system_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### System Health")
                status_color = {"healthy": "success", "degraded": "warning", "critical": "error"}
                getattr(st, status_color.get(health['overall_status'], 'info'))(
                    f"Status: {health['overall_status'].title()}"
                )
                
                if health.get('critical_issues'):
                    st.error("Critical Issues:")
                    for issue in health['critical_issues']:
                        st.write(f"• {issue}")
                
                if health.get('warnings'):
                    st.warning("Warnings:")
                    for warning in health['warnings']:
                        st.write(f"• {warning}")
            
            with col2:
                st.markdown("### Agent Status")
                agent_health = health.get('agent_health', {})
                for agent, status in agent_health.items():
                    status_icon = {"healthy": "✅", "failed": "❌", "degraded": "⚠️"}
                    st.write(f"{status_icon.get(status, '❓')} {agent}: {status}")
            
            st.markdown("### System Information")
            with st.expander("View Full System Info"):
                st.json(system_info)
        
        else:
            st.error("System not initialized - cannot display logs")

    elif st.session_state["section"] == "news":
        st.markdown("# 📰 Market News")
        st.info("News section coming soon...")
    
    elif st.session_state["section"] == "settings":
        st.markdown("# ⚙️ Settings")
        st.info("Settings section coming soon...")

# Entry point
if __name__ == "__main__":
    main()