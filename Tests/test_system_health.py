# diagnostic.py - Check why system health is degraded
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_diagnostic():
    st.title("🔧 System Health Diagnostic")
    
    try:
        from Controller.autogen_controller import FinancialAdvisoryController
        
        st.header("🚀 Initializing System...")
        with st.spinner("Loading financial advisory system..."):
            controller = FinancialAdvisoryController()
        
        st.header("📊 Health Check Results")
        health = controller.health_check()
        
        # Display overall status
        st.subheader(f"Overall Status: {health['overall_status'].upper()}")
        
        if health['overall_status'] == 'healthy':
            st.success("✅ System is fully operational!")
        else:
            st.warning("⚠️ System is degraded - some features may not work optimally")
        
        # Display detailed agent status
        st.subheader("🔍 Agent Status Details")
        for agent_name, status in health['agent_statuses'].items():
            col1, col2 = st.columns([1, 3])
            with col1:
                if "Active" in status or "Template" in status:
                    st.success(f"✅ {agent_name}")
                else:
                    st.error(f"❌ {agent_name}")
            with col2:
                st.write(f"Status: {status}")
        
        # Display memory stats if available
        if 'memory_stats' in health:
            st.subheader("💾 Memory Statistics")
            st.json(health['memory_stats'])
        
        # Run detailed diagnostics for each agent
        st.header("🔬 Detailed Diagnostics")
        
        # Check Web Agent
        with st.expander("🌐 Web Supplementation Agent", expanded=True):
            try:
                web_agent = controller.agents['web']
                st.write("✅ Web Agent loaded successfully")
                
                # Test basic functionality
                test_result = web_agent.get_relevant_info("MTN")
                st.write(f"📊 Test query result: {len(test_result.get('tickers_analyzed', []))} tickers found")
                st.write(f"📰 News articles: {len(test_result.get('articles', []))}")
                
            except Exception as e:
                st.error(f"❌ Web Agent error: {e}")
        
        # Check Advisor Agent
        with st.expander("🤖 Advisor Agent", expanded=True):
            try:
                advisor_agent = controller.agents['advisor']
                st.write("✅ Advisor Agent loaded successfully")
                
                # Test if model is responding
                test_response = advisor_agent.explain_concept("What is a stock?")
                st.write(f"💡 Test response length: {len(test_response)} characters")
                
            except Exception as e:
                st.error(f"❌ Advisor Agent error: {e}")
        
        # Check Data Agent
        with st.expander("📈 Data Analysis Agent", expanded=True):
            try:
                data_agent = controller.agents['data']
                st.write(f"✅ Data Agent: {data_agent.get_model_info()['primary_approach']}")
                
                # Test code generation
                test_request = {
                    'task_type': 'stock_price_trend',
                    'stocks': ['MTN', 'VOD'],
                    'time_period': '1mo'
                }
                test_code = data_agent.generate_analysis_code(test_request)
                st.write(f"🐍 Generated code length: {len(test_code)} characters")
                
            except Exception as e:
                st.error(f"❌ Data Agent error: {e}")
        
        # Check Summarizer Agent
        with st.expander("📝 Summarizer Agent", expanded=True):
            try:
                summarizer_agent = controller.agents['summarizer']
                st.write("✅ Summarizer Agent loaded successfully")
                
                # Test summarization
                test_text = "This is a test text to see if the summarizer is working properly."
                summary = summarizer_agent.summarize_text(test_text)
                st.write(f"📋 Test summary: {summary}")
                
            except Exception as e:
                st.error(f"❌ Summarizer Agent error: {e}")
        
        # Check Memory Manager
        with st.expander("💾 Memory Manager", expanded=True):
            try:
                memory_manager = controller.agents['memory']
                st.write("✅ Memory Manager loaded successfully")
                
                # Test memory operations
                stats = memory_manager.get_memory_stats()
                st.write(f"🗂️ Memory sessions: {stats.get('total_sessions', 0)}")
                
            except Exception as e:
                st.error(f"❌ Memory Manager error: {e}")
        
        # Check Embedding Agent
        with st.expander("🔤 Embedding Agent", expanded=True):
            try:
                embedding_agent = controller.agents['embedding']
                st.write("✅ Embedding Agent loaded successfully")
                
                # Test embedding
                test_embedding = embedding_agent.get_embedding("test text")
                st.write(f"📐 Embedding dimensions: {len(test_embedding)}")
                
            except Exception as e:
                st.error(f"❌ Embedding Agent error: {e}")
        
        # Provide solutions
        st.header("🛠️ Possible Solutions")
        
        if "Template-Based Fallback" in str(health['agent_statuses']):
            st.info("""
            **Issue: Some agents are using template-based fallbacks**
            - This is normal for the Data Agent if the language model fails to load
            - The system will still work but with limited AI capabilities
            - Check if you have enough GPU memory for the models
            """)
        
        if "Inactive" in str(health['agent_statuses']):
            st.error("""
            **Issue: Some agents are completely inactive**
            - Check your internet connection for model downloads
            - Verify all required packages are installed: `pip install transformers torch sentence-transformers`
            - Restart the application
            """)
        
        if "Error" in str(health['agent_statuses']):
            st.error("""
            **Issue: Some agents have errors**
            - Check the console for detailed error messages
            - You may need to reinstall some dependencies
            - Consider using smaller models if you have limited resources
            """)
        
        st.success("""
        **🎯 Even with 'degraded' status, the system should still work!**
        - Template-based fallbacks ensure basic functionality
        - You can still analyze JSE stocks and get financial advice
        - The main AI capabilities might be limited but core features work
        """)
        
    except Exception as e:
        st.error(f"💥 Failed to run diagnostic: {e}")
        st.info("""
        **Quick fixes to try:**
        1. Restart the application
        2. Check your internet connection
        3. Make sure all requirements are installed: `pip install -r requirements.txt`
        4. Check if you have enough system resources (RAM/GPU)
        """)

if __name__ == "__main__":
    run_diagnostic()
