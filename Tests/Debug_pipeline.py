# debug_test.py - Minimal test to identify the hanging issue

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_individual_agents():
    """Test each agent individually to find the problem"""
    
    print("=== DEBUGGING INDIVIDUAL AGENTS ===")
    
    # Test 1: Web Agent
    try:
        print("\n[1/6] Testing Web Agent...")
        from Model.web_model import WebSupplementationAgent
        web_agent = WebSupplementationAgent()
        result = web_agent.get_relevant_info("JSE banking", max_articles=2)
        print(f"? Web Agent: Found {len(result)} articles")
    except Exception as e:
        print(f"? Web Agent failed: {e}")
    
    # Test 2: Embedding Agent  
    try:
        print("\n[2/6] Testing Embedding Agent...")
        from Model.embeddings_model import EmbeddingAgent
        embed_agent = EmbeddingAgent()
        embedding = embed_agent.get_embedding("test text")
        print(f"? Embedding Agent: Generated embedding shape {embedding.shape}")
    except Exception as e:
        print(f"? Embedding Agent failed: {e}")
    
    # Test 3: Memory Manager
    try:
        print("\n[3/6] Testing Memory Manager...")
        from Controller.memory_manager import MemoryManager
        memory = MemoryManager()
        print("? Memory Manager: Initialized successfully")
    except Exception as e:
        print(f"? Memory Manager failed: {e}")
    
    # Test 4: Advisor Agent (most likely culprit)
    try:
        print("\n[4/6] Testing Advisor Agent...")
        from Model.advisor_model import AdvisorAgent
        advisor = AdvisorAgent()
        print("? Advisor Agent: Model loaded")
        
        # Test generation with simple input
        print("Testing simple generation...")
        response = advisor.get_financial_advice("What is portfolio diversification?")
        print(f"? Advisor Agent: Generated response ({len(response)} chars)")
        
    except Exception as e:
        print(f"? Advisor Agent failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Data Agent
    try:
        print("\n[5/6] Testing Data Agent...")
        from Model.data_model import DataAgent
        data_agent = DataAgent()
        code = data_agent.generate_analysis_code({"task_type": "stock_price_trend", "stocks": ["AGL"]})
        print(f"? Data Agent: Generated code ({len(code)} chars)")
    except Exception as e:
        print(f"? Data Agent failed: {e}")
    
    # Test 6: Summarizer Agent
    try:
        print("\n[6/6] Testing Summarizer Agent...")
        from Model.summarizer_model import SummarizerAgent
        summarizer = SummarizerAgent()
        summary, insights = summarizer.create_insights("This is a test financial advice response about JSE banking stocks.")
        print(f"? Summarizer Agent: Generated summary and insights")
    except Exception as e:
        print(f"? Summarizer Agent failed: {e}")

def test_minimal_controller():
    """Test minimal controller functionality"""
    print("\n=== TESTING MINIMAL CONTROLLER ===")
    
    try:
        from Controller.autogen_controller import FinancialAdvisoryController
        controller = FinancialAdvisoryController()
        
        # Test just the advisor part (skip web/data for now)
        print("\nTesting minimal query processing...")
        
        # Directly call advisor without web context
        advisor_response = controller.agents['advisor'].get_financial_advice(
            "Should I diversify my JSE banking portfolio?"
        )
        
        print(f"? Direct advisor call successful: {len(advisor_response)} chars")
        print(f"Response preview: {advisor_response[:200]}...")
        
    except Exception as e:
        print(f"? Minimal controller test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("?? DEBUGGING FINANCIAL ADVISORY SYSTEM")
    print("This will test each component individually to find issues.\n")
    
    test_individual_agents()
    test_minimal_controller()
    
    print("\n=== DEBUG COMPLETE ===")
    print("Check the results above to identify which component is causing issues.")
