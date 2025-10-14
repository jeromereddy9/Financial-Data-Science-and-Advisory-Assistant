# verify_memory_integration.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_memory_integration():
    """Test that memory integration is working"""
    print("🧠 TESTING MEMORY MANAGER INTEGRATION")
    print("=" * 50)
    
    try:
        from Controller.autogen_controller import FinancialAdvisoryController
        
        print("🔄 Initializing system with MemoryManager...")
        controller = FinancialAdvisoryController()
        
        # Test memory functionality
        memory_stats = controller.agents['memory'].get_memory_stats()
        print(f"📊 Memory Stats: {memory_stats}")
        
        # Test a query that should use memory
        test_query = "What's the current price of MTN?"
        print(f"\n📝 Test Query: {test_query}")
        
        response = controller.process_user_query(test_query)
        
        print(f"✅ Session ID: {response.get('session_id')}")
        print(f"📈 Tickers Analyzed: {response.get('tickers_analyzed', [])}")
        print(f"🧠 Memory Used: {response.get('memory_used')}")
        print(f"📋 Summary: {response.get('summary', '')[:100]}...")
        
        # Check memory stats after query
        updated_stats = controller.agents['memory'].get_memory_stats()
        print(f"\n🔄 Updated Memory Stats: {updated_stats.get('total_sessions', 0)} sessions")
        
        # Test memory search
        similar_sessions = controller.agents['memory'].search_memory("MTN stock price", top_k=2)
        print(f"🔍 Found {len(similar_sessions)} similar sessions in memory")
        
        print("\n🎉 MEMORY INTEGRATION SUCCESSFUL!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_integration()
