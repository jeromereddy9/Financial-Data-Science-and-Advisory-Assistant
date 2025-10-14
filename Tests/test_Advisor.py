# production_test.py
import sys
import os
import time

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def production_ready_test():
    """Test the fully fixed production system"""
    print("🎯 JSE STOCK ADVISOR - PRODUCTION READY TEST")
    print("=" * 55)
    
    try:
        from Controller.autogen_controller import FinancialAdvisoryController
        
        print("🔄 Initializing Production System...")
        start_time = time.time()
        controller = FinancialAdvisoryController()
        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        
        # Production test queries
        test_cases = [
            {
                "query": "Compare MTN and Vodacom stock performance with recent news",
                "description": "COMPARATIVE STOCK ANALYSIS"
            },
            {
                "query": "What are the current prices for Standard Bank, FirstRand, and Capitec?",
                "description": "MULTI-STOCK PRICE CHECK"
            },
            {
                "query": "Explain what dividend yield means for JSE investors",
                "description": "FINANCIAL CONCEPT EXPLANATION"
            },
            {
                "query": "Show me mining stocks performance: Anglo American, Sasol, and Gold Fields",
                "description": "SECTOR-SPECIFIC ANALYSIS"
            },
            {
                "query": "What's happening with Naspers and Prosus shares?",
                "description": "RELATED STOCKS ANALYSIS"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {test_case['description']}")
            print(f"QUERY: '{test_case['query']}'")
            print(f"{'='*60}")
            
            start_time = time.time()
            response = controller.process_user_query(test_case['query'])
            processing_time = time.time() - start_time
            
            print(f"⏰ Processing Time: {processing_time:.2f}s")
            print(f"📊 Request Type: {response.get('request_type')}")
            
            # Display advisor response
            advisor_response = response.get('advisor_full_response', '')
            if advisor_response:
                print(f"\n💡 ADVISOR ANALYSIS:")
                print("-" * 50)
                # Display in clean format
                lines = [line.strip() for line in advisor_response.split('\n') if line.strip()]
                for line in lines[:8]:  # Show first 8 lines
                    print(f"  {line}")
                if len(lines) > 8:
                    print(f"  ... (full response: {len(advisor_response)} characters)")
            
            # Show market data summary
            market_data = response.get('market_data_context')
            if market_data and "unavailable" not in str(market_data):
                print(f"\n📈 MARKET DATA: Available for {len(response.get('tickers_analyzed', []))} stocks")
            
            # Show news summary
            articles = response.get('web_context', [])
            if articles:
                print(f"📰 NEWS: {len(articles)} articles processed")
                for j, article in enumerate(articles[:2], 1):
                    print(f"     {j}. {article.get('headline', 'No headline')[:60]}...")
            
            # Show summary
            summary = response.get('summary', '')
            if summary and len(summary) > 50:
                print(f"\n📋 EXECUTIVE SUMMARY: {summary}")
            
            print(f"\n✅ TEST {i} COMPLETED")
            time.sleep(2)  # Brief pause between tests
        
        # Final system status
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        health = controller.health_check()
        print(f"🩺 FINAL SYSTEM STATUS: {health['overall_status']}")
        print(f"📊 Total Tests Run: {len(test_cases)}")
        print(f"💬 Conversation History: {len(controller.conversation_history)} exchanges")
        
        # Show working components
        print("\n✅ WORKING COMPONENTS:")
        for agent, status in health['agent_statuses'].items():
            if "Active" in status or "Template" in status:
                print(f"   ✓ {agent}: {status}")
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

def interactive_mode():
    """Interactive mode for live queries"""
    print("\n💬 INTERACTIVE JSE ADVISOR MODE")
    print("Type 'quit' to exit, 'status' for system info")
    print("=" * 50)
    
    from Controller.autogen_controller import FinancialAdvisoryController
    
    controller = FinancialAdvisoryController()
    
    while True:
        try:
            user_query = input("\n🎯 Ask about JSE stocks: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() == 'status':
                health = controller.health_check()
                print(f"System Status: {health['overall_status']}")
                continue
            elif user_query.lower() == 'reset':
                controller.reset_conversation()
                print("Conversation history reset")
                continue
            elif not user_query:
                continue
            
            print("⏳ Processing...")
            start_time = time.time()
            
            response = controller.process_user_query(user_query)
            processing_time = time.time() - start_time
            
            print(f"\n✅ Response ({processing_time:.2f}s):")
            print("-" * 60)
            print(response.get('advisor_full_response', 'No response generated'))
            
            # Quick stats
            market_data = response.get('market_data_context')
            if market_data and "unavailable" not in str(market_data):
                print(f"\n📊 Market data available")
            
            articles = response.get('web_context', [])
            if articles:
                print(f"📰 {len(articles)} news articles considered")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using JSE Advisor!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("JSE Stock Advisor - Production Test Suite")
    print("1. Run comprehensive production tests")
    print("2. Start interactive advisor mode")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        production_ready_test()
    elif choice == "2":
        interactive_mode()
    else:
        production_ready_test()