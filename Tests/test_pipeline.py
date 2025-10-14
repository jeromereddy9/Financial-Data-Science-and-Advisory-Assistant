# test_pipeline.py - Comprehensive JSE Advisory System Testing
import sys
import os
import time
from datetime import datetime
import json

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Controller.autogen_controller import FinancialAdvisoryController

class PipelineTester:
    def __init__(self):
        self.controller = None
        self.test_results = []
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the financial advisory system"""
        print("🚀 INITIALIZING JSE FINANCIAL ADVISORY SYSTEM...")
        print("=" * 60)
        
        try:
            start_time = time.time()
            self.controller = FinancialAdvisoryController()
            init_time = time.time() - start_time
            
            # Health check
            health = self.controller.health_check()
            
            print(f"✅ System initialized in {init_time:.2f} seconds")
            print(f"📊 Overall Status: {health['overall_status'].upper()}")
            
            # Display agent status
            print("\n🔧 AGENT STATUS:")
            for agent, status in health['agent_statuses'].items():
                status_icon = "✅" if "Active" in status or "Template" in status else "⚠️"
                print(f"   {status_icon} {agent}: {status}")
            
            # Memory stats
            if self.controller.agents.get('memory'):
                memory_stats = self.controller.agents['memory'].get_memory_stats()
                print(f"💾 Memory Sessions: {memory_stats.get('total_sessions', 0)}")
            
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    def run_test_query(self, query, category, expected_tickers=None):
        """Run a single test query and record results"""
        print(f"\n🎯 TESTING: {category}")
        print(f"📝 Query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Process the query
            response = self.controller.process_user_query(query)
            processing_time = time.time() - start_time
            
            # Extract results
            result = {
                'category': category,
                'query': query,
                'processing_time': processing_time,
                'success': True,
                'response_length': len(response.get('advisor_full_response', '')),
                'tickers_analyzed': response.get('tickers_analyzed', []),
                'request_type': response.get('request_type'),
                'articles_found': len(response.get('web_context', [])),
                'has_market_data': bool(response.get('market_data_context')),
                'has_analysis': bool(response.get('analysis_results')),
                'session_id': response.get('session_id'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Display results
            print(f"✅ Success: Processed in {processing_time:.2f}s")
            print(f"📊 Response: {result['response_length']} characters")
            print(f"📈 Tickers: {result['tickers_analyzed']}")
            print(f"📰 Articles: {result['articles_found']}")
            print(f"🔬 Analysis: {'Yes' if result['has_analysis'] else 'No'}")
            print(f"💾 Session: {result['session_id']}")
            
            # Show snippet of response
            advisor_response = response.get('advisor_full_response', '')[:200] + "..." if len(response.get('advisor_full_response', '')) > 200 else response.get('advisor_full_response', '')
            print(f"💡 Response snippet: {advisor_response}")
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            error_result = {
                'category': category,
                'query': query,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"❌ Error: {e}")
            self.test_results.append(error_result)
            return error_result
    
    def run_comprehensive_tests(self):
        """Run a comprehensive suite of test queries"""
        if not self.controller:
            print("❌ System not initialized. Cannot run tests.")
            return
        
        print("\n" + "=" * 60)
        print("🧪 COMPREHENSIVE PIPELINE TESTING")
        print("=" * 60)
        
        test_queries = [
            # Basic Stock Queries
            ("What is the current price of MTN?", "Basic Stock Price"),
            ("Show me MTN and Vodacom prices", "Multiple Stock Prices"),
            ("MTN share price today", "Stock Price Short"),
            
            # Comparative Analysis
            ("Compare MTN and Vodacom stock performance", "Stock Comparison"),
            ("Compare Standard Bank and FirstRand", "Banking Sector Comparison"),
            ("Show me mining stocks: Anglo American vs Sasol", "Mining Sector Comparison"),
            
            # Sector Analysis
            ("How is the banking sector performing?", "Sector Analysis"),
            ("What's happening with JSE telecom stocks?", "Telecom Sector"),
            ("Analyze the mining sector on JSE", "Mining Sector"),
            
            # News & Market Context
            ("Show me recent news about Naspers", "Stock News"),
            ("What's the latest news affecting JSE stocks?", "Market News"),
            ("News about Standard Bank shares", "Specific Stock News"),
            
            # Technical Analysis Requests
            ("Show me candlestick chart for MTN", "Candlestick Chart"),
            ("Plot moving averages for Anglo American", "Moving Averages"),
            ("Analyze trading volume for Vodacom", "Volume Analysis"),
            ("Create correlation matrix for mining stocks", "Correlation Analysis"),
            
            # Portfolio & Allocation
            ("Show portfolio allocation for MTN 40%, NPN 30%, SBK 30%", "Portfolio Allocation"),
            ("Analyze my holdings: MTN, Naspers, Standard Bank", "Portfolio Analysis"),
            
            # Conceptual Explanations
            ("Explain what P/E ratio means", "Financial Concept"),
            ("What is dividend yield?", "Dividend Concept"),
            ("Explain market capitalization", "Market Cap Concept"),
            
            # Complex Multi-part Queries
            ("Compare MTN and Vodacom performance with recent news and show me charts", "Complex Analysis"),
            ("Analyze Standard Bank stock with technical indicators and market context", "Technical + Fundamental"),
            
            # Edge Cases
            ("What's the weather today?", "Irrelevant Query"),
            ("", "Empty Query"),
            ("Show me stock for unknown company XYZ", "Unknown Company"),
        ]
        
        total_tests = len(test_queries)
        successful_tests = 0
        total_processing_time = 0
        
        for i, (query, category) in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}/{total_tests}")
            result = self.run_test_query(query, category)
            
            if result['success']:
                successful_tests += 1
                total_processing_time += result['processing_time']
            
            # Small delay between tests to avoid rate limiting
            if i < len(test_queries):
                time.sleep(2)
        
        # Generate test summary
        self.generate_test_summary(total_tests, successful_tests, total_processing_time)
    
    def generate_test_summary(self, total_tests, successful_tests, total_processing_time):
        """Generate a comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY REPORT")
        print("=" * 60)
        
        success_rate = (successful_tests / total_tests) * 100
        avg_processing_time = total_processing_time / successful_tests if successful_tests > 0 else 0
        
        print(f"✅ Successful Tests: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.2f}s")
        print(f"🕒 Total Testing Time: {total_processing_time:.2f}s")
        
        # Categorize results
        categories = {}
        for result in self.test_results:
            if result['success']:
                category = result['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
        
        print(f"\n📈 RESULTS BY CATEGORY:")
        for category, results in categories.items():
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            print(f"   {category}: {len(results)} tests, avg {avg_time:.2f}s")
        
        # Identify fastest and slowest tests
        if self.test_results:
            successful_results = [r for r in self.test_results if r['success']]
            if successful_results:
                fastest = min(successful_results, key=lambda x: x['processing_time'])
                slowest = max(successful_results, key=lambda x: x['processing_time'])
                
                print(f"\n⚡ Fastest Test: '{fastest['query']}' - {fastest['processing_time']:.2f}s")
                print(f"🐌 Slowest Test: '{slowest['query']}' - {slowest['processing_time']:.2f}s")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for failed in failed_tests:
                print(f"   - {failed['category']}: '{failed['query']}'")
                print(f"     Error: {failed['error']}")
        
        # Save detailed results to file
        self.save_test_results()
    
    def save_test_results(self):
        """Save test results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_test_results_{timestamp}.json"
        
        results_data = {
            'test_timestamp': datetime.now().isoformat(),
            'system_health': self.controller.health_check() if self.controller else {},
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results if r['success']]),
                'failed_tests': len([r for r in self.test_results if not r['success']])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
    
    def test_specific_scenarios(self):
        """Test specific scenarios of interest"""
        print("\n" + "=" * 60)
        print("🎯 SPECIFIC SCENARIO TESTING")
        print("=" * 60)
        
        scenarios = [
            # Data Agent Visualization Tests
            ("Generate code for MTN candlestick chart", "Candlestick Code Generation"),
            ("Create moving averages code for multiple stocks", "Technical Analysis Code"),
            ("Show me portfolio allocation visualization code", "Portfolio Visualization Code"),
            
            # Memory Functionality Tests
            ("What did we discuss about MTN?", "Memory Recall - First Query"),
            ("Tell me more about banking stocks we talked about", "Memory Recall - Context"),
            
            # Web Agent Comprehensive Tests
            ("Get comprehensive data for top 5 JSE stocks", "Multi-Stock Data Fetch"),
            ("Fetch latest news for financial sector", "Sector News Fetch"),
            
            # Advisor Agent Complex Reasoning
            ("Based on current market data, what's your outlook for telecom stocks?", "Market Outlook"),
            ("Considering recent news, should I invest in mining stocks?", "Investment Advice"),
        ]
        
        for query, scenario in scenarios:
            self.run_test_query(query, scenario)
    
    def stress_test(self, num_queries=10):
        """Run multiple similar queries to test performance"""
        print(f"\n⚡ STRESS TEST: {num_queries} consecutive queries")
        
        base_query = "Show me MTN stock price"
        start_time = time.time()
        
        for i in range(num_queries):
            query = f"{base_query} - test {i+1}"
            self.run_test_query(query, f"Stress Test {i+1}")
        
        total_time = time.time() - start_time
        print(f"\n📊 Stress Test Complete: {num_queries} queries in {total_time:.2f}s")
        print(f"📈 Average: {total_time/num_queries:.2f}s per query")

def main():
    """Main testing function"""
    print("JSE ADVISORY SYSTEM - PIPELINE TESTER")
    print("This will test the entire system with various query types.")
    print("It may take several minutes to complete.\n")
    
    tester = PipelineTester()
    
    if not tester.controller:
        print("❌ Cannot proceed with testing - system initialization failed.")
        return
    
    # Ask user what type of testing to perform
    print("\nSelect test type:")
    print("1. Comprehensive Test Suite (Recommended)")
    print("2. Specific Scenario Testing")
    print("3. Stress Test")
    print("4. All Tests")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            tester.run_comprehensive_tests()
        elif choice == "2":
            tester.test_specific_scenarios()
        elif choice == "3":
            num_queries = input("Number of queries for stress test (default 10): ").strip()
            num_queries = int(num_queries) if num_queries.isdigit() else 10
            tester.stress_test(num_queries)
        elif choice == "4":
            tester.run_comprehensive_tests()
            tester.test_specific_scenarios()
            tester.stress_test()
        else:
            print("Invalid choice. Running comprehensive tests...")
            tester.run_comprehensive_tests()
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Testing error: {e}")

if __name__ == "__main__":
    main()