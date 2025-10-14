# test_visualization.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_visualization_queries():
    """Test queries that should trigger visualization code generation"""
    from Controller.autogen_controller import FinancialAdvisoryController
    
    controller = FinancialAdvisoryController()
    
    visualization_queries = [
        "Show me MTN candlestick chart for the last 3 months",
        "Plot moving averages for Standard Bank and FirstRand",
        "Create volume analysis for Vodacom stock",
        "Generate correlation matrix for mining stocks: Anglo, Sasol, Gold Fields",
        "Show me portfolio allocation chart for MTN 40%, NPN 30%, SBK 30%"
    ]
    
    for query in visualization_queries:
        print(f"\n🎯 Testing: {query}")
        response = controller.process_user_query(query)
        
        analysis = response.get('analysis_results')
        if analysis:
            print(f"✅ VISUALIZATION TRIGGERED: {analysis['analysis_type']}")
            print(f"📊 Stocks: {analysis['stocks_analyzed']}")
            print(f"💡 Insights: {analysis['insights'][:100]}...")
            print(f"🐍 Code Length: {len(analysis['code'])} characters")
        else:
            print("❌ No visualization generated")
        
        print("-" * 50)

if __name__ == "__main__":
    test_visualization_queries()
