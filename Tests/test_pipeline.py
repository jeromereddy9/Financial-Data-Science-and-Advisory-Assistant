# test_pipeline.py
import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Controller.autogen_controller import FinancialAdvisoryController
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the Controller module is in the correct path")
    sys.exit(1)

def print_separator(title, char="=", length=60):
    """Print a formatted separator with title"""
    print(f"\n{char * length}")
    print(f" {title.center(length - 2)} ")
    print(f"{char * length}")

def print_dict_formatted(data, title="", indent=0):
    """Print dictionary in a formatted way"""
    if title:
        print(" " * indent + f"{title}:")
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict_formatted(value, indent=indent + 2)
        elif isinstance(value, list):
            print(" " * indent + f"{key}: [{len(value)} items]")
            for i, item in enumerate(value[:3]):  # Show first 3 items
                if isinstance(item, dict):
                    print(" " * (indent + 2) + f"{i+1}. {item.get('headline', item.get('summary', str(item)[:50]))}")
                else:
                    print(" " * (indent + 2) + f"{i+1}. {str(item)[:50]}")
            if len(value) > 3:
                print(" " * (indent + 2) + f"... and {len(value) - 3} more")
        else:
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(" " * indent + f"{key}: {value_str}")

def test_single_query(controller, query, description=""):
    """Test a single query and display results"""
    print_separator(f"Testing Query: {description}" if description else "Testing Query")
    print(f"Query: {query}")
    
    try:
        # Process the query
        print("\n[Processing...] This may take a moment while agents load and process...")
        results = controller.process_user_query(query)
        
        # Display results
        if results.get("error"):
            print_separator("ERROR OCCURRED", "!")
            print(f"Error: {results.get('executive_summary', {}).get('error', 'Unknown error')}")
            print(f"System Status: {results.get('system_status', {})}")
            return False
        
        # System Information
        print_separator("SYSTEM STATUS")
        system_status = results.get("system_status", {})
        print(f"Request Type: {results.get('request_type', 'unknown')}")
        print(f"Session ID: {results.get('session_id', 'unknown')}")
        print(f"Memory Used: {'Yes' if results.get('memory_used') else 'No'}")
        
        active_agents = [agent for agent, status in system_status.items() if status]
        inactive_agents = [agent for agent, status in system_status.items() if not status]
        
        print(f"Active Agents: {', '.join(active_agents)}")
        if inactive_agents:
            print(f"Inactive Agents: {', '.join(inactive_agents)}")
        
        # Main Advisor Response
        print_separator("FINANCIAL ADVISOR RESPONSE")
        advisor_response = results.get("advisor_full_response", "No response available")
        print(advisor_response)
        
        # Summary Information
        print_separator("SUMMARY & INSIGHTS")
        
        summary = results.get("summary", "")
        if summary:
            print("Quick Summary:")
            print(summary)
        
        detailed_insights = results.get("detailed_insights", "")
        if detailed_insights:
            print("\nDetailed Insights:")
            print(detailed_insights)
        
        # Executive Summary
        executive_summary = results.get("executive_summary", {})
        if executive_summary and not executive_summary.get("error"):
            print_separator("EXECUTIVE SUMMARY")
            print_dict_formatted(executive_summary)
        
        # Data Analysis Results
        analysis_results = results.get("analysis_results")
        if analysis_results:
            print_separator("DATA ANALYSIS & VISUALIZATION")
            
            if analysis_results.get("error"):
                print(f"Analysis Error: {analysis_results['error']}")
                print(f"Fallback: {analysis_results.get('fallback_message', '')}")
            else:
                print(f"Chart Type: {analysis_results.get('chart_type', 'unknown')}")
                print(f"Analysis Request: {analysis_results.get('analysis_request', {})}")
                
                data_insights = analysis_results.get("data_insights", "")
                if data_insights:
                    print(f"\nData Insights: {data_insights}")
                
                analysis_code = analysis_results.get("analysis_code", "")
                if analysis_code:
                    print("\nGenerated Analysis Code:")
                    print("-" * 40)
                    print(analysis_code[:500] + "..." if len(analysis_code) > 500 else analysis_code)
                    print("-" * 40)
        
        # Web Context Information
        web_context = results.get("web_context", [])
        if web_context:
            print_separator("WEB CONTEXT (Recent News)")
            for i, article in enumerate(web_context, 1):
                print(f"{i}. {article.get('headline', 'No headline')}")
                print(f"   Source: {article.get('source', 'Unknown')}")
                print(f"   Summary: {article.get('summary', 'No summary')[:100]}...")
                print()
        
        print_separator("QUERY COMPLETED SUCCESSFULLY", "?")
        return True
        
    except Exception as e:
        print_separator("CRITICAL ERROR", "!")
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test(controller):
    """Run multiple test queries to verify system functionality"""
    
    test_queries = [
        {
            "query": "Should I diversify my portfolio if I only hold shares in banks on the JSE?",
            "description": "Portfolio Diversification Query"
        },
        {
            "query": "What are the trends for AGL and SBK stocks? Show me a chart comparing their performance.",
            "description": "Stock Analysis with Visualization"
        },
        {
            "query": "How is the JSE mining sector performing compared to banking? I want to see the data.",
            "description": "Sector Comparison Query"
        },
        {
            "query": "What should I consider when investing in MTN shares?",
            "description": "Individual Stock Analysis"
        }
    ]
    
    print_separator("COMPREHENSIVE SYSTEM TEST", "=", 80)
    print(f"Running {len(test_queries)} test queries...")
    
    successful_tests = 0
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[TEST {i}/{len(test_queries)}]")
        
        success = test_single_query(
            controller, 
            test["query"], 
            test["description"]
        )
        
        if success:
            successful_tests += 1
        
        # Small delay between tests
        import time
        time.sleep(2)
    
    # Final Results
    print_separator("TEST RESULTS SUMMARY", "=", 80)
    print(f"Successful Tests: {successful_tests}/{len(test_queries)}")
    print(f"Success Rate: {(successful_tests/len(test_queries)*100):.1f}%")
    
    if successful_tests == len(test_queries):
        print("?? All tests passed! System is functioning correctly.")
    elif successful_tests > 0:
        print("??  Some tests passed. System has partial functionality.")
    else:
        print("? All tests failed. System requires attention.")

def main():
    """Main test function"""
    print_separator("FINANCIAL ADVISORY SYSTEM - TEST PIPELINE", "=", 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize the controller
        print("\n[INITIALIZATION] Loading Financial Advisory System...")
        controller = FinancialAdvisoryController()
        
        # Display system information
        print_separator("SYSTEM INFORMATION")
        system_info = controller.get_system_info()
        
        print(f"System: {system_info['system_name']}")
        print(f"Version: {system_info['version']}")
        print(f"Target Market: {system_info['target_market']}")
        
        # Display agent status
        print("\nAgent Status:")
        for agent_name, status in system_info['system_status'].items():
            status_icon = "?" if status else "?"
            print(f"  {status_icon} {agent_name}: {'Active' if status else 'Inactive'}")
        
        # Health check
        health = controller.health_check()
        print(f"\nOverall System Health: {health['overall_status'].upper()}")
        
        if health['critical_issues']:
            print("Critical Issues:")
            for issue in health['critical_issues']:
                print(f"  - {issue}")
        
        if health['warnings']:
            print("Warnings:")
            for warning in health['warnings']:
                print(f"  - {warning}")
        
        # Run tests based on system health
        if health['overall_status'] in ['healthy', 'degraded']:
            print("\n[STARTING TESTS] System ready for testing...")
            run_comprehensive_test(controller)
        else:
            print("\n[SYSTEM CRITICAL] Cannot run tests due to critical system issues.")
            print("Please check agent initialization and resolve issues before testing.")
    
    except Exception as e:
        print_separator("INITIALIZATION ERROR", "!")
        print(f"Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting Tips:")
        print("1. Check that all model files are accessible")
        print("2. Verify internet connection for downloading models")
        print("3. Ensure sufficient disk space and memory")
        print("4. Check file permissions in the project directory")

if __name__ == "__main__":
    main()