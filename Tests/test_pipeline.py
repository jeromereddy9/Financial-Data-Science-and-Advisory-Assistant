# test_pipeline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Controller import autogen_controller as autogen

def main():
    controller = autogen.AutoGenController()

    # Example user query
    user_query = "Should I diversify my portfolio if I only hold shares in banks on the JSE?"

    # Run the full pipeline
    results = controller.process_user_query(user_query)

    print("\n=== Full Advisor Response ===")
    print(results["advisor_full_response"])

    print("\n=== Summarized Advice ===")
    print(results["summary"])

    print("\n=== Detailed Insights ===")
    print(results["detailed_insights"])

    if results["analysis_results"]:
        print("\n=== Data Analysis Output ===")
        print(results["analysis_results"])

if __name__ == "__main__":
    main()

