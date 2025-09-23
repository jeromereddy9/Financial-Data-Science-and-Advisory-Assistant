# test_pipeline.py
from Controller import autogen_controller as autogen

def main():
    controller = autogen.AutoGenController()

    # Example user query
    user_query = "Should I diversify my portfolio if I only hold shares in banks on the JSE?"

    # Run the full pipeline
    results = controller.handle_user_query(user_query)

    print("\n=== Full Advisor Response ===")
    print(results["advisor_full_response"])

    print("\n=== Summarized Advice ===")
    print(results["summary"])

    print("\n=== Detailed Insights ===")
    print(results["detailed_insights"])

    if results["analysis_output"]:
        print("\n=== Data Analysis Output ===")
        print(results["analysis_output"])

if __name__ == "__main__":
    main()

