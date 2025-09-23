from Model.advisor_model import AdvisorAgent
from Model.data_model import DataAgent
from Model.web_model import WebAgent
from Model.summarizer_model import SummarizerAgent
from Model.memory_manager import MemoryManager
from Model.embedding_model import EmbeddingAgent  # wrapper around sentence-transformers

class AutoGenController:
    def __init__(self):
        self.web_agent = WebAgent()
        self.embedding_agent = EmbeddingAgent()
        self.advisor_agent = AdvisorAgent()
        self.data_agent = DataAgent()
        self.summarizer_agent = SummarizerAgent()
        self.memory_manager = MemoryManager()

    def handle_user_query(self, user_query):
        # Step 1: Web Agent retrieves relevant info
        web_texts = self.web_agent.fetch_financial_news(user_query)

        # Step 2: Embed the retrieved info
        embedded_texts = self.embedding_agent.embed_texts(web_texts)

        # Step 3: Check memory for context
        memory_contexts = self.memory_manager.search_memory(user_query)

        # Combine everything for the Advisor
        combined_context = "\n".join([t['summary'] for t in memory_contexts] + embedded_texts)

        # Step 4: Advisor generates financial advice / explanations
        advisor_response = self.advisor_agent.get_financial_advice(user_query, context=combined_context)

        # Step 5: Decide if data model should be invoked (graphs/analysis)
        if self.data_agent.should_generate_analysis(user_query, advisor_response):
            analysis_output = self.data_agent.generate_analysis(advisor_response)
            advisor_response += "\n\n" + analysis_output

        # Step 6: Summarizer condenses response
        summary, detailed_insights = self.summarizer_agent.create_insights(advisor_response)

        # Step 7: Store session in memory
        self.memory_manager.add_session(advisor_response)

        # Step 8: Return results for GUI
        return {
            "advisor_full_response": advisor_response,
            "summary": summary,
            "detailed_insights": detailed_insights,
            "analysis_output": analysis_output if 'analysis_output' in locals() else None
        }
