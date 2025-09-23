import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.advisor_model import AdvisorAgent
from Model.data_model import DataAgent
from Model.web_model import WebSupplementationAgent
from Model.summarizer_model import SummarizerAgent
from .memory_manager import MemoryManager
from Model.embeddings_model import EmbeddingAgent  # wrapper around sentence-transformers

class AutoGenController:
    def __init__(self):
        self.web_agent = WebSupplementationAgent()
        self.embedding_agent = EmbeddingAgent()
        self.advisor_agent = AdvisorAgent()
        self.data_agent = DataAgent()
        self.summarizer_agent = SummarizerAgent()
        self.memory_manager = MemoryManager()

    def process_user_query(self, user_query):
        # 1. Fetch relevant web context
        external_texts = self.web_agent.get_relevant_info(user_query)
        
        # 2. Include memory context if needed
        memory_contexts = self.memory_manager.search_memory(user_query)
        memory_summaries = [m['summary'] for m in memory_contexts]

        external_summaries = [a["summary"] for a in external_texts if "summary" in a]

        # Combine web info + memory context for advisor
        combined_context = "\n".join(memory_summaries + external_texts)

        # 3. Advisor generates advice using combined context
        advisor_response = self.advisor_agent.get_advice(user_query, context=combined_context)

        # 4. Check if the advisor mentions visualizations
        if "visualize" in advisor_response.lower():
            analysis_code = self.data_agent.generate_analysis(advisor_response)
            analysis_results = execute_code(analysis_code)  # run the code safely
        else:
            analysis_results = None

        # 5. Summarize advisor response for display
        summary, detailed_insights = self.summarizer_agent.summarize(advisor_response)

        # 6. Add the current session to memory
        self.memory_manager.add_session(advisor_response)

        return {
            "advisor_full_response": advisor_response,
            "summary": summary,
            "detailed_insights": detailed_insights,
            "analysis_results": analysis_results
        }
