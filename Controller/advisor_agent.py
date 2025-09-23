import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))
from advisor_model import AdvisorAgent

class AdvisorController:
    def __init__(self):
        self.advisor = AdvisorAgent()
    
    def handle_advice_request(self, query, context=""):
        return self.advisor.get_financial_advice(query, context)
    
    def handle_explanation_request(self, concept, level="beginner"):
        return self.advisor.explain_concept(concept, level)
