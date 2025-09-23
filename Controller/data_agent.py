import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model'))
from data_model import DataAgent

class DataController:
    def __init__(self):
        self.data_agent = DataAgent()
    
    def handle_analysis_request(self, data_description, analysis_type="basic"):
        return self.data_agent.generate_analysis_code(data_description, analysis_type)
    
    def handle_visualization_request(self, chart_type, data_columns):
        return self.data_agent.create_visualization_code(chart_type, data_columns)
    
    def handle_explanation_request(self, code, results):
        return self.data_agent.explain_analysis(code, results)
