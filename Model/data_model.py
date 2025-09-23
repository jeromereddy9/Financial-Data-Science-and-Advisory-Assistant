from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class DataAgent:
    def __init__(self):
        self.model_name = "bigcode/starcoderbase-1b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_analysis_code(self, data_description, analysis_type="basic"):
        """
        Generate Python code for data analysis
        """
        prompt = f"""# Generate Python code for {analysis_type} analysis
# Data: {data_description}
# Requirements: Use pandas, matplotlib, seaborn for visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and analyze data
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=["# End", "```"]
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = response.replace(prompt, "").strip()
        return code
    
    def create_visualization_code(self, chart_type, data_columns):
        """
        Generate code for specific visualizations
        """
        prompt = f"""# Create a {chart_type} visualization
# Data columns: {', '.join(data_columns)}

import matplotlib.pyplot as plt
import seaborn as sns

# Create {chart_type} plot
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = response.replace(prompt, "").strip()
        return code
    
    def explain_analysis(self, code, results):
        """
        Generate explanations for analysis results
        """
        prompt = f"""# Explain the following data analysis results:
# Code used: {code[:200]}...
# Results: {results}

# Analysis Explanation:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = response.split("# Analysis Explanation:")[-1].strip()
        return explanation
