# Model/data_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List, Optional
import re

class DataAgent:
    """
    Data analysis agent that generates Python visualization code for JSE stock analysis.
    Uses template-based approach (primary) with optional LLM fallback for reliability.
    Templates ensure consistent, working code without hallucination risks.
    """
    
    def __init__(self):
        """
        Initialize DataAgent with optional LLM and pre-defined code templates.
        LLM is attempted but templates provide guaranteed functionality.
        """
        # Using a better instruction-following model for code generation (if available)
        self.model_name = "microsoft/DialoGPT-medium"  # Alternative: "google/flan-t5-base" for instruction following
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Attempt to load LLM for code generation (optional enhancement)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            
            # Ensure tokenizer has pad token for batching
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"[DataAgent] Successfully loaded model: {self.model_name}")
        except Exception as e:
            # Graceful fallback: templates work without LLM
            print(f"[DataAgent] Error loading model, falling back to template-based approach: {e}")
            self.model = None
            self.tokenizer = None
        
        # Pre-defined code templates for common JSE analysis tasks
        # These are production-ready, tested code snippets that avoid LLM hallucination
        self.code_templates = {
            "stock_price_trend": self._get_stock_trend_template(),
            "portfolio_allocation": self._get_portfolio_allocation_template(),
            "sector_comparison": self._get_sector_comparison_template(),
            "volume_analysis": self._get_volume_analysis_template(),
            "candlestick_chart": self._get_candlestick_template(),
            "correlation_matrix": self._get_correlation_template(),
            "moving_averages": self._get_moving_averages_template()
        }
    
    def generate_analysis_code(self, analysis_request: Dict[str, Any]) -> str:
        """
        Generate Python code for JSE stock analysis and visualization.
        Primary method: uses reliable templates. Fallback: LLM generation.
        
        Args:
            analysis_request: Dictionary containing:
                - task_type: Type of analysis (trend, allocation, comparison, etc.)
                - stocks: List of JSE stock codes (without .JO suffix)
                - time_period: Analysis period (1y, 6mo, 3mo, etc.)
                - additional_params: Any extra parameters (e.g., holdings for portfolio)
        
        Returns:
            Python code string for data analysis and visualization
        """
        try:
            task_type = analysis_request.get('task_type', 'stock_price_trend')
            stocks = analysis_request.get('stocks', ['AGL', 'SBK', 'NPN'])  # Default JSE blue chips
            time_period = analysis_request.get('time_period', '1Y')
            
            # Use template-based approach (more reliable than LLM for code generation)
            # Templates are pre-tested, production-ready code that always works
            if task_type in self.code_templates:
                code = self.code_templates[task_type]
                # Customize template with actual stock symbols and parameters
                code = self._customize_template(code, analysis_request)
                return code
            else:
                # Fallback: generate using LLM if available and task type not in templates
                return self._generate_code_with_llm(analysis_request)
        
        except Exception as e:
            print(f"[DataAgent] Error generating analysis code: {e}")
            # Final fallback: simple, guaranteed-to-work code
            return self._get_fallback_code(analysis_request)
    
    def generate_data_insights(self, data_context: str, analysis_type: str = "general") -> str:
        """
        Generate textual insights about financial data trends.
        Provides natural language interpretation of numerical analysis.
        
        Args:
            data_context: Description of the data or analysis results
            analysis_type: Type of analysis (trend, risk, performance, etc.)
        
        Returns:
            Textual analysis and insights for non-technical users
        """
        try:
            if self.model and self.tokenizer:
                # Use LLM for dynamic insights generation
                return self._generate_insights_with_llm(data_context, analysis_type)
            else:
                # Use template-based insights (reliable but less dynamic)
                return self._generate_template_insights(data_context, analysis_type)
        
        except Exception as e:
            print(f"[DataAgent] Error generating insights: {e}")
            return f"Analysis of {data_context}: Please review the data visualization for key trends and patterns."
    
    def _generate_code_with_llm(self, analysis_request: Dict[str, Any]) -> str:
        """
        Generate code using the language model.
        Optional fallback for custom analysis types not in templates.
        """
        if not self.model or not self.tokenizer:
            return self._get_fallback_code(analysis_request)
        
        try:
            task_type = analysis_request.get('task_type', 'stock_analysis')
            stocks = analysis_request.get('stocks', ['AGL', 'SBK'])
            
            # Construct detailed prompt for code generation
            prompt = f"""Generate Python code for JSE stock analysis.

Requirements:
- Create {task_type} visualization
- Use stocks: {stocks}
- Include proper imports (matplotlib, pandas, numpy, yfinance)
- Add JSE suffix (.JO) to stock symbols
- Include error handling
- Add clear titles and labels
- Make it production-ready

Python code:
```python"""
            
            # Tokenize with truncation to fit context window
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate code with low temperature for consistency
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=400,
                    temperature=0.3,  # Lower temperature for code generation (more deterministic)
                    do_sample=True,
                    top_p=0.8,  # Nucleus sampling for quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = result.replace(prompt, "").strip()
            
            # Clean up the generated code (remove markdown formatting)
            if "```python" in code:
                code = code.split("```python")[-1].split("```")[0].strip()
            
            # Validate generated code before returning
            return code if self._is_valid_python_code(code) else self._get_fallback_code(analysis_request)
        
        except Exception as e:
            print(f"[DataAgent] Error in LLM code generation: {e}")
            return self._get_fallback_code(analysis_request)
    
    def _generate_insights_with_llm(self, data_context: str, analysis_type: str) -> str:
        """
        Generate insights using the language model.
        Creates natural language interpretation of data analysis.
        """
        try:
            # Construct prompt for insight generation
            prompt = f"""Analyze this JSE financial data and provide insights.

Analysis Type: {analysis_type}
Data Context: {data_context}

Provide clear, actionable insights about:
1. Key trends and patterns
2. Risk factors to consider
3. Potential opportunities
4. Market implications for JSE investors

Analysis:"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate with higher temperature for natural text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=300,
                    temperature=0.7,  # Higher temperature for creative insights
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            insights = result.replace(prompt, "").strip()
            
            # Fallback to template if LLM output is insufficient
            return insights if insights else self._generate_template_insights(data_context, analysis_type)
        
        except Exception as e:
            print(f"[DataAgent] Error generating LLM insights: {e}")
            return self._generate_template_insights(data_context, analysis_type)
    
    def _get_stock_trend_template(self) -> str:
        """
        Template for stock price trend analysis.
        Line chart showing historical closing prices for multiple JSE stocks.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def analyze_jse_stock_trends(stocks=None, period="1y"):
    """Analyze JSE stock price trends"""
    if stocks is None:
        stocks = ["{stocks}"]  # Will be replaced with actual stocks
    
    # Add JSE suffix (.JO) to stock symbols if not present
    jse_stocks = [stock + ".JO" if not stock.endswith(".JO") else stock for stock in stocks]
    
    plt.figure(figsize=(12, 8))
    
    for stock in jse_stocks:
        try:
            # Download historical stock data from Yahoo Finance
            data = yf.download(stock, period=period, progress=False)
            if not data.empty:
                # Plot closing prices with stock ticker as label
                plt.plot(data.index, data['Close'], label=stock.replace('.JO', ''), linewidth=2)
        except Exception as e:
            print(f"Error downloading data for {stock}: {e}")
    
    # Customize plot appearance
    plt.title("JSE Stock Price Trends - {time_period}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (ZAR)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return "Stock trend analysis completed"

# Execute analysis
result = analyze_jse_stock_trends()
print(result)
'''
    
    def _get_portfolio_allocation_template(self) -> str:
        """
        Template for portfolio allocation visualization.
        Pie chart showing percentage distribution of holdings.
        """
        return '''
import matplotlib.pyplot as plt
import numpy as np

def create_portfolio_allocation_chart(holdings=None):
    """Create portfolio allocation pie chart"""
    if holdings is None:
        holdings = {holdings_data}  # Will be replaced with actual holdings dict
    
    # Extract labels (stock symbols) and sizes (percentages)
    labels = list(holdings.keys())
    sizes = list(holdings.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    plt.figure(figsize=(10, 8))
    
    # Create pie chart with percentage labels
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    
    # Enhance appearance: white text on colored wedges
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title("JSE Portfolio Allocation", fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures circular pie
    
    # Add legend with actual percentage values
    legend_labels = [f"{label}: {size:.1f}%" for label, size in zip(labels, sizes)]
    plt.legend(wedges, legend_labels, title="Holdings", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.show()
    
    return "Portfolio allocation chart created"

# Execute analysis
result = create_portfolio_allocation_chart()
print(result)
'''
    
    def _get_candlestick_template(self) -> str:
        """
        Template for candlestick chart.
        Shows OHLC (Open-High-Low-Close) data for detailed price action analysis.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

def create_candlestick_chart(stock="{stock}", period="3mo"):
    """Create candlestick chart for JSE stock"""
    jse_stock = stock + ".JO" if not stock.endswith(".JO") else stock
    
    try:
        # Download OHLC data
        data = yf.download(jse_stock, period=period, progress=False)
        
        if data.empty:
            print(f"No data available for {jse_stock}")
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each candlestick
        for i, (date, row) in enumerate(data.iterrows()):
            open_price, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
            
            # Color convention: green if close > open (bullish), red if close < open (bearish)
            color = 'green' if close >= open_price else 'red'
            
            # Draw the wick (high-low line representing full price range)
            ax.plot([i, i], [low, high], color='black', linewidth=1)
            
            # Draw the body (rectangle representing open-close range)
            height = abs(close - open_price)
            bottom = min(open_price, close)
            rect = Rectangle((i-0.3, bottom), 0.6, height, facecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        # Customize chart appearance
        ax.set_title(f"{stock} - JSE Candlestick Chart ({period})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel("Price (ZAR)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (show every 10th date to avoid clutter)
        step = max(1, len(data) // 10)
        ax.set_xticks(range(0, len(data), step))
        ax.set_xticklabels([data.index[i].strftime('%Y-%m-%d') for i in range(0, len(data), step)], rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return f"Candlestick chart created for {stock}"
        
    except Exception as e:
        print(f"Error creating candlestick chart: {e}")
        return "Error in chart creation"

# Execute analysis
result = create_candlestick_chart()
print(result)
'''
    
    def _get_volume_analysis_template(self) -> str:
        """
        Template for volume analysis.
        Dual-panel chart showing price movement and trading volume together.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def analyze_volume_trends(stocks=None, period="6mo"):
    """Analyze trading volume trends for JSE stocks"""
    if stocks is None:
        stocks = ["{stocks}"]
    
    # Create two subplots: price on top, volume on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    for stock in stocks:
        jse_stock = stock + ".JO" if not stock.endswith(".JO") else stock
        
        try:
            data = yf.download(jse_stock, period=period, progress=False)
            
            if not data.empty:
                # Plot price in top panel
                ax1.plot(data.index, data['Close'], label=f"{stock} Price", linewidth=2)
                
                # Plot volume as bar chart in bottom panel
                ax2.bar(data.index, data['Volume'], label=f"{stock} Volume", alpha=0.7, width=1)
        
        except Exception as e:
            print(f"Error processing {stock}: {e}")
    
    # Customize charts
    ax1.set_title("JSE Stock Price and Volume Analysis", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (ZAR)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return "Volume analysis completed"

# Execute analysis
result = analyze_volume_trends()
print(result)
'''
    
    def _get_sector_comparison_template(self) -> str:
        """
        Template for sector comparison.
        Bar chart comparing performance of different JSE sectors.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def compare_jse_sectors():
    """Compare major JSE sector performance"""
    # Representative stocks for major JSE sectors
    sector_stocks = {
        'Banking': ['SBK.JO', 'FSR.JO', 'NED.JO'],
        'Mining': ['AGL.JO', 'BIL.JO', 'GFI.JO'],
        'Retail': ['SHP.JO', 'TRU.JO', 'WHL.JO'],
        'Telecom': ['MTN.JO', 'VOD.JO']
    }
    
    sector_performance = {}
    
    plt.figure(figsize=(14, 8))
    
    # Calculate average return for each sector
    for sector, stocks in sector_stocks.items():
        sector_returns = []
        
        for stock in stocks:
            try:
                data = yf.download(stock, period="1y", progress=False)
                if not data.empty:
                    # Calculate percentage return over the period
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    return_pct = ((end_price - start_price) / start_price) * 100
                    sector_returns.append(return_pct)
            except:
                continue
        
        # Average returns across stocks in sector
        if sector_returns:
            avg_return = sum(sector_returns) / len(sector_returns)
            sector_performance[sector] = avg_return
    
    # Create bar chart with color-coded performance
    sectors = list(sector_performance.keys())
    returns = list(sector_performance.values())
    colors = ['green' if r >= 0 else 'red' for r in returns]  # Green for positive, red for negative
    
    bars = plt.bar(sectors, returns, color=colors, alpha=0.7)
    
    # Add percentage labels on top of bars
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{return_val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.title("JSE Sector Performance Comparison (1 Year)", fontsize=16, fontweight='bold')
    plt.xlabel("Sector", fontsize=12)
    plt.ylabel("Return (%)", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Zero line for reference
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return f"Sector comparison completed. Results: {sector_performance}"

# Execute analysis
result = compare_jse_sectors()
print(result)
'''
    
    def _get_moving_averages_template(self) -> str:
        """
        Template for moving averages analysis.
        Shows price with 20-day, 50-day, and 200-day moving averages for trend analysis.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def plot_moving_averages(stock="{stock}", period="1y"):
    """Plot stock price with moving averages"""
    jse_stock = stock + ".JO" if not stock.endswith(".JO") else stock
    
    try:
        # Download data
        data = yf.download(jse_stock, period=period, progress=False)
        
        if data.empty:
            return f"No data available for {stock}"
        
        # Calculate moving averages (common technical analysis indicators)
        data['MA20'] = data['Close'].rolling(window=20).mean()   # Short-term trend
        data['MA50'] = data['Close'].rolling(window=50).mean()   # Medium-term trend
        data['MA200'] = data['Close'].rolling(window=200).mean() # Long-term trend
        
        # Create plot with all moving averages
        plt.figure(figsize=(14, 8))
        
        plt.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
        plt.plot(data.index, data['MA20'], label='20-day MA', linewidth=1.5, color='orange', alpha=0.8)
        plt.plot(data.index, data['MA50'], label='50-day MA', linewidth=1.5, color='green', alpha=0.8)
        plt.plot(data.index, data['MA200'], label='200-day MA', linewidth=1.5, color='red', alpha=0.8)
        
        plt.title(f"{stock} - Price with Moving Averages", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (ZAR)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Provide textual analysis of current position relative to MAs
        current_price = data['Close'].iloc[-1]
        ma20 = data['MA20'].iloc[-1]
        ma50 = data['MA50'].iloc[-1]
        
        analysis = f"\\nCurrent Analysis for {stock}:\\n"
        analysis += f"Current Price: R{current_price:.2f}\\n"
        analysis += f"20-day MA: R{ma20:.2f} ({'Above' if current_price > ma20 else 'Below'})\\n"
        analysis += f"50-day MA: R{ma50:.2f} ({'Above' if current_price > ma50 else 'Below'})\\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in moving averages analysis: {e}"

# Execute analysis
result = plot_moving_averages()
print(result)
'''
    
    def _get_correlation_template(self) -> str:
        """
        Template for correlation matrix analysis.
        Heatmap showing how strongly different stocks move together.
        """
        return '''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def create_correlation_matrix(stocks=None, period="1y"):
    """Create correlation matrix for JSE stocks"""
    if stocks is None:
        stocks = ["{stocks}"]
    
    # Add JSE suffix
    jse_stocks = [stock + ".JO" if not stock.endswith(".JO") else stock for stock in stocks]
    
    # Download closing prices for all stocks
    price_data = pd.DataFrame()
    
    for stock in jse_stocks:
        try:
            data = yf.download(stock, period=period, progress=False)
            if not data.empty:
                price_data[stock.replace('.JO', '')] = data['Close']
        except Exception as e:
            print(f"Error downloading {stock}: {e}")
    
    if price_data.empty:
        return "No data available for correlation analysis"
    
    # Calculate correlation matrix (values from -1 to 1)
    # 1 = perfect positive correlation, -1 = perfect negative correlation, 0 = no correlation
    correlation_matrix = price_data.corr()
    
    # Create heatmap visualization
    plt.figure(figsize=(10, 8))
    
    # Mask upper triangle to avoid redundancy (correlation matrix is symmetric)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    
    plt.title("JSE Stock Correlation Matrix", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return f"Correlation analysis completed for {len(price_data.columns)} stocks"

# Execute analysis
result = create_correlation_matrix()
print(result)
'''
    
    def _customize_template(self, template: str, analysis_request: Dict[str, Any]) -> str:
        """
        Customize template with actual parameters.
        Replaces placeholder strings with real stock symbols, time periods, etc.
        """
        stocks = analysis_request.get('stocks', ['AGL', 'SBK', 'NPN'])
        time_period = analysis_request.get('time_period', '1y')
        
        # Replace stock placeholders (handles both single and multiple stocks)
        if isinstance(stocks, list):
            if len(stocks) == 1:
                template = template.replace('"{stock}"', f'"{stocks[0]}"')
            template = template.replace('"{stocks}"', str(stocks))
            template = template.replace('["{stocks}"]', str(stocks))
        
        # Replace time period placeholder
        template = template.replace('{time_period}', time_period)
        
        # Handle portfolio holdings if present (for allocation charts)
        holdings = analysis_request.get('holdings', {})
        if holdings:
            template = template.replace('{holdings_data}', str(holdings))
        
        return template
    
    def _get_fallback_code(self, analysis_request: Dict[str, Any]) -> str:
        """
        Fallback code when other methods fail.
        Simple, guaranteed-to-work visualization code.
        """
        stocks = analysis_request.get('stocks', ['AGL', 'SBK'])
        return f'''
# Basic JSE Stock Analysis
import yfinance as yf
import matplotlib.pyplot as plt

stocks = {stocks}
for stock in stocks:
    try:
        jse_stock = stock + ".JO"
        data = yf.download(jse_stock, period="6mo", progress=False)
        
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'])
        plt.title(f"{stock} - JSE Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price (ZAR)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error analyzing {stock}: {e}")
'''
    
    def _generate_template_insights(self, data_context: str, analysis_type: str) -> str:
        """
        Generate template-based insights.
        Pre-written insights for common analysis types when LLM unavailable.
        """
        # Library of generic but useful insights by analysis type
        insights_templates = {
            "trend": "The stock shows clear directional movement. Monitor key support and resistance levels for potential entry/exit points.",
            "volatility": "Price volatility indicates increased market uncertainty. Consider position sizing and risk management strategies.",
            "volume": "Trading volume patterns suggest institutional activity. Higher volume confirms price movements.",
            "correlation": "Stock correlations reveal sector and market relationships. Diversification opportunities exist in uncorrelated assets.",
            "general": "Key market trends are evident in the data. Consider broader economic factors affecting JSE performance."
        }
        
        base_insight = insights_templates.get(analysis_type, insights_templates["general"])
        return f"Data Analysis Insights: {base_insight}\n\nContext: {data_context}\n\nRecommendation: Review the visualization for specific entry/exit signals and consult with a financial advisor for personalized guidance."
    
    def _is_valid_python_code(self, code: str) -> bool:
        """
        Basic validation of generated Python code.
        Checks for presence of essential Python keywords.
        """
        if not code:
            return False
        
        # Check for basic Python structure (imports and plotting)
        python_keywords = ['import', 'def', 'plt.', 'yf.', 'data']
        return any(keyword in code for keyword in python_keywords)
    
    def get_available_analysis_types(self) -> List[str]:
        """
        Get list of available analysis types.
        Useful for UI dropdowns or validation.
        """
        return list(self.code_templates.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the data agent.
        Returns configuration and capabilities for system introspection.
        """
        return {
            "model_name": self.model_name if self.model else "Template-based",
            "available_analysis_types": self.get_available_analysis_types(),
            "primary_approach": "Template-based code generation with LLM fallback",
            "supported_exchanges": ["JSE (Johannesburg Stock Exchange)"]
        }