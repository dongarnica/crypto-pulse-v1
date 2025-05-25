import os
from dotenv import load_dotenv
from datetime import datetime
import pytz

load_dotenv()

class TradingConfig:
    def __init__(self, ticker="BTC"):
        self.ticker_short = ticker.upper()
        self.ticker_slash = f'{self.ticker_short}/USD'
        self.ticker_strip = f'{self.ticker_short}USD'
        self.model_name = f'ticker_model_{self.ticker_short.lower()}.keras'
        self.model_dir = os.path.join(os.getcwd(), "trade_models")
        
        # API Keys
        self.ai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_llm = os.getenv("MODEL_LLM")
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        
        # Trading Parameters
        self.risk_params = {
            'position_size': 0.02,  # 2% of portfolio
            'stop_loss_mult': 1.5,  # 1.5x ATR
            'take_profit_mult': 3   # 3x ATR
        }
        
        # Time Settings
        self.timezone = pytz.timezone('America/Denver')
        self.now = datetime.now(self.timezone)
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Training parameters
        self.epochs: int = 50
        self.patience: int = 5  # Early stopping patience