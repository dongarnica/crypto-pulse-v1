import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
from typing import Dict, Any, Optional

load_dotenv()

class AppConfig:
    """
    Centralized configuration class for managing API keys, secrets, and application settings.
    Provides both class-based and dictionary-like access to configuration values.
    """
    
    def __init__(self, ticker: str = "BTC"):
        self.ticker_short = ticker.upper()
        self.ticker_slash = f'{self.ticker_short}/USD'
        self.ticker_strip = f'{self.ticker_short}USD'
        self.model_name = f'ticker_model_{self.ticker_short.lower()}.keras'
        self.model_dir = os.path.join(os.getcwd(), "trade_models")
        
        # API Keys and Secrets
        self.ai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_llm = os.getenv("MODEL_LLM", "gpt-4")
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        
        # Alpaca Configuration
        self.alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.alpaca_data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Trading Parameters
        self.risk_params = {
            'position_size': float(os.getenv("POSITION_SIZE", "0.02")),  # 2% of portfolio
            'stop_loss_mult': float(os.getenv("STOP_LOSS_MULT", "1.5")),  # 1.5x ATR
            'take_profit_mult': float(os.getenv("TAKE_PROFIT_MULT", "3")),  # 3x ATR
            'max_positions': int(os.getenv("MAX_POSITIONS", "5"))
        }
        
        # Time Settings
        self.timezone = pytz.timezone('America/Denver')
        self.now = datetime.now(self.timezone)
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Training parameters
        self.epochs: int = int(os.getenv("LSTM_EPOCHS", "50"))
        self.patience: int = int(os.getenv("LSTM_PATIENCE", "5"))  # Early stopping patience
        self.batch_size: int = int(os.getenv("LSTM_BATCH_SIZE", "32"))
        self.lookback: int = int(os.getenv("LSTM_LOOKBACK", "60"))
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/trading.log")
        
        # Validate required keys
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that required configuration values are present."""
        required_keys = {
            'ALPACA_API_KEY': self.alpaca_api_key,
            'ALPACA_SECRET_KEY': self.alpaca_secret,
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            print(f"Warning: Missing required environment variables: {', '.join(missing_keys)}")
            print("Please ensure these are set in your .env file or environment.")
    
    def get_alpaca_config(self) -> Dict[str, str]:
        """Get Alpaca-specific configuration as a dictionary."""
        return {
            'api_key': self.alpaca_api_key,
            'api_secret': self.alpaca_secret,
            'base_url': self.alpaca_base_url,
            'data_url': self.alpaca_data_url
        }
    
    def get_llm_config(self) -> Dict[str, str]:
        """Get LLM-specific configuration as a dictionary."""
        return {
            'api_key': self.ai_api_key,
            'model': self.model_llm,
            'perplexity_key': self.perplexity_key
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration as a dictionary."""
        return {
            'ticker_short': self.ticker_short,
            'ticker_slash': self.ticker_slash,
            'ticker_strip': self.ticker_strip,
            'risk_params': self.risk_params,
            'model_dir': self.model_dir,
            'model_name': self.model_name
        }
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to config values."""
        config_map = {
            'ALPACA_API_KEY': self.alpaca_api_key,
            'ALPACA_SECRET_KEY': self.alpaca_secret,
            'ALPACA_API_SECRET': self.alpaca_secret,  # Alternative name
            'ALPACA_BASE_URL': self.alpaca_base_url,
            'OPENAI_API_KEY': self.ai_api_key,
            'MODEL_LLM': self.model_llm,
            'PERPLEXITY_API_KEY': self.perplexity_key,
            'TICKER_SHORT': self.ticker_short,
            'TICKER_SLASH': self.ticker_slash,
            'TICKER_STRIP': self.ticker_strip,
            'MODEL_DIR': self.model_dir,
            'MODEL_NAME': self.model_name,
            'LOG_LEVEL': self.log_level,
            'LOG_FILE': self.log_file
        }
        
        if key in config_map:
            return config_map[key]
        
        # Try to get from risk_params
        if key in self.risk_params:
            return self.risk_params[key]
        
        raise KeyError(f"Configuration key '{key}' not found")
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire configuration to a dictionary."""
        return {
            # API Keys
            'ALPACA_API_KEY': self.alpaca_api_key,
            'ALPACA_SECRET_KEY': self.alpaca_secret,
            'ALPACA_BASE_URL': self.alpaca_base_url,
            'OPENAI_API_KEY': self.ai_api_key,
            'MODEL_LLM': self.model_llm,
            'PERPLEXITY_API_KEY': self.perplexity_key,
            
            # Trading Configuration
            'TICKER_SHORT': self.ticker_short,
            'TICKER_SLASH': self.ticker_slash,
            'TICKER_STRIP': self.ticker_strip,
            'MODEL_DIR': self.model_dir,
            'MODEL_NAME': self.model_name,
            
            # Risk Parameters
            **self.risk_params,
            
            # Training Parameters
            'EPOCHS': self.epochs,
            'PATIENCE': self.patience,
            'BATCH_SIZE': self.batch_size,
            'LOOKBACK': self.lookback,
            
            # Logging
            'LOG_LEVEL': self.log_level,
            'LOG_FILE': self.log_file
        }


# Create default config instance
config = AppConfig()

# For backward compatibility, also create a simple dictionary
config_dict = config.to_dict()