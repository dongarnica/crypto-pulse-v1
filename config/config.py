import os
from dotenv import load_dotenv
from datetime import datetime
import pytz
import logging
import logging.handlers
from typing import Dict, Any, Optional, List

load_dotenv()

class AppConfig:
    """
    Centralized configuration class for managing API keys, secrets, and application settings.
    Provides both class-based and dictionary-like access to configuration values.
    Supports multiple ticker symbols loaded from tickers.txt.
    """
    
    def __init__(self, ticker: str = "BTC"):
        # Load all available tickers from file
        self.available_tickers = self._load_tickers()
        
        # Set default ticker
        self.ticker_short = ticker.upper()
        self.ticker_slash = f'{self.ticker_short}/USD'
        self.ticker_strip = f'{self.ticker_short}USD'
        self.model_name = f'ticker_model_{self.ticker_short.lower()}.keras'
        self.model_dir = os.path.join(os.getcwd(), "trade_models")
        
        # Validate ticker is in available list
        if self.ticker_slash not in self.available_tickers:
            print(f"Warning: {self.ticker_slash} not found in tickers.txt. Available tickers: {self.available_tickers}")
        
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
        
        # Multi-ticker trading parameters
        self.multi_ticker_params = {
            'enable_multi_ticker': os.getenv("ENABLE_MULTI_TICKER", "false").lower() == "true",
            'max_tickers_active': int(os.getenv("MAX_TICKERS_ACTIVE", "3")),
            'portfolio_allocation_per_ticker': float(os.getenv("PORTFOLIO_ALLOCATION_PER_TICKER", "0.33"))  # 33% per ticker max
        }
        
        # Multi-ticker settings (can be overridden by command line args)
        self.multi_ticker_enabled = self.multi_ticker_params['enable_multi_ticker']
        self.max_active_tickers = self.multi_ticker_params['max_tickers_active']
        self.ticker_allocation = self.multi_ticker_params['portfolio_allocation_per_ticker']
        
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
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Validate required keys
        self._validate_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_tickers(self) -> List[str]:
        """Load ticker symbols from tickers.txt file."""
        tickers_file = os.path.join(os.path.dirname(__file__), '..', 'tickers.txt')
        tickers = []
        
        try:
            with open(tickers_file, 'r') as f:
                for line in f:
                    ticker = line.strip()
                    if ticker and not ticker.startswith('#') and not ticker.startswith('//'):
                        tickers.append(ticker)
            
            if not tickers:
                print("Warning: No tickers found in tickers.txt, using default BTC/USD")
                return ["BTC/USD"]
                
            print(f"Loaded {len(tickers)} tickers from tickers.txt: {tickers}")
            return tickers
            
        except FileNotFoundError:
            print("Warning: tickers.txt not found, using default BTC/USD")
            return ["BTC/USD"]
        except Exception as e:
            print(f"Error loading tickers.txt: {e}, using default BTC/USD")
            return ["BTC/USD"]
    
    def get_active_tickers(self) -> List[str]:
        """Get list of tickers that should be actively traded."""
        if self.multi_ticker_enabled:
            max_active = self.max_active_tickers
            return self.available_tickers[:max_active]
        else:
            return [self.ticker_slash]
    
    def get_ticker_config(self, ticker: str) -> Dict[str, str]:
        """Get configuration for a specific ticker."""
        ticker_short = ticker.split('/')[0] if '/' in ticker else ticker.replace('USD', '')
        return {
            'ticker_short': ticker_short,
            'ticker_slash': f'{ticker_short}/USD',
            'ticker_strip': f'{ticker_short}USD',
            'model_name': f'ticker_model_{ticker_short.lower()}.keras'
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration for the application."""
        # Convert string log level to logging constant
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        level = log_levels.get(self.log_level.upper(), logging.INFO)
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log the initialization
        logging.info(f"Logging initialized - Level: {self.log_level}, File: {self.log_file}")
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance with the specified name."""
        return logging.getLogger(name)
    
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
            'available_tickers': self.available_tickers,
            'active_tickers': self.get_active_tickers(),
            'risk_params': self.risk_params,
            'multi_ticker_params': self.multi_ticker_params,
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
            'AVAILABLE_TICKERS': self.available_tickers,
            'ACTIVE_TICKERS': self.get_active_tickers(),
            'MODEL_DIR': self.model_dir,
            'MODEL_NAME': self.model_name,
            
            # Risk Parameters
            **self.risk_params,
            
            # Multi-ticker Parameters
            **self.multi_ticker_params,
            
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