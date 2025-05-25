"""
Logging utilities for the crypto trading bot.
Provides standardized logging configuration and helper functions.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Set up standardized logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatters
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, *args, **kwargs) -> None:
    """
    Log function call with arguments.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    args_str = ", ".join([str(arg) for arg in args])
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    
    all_args = []
    if args_str:
        all_args.append(args_str)
    if kwargs_str:
        all_args.append(kwargs_str)
    
    args_combined = ", ".join(all_args)
    logger.debug(f"Calling {func_name}({args_combined})")


def log_performance(logger: logging.Logger, operation: str, duration: float, **metadata) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        **metadata: Additional metadata to log
    """
    metadata_str = ", ".join([f"{k}={v}" for k, v in metadata.items()])
    message = f"Performance: {operation} completed in {duration:.3f}s"
    if metadata_str:
        message += f" ({metadata_str})"
    logger.info(message)


def log_api_request(logger: logging.Logger, method: str, url: str, status_code: int, 
                   duration: float, **extras) -> None:
    """
    Log API request details.
    
    Args:
        logger: Logger instance
        method: HTTP method
        url: Request URL
        status_code: HTTP status code
        duration: Request duration in seconds
        **extras: Additional request metadata
    """
    extras_str = ", ".join([f"{k}={v}" for k, v in extras.items()])
    message = f"API {method} {url} -> {status_code} ({duration:.3f}s)"
    if extras_str:
        message += f" [{extras_str}]"
    
    if status_code >= 400:
        logger.error(message)
    elif status_code >= 300:
        logger.warning(message)
    else:
        logger.info(message)


def log_trade_signal(logger: logging.Logger, symbol: str, signal: str, confidence: float,
                    price: float, **indicators) -> None:
    """
    Log trading signal generation.
    
    Args:
        logger: Logger instance
        symbol: Trading symbol
        signal: Signal type (BUY, SELL, HOLD)
        confidence: Signal confidence (0-1)
        price: Current price
        **indicators: Technical indicators
    """
    indicators_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in indicators.items()])
    
    message = f"Signal {signal} for {symbol} @ ${price:.2f} (confidence: {confidence:.3f})"
    if indicators_str:
        message += f" [indicators: {indicators_str}]"
    
    logger.info(message)


def log_portfolio_update(logger: logging.Logger, symbol: str, action: str, quantity: float,
                        price: float, balance: float, **extras) -> None:
    """
    Log portfolio/position updates.
    
    Args:
        logger: Logger instance
        symbol: Trading symbol
        action: Action taken (BUY, SELL)
        quantity: Quantity traded
        price: Trade price
        balance: Remaining balance
        **extras: Additional trade metadata
    """
    trade_value = quantity * price
    extras_str = ", ".join([f"{k}={v}" for k, v in extras.items()])
    
    message = f"Portfolio: {action} {quantity:.6f} {symbol} @ ${price:.2f} " \
             f"(value: ${trade_value:.2f}, balance: ${balance:.2f})"
    if extras_str:
        message += f" [{extras_str}]"
    
    logger.info(message)


class PerformanceLogger:
    """Context manager for logging operation performance."""
    
    def __init__(self, logger: logging.Logger, operation: str, **metadata):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(f"Operation {self.operation} failed after {duration:.3f}s: {exc_val}")
        else:
            log_performance(self.logger, self.operation, duration, **self.metadata)


# Default application logging setup
def setup_default_logging():
    """Set up default logging configuration for the crypto trading bot."""
    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Set up logging with file and console output
    log_file = os.path.join(logs_dir, f'trading_{datetime.now().strftime("%Y%m%d")}.log')
    
    setup_logging(
        log_level="INFO",
        log_file=log_file,
        console_output=True
    )
    
    # Log startup
    logger = get_logger(__name__)
    logger.info("=" * 50)
    logger.info("Crypto Trading Bot - Logging Initialized")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Demo usage
    setup_default_logging()
    
    logger = get_logger(__name__)
    logger.info("Testing logging utilities")
    
    # Test performance logging
    with PerformanceLogger(logger, "test_operation", test_param="value"):
        import time
        time.sleep(0.1)
    
    # Test API logging
    log_api_request(logger, "GET", "https://api.example.com/data", 200, 0.234)
    
    # Test signal logging
    log_trade_signal(logger, "BTC/USD", "BUY", 0.85, 50000.0, rsi=65.5, macd=0.123)
    
    logger.info("Logging utilities test completed")
