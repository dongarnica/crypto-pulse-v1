#!/usr/bin/env python3
"""
Entry point for the AI-assisted crypto trading bot.
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import AppConfig
from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger

def main():
    """Main entry point for the crypto trading bot."""
    
    # Set up comprehensive logging first
    setup_default_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("STARTING AI-ASSISTED CRYPTO TRADING BOT")
    logger.info("=" * 50)
    
    print("Starting AI-assisted crypto trading bot...")
    
    try:        # Initialize configuration
        with PerformanceLogger(logger, "configuration_initialization"):
            config = AppConfig()
            
        # Log configuration status
        logger.info(f"Ticker: {config.ticker_short}")
        logger.info(f"Model directory: {config.model_dir}")
        logger.info(f"Log level: {config.log_level}")
        logger.info(f"Log file: {config.log_file}")
        
        # Check API configurations
        alpaca_config = config.get_alpaca_config()
        llm_config = config.get_llm_config()
        
        api_status = {
            'Alpaca': bool(alpaca_config['api_key'] and alpaca_config['api_secret']),
            'OpenAI': bool(llm_config['api_key']),
            'Perplexity': bool(llm_config['perplexity_key'])
        }
        
        logger.info("API Configuration Status:")
        for service, configured in api_status.items():
            status = "✓ Configured" if configured else "✗ Missing"
            logger.info(f"  {service}: {status}")
            print(f"  {service}: {status}")
        
        if not api_status['Alpaca']:
            logger.error("Alpaca API credentials are required for trading")
            print("❌ Error: Alpaca API credentials are required for trading")
            print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
            return 1
        
        # TODO: Add trading bot logic here
        logger.info("Bot initialization complete")
        logger.warning("Trading logic not yet implemented")
        print("✅ Bot initialization complete")
        print("⚠️  Trading logic not yet implemented")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start crypto trading bot: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
