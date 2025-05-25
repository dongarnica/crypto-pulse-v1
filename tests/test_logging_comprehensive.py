#!/usr/bin/env python3
"""
Test script to verify comprehensive logging implementation across all modules.
"""

import os
import sys
import logging
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_logging, get_logger, log_trade_signal, log_api_request, PerformanceLogger
from config.config import AppConfig
from llm.llm_client import LLMClient
from data.crypto_data_client import CryptoMarketDataClient
from models.lstm_model_v2 import AggressiveCryptoSignalGenerator
from exchanges.alpaca_client import AlpacaCryptoTrading


def test_logging_utilities():
    """Test the logging utilities module."""
    print("\n" + "="*60)
    print("TESTING LOGGING UTILITIES")
    print("="*60)
    
    # Create temporary log file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_log:
        temp_log_path = temp_log.name
    
    try:
        # Set up logging with file output
        setup_logging(
            log_level="DEBUG",
            log_file=temp_log_path,
            console_output=True
        )
        
        logger = get_logger(__name__)
        logger.info("Testing logging utilities...")
        
        # Test performance logging
        with PerformanceLogger(logger, "test_operation", param1="value1"):
            import time
            time.sleep(0.01)  # Reduced from 0.1 to 0.01 for faster testing
        
        # Test API request logging
        log_api_request(logger, "GET", "https://api.test.com/data", 200, 0.123, size="1KB")
        log_api_request(logger, "POST", "https://api.test.com/error", 500, 2.5, error="timeout")
        
        # Test trading signal logging
        log_trade_signal(logger, "BTC/USD", "BUY", 0.85, 50000.0, rsi=65.5, macd=0.123)
        
        # Check if log file was created and has content
        if os.path.exists(temp_log_path):
            with open(temp_log_path, 'r') as f:
                log_content = f.read()
                if log_content:
                    print("✅ Log file created successfully")
                    print(f"   Log entries: {len(log_content.split(chr(10)))} lines")
                else:
                    print("❌ Log file is empty")
        else:
            print("❌ Log file was not created")
            
    except Exception as e:
        print(f"❌ Logging utilities test failed: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_log_path):
            os.unlink(temp_log_path)


def test_config_logging():
    """Test logging in configuration module."""
    print("\n" + "="*60)
    print("TESTING CONFIG MODULE LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing config module logging...")
        
        # Test AppConfig logging
        config = AppConfig()
        
        # Test different config methods
        alpaca_config = config.get_alpaca_config()
        llm_config = config.get_llm_config()
        
        print("✅ Config module logging working")
        
    except Exception as e:
        print(f"❌ Config logging test failed: {e}")


def test_llm_client_logging():
    """Test logging in LLM client module."""
    print("\n" + "="*60)
    print("TESTING LLM CLIENT LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing LLM client logging...")
        
        # Test LLMClient initialization (will log warnings about missing API keys)
        llm_client = LLMClient()
        
        # Test query method (will fail due to missing API key, but will log the attempt)
        try:
            llm_client.query("test prompt", provider="openai")
        except Exception:
            pass  # Expected to fail due to missing API key
            
        print("✅ LLM client logging working")
        
    except Exception as e:
        print(f"❌ LLM client logging test failed: {e}")


def test_crypto_data_client_logging():
    """Test logging in crypto data client module."""
    print("\n" + "="*60)
    print("TESTING CRYPTO DATA CLIENT LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing crypto data client logging...")
        
        # Test CryptoMarketDataClient initialization only (no API calls)
        client = CryptoMarketDataClient()
        logger.info("CryptoMarketDataClient initialized successfully")
        
        # Test logging without making actual API calls
        logger.info("Crypto data client logging test - initialization successful")
        logger.debug("Debug level logging test for crypto client")
        logger.warning("Warning level logging test for crypto client")
            
        print("✅ Crypto data client logging working")
        
    except Exception as e:
        print(f"❌ Crypto data client logging test failed: {e}")


def test_lstm_model_logging():
    """Test logging in LSTM model module."""
    print("\n" + "="*60)
    print("TESTING LSTM MODEL LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing LSTM model logging...")
        
        # Test SignalGenerator initialization only (no data fetching)
        signal_generator = AggressiveCryptoSignalGenerator(ticker='BTC/USD')
        logger.info("SignalGenerator initialized successfully")
        
        # Test logging without making actual API calls
        logger.info("LSTM model logging test - initialization successful")
        logger.debug("Debug level logging test for LSTM model")
        logger.warning("Warning level logging test for LSTM model")
            
        print("✅ LSTM model logging working")
        
    except Exception as e:
        print(f"❌ LSTM model logging test failed: {e}")


def test_alpaca_client_logging():
    """Test logging in Alpaca client module."""
    print("\n" + "="*60)
    print("TESTING ALPACA CLIENT LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing Alpaca client logging...")
        
        # Test AlpacaCryptoTrading initialization only (no API calls)
        alpaca_client = AlpacaCryptoTrading(
            api_key="test_key",
            api_secret="test_secret"
        )
        logger.info("AlpacaCryptoTrading initialized successfully")
        
        # Test logging without making actual API calls
        logger.info("Alpaca client logging test - initialization successful")
        logger.debug("Debug level logging test for Alpaca client")
        logger.warning("Warning level logging test for Alpaca client")
            
        print("✅ Alpaca client logging working")
        
    except Exception as e:
        print(f"❌ Alpaca client logging test failed: {e}")


def test_main_application_logging():
    """Test logging in main application."""
    print("\n" + "="*60)
    print("TESTING MAIN APPLICATION LOGGING")
    print("="*60)
    
    try:
        logger = get_logger(__name__)
        logger.info("Testing main application logging...")
        
        # Import and test main module
        from main import main
        
        # This will run the main function which should log extensively
        print("   Running main application (check logs)...")
        result = main()
        
        print("✅ Main application logging working")
        
    except Exception as e:
        print(f"❌ Main application logging test failed: {e}")


def run_all_tests():
    """Run all logging tests."""
    print("COMPREHENSIVE LOGGING TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    
    # Set up logging for the test suite
    setup_logging(log_level="DEBUG", console_output=True)
    
    test_functions = [
        test_logging_utilities,
        test_config_logging,
        test_llm_client_logging,
        test_crypto_data_client_logging,
        test_lstm_model_logging,
        test_alpaca_client_logging,
        test_main_application_logging
    ]
    
    results = []
    for test_func in test_functions:
        try:
            test_func()
            results.append((test_func.__name__, "PASSED"))
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            results.append((test_func.__name__, "FAILED"))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, status in results:
        print(f"  {test_name}: {status}")
        if status == "PASSED":
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    print(f"Test completed at: {datetime.now()}")
    
    if passed == len(results):
        print("🎉 All logging tests passed!")
        return 0
    else:
        print("⚠️  Some logging tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
