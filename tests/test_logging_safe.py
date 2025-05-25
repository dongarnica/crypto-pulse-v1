#!/usr/bin/env python3
"""
Enhanced logging comprehensive test - safer version that avoids hanging.
Tests core logging functionality without external API calls.
"""

import os
import sys
import logging
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_logging_utilities():
    """Test the logging utilities module."""
    print("\n" + "="*60)
    print("🔧 TESTING LOGGING UTILITIES")
    print("="*60)
    
    results = {
        'test_name': 'Logging Utilities Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    try:
        from utils.logging_utils import setup_logging, get_logger, log_trade_signal, log_api_request, PerformanceLogger
        results['details'].append("✓ Logging utilities imported successfully")
        results['passed'] += 1
        
        # Create temporary log file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        try:
            # Set up logging with file output
            setup_logging(
                log_level="DEBUG",
                log_file=temp_log_path,
                console_output=False  # Avoid console spam
            )
            
            logger = get_logger(__name__)
            logger.info("Testing logging utilities...")
            results['details'].append("✓ Logging setup successful")
            results['passed'] += 1
            
            # Test performance logging (minimal sleep)
            with PerformanceLogger(logger, "test_operation", param1="value1"):
                import time
                time.sleep(0.01)  # Very short sleep
            
            results['details'].append("✓ PerformanceLogger working")
            results['passed'] += 1
            
            # Test API request logging
            log_api_request(logger, "GET", "https://api.test.com/data", 200, 0.123, size="1KB")
            log_api_request(logger, "POST", "https://api.test.com/error", 500, 2.5, error="timeout")
            results['details'].append("✓ API request logging working")
            results['passed'] += 1
            
            # Test trade signal logging
            log_trade_signal(logger, "buy", "BTC/USD", 0.5, 50000.0, 0.8)
            results['details'].append("✓ Trade signal logging working")
            results['passed'] += 1
            
            print("✅ Logging utilities test PASSED")
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_log_path)
            except:
                pass
                
    except Exception as e:
        error_msg = f"Logging utilities test failed: {str(e)}"
        results['details'].append(f"✗ {error_msg}")
        results['errors'].append(error_msg)
        results['failed'] += 1
        print(f"❌ Logging utilities test FAILED: {e}")
    
    return results

def test_basic_config_logging():
    """Test basic configuration logging."""
    print("\n" + "="*60)
    print("⚙️ TESTING CONFIG MODULE LOGGING")
    print("="*60)
    
    results = {
        'test_name': 'Config Logging Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    try:
        from utils.logging_utils import get_logger
        logger = get_logger(__name__)
        logger.info("Testing config module logging...")
        
        # Test config import without API calls
        from config.config import AppConfig
        config = AppConfig()
        logger.info(f"Config loaded for ticker: {config.ticker_short}")
        
        results['details'].append("✓ AppConfig logging working")
        results['passed'] += 1
        
        print("✅ Config logging test PASSED")
        
    except Exception as e:
        error_msg = f"Config logging test failed: {str(e)}"
        results['details'].append(f"✗ {error_msg}")
        results['errors'].append(error_msg)
        results['failed'] += 1
        print(f"❌ Config logging test FAILED: {e}")
    
    return results

def test_module_initialization_logging():
    """Test module initialization logging without API calls."""
    print("\n" + "="*60)
    print("🏗️ TESTING MODULE INITIALIZATION LOGGING") 
    print("="*60)
    
    results = {
        'test_name': 'Module Initialization Logging Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    try:
        from utils.logging_utils import get_logger
        logger = get_logger(__name__)
        
        # Test crypto data client initialization only
        try:
            from data.crypto_data_client import CryptoMarketDataClient
            client = CryptoMarketDataClient()
            logger.info("CryptoMarketDataClient initialized successfully")
            results['details'].append("✓ CryptoMarketDataClient initialization logging working")
            results['passed'] += 1
        except Exception as e:
            logger.error(f"CryptoMarketDataClient initialization failed: {e}")
            results['details'].append(f"✗ CryptoMarketDataClient initialization failed: {e}")
            results['failed'] += 1
        
        # Test Alpaca client initialization only
        try:
            from exchanges.alpaca_client import AlpacaCryptoTrading
            alpaca_client = AlpacaCryptoTrading(
                api_key="test_key",
                api_secret="test_secret"
            )
            logger.info("AlpacaCryptoTrading initialized successfully")
            results['details'].append("✓ AlpacaCryptoTrading initialization logging working")
            results['passed'] += 1
        except Exception as e:
            logger.error(f"AlpacaCryptoTrading initialization failed: {e}")
            results['details'].append(f"✗ AlpacaCryptoTrading initialization failed: {e}")
            results['failed'] += 1
        
        # Test LSTM model initialization only  
        try:
            from models.lstm_model import SignalGenerator
            signal_generator = SignalGenerator(ticker='BTC/USD')
            logger.info("SignalGenerator initialized successfully")
            results['details'].append("✓ SignalGenerator initialization logging working")
            results['passed'] += 1
        except Exception as e:
            logger.error(f"SignalGenerator initialization failed: {e}")
            results['details'].append(f"✗ SignalGenerator initialization failed: {e}")
            results['failed'] += 1
            
        if results['passed'] > 0:
            print("✅ Module initialization logging test PASSED")
        else:
            print("❌ Module initialization logging test FAILED")
        
    except Exception as e:
        error_msg = f"Module initialization logging test failed: {str(e)}"
        results['details'].append(f"✗ {error_msg}")
        results['errors'].append(error_msg)
        results['failed'] += 1
        print(f"❌ Module initialization logging test FAILED: {e}")
    
    return results

def run_all_tests():
    """Run all logging tests."""
    print("\n" + "="*80)
    print("🚀 COMPREHENSIVE LOGGING TEST SUITE (SAFE VERSION)")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    test_functions = [
        test_logging_utilities,
        test_basic_config_logging,
        test_module_initialization_logging
    ]
    
    all_results = []
    total_passed = 0
    total_failed = 0
    
    for test_func in test_functions:
        try:
            result = test_func()
            all_results.append(result)
            total_passed += result['passed']
            total_failed += result['failed']
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            all_results.append({
                'test_name': test_func.__name__,
                'passed': 0,
                'failed': 1,
                'details': [f"Test execution failed: {str(e)}"],
                'errors': [str(e)]
            })
            total_failed += 1
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("📈 FINAL LOGGING TEST RESULTS")
    print("="*80)
    
    for result in all_results:
        print(f"\n📊 {result['test_name']} Summary:")
        print(f"   ✅ Passed: {result['passed']}")
        print(f"   ❌ Failed: {result['failed']}")
        if result['details']:
            print("   Details:")
            for detail in result['details']:
                print(f"     {detail}")
        if result['errors']:
            print("   Errors:")
            for error in result['errors']:
                print(f"     {error}")
    
    print(f"\nTotal Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Test completed at: {datetime.now()}")
    
    if total_failed == 0:
        print("🎉 ALL LOGGING TESTS PASSED!")
        return 0
    else:
        print(f"⚠️ {total_failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
