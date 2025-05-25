#!/usr/bin/env python3
"""
Enhanced crypto data testing module with comprehensive output reporting.
"""

import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger

def handle_rate_limit(func, max_retries=2, delay=1.0):
    """Simple rate limiting handler for API calls."""
    import time
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    return None  # Skip on rate limit
            else:
                raise e  # Re-raise non-rate-limit errors
    return None

def test_crypto_imports() -> Dict[str, Any]:
    """Test basic crypto-related imports."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Crypto Imports Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("1. Testing basic imports...")
    
    try:
        import requests
        results['details'].append("✓ requests imported successfully")
        results['passed'] += 1
        logger.info("Successfully imported requests")
    except ImportError as e:
        results['details'].append(f"✗ Failed to import requests: {e}")
        results['failed'] += 1
        results['errors'].append(str(e))
    
    try:
        import pandas as pd
        results['details'].append("✓ pandas imported successfully")
        results['passed'] += 1
        logger.info("Successfully imported pandas")
    except ImportError as e:
        results['details'].append(f"✗ Failed to import pandas: {e}")
        results['failed'] += 1
        results['errors'].append(str(e))
    
    try:
        import numpy as np
        results['details'].append("✓ numpy imported successfully")
        results['passed'] += 1
        logger.info("Successfully imported numpy")
    except ImportError as e:
        results['details'].append(f"✗ Failed to import numpy: {e}")
        results['failed'] += 1
        results['errors'].append(str(e))
    
    return results

def test_api_connectivity() -> Dict[str, Any]:
    """Test API connectivity to external services."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'API Connectivity Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("2. Testing API connectivity...")
    
    try:
        import requests
        
        def api_ping():
            return requests.get('https://api.coingecko.com/api/v3/ping', timeout=10)
        
        with PerformanceLogger(logger, "CoinGecko API ping") as perf:
            response = handle_rate_limit(api_ping, max_retries=2, delay=1.0)
            
        if response and response.status_code == 200:
            results['details'].append(f"✓ CoinGecko API ping successful (status: {response.status_code})")
            results['passed'] += 1
            logger.info(f"CoinGecko API ping successful: {response.status_code}")
        elif response:
            results['details'].append(f"✗ CoinGecko API ping failed (status: {response.status_code})")
            results['failed'] += 1
            results['errors'].append(f"HTTP {response.status_code}")
        else:
            results['details'].append("⚠️ CoinGecko API ping skipped due to rate limiting")
            # Don't count as failed - just skip
            logger.warning("API ping skipped due to rate limiting")
            
    except Exception as e:
        results['details'].append(f"✗ API connectivity test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"API connectivity test failed: {e}")
    
    return results

def test_crypto_data_client() -> Dict[str, Any]:
    """Test crypto data client functionality."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Crypto Data Client Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("3. Testing crypto data client...")
    
    try:
        from data.crypto_data_client import CryptoMarketDataClient
        results['details'].append("✓ CryptoMarketDataClient imported successfully")
        results['passed'] += 1
        logger.info("Successfully imported CryptoMarketDataClient")
        
        # Create client
        client = CryptoMarketDataClient()
        results['details'].append("✓ CryptoMarketDataClient instance created")
        results['passed'] += 1
        logger.info("CryptoMarketDataClient instance created")
        
        # Test real-time price
        print("   Testing real-time BTC price...")
        def get_real_time_price():
            return client.get_realtime_price("BTC/USD")
            
        with PerformanceLogger(logger, "BTC real-time price fetch") as perf:
            btc_data = handle_rate_limit(get_real_time_price, max_retries=2, delay=1.0)
            
        if btc_data:
            results['details'].append(f"✓ Real-time BTC data retrieved")
            results['passed'] += 1
            logger.info(f"Real-time BTC data retrieved successfully")
        else:
            results['details'].append("⚠️ Real-time BTC data skipped due to rate limiting")
            # Don't count as failed for rate limiting
            logger.warning("Real-time data fetch skipped due to rate limiting")
        
        # Test historical bars
        print("   Testing historical BTC bars...")
        print("Fetching 24 hours (1.0 days) of historical data for BTC/USD...")
        
        def get_historical_bars():
            return client.get_historical_bars('BTC/USD', hours=24)
            
        with PerformanceLogger(logger, "BTC historical bars fetch") as perf:
            bars = handle_rate_limit(get_historical_bars, max_retries=2, delay=1.0)
            
        if bars is not None:
            bars_info = f"type: {type(bars)}"
            if hasattr(bars, 'shape'):
                bars_info += f", shape: {bars.shape}"
            elif hasattr(bars, '__len__'):
                bars_info += f", length: {len(bars)}"
                
            results['details'].append(f"✓ Historical BTC bars retrieved: {bars_info}")
            results['passed'] += 1
            logger.info(f"Historical BTC bars: {bars_info}")
        else:
            print("No historical data available for BTC/USD")
            results['details'].append("⚠️ Historical BTC bars skipped due to rate limiting")
            # Don't count as failed for rate limiting
            logger.warning("Historical bars fetch skipped due to rate limiting")
            
    except Exception as e:
        results['details'].append(f"✗ Crypto data client test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Crypto data client test failed: {e}")
    
    return results

def run_comprehensive_crypto_test() -> Dict[str, Any]:
    """Run comprehensive crypto testing suite."""
    logger = get_logger(__name__)
    
    print("=" * 80)
    print("🔍 COMPREHENSIVE CRYPTO DATA TESTING")
    print("=" * 80)
    
    start_time = time.time()
    all_results = []
    
    # Run all tests
    test_functions = [
        test_crypto_imports,
        test_api_connectivity,
        test_crypto_data_client
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            all_results.append(result)
            
            # Print test summary
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
                    
        except Exception as e:
            logger.error(f"Failed to run test {test_func.__name__}: {e}")
            print(f"❌ Failed to run test {test_func.__name__}: {e}")
    
    # Calculate overall results
    total_passed = sum(r['passed'] for r in all_results)
    total_failed = sum(r['failed'] for r in all_results)
    total_duration = time.time() - start_time
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📈 FINAL CRYPTO TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    print(f"Total Duration: {total_duration:.3f}s")
    
    if total_failed == 0:
        print("🎉 ALL CRYPTO TESTS PASSED!")
    else:
        print(f"⚠️ {total_failed} TEST(S) FAILED")
    
    return {
        'total_passed': total_passed,
        'total_failed': total_failed,
        'duration': total_duration,
        'results': all_results
    }

if __name__ == "__main__":
    # Setup logging
    setup_default_logging()
    
    try:
        final_results = run_comprehensive_crypto_test()
        
        # Exit with error code if tests failed
        if final_results['total_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Crypto testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
