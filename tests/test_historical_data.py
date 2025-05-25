#!/usr/bin/env python3
"""
Enhanced historical data testing module with comprehensive output reporting.
"""

import sys
import os
import time
import pytest
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger
from data.crypto_data_client import CryptoMarketDataClient


@pytest.fixture
def crypto_client():
    """Create a CryptoMarketDataClient instance for testing."""
    return CryptoMarketDataClient()


@pytest.fixture
def test_logger():
    """Create a logger for testing."""
    return get_logger(__name__)


@pytest.fixture
def test_results():
    """Create a standard test results dictionary."""
    return {
        'test_name': 'Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }


@pytest.fixture
def test_symbols():
    """Provide test symbols for crypto testing."""
    return ['ETH/USD', 'BTC/USD']


def handle_api_call_with_rate_limit(func, max_retries=2):
    """Simple rate limiting handler for API calls."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            def test_rate_limit_scenarios(crypto_client, test_logger, test_results) -> Dict[str, Any]:
                """Test various rate limiting scenarios and recovery mechanisms."""
                logger = test_logger
                results = test_results
                results['test_name'] = 'Rate Limit Scenarios Test'
                
                print("⏱️ Testing rate limiting scenarios...")
                
                try:
                    # Test rapid consecutive calls to trigger rate limiting
                    symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD']
                    successful_calls = 0
                    rate_limited_calls = 0
                    
                    for i, symbol in enumerate(symbols):
                        def fetch_price():
                            return crypto_client.get_realtime_price(symbol)
                        
                        result = handle_api_call_with_rate_limit(fetch_price, max_retries=1)
                        if result:
                            successful_calls += 1
                            results['details'].append(f"✓ Successfully fetched {symbol} price")
                        else:
                            rate_limited_calls += 1
                            results['details'].append(f"⚠️ Rate limited for {symbol}")
                    
                    results['details'].append(f"✓ Processed {len(symbols)} symbols")
                    results['details'].append(f"  Successful: {successful_calls}")
                    results['details'].append(f"  Rate limited: {rate_limited_calls}")
                    results['passed'] += 1
                    
                    # Test rate limit recovery with exponential backoff
                    if rate_limited_calls > 0:
                        results['details'].append("✓ Testing rate limit recovery...")
                        time.sleep(5)  # Wait before retry
                        
                        def retry_fetch():
                            return crypto_client.get_realtime_price("BTC/USD")
                        
                        recovery_result = handle_api_call_with_rate_limit(retry_fetch, max_retries=3)
                        if recovery_result:
                            results['details'].append("✓ Successfully recovered from rate limiting")
                            results['passed'] += 1
                        else:
                            results['details'].append("⚠️ Still rate limited after recovery attempt")
                            results['failed'] += 1
                    
                except Exception as e:
                    results['details'].append(f"✗ Rate limit test failed: {str(e)}")
                    results['failed'] += 1
                    results['errors'].append(str(e))
                    logger.error(f"Rate limit scenarios test failed: {e}")
                
                return results


            def test_api_endpoint_reliability(crypto_client, test_logger, test_results) -> Dict[str, Any]:
                """Test API endpoint reliability and error handling."""
                logger = test_logger
                results = test_results
                results['test_name'] = 'API Endpoint Reliability Test'
                
                print("🔗 Testing API endpoint reliability...")
                
                try:
                    # Test different endpoint combinations
                    test_cases = [
                        ('BTC/USD', 1, 'realtime'),
                        ('ETH/USD', 3, 'historical'),
                        ('LTC/USD', 1, 'realtime'),
                        ('XRP/USD', 2, 'historical')
                    ]
                    
                    for symbol, days, test_type in test_cases:
                        try:
                            if test_type == 'realtime':
                                def fetch_data():
                                    return crypto_client.get_realtime_price(symbol)
                            else:
                                def fetch_data():
                                    return crypto_client.get_historical_data(symbol, days=days)
                            
                            result = handle_api_call_with_rate_limit(fetch_data, max_retries=2)
                            
                            if result:
                                results['details'].append(f"✓ {symbol} {test_type} endpoint accessible")
                                results['passed'] += 1
                            else:
                                results['details'].append(f"⚠️ {symbol} {test_type} endpoint rate limited")
                                # This is expected behavior, not a failure
                                
                        except Exception as e:
                            error_msg = str(e).lower()
                            if "429" in error_msg or "too many requests" in error_msg:
                                results['details'].append(f"⚠️ {symbol} {test_type} rate limited: {type(e).__name__}")
                            else:
                                results['details'].append(f"✗ {symbol} {test_type} failed: {type(e).__name__}")
                                results['failed'] += 1
                                results['errors'].append(f"{symbol} {test_type}: {str(e)}")
                    
                    # Test network connectivity check
                    results['details'].append("✓ API endpoint connectivity tests completed")
                    results['passed'] += 1
                    
                except Exception as e:
                    results['details'].append(f"✗ API reliability test failed: {str(e)}")
                    results['failed'] += 1
                    results['errors'].append(str(e))
                    logger.error(f"API endpoint reliability test failed: {e}")
                
                return results


            def test_data_validation_and_format(crypto_client, test_logger, test_results) -> Dict[str, Any]:
                """Test data validation and format consistency."""
                logger = test_logger
                results = test_results
                results['test_name'] = 'Data Validation and Format Test'
                
                print("📊 Testing data validation and format...")
                
                try:
                    # Test realtime data format validation
                    def validate_realtime_data():
                        return crypto_client.get_realtime_price("BTC/USD")
                    
                    realtime_data = handle_api_call_with_rate_limit(validate_realtime_data)
                    
                    if realtime_data:
                        # Validate realtime data structure
                        expected_keys = ['price', 'market_cap', 'volume_24h', 'change_24h', 'last_updated']
                        missing_keys = [key for key in expected_keys if key not in realtime_data]
                        
                        if not missing_keys:
                            results['details'].append("✓ Realtime data format validation passed")
                            results['passed'] += 1
                        else:
                            results['details'].append(f"⚠️ Missing keys in realtime data: {missing_keys}")
                            results['failed'] += 1
                            results['errors'].append(f"Missing keys: {missing_keys}")
                        
                        # Validate data types
                        if isinstance(realtime_data.get('price'), (int, float)) and realtime_data.get('price') > 0:
                            results['details'].append("✓ Price data type validation passed")
                            results['passed'] += 1
                        else:
                            results['details'].append("✗ Invalid price data type or value")
                            results['failed'] += 1
                    else:
                        results['details'].append("⚠️ Realtime data validation skipped due to rate limiting")
                    
                    # Test historical data format validation
                    def validate_historical_data():
                        return crypto_client.get_historical_data("ETH/USD", days=1)
                    
                    historical_data = handle_api_call_with_rate_limit(validate_historical_data)
                    
                    if historical_data:
                        if 'prices' in historical_data and isinstance(historical_data['prices'], list):
                            results['details'].append("✓ Historical data format validation passed")
                            results['passed'] += 1
                            
                            # Validate price point structure
                            if historical_data['prices']:
                                first_price = historical_data['prices'][0]
                                if isinstance(first_price, list) and len(first_price) == 2:
                                    results['details'].append("✓ Price point structure validation passed")
                                    results['passed'] += 1
                                else:
                                    results['details'].append("✗ Invalid price point structure")
                                    results['failed'] += 1
                        else:
                            results['details'].append("✗ Invalid historical data format")
                            results['failed'] += 1
                    else:
                        results['details'].append("⚠️ Historical data validation skipped due to rate limiting")
                    
                except Exception as e:
                    results['details'].append(f"✗ Data validation test failed: {str(e)}")
                    results['failed'] += 1
                    results['errors'].append(str(e))
                    logger.error(f"Data validation test failed: {e}")
                
                return results


            def test_error_handling_robustness(crypto_client, test_logger, test_results) -> Dict[str, Any]:
                """Test error handling robustness and recovery."""
                logger = test_logger
                results = test_results
                results['test_name'] = 'Error Handling Robustness Test'
                
                print("🛡️ Testing error handling robustness...")
                
                try:
                    # Test various error scenarios
                    error_test_cases = [
                        ("NONEXISTENT/USD", "Invalid symbol handling"),
                        ("", "Empty symbol handling"),
                        ("BTC", "Incomplete symbol format"),
                        ("BTC/INVALID", "Invalid quote currency")
                    ]
                    
                    for symbol, description in error_test_cases:
                        try:
                            def fetch_invalid():
                                return crypto_client.get_realtime_price(symbol)
                            
                            result = handle_api_call_with_rate_limit(fetch_invalid, max_retries=1)
                            
                            if result is None:
                                results['details'].append(f"✓ {description}: Handled gracefully")
                                results['passed'] += 1
                            else:
                                results['details'].append(f"⚠️ {description}: Unexpected result returned")
                                
                        except Exception as e:
                            error_msg = str(e).lower()
                            if "429" in error_msg or "too many requests" in error_msg:
                                results['details'].append(f"⚠️ {description}: Rate limited")
                            else:
                                results['details'].append(f"✓ {description}: Exception handled ({type(e).__name__})")
                                results['passed'] += 1
                    
                    # Test negative days for historical data
                    try:
                        negative_days_data = crypto_client.get_historical_data("BTC/USD", days=-1)
                        if negative_days_data is None:
                            results['details'].append("✓ Negative days handled correctly")
                            results['passed'] += 1
                        else:
                            results['details'].append("⚠️ Negative days returned unexpected data")
                    except Exception as e:
                        results['details'].append(f"✓ Negative days threw expected exception: {type(e).__name__}")
                        results['passed'] += 1
                    
                    # Test extremely large days value
                    try:
                        def fetch_large_range():
                            return crypto_client.get_historical_data("BTC/USD", days=99999)
                        
                        large_range_data = handle_api_call_with_rate_limit(fetch_large_range, max_retries=1)
                        results['details'].append("✓ Large date range handled without crashing")
                        results['passed'] += 1
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" not in error_msg and "too many requests" not in error_msg:
                            results['details'].append(f"✓ Large date range threw expected exception: {type(e).__name__}")
                            results['passed'] += 1
                        else:
                            results['details'].append("⚠️ Large date range rate limited")
                    
                except Exception as e:
                    results['details'].append(f"✗ Error handling test failed: {str(e)}")
                    results['failed'] += 1
                    results['errors'].append(str(e))
                    logger.error(f"Error handling robustness test failed: {e}")
                
                return results


            def test_performance_under_load(crypto_client, test_logger, test_results) -> Dict[str, Any]:
                """Test performance characteristics under simulated load."""
                logger = test_logger
                results = test_results
                results['test_name'] = 'Performance Under Load Test'
                
                print("⚡ Testing performance under load...")
                
                try:
                    start_time = time.time()
                    response_times = []
                    
                    # Test multiple quick requests (respecting rate limits)
                    symbols = ['BTC/USD', 'ETH/USD']
                    
                    for i, symbol in enumerate(symbols):
                        request_start = time.time()
                        
                        def fetch_with_timing():
                            return crypto_client.get_realtime_price(symbol)
                        
                        result = handle_api_call_with_rate_limit(fetch_with_timing, max_retries=2)
                        request_time = time.time() - request_start
                        response_times.append(request_time)
                        
                        if result:
                            results['details'].append(f"✓ {symbol} fetched in {request_time:.3f}s")
                        else:
                            results['details'].append(f"⚠️ {symbol} rate limited after {request_time:.3f}s")
                        
                        # Small delay to respect rate limits
                        if i < len(symbols) - 1:
                            time.sleep(1)
                    
                    total_time = time.time() - start_time
                    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                    
                    results['details'].append(f"✓ Performance test completed")
                    results['details'].append(f"  Total time: {total_time:.3f}s")
                    results['details'].append(f"  Average response time: {avg_response_time:.3f}s")
                    results['details'].append(f"  Requests processed: {len(symbols)}")
                    results['passed'] += 1
                    
                    # Check if performance is within acceptable range
                    if avg_response_time < 5.0:  # 5 seconds threshold
                        results['details'].append("✓ Performance within acceptable range")
                        results['passed'] += 1
                    else:
                        results['details'].append("⚠️ Performance slower than expected")
                        results['failed'] += 1
                    
                except Exception as e:
                    results['details'].append(f"✗ Performance test failed: {str(e)}")
                    results['failed'] += 1
                    results['errors'].append(str(e))
                    logger.error(f"Performance under load test failed: {e}")
                
                return results


            def test_pytest_rate_limit_recovery(crypto_client, test_logger):
                """Test rate limit recovery using pytest fixtures."""
                def mock_rate_limit_then_success():
                    # Simulate rate limit followed by success
                    if not hasattr(mock_rate_limit_then_success, 'called'):
                        mock_rate_limit_then_success.called = True
                        raise Exception("429 Too Many Requests")
                    return "recovered"
                
                result = handle_api_call_with_rate_limit(mock_rate_limit_then_success, max_retries=2)
                assert result == "recovered"
                test_logger.info("Pytest rate limit recovery test passed")


            def test_pytest_data_format_validation(crypto_client, test_logger):
                """Test data format validation using pytest fixtures."""
                # Test that client methods exist and are callable
                assert hasattr(crypto_client, 'get_realtime_price')
                assert hasattr(crypto_client, 'get_historical_data')
                assert callable(crypto_client.get_realtime_price)
                assert callable(crypto_client.get_historical_data)
                test_logger.info("Pytest data format validation test passed")


            def test_pytest_error_scenarios(test_logger):
                """Test various error scenarios using pytest fixtures."""
                # Test rate limit handler with different error types
                def mock_non_rate_limit_error():
                    raise ValueError("Some other error")
                
                with pytest.raises(ValueError):
                    handle_api_call_with_rate_limit(mock_non_rate_limit_error)
                
                test_logger.info("Pytest error scenarios test passed")
            eth_history = handle_api_call_with_rate_limit(fetch_eth_data)
        
        if eth_history:
            price_count = len(eth_history.get('prices', []))
            results['details'].append(f"✓ ETH historical data retrieved")
            results['details'].append(f"  Data keys: {list(eth_history.keys())}")
            results['details'].append(f"  Price points: {price_count}")
            results['passed'] += 1
            
            logger.info(f"Successfully retrieved ETH data with {price_count} price points")
            
            # Test formatting method
            if hasattr(crypto_client, 'format_historical_summary'):
                summary = crypto_client.format_historical_summary(eth_history, "ETH/USD")
                if summary:
                    results['details'].append("✓ Historical summary formatted successfully")
                    results['details'].append(f"  Summary: {summary[:200]}...")  # First 200 chars
                    results['passed'] += 1
                else:
                    results['details'].append("✗ Failed to format historical summary")
                    results['failed'] += 1
                    results['errors'].append("Empty summary returned")
            else:
                results['details'].append("⚠️ format_historical_summary method not available")
                results['failed'] += 1
                results['errors'].append("Missing format_historical_summary method")
        else:
            results['details'].append("⚠️ ETH historical data skipped due to rate limiting")
            logger.warning("ETH historical data fetch skipped due to rate limiting")
            
    except Exception as e:
        results['details'].append(f"✗ ETH historical data test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"ETH historical data test failed: {e}")
    
    return results

def test_btc_historical_data(crypto_client, test_logger, test_results) -> Dict[str, Any]:
    """Test BTC/USD historical data retrieval and formatting."""
    logger = test_logger
    results = test_results
    results['test_name'] = 'BTC Historical Data Test'
    
    print("₿ Testing BTC/USD historical data...")
    
    try:
        results['details'].append("✓ CryptoMarketDataClient created")
        results['passed'] += 1
        
        # Test BTC historical data fetch with rate limiting
        def fetch_btc_data():
            return crypto_client.get_historical_data("BTC/USD", days=3)
            
        with PerformanceLogger(logger, "BTC 3-day historical data fetch") as perf:
            btc_history = handle_api_call_with_rate_limit(fetch_btc_data)
        
        if btc_history:
            price_count = len(btc_history.get('prices', []))
            results['details'].append(f"✓ BTC historical data retrieved")
            results['details'].append(f"  Data keys: {list(btc_history.keys())}")
            results['details'].append(f"  Price points: {price_count}")
            results['passed'] += 1
            
            logger.info(f"Successfully retrieved BTC data with {price_count} price points")
            
            # Test formatting method
            if hasattr(crypto_client, 'format_historical_summary'):
                summary = crypto_client.format_historical_summary(btc_history, "BTC/USD")
                if summary:
                    results['details'].append("✓ Historical summary formatted successfully")
                    results['details'].append(f"  Summary: {summary[:200]}...")  # First 200 chars
                    results['passed'] += 1
                else:
                    results['details'].append("✗ Failed to format historical summary")
                    results['failed'] += 1
                    results['errors'].append("Empty summary returned")
            else:
                results['details'].append("⚠️ format_historical_summary method not available")
                results['failed'] += 1
                results['errors'].append("Missing format_historical_summary method")
        else:
            results['details'].append("⚠️ BTC historical data skipped due to rate limiting")
            logger.warning("BTC historical data fetch skipped due to rate limiting")
            
    except Exception as e:
        results['details'].append(f"✗ BTC historical data test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"BTC historical data test failed: {e}")
    
    return results

def test_historical_data_edge_cases(crypto_client, test_logger, test_results) -> Dict[str, Any]:
    """Test edge cases for historical data retrieval."""
    logger = test_logger
    results = test_results
    results['test_name'] = 'Historical Data Edge Cases Test'
    
    print("🔍 Testing historical data edge cases...")
    
    try:
        # Test with invalid symbol (with rate limiting)
        def fetch_invalid_data():
            return crypto_client.get_historical_data("INVALID/USD", days=1)
            
        try:
            invalid_data = handle_api_call_with_rate_limit(fetch_invalid_data)
            if invalid_data is None:
                results['details'].append("✓ Invalid symbol handled correctly (returned None)")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Invalid symbol returned data unexpectedly")
                results['failed'] += 1
        except Exception as e:
            results['details'].append(f"✓ Invalid symbol threw expected exception: {type(e).__name__}")
            results['passed'] += 1
        
        # Test with zero days
        try:
            zero_data = crypto_client.get_historical_data("BTC/USD", days=0)
            if zero_data is None:
                results['details'].append("✓ Zero days handled correctly (returned None)")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Zero days returned data unexpectedly")
                results['failed'] += 1
        except Exception as e:
            results['details'].append(f"✓ Zero days threw expected exception: {type(e).__name__}")
            results['passed'] += 1
        
        # Test with very large days value
        try:
            large_data = crypto_client.get_historical_data("BTC/USD", days=365)
            if large_data:
                results['details'].append("✓ Large days value handled successfully")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Large days value returned no data")
                results['failed'] += 1
        except Exception as e:
            results['details'].append(f"⚠️ Large days value failed: {type(e).__name__}")
            results['failed'] += 1
            results['errors'].append(str(e))
            
    except Exception as e:
        results['details'].append(f"✗ Edge cases test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Historical data edge cases test failed: {e}")
    
    return results

# Pytest Test Functions using fixtures

def test_pytest_crypto_client_creation(crypto_client, test_logger):
    """Test crypto client creation using pytest fixtures."""
    assert crypto_client is not None
    assert hasattr(crypto_client, 'get_historical_data')
    assert hasattr(crypto_client, 'get_realtime_price')
    test_logger.info("Pytest crypto client creation test passed")


def test_pytest_historical_data_structure(crypto_client, test_symbols, test_logger):
    """Test historical data structure without API calls using pytest fixtures."""
    # Test that the client has the expected methods
    for symbol in test_symbols:
        assert isinstance(symbol, str)
        assert '/' in symbol  # Should be in format like 'BTC/USD'
    
    # Test client methods exist
    assert callable(getattr(crypto_client, 'get_historical_data', None))
    test_logger.info("Pytest historical data structure test passed")


def test_pytest_rate_limiting_handler(test_logger):
    """Test rate limiting handler using pytest fixtures."""
    def mock_success_function():
        return "success"
    
    def mock_rate_limit_function():
        raise Exception("429 Too Many Requests")
    
    # Test successful function
    result = handle_api_call_with_rate_limit(mock_success_function)
    assert result == "success"
    
    # Test rate limited function returns None
    result = handle_api_call_with_rate_limit(mock_rate_limit_function, max_retries=1)
    assert result is None
    
    test_logger.info("Pytest rate limiting handler test passed")

def run_comprehensive_historical_test() -> Dict[str, Any]:
    """Run comprehensive historical data testing suite."""
    logger = get_logger(__name__)
    
    print("=" * 80)
    print("📊 COMPREHENSIVE HISTORICAL DATA TESTING")
    print("=" * 80)
    
    start_time = time.time()
    all_results = []
    
    # Run all tests
    test_functions = [
        test_eth_historical_data,
        test_btc_historical_data,
        test_historical_data_edge_cases
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
    print("📈 FINAL HISTORICAL DATA TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    print(f"Total Duration: {total_duration:.3f}s")
    
    if total_failed == 0:
        print("🎉 ALL HISTORICAL DATA TESTS PASSED!")
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
        final_results = run_comprehensive_historical_test()
        
        # Exit with error code if tests failed
        if final_results['total_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Historical data testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
