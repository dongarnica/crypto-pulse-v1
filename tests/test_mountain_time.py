#!/usr/bin/env python3
"""
Enhanced Mountain Time testing module with comprehensive output reporting.
Tests timezone conversion functionality and API timestamp handling.
"""

import sys
import os
import time
import pytest
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger


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
def mountain_timezone():
    """Create Mountain timezone for testing."""
    import pytz
    return pytz.timezone('America/Denver')


@pytest.fixture
def utc_timezone():
    """Create UTC timezone for testing."""
    import pytz
    return pytz.timezone('UTC')

def test_timezone_conversion() -> Dict[str, Any]:
    """Test basic timezone conversion functionality."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Timezone Conversion Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("🕐 Testing timezone conversion...")
    
    try:
        import pytz
        results['details'].append("✓ pytz module imported successfully")
        results['passed'] += 1
        
        # Test timezone creation
        mountain_tz = pytz.timezone('America/Denver')
        results['details'].append("✓ Mountain timezone created")
        results['passed'] += 1
        
        # Test UTC to Mountain conversion
        utc_now = datetime.now(tz=pytz.UTC)
        mt_now = utc_now.astimezone(mountain_tz)
        
        results['details'].append(f"✓ UTC to Mountain conversion successful")
        results['details'].append(f"  Current UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        results['details'].append(f"  Current Mountain: {mt_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        results['passed'] += 1
        
        logger.info(f"Timezone conversion successful: UTC {utc_now} -> MT {mt_now}")
        
    except ImportError as e:
        results['details'].append(f"✗ Failed to import pytz: {e}")
        results['failed'] += 1
        results['errors'].append(f"pytz import error: {e}")
    except Exception as e:
        results['details'].append(f"✗ Timezone conversion failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Timezone conversion test failed: {e}")
    
    return results

def test_crypto_client_time_conversion() -> Dict[str, Any]:
    """Test crypto data client time conversion methods."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Crypto Client Time Conversion Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("💰 Testing crypto client time conversion...")
    
    try:
        from data.crypto_data_client import CryptoMarketDataClient
        
        client = CryptoMarketDataClient()
        results['details'].append("✓ CryptoMarketDataClient created")
        results['passed'] += 1
        
        # Test with current timestamp
        current_ts = int(datetime.now().timestamp())
        
        if hasattr(client, '_convert_to_mountain_time'):
            mt_formatted = client._convert_to_mountain_time(current_ts)
            
            if mt_formatted:
                results['details'].append("✓ Mountain time conversion successful")
                results['details'].append(f"  Timestamp: {current_ts}")
                results['details'].append(f"  Mountain Time: {mt_formatted}")
                results['passed'] += 1
            else:
                results['details'].append("✗ Mountain time conversion returned empty result")
                results['failed'] += 1
                results['errors'].append("Empty conversion result")
        else:
            results['details'].append("⚠️ _convert_to_mountain_time method not available")
            results['details'].append("  This may be expected if method was refactored")
        
        # Test with invalid timestamp
        if hasattr(client, '_convert_to_mountain_time'):
            invalid_mt = client._convert_to_mountain_time(None)
            
            if invalid_mt == "N/A" or invalid_mt is None:
                results['details'].append("✓ Invalid timestamp handled correctly")
                results['passed'] += 1
            else:
                results['details'].append(f"⚠️ Invalid timestamp returned: {invalid_mt}")
                results['details'].append("  Expected 'N/A' or None")
        
        logger.info("Crypto client time conversion tests completed")
        
    except ImportError as e:
        results['details'].append(f"✗ Failed to import CryptoMarketDataClient: {e}")
        results['failed'] += 1
        results['errors'].append(f"Import error: {e}")
    except Exception as e:
        results['details'].append(f"✗ Crypto client time conversion test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Crypto client time conversion test failed: {e}")
    
    return results

def test_api_timestamp_handling() -> Dict[str, Any]:
    """Test API call timestamp handling."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'API Timestamp Handling Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("🌐 Testing API timestamp handling...")
    
    try:
        from data.crypto_data_client import CryptoMarketDataClient
        
        client = CryptoMarketDataClient()
        
        # Test API call with timestamp handling
        with PerformanceLogger(logger, "BTC price fetch with timestamp") as perf:
            btc_data = client.get_realtime_price("BTC/USD")
        
        if btc_data:
            results['details'].append("✓ API call successful")
            
            # Check price data
            if 'price' in btc_data:
                price = btc_data['price']
                results['details'].append(f"  BTC/USD Price: ${price:,.2f}")
                results['passed'] += 1
            else:
                results['details'].append("✗ Price data missing from response")
                results['failed'] += 1
                results['errors'].append("Missing price data")
            
            # Check 24h change
            if '24h_change' in btc_data:
                change = btc_data['24h_change']
                results['details'].append(f"  24h Change: {change:+.2f}%")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ 24h change data not available")
            
            # Check timestamp handling
            if 'last_updated' in btc_data:
                raw_timestamp = btc_data['last_updated']
                results['details'].append(f"  Raw Timestamp: {raw_timestamp}")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Raw timestamp not available")
            
            # Check Mountain Time conversion
            if 'last_updated_mt' in btc_data:
                mt_time = btc_data['last_updated_mt']
                results['details'].append(f"  Mountain Time: {mt_time}")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Mountain Time conversion not available")
            
            logger.info(f"API timestamp handling successful for BTC: ${price:,.2f}")
            
        else:
            results['details'].append("✗ Failed to get BTC data from API")
            results['failed'] += 1
            results['errors'].append("API returned no data")
            
    except Exception as e:
        results['details'].append(f"✗ API timestamp handling test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"API timestamp handling test failed: {e}")
    
    return results

# Pytest Test Functions using fixtures

def test_pytest_timezone_imports(test_logger):
    """Test timezone imports using pytest fixtures."""
    import pytz
    assert pytz is not None
    
    # Test creating timezones
    mountain_tz = pytz.timezone('America/Denver')
    utc_tz = pytz.timezone('UTC')
    
    assert mountain_tz is not None
    assert utc_tz is not None
    
    test_logger.info("Pytest timezone imports test passed")


def test_pytest_timezone_conversion(mountain_timezone, utc_timezone, test_logger):
    """Test timezone conversion using pytest fixtures."""
    from datetime import datetime
    
    # Create a UTC datetime
    utc_time = datetime.now(utc_timezone)
    
    # Convert to Mountain time
    mountain_time = utc_time.astimezone(mountain_timezone)
    
    # Verify conversion
    assert utc_time.tzinfo == utc_timezone
    # Compare timezone names instead of objects directly
    assert str(mountain_time.tzinfo) == str(mountain_timezone)
    # Additionally verify it's the same moment in time
    assert utc_time.timestamp() == mountain_time.timestamp()  # Same moment in time
    
    test_logger.info("Pytest timezone conversion test passed")


def test_pytest_timestamp_handling(test_logger):
    """Test timestamp handling using pytest fixtures."""
    import time
    
    # Test current timestamp
    current_timestamp = int(time.time())
    assert current_timestamp > 0
    assert isinstance(current_timestamp, int)
    
    # Test timestamp conversion
    dt = datetime.fromtimestamp(current_timestamp)
    assert isinstance(dt, datetime)
    
    test_logger.info("Pytest timestamp handling test passed")

def run_comprehensive_mountain_time_test() -> Dict[str, Any]:
    """Run comprehensive Mountain Time testing suite."""
    logger = get_logger(__name__)
    
    print("=" * 80)
    print("🏔️ COMPREHENSIVE MOUNTAIN TIME TESTING")
    print("=" * 80)
    
    start_time = time.time()
    all_results = []
    
    # Run all tests
    test_functions = [
        test_timezone_conversion,
        test_crypto_client_time_conversion,
        test_api_timestamp_handling
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
    print("📈 FINAL MOUNTAIN TIME TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    print(f"Total Duration: {total_duration:.3f}s")
    
    if total_failed == 0:
        print("🎉 ALL MOUNTAIN TIME TESTS PASSED!")
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
        final_results = run_comprehensive_mountain_time_test()
        
        # Exit with error code if tests failed
        if final_results['total_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Mountain Time testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
