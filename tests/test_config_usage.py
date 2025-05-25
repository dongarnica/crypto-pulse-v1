#!/usr/bin/env python3
"""
Enhanced configuration testing module with comprehensive output reporting.
Tests AppConfig system functionality, API key validation, and module integration.
"""

import os
import sys
import time
import pytest
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger
from config.config import AppConfig


@pytest.fixture
def app_config():
    """Create an AppConfig instance for testing."""
    return AppConfig(ticker="BTC")


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

def test_config_initialization() -> Dict[str, Any]:
    """Test configuration initialization and basic functionality."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Configuration Initialization Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("⚙️ Testing configuration initialization...")
    
    try:
        # Test config creation
        with PerformanceLogger(logger, "Config initialization") as perf:
            config = AppConfig(ticker="BTC")
        
        results['details'].append("✓ AppConfig initialized successfully")
        results['details'].append(f"  Ticker: {config.ticker_short}")
        results['details'].append(f"  Model directory: {config.model_dir}")
        results['details'].append(f"  Model name: {config.model_name}")
        results['passed'] += 1
        
        # Test logger creation
        config_logger = config.get_logger(__name__)
        if config_logger:
            results['details'].append("✓ Logger creation successful")
            results['passed'] += 1
        else:
            results['details'].append("✗ Logger creation failed")
            results['failed'] += 1
            results['errors'].append("get_logger returned None")
        
        # Test attribute access
        try:
            ticker = config.ticker_short
            model_name = config.model_name
            results['details'].append("✓ Direct attribute access working")
            results['passed'] += 1
        except AttributeError as e:
            results['details'].append(f"✗ Attribute access failed: {e}")
            results['failed'] += 1
            results['errors'].append(str(e))
            
    except Exception as e:
        results['details'].append(f"✗ Configuration initialization failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Configuration initialization test failed: {e}")
    
    return results

def test_api_key_configuration() -> Dict[str, Any]:
    """Test API key configuration and validation."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'API Key Configuration Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("🔑 Testing API key configuration...")
    
    try:
        config = AppConfig(ticker="BTC")
        
        # Test Alpaca configuration
        alpaca_config = config.get_alpaca_config()
        if alpaca_config:
            results['details'].append("✓ Alpaca config retrieved")
            alpaca_configured = bool(alpaca_config.get('api_key'))
            status = "✅ Configured" if alpaca_configured else "❌ Not configured"
            results['details'].append(f"  Alpaca API key: {status}")
            results['passed'] += 1
        else:
            results['details'].append("✗ Failed to retrieve Alpaca config")
            results['failed'] += 1
            results['errors'].append("get_alpaca_config returned None")
        
        # Test LLM configuration
        llm_config = config.get_llm_config()
        if llm_config:
            results['details'].append("✓ LLM config retrieved")
            openai_configured = bool(llm_config.get('api_key'))
            perplexity_configured = bool(llm_config.get('perplexity_key'))
            
            openai_status = "✅ Configured" if openai_configured else "❌ Not configured"
            perplexity_status = "✅ Configured" if perplexity_configured else "❌ Not configured"
            
            results['details'].append(f"  OpenAI API key: {openai_status}")
            results['details'].append(f"  Perplexity API key: {perplexity_status}")
            results['passed'] += 1
        else:
            results['details'].append("✗ Failed to retrieve LLM config")
            results['failed'] += 1
            results['errors'].append("get_llm_config returned None")
        
        # Test trading configuration
        trading_config = config.get_trading_config()
        if trading_config:
            results['details'].append("✓ Trading config retrieved")
            results['details'].append(f"  Config keys: {list(trading_config.keys())}")
            results['passed'] += 1
        else:
            results['details'].append("✗ Failed to retrieve trading config")
            results['failed'] += 1
            results['errors'].append("get_trading_config returned None")
            
    except Exception as e:
        results['details'].append(f"✗ API key configuration test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"API key configuration test failed: {e}")
    
    return results

def test_environment_variables() -> Dict[str, Any]:
    """Test environment variable configuration status."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Environment Variables Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("🌍 Testing environment variables...")
    
    env_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'OPENAI_API_KEY',
        'PERPLEXITY_API_KEY'
    ]
    
    configured_count = 0
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            results['details'].append(f"✅ {var}: Configured")
            configured_count += 1
        else:
            results['details'].append(f"❌ {var}: Not configured")
    
    results['details'].append(f"Total configured: {configured_count}/{len(env_vars)}")
    
    if configured_count > 0:
        results['passed'] += configured_count
        logger.info(f"{configured_count}/{len(env_vars)} environment variables configured")
    
    missing_count = len(env_vars) - configured_count
    if missing_count > 0:
        results['failed'] += missing_count
        results['errors'].append(f"{missing_count} environment variables not configured")
        logger.warning(f"{missing_count} environment variables missing")
    
    return results

def test_module_integration() -> Dict[str, Any]:
    """Test configuration integration with various modules."""
    logger = get_logger(__name__)
    results = {
        'test_name': 'Module Integration Test',
        'passed': 0,
        'failed': 0,
        'details': [],
        'errors': []
    }
    
    print("🔗 Testing module integration...")
    
    try:
        config = AppConfig(ticker="BTC")
        
        # Test Alpaca client integration
        try:
            from exchanges.alpaca_client import AlpacaCryptoTrading
            alpaca_config = config.get_alpaca_config()
            
            if alpaca_config.get('api_key') and alpaca_config.get('api_secret'):
                alpaca_client = AlpacaCryptoTrading(
                    api_key=alpaca_config['api_key'],
                    api_secret=alpaca_config['api_secret'],
                    base_url="https://paper-api.alpaca.markets"
                )
                results['details'].append("✓ Alpaca client integration successful")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ Alpaca client not tested (missing API credentials)")
                results['details'].append("  Configure ALPACA_API_KEY and ALPACA_SECRET_KEY to test")
                
        except ImportError as e:
            results['details'].append(f"✗ Alpaca client import failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"Alpaca import error: {e}")
        except Exception as e:
            results['details'].append(f"✗ Alpaca client integration failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"Alpaca integration error: {e}")
        
        # Test LLM client integration
        try:
            from llm.llm_client import LLMClient
            llm_config = config.get_llm_config()
            
            if llm_config.get('api_key') or llm_config.get('perplexity_key'):
                llm_client = LLMClient(config)
                results['details'].append("✓ LLM client integration successful")
                results['passed'] += 1
            else:
                results['details'].append("⚠️ LLM client not tested (no API keys)")
                results['details'].append("  Configure OPENAI_API_KEY or PERPLEXITY_API_KEY to test")
                
        except ImportError as e:
            results['details'].append(f"✗ LLM client import failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"LLM import error: {e}")
        except Exception as e:
            results['details'].append(f"✗ LLM client integration failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"LLM integration error: {e}")
        
        # Test crypto data client integration
        try:
            from data.crypto_data_client import CryptoMarketDataClient
            data_client = CryptoMarketDataClient()
            results['details'].append("✓ Crypto data client integration successful")
            results['passed'] += 1
        except ImportError as e:
            results['details'].append(f"✗ Crypto data client import failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"Data client import error: {e}")
        except Exception as e:
            results['details'].append(f"✗ Crypto data client integration failed: {e}")
            results['failed'] += 1
            results['errors'].append(f"Data client integration error: {e}")
            
    except Exception as e:
        results['details'].append(f"✗ Module integration test failed: {str(e)}")
        results['failed'] += 1
        results['errors'].append(str(e))
        logger.error(f"Module integration test failed: {e}")
    
    return results

# Pytest Test Functions using fixtures

def test_pytest_config_initialization(app_config, test_logger):
    """Test configuration initialization using pytest fixtures."""
    assert app_config is not None
    assert app_config.ticker_short == "BTC"
    assert hasattr(app_config, 'model_dir')
    assert hasattr(app_config, 'model_name')
    test_logger.info("Pytest config initialization test passed")


def test_pytest_api_keys(app_config, test_logger):
    """Test API key configuration using pytest fixtures."""
    # Test Alpaca config
    alpaca_config = app_config.get_alpaca_config()
    assert alpaca_config is not None
    assert 'api_key' in alpaca_config
    assert 'api_secret' in alpaca_config
    
    # Test LLM config  
    llm_config = app_config.get_llm_config()
    assert llm_config is not None
    assert 'api_key' in llm_config, "LLM config should contain 'api_key'"
    
    test_logger.info("Pytest API keys test passed")


def test_pytest_environment_variables(test_logger):
    """Test environment variables using pytest fixtures."""
    import os
    
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'OPENAI_API_KEY',
        'PERPLEXITY_API_KEY'
    ]
    
    configured_count = 0
    for var in required_vars:
        if os.getenv(var):
            configured_count += 1
    
    # At least some environment variables should be configured
    assert configured_count > 0, "No environment variables configured"
    test_logger.info(f"Pytest environment variables test passed: {configured_count}/{len(required_vars)} configured")


def test_pytest_module_integration(app_config, test_logger):
    """Test module integration using pytest fixtures."""
    # Test that we can get trading config
    trading_config = app_config.get_trading_config()
    assert trading_config is not None
    assert isinstance(trading_config, dict)
    
    # Test that logger works
    config_logger = app_config.get_logger('test')
    assert config_logger is not None
    
    test_logger.info("Pytest module integration test passed")

def run_comprehensive_config_test() -> Dict[str, Any]:
    """Run comprehensive configuration testing suite."""
    logger = get_logger(__name__)
    
    print("=" * 80)
    print("⚙️ COMPREHENSIVE CONFIGURATION TESTING")
    print("=" * 80)
    
    start_time = time.time()
    all_results = []
    
    # Run all tests
    test_functions = [
        test_config_initialization,
        test_api_key_configuration,
        test_environment_variables,
        test_module_integration
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
    print("📈 FINAL CONFIGURATION TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed) * 100):.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    print(f"Total Duration: {total_duration:.3f}s")
    
    if total_failed == 0:
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
    else:
        print(f"⚠️ {total_failed} TEST(S) FAILED")
    
    # Print configuration guide if needed
    if total_failed > 0:
        print("\n" + "=" * 80)
        print("📋 CONFIGURATION GUIDE")
        print("=" * 80)
        print("To configure missing API keys:")
        print("1. Create a .env file in the project root")
        print("2. Add your API keys like:")
        print("   ALPACA_API_KEY=your_key_here")
        print("   ALPACA_SECRET_KEY=your_secret_here")
        print("   OPENAI_API_KEY=your_openai_key_here")
        print("   PERPLEXITY_API_KEY=your_perplexity_key_here")
    
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
        final_results = run_comprehensive_config_test()
        
        # Exit with error code if tests failed
        if final_results['total_failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Configuration testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        print(f"   {service} API configured: {status_symbol}")
        logger.info(f"{service} API configured: {configured}")
    
    # 3. Alpaca Trading Client Example
    logger.info("Testing Alpaca Trading Client")
    print("\n2. Alpaca Trading Client:")
    try:
        if alpaca_config['api_key'] and alpaca_config['api_secret']:
            alpaca_client = AlpacaCryptoTrading(
                api_key=alpaca_config['api_key'],
                api_secret=alpaca_config['api_secret'],
                base_url=alpaca_config['base_url']
            )
            print("   ✅ Alpaca client initialized successfully")
            logger.info("Alpaca client initialized successfully")
            
            # Example: Get account info (if credentials are valid)
            try:
                account = alpaca_client.get_account()
                print(f"   Account buying power: ${account.get('buying_power', 'N/A')}")
                print(f"   Account cash: ${account.get('cash', 'N/A')}")
                print(f"   Portfolio value: ${account.get('portfolio_value', 'N/A')}")
                logger.info(f"Account retrieved - Portfolio value: ${account.get('portfolio_value', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️  Could not fetch account info: {str(e)}")
                logger.warning(f"Could not fetch account info: {str(e)}")
            
            # Example: Get current positions
            try:
                positions = alpaca_client.list_positions()
                position_count = len(positions) if positions else 0
                print(f"   Current positions: {position_count}")
                logger.info(f"Retrieved {position_count} positions")
                
                if positions:
                    print("   Open positions:")
                    for pos in positions:
                        symbol = pos.get('symbol', 'N/A')
                        qty = pos.get('qty', 'N/A')
                        market_value = pos.get('market_value', 'N/A')
                        unrealized_pl = pos.get('unrealized_pl', 'N/A')
                        side = pos.get('side', 'N/A')
                        
                        print(f"     {symbol}: {qty} shares ({side})")
                        print(f"       Market value: ${market_value}")
                        print(f"       Unrealized P&L: ${unrealized_pl}")
                        logger.debug(f"Position {symbol}: {qty} shares, value: ${market_value}")
                else:
                    print("   No open positions")
                    logger.info("No open positions found")
            except Exception as e:
                print(f"   ⚠️  Could not fetch positions: {str(e)}")
                logger.error(f"Could not fetch positions: {str(e)}")
                
            # Example: Portfolio summary analysis
            try:
                portfolio = alpaca_client.get_portfolio_summary()
                if 'error' not in portfolio:
                    total_value = portfolio['account']['portfolio_value']
                    total_pl = portfolio['positions']['total_unrealized_pl']
                    total_return = portfolio['performance']['total_return_pct']
                    
                    print(f"   Portfolio value: ${total_value:,.2f}")
                    print(f"   Total P&L: ${total_pl:,.2f}")
                    print(f"   Total return: {total_return:+.2f}%")
                    logger.info(f"Portfolio analysis - Value: ${total_value:,.2f}, Return: {total_return:+.2f}%")
                    
                    if portfolio['performance']['best_performer']:
                        best = portfolio['performance']['best_performer']
                        print(f"   Best performer: {best['symbol']} ({best['return_pct']:+.2f}%)")
                        logger.info(f"Best performer: {best['symbol']} ({best['return_pct']:+.2f}%)")
                else:
                    print(f"   ⚠️  Portfolio analysis failed: {portfolio['error']}")
                    logger.warning(f"Portfolio analysis failed: {portfolio['error']}")
            except Exception as e:
                print(f"   ⚠️  Portfolio analysis error: {str(e)}")
                logger.error(f"Portfolio analysis error: {str(e)}")
        else:
            print("   ❌ Alpaca credentials not configured")
            logger.warning("Alpaca credentials not configured")
    except Exception as e:
        print(f"   ❌ Error initializing Alpaca client: {str(e)}")
        logger.error(f"Error initializing Alpaca client: {str(e)}")
    
    # 4. LLM Client Example
    logger.info("Testing LLM Client")
    print("\n3. LLM Client:")
    try:
        llm_client = LLMClient(config=config)
        print("   ✅ LLM client initialized successfully")
        logger.info("LLM client initialized successfully")
        
        # Example query (only if API key is available)
        if llm_config['api_key']:
            try:
                response = llm_client.query(
                    "What is Bitcoin?", 
                    provider="openai", 
                    model="gpt-3.5-turbo"  # Use cheaper model for testing
                )
                print(f"   Sample response length: {len(response)} characters")
                logger.info(f"LLM query successful - Response length: {len(response)} characters")
            except Exception as e:
                print(f"   ⚠️  LLM query failed: {str(e)}")
                logger.error(f"LLM query failed: {str(e)}")
        else:
            print("   ⚠️  No OpenAI API key configured, skipping query test")
            logger.warning("No OpenAI API key configured, skipping query test")
    except Exception as e:
        print(f"   ❌ Error initializing LLM client: {str(e)}")
        logger.error(f"Error initializing LLM client: {str(e)}")
    
    # 5. Crypto Data Client Example
    logger.info("Testing Crypto Data Client")
    print("\n4. Crypto Data Client:")
    try:
        data_client = CryptoMarketDataClient()
        print("   ✅ Crypto data client initialized successfully")
        logger.info("Crypto data client initialized successfully")
        
        # Example: Get current price
        try:
            btc_data = data_client.get_realtime_price("BTC/USD")
            if btc_data:
                current_price = btc_data['price']
                change_24h = btc_data.get('24h_change', 0)
                print(f"   Current Bitcoin price: ${current_price:,.2f}")
                print(f"   24h change: {change_24h:+.2f}%")
                logger.info(f"Bitcoin price retrieved: ${current_price:,.2f}, 24h change: {change_24h:+.2f}%")
            else:
                print("   ⚠️  No price data returned")
                logger.warning("No price data returned from crypto data client")
        except Exception as e:
            print(f"   ⚠️  Could not fetch price data: {str(e)}")
            logger.error(f"Could not fetch price data: {str(e)}")
    except Exception as e:
        print(f"   ❌ Error initializing data client: {str(e)}")
        logger.error(f"Error initializing data client: {str(e)}")
    
    # 6. Configuration Access Examples
    logger.info("Demonstrating configuration access methods")
    print("\n5. Configuration Access Methods:")
    
    # Dictionary-style access
    print("   Dictionary-style access:")
    try:
        print(f"   config['ALPACA_API_KEY']: {'***configured***' if config['ALPACA_API_KEY'] else 'Not set'}")
        print(f"   config['TICKER_SHORT']: {config['TICKER_SHORT']}")
        logger.debug("Dictionary-style access demonstrated successfully")
    except KeyError as e:
        print(f"   KeyError: {e}")
        logger.error(f"Dictionary-style access KeyError: {e}")
    
    # Direct attribute access
    print("   Direct attribute access:")
    print(f"   config.ticker_short: {config.ticker_short}")
    print(f"   config.model_name: {config.model_name}")
    logger.debug("Direct attribute access demonstrated successfully")
    
    # Specialized getters
    print("   Specialized getter methods:")
    trading_config = config.get_trading_config()
    print(f"   Trading config keys: {list(trading_config.keys())}")
    logger.debug(f"Trading config retrieved with keys: {list(trading_config.keys())}")
    
    # 7. Environment Variables Check
    logger.info("Checking environment variables status")
    print("\n6. Environment Variables Status:")
    env_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY', 
        'OPENAI_API_KEY',
        'PERPLEXITY_API_KEY'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Not set"
        print(f"   {var}: {status}")
        logger.debug(f"Environment variable {var}: {'Set' if value else 'Not set'}")
    
    print("\n=== Configuration Complete ===")
    print("\nTo configure missing API keys:")
    print("1. Create a .env file in the project root")
    print("2. Add your API keys like:")
    print("   ALPACA_API_KEY=your_key_here")
    print("   ALPACA_SECRET_KEY=your_secret_here")
    print("   OPENAI_API_KEY=your_openai_key_here")
    print("   PERPLEXITY_API_KEY=your_perplexity_key_here")
    
    logger.info("AppConfig usage demonstration completed successfully")

if __name__ == "__main__":
    main()
