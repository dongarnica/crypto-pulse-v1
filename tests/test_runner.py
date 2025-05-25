#!/usr/bin/env python3
"""
Comprehensive test runner for the crypto trading bot.
Provides detailed output, test results, and logging for all test cases.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger

class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.failed = False
        self.skipped = False
        self.error_message = None
        self.duration = 0.0
        self.details = []
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Mark test as started."""
        self.start_time = time.time()
        
    def pass_test(self, details: str = None):
        """Mark test as passed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.passed = True
        if details:
            self.details.append(details)
            
    def fail_test(self, error_message: str, details: str = None):
        """Mark test as failed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.failed = True
        self.error_message = error_message
        if details:
            self.details.append(details)
            
    def skip_test(self, reason: str):
        """Mark test as skipped."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.skipped = True
        self.error_message = reason
    
    def get_status(self) -> str:
        """Get test status as string."""
        if self.passed:
            return "PASSED"
        elif self.failed:
            return "FAILED"
        elif self.skipped:
            return "SKIPPED"
        else:
            return "UNKNOWN"
    
    def get_status_icon(self) -> str:
        """Get status icon."""
        if self.passed:
            return "✅"
        elif self.failed:
            return "❌"
        elif self.skipped:
            return "⚠️"
        else:
            return "❓"


class TestRunner:
    """Test runner with comprehensive reporting."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test with error handling and reporting."""
        result = TestResult(test_name)
        result.start()
        
        self.logger.info(f"Starting test: {test_name}")
        print(f"\n{'='*60}")
        print(f"🧪 RUNNING TEST: {test_name}")
        print(f"{'='*60}")
        
        try:
            # Run the test function
            test_output = test_func(*args, **kwargs)
            
            # If test function returns output, use it
            if test_output:
                result.pass_test(str(test_output))
            else:
                result.pass_test()
                
            self.logger.info(f"Test {test_name} PASSED in {result.duration:.3f}s")
            print(f"\n{result.get_status_icon()} TEST {result.get_status()}: {test_name}")
            print(f"   Duration: {result.duration:.3f}s")
            
        except Exception as e:
            error_msg = str(e)
            result.fail_test(error_msg, traceback.format_exc())
            
            self.logger.error(f"Test {test_name} FAILED in {result.duration:.3f}s: {error_msg}")
            print(f"\n{result.get_status_icon()} TEST {result.get_status()}: {test_name}")
            print(f"   Duration: {result.duration:.3f}s")
            print(f"   Error: {error_msg}")
            
        self.test_results.append(result)
        return result
    
    def run_test_suite(self, test_functions: List[Tuple[str, callable]]) -> Dict[str, Any]:
        """Run a complete test suite."""
        self.start_time = time.time()
        
        print("="*80)
        print("🚀 CRYPTO TRADING BOT - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running {len(test_functions)} tests...")
        
        self.logger.info(f"Starting test suite with {len(test_functions)} tests")
        
        # Run all tests
        for test_name, test_func in test_functions:
            try:
                self.run_test(test_name, test_func)
            except KeyboardInterrupt:
                self.logger.warning("Test suite interrupted by user")
                print("\n⚠️ Test suite interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error running test {test_name}: {str(e)}")
                print(f"\n❌ Unexpected error in test {test_name}: {str(e)}")
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Generate summary
        summary = self.generate_summary()
        
        # Print final report
        self.print_final_report(total_duration)
        
        self.logger.info(f"Test suite completed in {total_duration:.3f}s - {summary['passed']}/{summary['total']} passed")
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = sum(1 for r in self.test_results if r.failed)
        skipped = sum(1 for r in self.test_results if r.skipped)
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'total_duration': sum(r.duration for r in self.test_results),
            'results': self.test_results
        }
    
    def print_final_report(self, total_duration: float):
        """Print comprehensive final test report."""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("📊 FINAL TEST REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"Total Tests Run: {summary['total']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️ Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {total_duration:.3f}s")
        print(f"Average Test Duration: {(total_duration / summary['total']):.3f}s" if summary['total'] > 0 else "N/A")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 60)
        
        for result in self.test_results:
            status_line = f"{result.get_status_icon()} {result.name:<40} {result.get_status():<8} ({result.duration:.3f}s)"
            print(status_line)
            
            if result.failed and result.error_message:
                print(f"    └─ Error: {result.error_message}")
            
            if result.details:
                for detail in result.details[:2]:  # Show first 2 details
                    print(f"    └─ {detail}")
        
        # Failed tests summary
        failed_tests = [r for r in self.test_results if r.failed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS SUMMARY:")
            print("-" * 40)
            for result in failed_tests:
                print(f"• {result.name}: {result.error_message}")
        
        print("\n" + "="*80)
        
        if summary['failed'] == 0:
            print("🎉 ALL TESTS PASSED! The crypto trading bot is working correctly.")
        elif summary['passed'] > 0:
            print(f"⚠️ PARTIAL SUCCESS: {summary['passed']}/{summary['total']} tests passed.")
        else:
            print("🚨 ALL TESTS FAILED: Please check the error messages above.")
        
        print("="*80)


def test_configuration():
    """Test configuration loading and validation."""
    from config.config import AppConfig
    
    print("🔧 Testing configuration loading...")
    config = AppConfig()
    
    # Test basic config loading
    assert config.ticker_short is not None, "ticker_short should be loaded"
    assert config.model_dir is not None, "model_dir should be loaded"
    
    print(f"✅ Configuration loaded successfully")
    print(f"   - Ticker: {config.ticker_short}")
    print(f"   - Model Directory: {config.model_dir}")
    print(f"   - Log Level: {config.log_level}")
    
    # Test API configurations
    alpaca_config = config.get_alpaca_config()
    llm_config = config.get_llm_config()
    
    print(f"   - Alpaca API Key: {'✅ Present' if alpaca_config['api_key'] else '❌ Missing'}")
    print(f"   - OpenAI API Key: {'✅ Present' if llm_config['api_key'] else '❌ Missing'}")
    
    return "Configuration test completed successfully"


def test_logging_system():
    """Test logging system functionality."""
    logger = get_logger(__name__)
    
    print("📝 Testing logging system...")
    
    # Test different log levels
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    
    print("✅ Logging system test completed")
    
    # Check if log file exists
    log_files = []
    logs_dir = "/workspaces/crypto-refactor/logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    print(f"   - Log files found: {len(log_files)}")
    for log_file in log_files[:3]:  # Show first 3 log files
        print(f"     • {log_file}")
    
    return f"Logging test completed - {len(log_files)} log files found"


def test_alpaca_client_initialization():
    """Test Alpaca client initialization."""
    from config.config import AppConfig
    from exchanges.alpaca_client import AlpacaCryptoTrading
    
    print("🏦 Testing Alpaca client initialization...")
    
    config = AppConfig()
    alpaca_config = config.get_alpaca_config()
    
    if not alpaca_config['api_key'] or not alpaca_config['api_secret']:
        raise Exception("Missing Alpaca API credentials - Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    
    # Initialize client
    client = AlpacaCryptoTrading(
        api_key=alpaca_config['api_key'],
        api_secret=alpaca_config['api_secret'],
        base_url=alpaca_config['base_url']
    )
    
    print(f"✅ Alpaca client initialized successfully")
    print(f"   - Base URL: {alpaca_config['base_url']}")
    print(f"   - API Key: {alpaca_config['api_key'][:8]}...{alpaca_config['api_key'][-4:]}")
    
    return "Alpaca client initialization successful"


def test_alpaca_account_access():
    """Test Alpaca account access."""
    from config.config import AppConfig
    from exchanges.alpaca_client import AlpacaCryptoTrading
    
    print("💰 Testing Alpaca account access...")
    
    config = AppConfig()
    alpaca_config = config.get_alpaca_config()
    
    if not alpaca_config['api_key'] or not alpaca_config['api_secret']:
        raise Exception("Missing Alpaca API credentials")
    
    client = AlpacaCryptoTrading(
        api_key=alpaca_config['api_key'],
        api_secret=alpaca_config['api_secret'],
        base_url=alpaca_config['base_url']
    )
    
    # Test account access
    account = client.get_account()
    
    print(f"✅ Account access successful")
    print(f"   - Account ID: {account.get('id', 'N/A')}")
    print(f"   - Account Status: {account.get('status', 'N/A')}")
    print(f"   - Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
    print(f"   - Cash Available: ${float(account.get('cash', 0)):,.2f}")
    
    return f"Account access test passed - Portfolio: ${float(account.get('portfolio_value', 0)):,.2f}"


def test_crypto_data_client():
    """Test crypto data client functionality."""
    from data.crypto_data_client import CryptoMarketDataClient
    
    print("📊 Testing crypto data client...")
    
    client = CryptoMarketDataClient()
    
    # Test real-time price
    btc_data = client.get_realtime_price("BTC/USD")
    
    if not btc_data:
        raise Exception("Failed to get BTC price data")
    
    print(f"✅ Crypto data client working")
    print(f"   - BTC/USD Price: ${btc_data['price']:,.2f}")
    print(f"   - 24h Change: {btc_data['24h_change']:+.2f}%")
    print(f"   - Last Updated: {btc_data['last_updated_mt']}")
    
    return f"Crypto data test passed - BTC: ${btc_data['price']:,.2f}"


def test_llm_client():
    """Test LLM client functionality."""
    from llm.llm_client import LLMClient
    from config.config import AppConfig
    
    print("🤖 Testing LLM client...")
    
    config = AppConfig()
    llm_config = config.get_llm_config()
    
    if not llm_config.get('api_key'):
        print("⚠️ OpenAI API key not found - skipping LLM test")
        return "LLM test skipped - no API key"
    
    client = LLMClient(config=config)
    
    # Test simple query
    test_prompt = "Say 'Hello from crypto bot' and nothing else."
    response = client.query(test_prompt, provider="openai")
    
    print(f"✅ LLM client working")
    print(f"   - Response: {response[:50]}...")
    
    return f"LLM test passed - Response length: {len(response)} chars"


def main():
    """Run the comprehensive test suite."""
    # Set up logging
    setup_default_logging()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Define test functions
    test_functions = [
        ("Configuration Loading", test_configuration),
        ("Logging System", test_logging_system),
        ("Alpaca Client Init", test_alpaca_client_initialization),
        ("Alpaca Account Access", test_alpaca_account_access),
        ("Crypto Data Client", test_crypto_data_client),
        ("LLM Client", test_llm_client),
    ]
    
    # Run test suite
    summary = runner.run_test_suite(test_functions)
    
    # Return appropriate exit code
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
