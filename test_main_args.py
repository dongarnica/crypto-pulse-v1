#!/usr/bin/env python3
"""Test script for main application argument parsing."""

import sys
import os
import argparse

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

def test_argument_parsing():
    """Test the argument parsing functionality."""
    
    print("=== Testing Main Application Argument Parsing ===")
    
    # Simulate the argument parser from main_app.py
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot - Comprehensive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_app.py --mode dashboard                    # Interactive dashboard
  python main_app.py --mode trading --duration 24       # Automated trading for 24 hours
  python main_app.py --mode analysis --period 30        # 30-day performance analysis
  python main_app.py --mode backtest --hours 720        # Backtest for 720 hours
  python main_app.py --mode data --duration 1           # Data collection for 1 hour
  python main_app.py --mode setup                       # System setup and validation
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['dashboard', 'trading', 'analysis', 'backtest', 'data', 'setup'],
        default='dashboard',
        help='Execution mode (default: dashboard)'
    )
    
    parser.add_argument(
        '--ticker',
        default='BTC',
        help='Crypto ticker symbol (default: BTC)'
    )
    
    parser.add_argument(
        '--multi-ticker',
        action='store_true',
        help='Enable multi-ticker trading mode'
    )
    
    parser.add_argument(
        '--max-tickers',
        type=int,
        default=3,
        help='Maximum number of active tickers in multi-ticker mode (default: 3)'
    )
    
    parser.add_argument(
        '--ticker-allocation',
        type=float,
        default=0.33,
        help='Portfolio allocation per ticker in multi-ticker mode (default: 0.33)'
    )
    
    # Test different argument combinations
    test_cases = [
        ['--mode', 'setup'],
        ['--mode', 'setup', '--multi-ticker'],
        ['--mode', 'setup', '--multi-ticker', '--max-tickers', '5'],
        ['--mode', 'setup', '--multi-ticker', '--max-tickers', '2', '--ticker-allocation', '0.5'],
    ]
    
    for i, test_args in enumerate(test_cases, 1):
        print(f"\nTest {i}: {' '.join(test_args)}")
        try:
            args = parser.parse_args(test_args)
            print(f"   Mode: {args.mode}")
            print(f"   Ticker: {args.ticker}")
            print(f"   Multi-ticker: {getattr(args, 'multi_ticker', False)}")
            print(f"   Max tickers: {getattr(args, 'max_tickers', 3)}")
            print(f"   Ticker allocation: {getattr(args, 'ticker_allocation', 0.33)}")
        except Exception as e:
            print(f"   Error: {str(e)}")
    
    print("\n✅ Argument parsing test completed!")

def test_config_integration():
    """Test configuration integration with arguments."""
    
    print("\n=== Testing Configuration Integration ===")
    
    try:
        from config.config import AppConfig
        
        # Test configuration with multi-ticker enabled
        config = AppConfig(ticker='BTC')
        config.multi_ticker_enabled = True
        config.max_active_tickers = 5
        config.ticker_allocation = 0.2
        
        active_tickers = config.get_active_tickers()
        
        print(f"Multi-ticker enabled: {config.multi_ticker_enabled}")
        print(f"Max active tickers: {config.max_active_tickers}")
        print(f"Ticker allocation: {config.ticker_allocation}")
        print(f"Active tickers ({len(active_tickers)}): {active_tickers}")
        
        print("\n✅ Configuration integration test completed!")
        
    except Exception as e:
        print(f"❌ Configuration test error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_argument_parsing()
    test_config_integration()
