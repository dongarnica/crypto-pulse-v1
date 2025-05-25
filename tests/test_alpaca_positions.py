#!/usr/bin/env python3
"""
Focused test script for Alpaca positions functionality.
Tests position retrieval, portfolio summaries, and position analysis.
"""

import os
import sys
import time
import logging

# Add parent directory to path so we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig
from exchanges.alpaca_client import AlpacaCryptoTrading

def print_test_summary(results, duration):
    """Print test summary."""
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⚠️ Skipped: {results['skipped']}")
    print(f"Success Rate: {(results['passed']/results['total_tests']*100):.1f}%")
    print(f"Duration: {duration:.2f}s")
    print(f"\nTest Details:")
    for detail in results['details']:
        print(f"  • {detail}")
    print(f"{'='*60}")
    
    if results['failed'] == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️ {results['failed']} TEST(S) FAILED")

def format_currency(amount):
    """Format currency values consistently."""
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return f"${amount}"

def format_percentage(value):
    """Format percentage values consistently."""
    try:
        return f"{float(value):+.2f}%"
    except (ValueError, TypeError):
        return f"{value}%"

def test_alpaca_positions():
    """Test Alpaca positions functionality."""
    
    test_results = {
        'total_tests': 5,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    # Initialize configuration (this also sets up logging)
    config = AppConfig()
    logger = config.get_logger(__name__)
    
    logger.info("Starting Alpaca positions test")
    print("=== Alpaca Positions Test ===\n")
    
    start_time = time.time()
    
    # Test 1: Validate credentials
    print("🔐 TEST 1: Validating Alpaca Credentials")
    print("-" * 40)
    
    # Initialize configuration
    alpaca_config = config.get_alpaca_config()
    
    # Validate credentials
    if not alpaca_config['api_key'] or not alpaca_config['api_secret']:
        error_msg = "Missing Alpaca API credentials!"
        logger.error(error_msg)
        print(f"❌ FAILED: {error_msg}")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
        test_results['failed'] += 1
        test_results['details'].append(f"Credentials validation: FAILED - {error_msg}")
        print_test_summary(test_results, time.time() - start_time)
        return test_results
    
    logger.info("Alpaca credentials found")
    print("✅ PASSED: Alpaca credentials found")
    print(f"📊 Using base URL: {alpaca_config['base_url']}")
    logger.info(f"Using Alpaca base URL: {alpaca_config['base_url']}")
    test_results['passed'] += 1
    test_results['details'].append("Credentials validation: PASSED")
    print()
    
    logger.info("Alpaca credentials found")
    print("✅ Alpaca credentials found")
    print(f"📊 Using base URL: {alpaca_config['base_url']}")
    logger.info(f"Using Alpaca base URL: {alpaca_config['base_url']}")
    print()
    # Test 2: Initialize client
    print("🏦 TEST 2: Initializing Alpaca Client")
    print("-" * 40)
    
    # Initialize client
    try:
        client = AlpacaCryptoTrading(
            api_key=alpaca_config['api_key'],
            api_secret=alpaca_config['api_secret'],
            base_url=alpaca_config['base_url']
        )
        print("✅ PASSED: Alpaca client initialized")
        logger.info("Alpaca client initialized successfully")
        test_results['passed'] += 1
        test_results['details'].append("Client initialization: PASSED")
    except Exception as e:
        error_msg = f"Failed to initialize Alpaca client: {str(e)}"
        logger.error(error_msg)
        print(f"❌ FAILED: {error_msg}")
        test_results['failed'] += 1
        test_results['details'].append(f"Client initialization: FAILED - {error_msg}")
        print_test_summary(test_results, time.time() - start_time)
        return test_results
    print()
    
    # Test 3: Account Information
    print("💰 TEST 3: Account Information Retrieval")
    print("-" * 40)
    
    logger.info("Testing account information retrieval")
    
    try:
        account = client.get_account()
        
        account_info = {
            'id': account.get('id', 'N/A'),
            'status': account.get('status', 'N/A'),
            'cash': account.get('cash', 'N/A'),
            'buying_power': account.get('buying_power', 'N/A'),
            'portfolio_value': account.get('portfolio_value', 'N/A')
        }
        
        print("✅ PASSED: Account information retrieved")
        print(f"Account ID: {account_info['id']}")
        print(f"Account Status: {account_info['status']}")
        print(f"Cash Available: {format_currency(account_info['cash'])}")
        print(f"Buying Power: {format_currency(account_info['buying_power'])}")
        print(f"Portfolio Value: {format_currency(account_info['portfolio_value'])}")
        print(f"Day Trade Count: {account.get('daytrade_count', 'N/A')}")
        print(f"Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
        
        logger.info(f"Account retrieved - ID: {account_info['id']}, Status: {account_info['status']}")
        logger.info(f"Portfolio value: {account_info['portfolio_value']}, Cash: {account_info['cash']}")
        
        test_results['passed'] += 1
        test_results['details'].append(f"Account info: PASSED - Portfolio: {format_currency(account_info['portfolio_value'])}")
        
    except Exception as e:
        error_msg = f"Failed to get account information: {str(e)}"
        logger.error(error_msg)
        print(f"❌ FAILED: {error_msg}")
        test_results['failed'] += 1
        test_results['details'].append(f"Account info: FAILED - {error_msg}")
    print()
    
    # Test 4: Positions Information
    print("📈 TEST 4: Positions Analysis")
    print("-" * 40)
    
    logger.info("Testing positions retrieval")
    
    try:
        positions = client.list_positions()
        
        if not positions or len(positions) == 0:
            logger.warning("No open positions found")
            print("⚠️ SKIPPED: No open positions found")
            print("\n💡 Tips for testing positions:")
            print("   • Place a small test order first")
            print("   • Check if you're using paper trading vs live")
            print("   • Verify your account has sufficient buying power")
            test_results['skipped'] += 1
            test_results['details'].append("Positions analysis: SKIPPED - No positions found")
        else:
            position_count = len(positions)
            logger.info(f"Retrieved {position_count} open positions")
            print(f"✅ PASSED: Found {position_count} open position(s)")
            
            total_market_value = 0
            total_unrealized_pl = 0
            total_cost_basis = 0
            
            for i, pos in enumerate(positions, 1):
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('qty', 0)
                side = pos.get('side', 'N/A')
                market_value = pos.get('market_value', 0)
                avg_entry_price = pos.get('avg_entry_price', 0)
                current_price = pos.get('current_price', 0)
                unrealized_pl = pos.get('unrealized_pl', 0)
                unrealized_plpc = pos.get('unrealized_plpc', 0)
                cost_basis = pos.get('cost_basis', 0)
                
                # Convert string values to float for calculations
                try:
                    qty = float(qty)
                    market_value = float(market_value)
                    avg_entry_price = float(avg_entry_price)
                    current_price = float(current_price)
                    unrealized_pl = float(unrealized_pl)
                    unrealized_plpc = float(unrealized_plpc) * 100 if unrealized_plpc else 0
                    cost_basis = float(cost_basis)
                    
                    logger.debug(f"Position {symbol}: {qty} units, market value: {market_value}, P&L: {unrealized_pl}")
                except ValueError as ve:
                    logger.warning(f"Could not parse numeric values for {symbol}: {str(ve)}")
                    print(f"⚠️  Warning: Could not parse numeric values for {symbol}")
                    continue
                
                print(f"Position {i}: {symbol}")
                print(f"├─ Quantity: {qty:,.6f} ({side})")
                print(f"├─ Avg Entry Price: {format_currency(avg_entry_price)}")
                print(f"├─ Current Price: {format_currency(current_price)}")
                print(f"├─ Cost Basis: {format_currency(cost_basis)}")
                print(f"├─ Market Value: {format_currency(market_value)}")
                print(f"└─ Unrealized P&L: {format_currency(unrealized_pl)} ({format_percentage(unrealized_plpc)})")
                print()
                
                # Add to totals
                total_market_value += market_value
                total_unrealized_pl += unrealized_pl
                total_cost_basis += cost_basis
            
            # Portfolio Summary
            logger.info(f"Portfolio totals - Market value: {total_market_value}, P&L: {total_unrealized_pl}")
            print("="*30)
            print("📊 PORTFOLIO SUMMARY")
            print("="*30)
            print(f"Total Positions: {len(positions)}")
            print(f"Total Cost Basis: {format_currency(total_cost_basis)}")
            print(f"Total Market Value: {format_currency(total_market_value)}")
            print(f"Total Unrealized P&L: {format_currency(total_unrealized_pl)}")
            
            if total_cost_basis > 0:
                total_return_pct = (total_unrealized_pl / total_cost_basis) * 100
                print(f"Total Return: {format_percentage(total_return_pct)}")
                logger.info(f"Total return: {total_return_pct:.2f}%")
            
            test_results['passed'] += 1
            test_results['details'].append(f"Positions analysis: PASSED - {position_count} positions, Total P&L: {format_currency(total_unrealized_pl)}")
            
    except Exception as e:
        error_msg = f"Failed to get positions: {str(e)}"
        logger.error(f"{error_msg} - Error type: {type(e).__name__}")
        print(f"❌ FAILED: {error_msg}")
        print(f"Error details: {type(e).__name__}")
        test_results['failed'] += 1
        test_results['details'].append(f"Positions analysis: FAILED - {error_msg}")
    print()
    
    # Test 3: Portfolio Summary Analysis
    print("\n" + "="*50)
    print("📊 PORTFOLIO SUMMARY ANALYSIS")
    print("="*50)
    
    try:
        portfolio_summary = client.get_portfolio_summary()
        
        if 'error' in portfolio_summary:
            print(f"❌ Portfolio analysis failed: {portfolio_summary['error']}")
        else:
            # Account summary
            account = portfolio_summary['account']
            print(f"💰 Account Overview:")
            print(f"   Cash Available: {format_currency(account['cash'])}")
            print(f"   Portfolio Value: {format_currency(account['portfolio_value'])}")
            print(f"   Account Status: {account['status']}")
            
            # Positions summary
            positions_data = portfolio_summary['positions']
            print(f"\n📈 Positions Overview:")
            print(f"   Total Positions: {positions_data['count']}")
            print(f"   Total Market Value: {format_currency(positions_data['total_market_value'])}")
            print(f"   Total Cost Basis: {format_currency(positions_data['total_cost_basis'])}")
            print(f"   Total Unrealized P&L: {format_currency(positions_data['total_unrealized_pl'])}")
            
            # Performance summary
            performance = portfolio_summary['performance']
            print(f"\n🎯 Performance Summary:")
            print(f"   Total Return: {format_percentage(performance['total_return_pct'])}")
            
            if performance['best_performer']:
                best = performance['best_performer']
                print(f"   Best Performer: {best['symbol']} ({format_percentage(best['return_pct'])})")
            
            if performance['worst_performer']:
                worst = performance['worst_performer']
                print(f"   Worst Performer: {worst['symbol']} ({format_percentage(worst['return_pct'])})")
            
            # Individual position details
            if positions_data['by_symbol']:
                print(f"\n📋 Position Details:")
                for symbol, pos_data in positions_data['by_symbol'].items():
                    print(f"   {symbol}:")
                    print(f"     Market Value: {format_currency(pos_data['market_value'])}")
                    print(f"     Return: {format_percentage(pos_data['return_pct'])}")
                    print(f"     P&L: {format_currency(pos_data['unrealized_pl'])}")
    
    except Exception as e:
        print(f"❌ Portfolio summary failed: {str(e)}")

    # Test 4: Individual Position Risk Analysis
    print("\n" + "="*50)
    print("⚠️  POSITION RISK ANALYSIS")
    print("="*50)
    
    # Get positions for risk analysis
    try:
        positions = client.list_positions()
        if positions and len(positions) > 0:
            for pos in positions[:3]:  # Analyze first 3 positions
                symbol = pos.get('symbol', 'N/A')
                risk_analysis = client.analyze_position_risk(symbol)
                
                if 'error' in risk_analysis:
                    print(f"❌ {symbol}: {risk_analysis['error']}")
                else:
                    print(f"\n🔍 {symbol} Risk Analysis:")
                    print(f"   Current Price: {format_currency(risk_analysis['current_price'])}")
                    print(f"   Entry Price: {format_currency(risk_analysis['entry_price'])}")
                    print(f"   Price Change: {format_percentage(risk_analysis['price_change_pct'])}")
                    print(f"   Risk Level: {risk_analysis['risk_level']}")
                    print(f"   Is Profitable: {'✅' if risk_analysis['analysis']['is_profitable'] else '❌'}")
                    print(f"   High Risk: {'⚠️ ' if risk_analysis['analysis']['is_high_risk'] else '✅'}")
                    print(f"   Needs Attention: {'🚨' if risk_analysis['analysis']['needs_attention'] else '✅'}")
        else:
            print("📭 No positions available for risk analysis")
    
    except Exception as e:
        print(f"❌ Risk analysis failed: {str(e)}")
    
    # Test 5: Individual Position Lookup
    print("\n" + "="*50)
    print("🔍 INDIVIDUAL POSITION TEST")
    print("="*50)
    
    test_symbols = ["BTCUSD", "ETHUSD", "ADAUSD"]
    
    for symbol in test_symbols:
        try:
            position = client.get_position(symbol)
            if position:
                print(f"✅ {symbol}: Found position")
                print(f"   Quantity: {position.get('qty', 'N/A')}")
                print(f"   Market Value: {format_currency(position.get('market_value', 'N/A'))}")
            else:
                print(f"📭 {symbol}: No position found")
        except Exception as e:
            print(f"📭 {symbol}: No position (or error: {str(e)})")
    
    print("\n" + "="*50)
    print("✅ POSITIONS TEST COMPLETE")
    print("="*50)

if __name__ == "__main__":
    test_alpaca_positions()
