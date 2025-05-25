#!/usr/bin/env python3
"""
Comprehensive Alpaca Positions Test Script
Combines simple diagnostic checks with advanced portfolio analysis.
Tests position retrieval, portfolio summaries, risk analysis, and account information.
"""

import os
import sys
import traceback
import pytest

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

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

def print_section_header(title, char="=", width=60):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f"{title:^{width}}")
    print(char * width)

def print_subsection(title, char="─", width=40):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

@pytest.fixture
def alpaca_client():
    """Create an AlpacaCryptoTrading client for testing."""
    try:
        from config.config import AppConfig
        config = AppConfig()
        alpaca_config = config.get_alpaca_config()
        
        api_key = alpaca_config.get('api_key', '')
        api_secret = alpaca_config.get('api_secret', '')
        
        if not api_key or not api_secret:
            pytest.skip("API credentials not available")
        
        from exchanges.alpaca_client import AlpacaCryptoTrading
        
        client = AlpacaCryptoTrading(
            api_key=api_key,
            api_secret=api_secret,
            base_url=alpaca_config['base_url']
        )
        return client
        
    except Exception as e:
        pytest.skip(f"Could not initialize Alpaca client: {e}")

def test_account_access(alpaca_client):
    """Test account access and display basic account information."""
    print_section_header("📋 ACCOUNT INFORMATION")
    
    try:
        account = alpaca_client.get_account()
        
        print(f"Account ID: {account.get('id', 'N/A')}")
        print(f"Account Status: {account.get('status', 'N/A')}")
        print(f"Cash Available: {format_currency(account.get('cash', 'N/A'))}")
        print(f"Buying Power: {format_currency(account.get('buying_power', 'N/A'))}")
        print(f"Portfolio Value: {format_currency(account.get('portfolio_value', 'N/A'))}")
        print(f"Day Trade Count: {account.get('daytrade_count', 'N/A')}")
        print(f"Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
        
        assert account is not None
        assert 'id' in account
        
    except Exception as e:
        print(f"❌ Failed to get account information: {str(e)[:200]}...")
        pytest.fail(f"Account access failed: {e}")

def test_positions_basic(alpaca_client):
    """Test basic positions retrieval and display."""
    print_section_header("📈 BASIC POSITIONS ANALYSIS")
    
    try:
        positions = alpaca_client.list_positions()
        print(f"✅ Positions retrieved - Count: {len(positions) if positions else 0}")
        
        if not positions or len(positions) == 0:
            print("📭 No open positions found")
            print("\n💡 Tips for testing positions:")
            print("   • Place a small test order first")
            print("   • Check if you're using paper trading vs live")
            print("   • Verify your account has sufficient buying power")
            pytest.skip("No positions available for testing")
        
        print(f"📊 Found {len(positions)} open position(s):")
        
        for i, pos in enumerate(positions[:5], 1):  # Show first 5 positions
            symbol = pos.get('symbol', 'N/A')
            qty = pos.get('qty', 'N/A')
            market_value = pos.get('market_value', 'N/A')
            unrealized_pl = pos.get('unrealized_pl', 'N/A')
            side = pos.get('side', 'N/A')
            
            print(f"   Position {i}: {symbol}")
            print(f"     Quantity: {qty} ({side})")
            print(f"     Market Value: {format_currency(market_value)}")
            print(f"     Unrealized P&L: {format_currency(unrealized_pl)}")
        
        if len(positions) > 5:
            print(f"   ... and {len(positions) - 5} more positions")
        
        assert len(positions) > 0
        
    except Exception as e:
        print(f"❌ Positions access failed: {str(e)[:200]}...")
        pytest.fail(f"Positions retrieval failed: {e}")
        print(f"❌ Positions access failed: {str(e)[:200]}...")
        return []

def test_positions_detailed(alpaca_client):
    """Test detailed positions analysis with calculations."""
    print_section_header("📊 DETAILED POSITIONS ANALYSIS")
    
    try:
        positions = alpaca_client.list_positions()
        
        if not positions or len(positions) == 0:
            print("📭 No positions for detailed analysis")
            pytest.skip("No positions available for detailed analysis")
        
        print(f"📊 Analyzing {len(positions)} position(s) in detail:\n")
        
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
            except ValueError:
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
        print_subsection("📊 PORTFOLIO TOTALS")
        print(f"Total Positions: {len(positions)}")
        print(f"Total Cost Basis: {format_currency(total_cost_basis)}")
        print(f"Total Market Value: {format_currency(total_market_value)}")
        print(f"Total Unrealized P&L: {format_currency(total_unrealized_pl)}")
        
        if total_cost_basis > 0:
            total_return_pct = (total_unrealized_pl / total_cost_basis) * 100
            print(f"Total Return: {format_percentage(total_return_pct)}")
        
        assert len(positions) > 0
        assert total_market_value >= 0
        
    except Exception as e:
        print(f"❌ Failed detailed positions analysis: {str(e)}")
        pytest.fail(f"Detailed positions analysis failed: {e}")

def test_portfolio_summary(alpaca_client):
    """Test comprehensive portfolio summary analysis."""
    print_section_header("🎯 PORTFOLIO SUMMARY ANALYSIS")
    
    try:
        portfolio_summary = alpaca_client.get_portfolio_summary()
        
        if 'error' in portfolio_summary:
            print(f"❌ Portfolio analysis failed: {portfolio_summary['error']}")
            pytest.fail(f"Portfolio summary failed: {portfolio_summary['error']}")
        
        # Account summary
        account = portfolio_summary['account']
        print_subsection("💰 ACCOUNT OVERVIEW")
        print(f"Cash Available: {format_currency(account['cash'])}")
        print(f"Portfolio Value: {format_currency(account['portfolio_value'])}")
        print(f"Buying Power: {format_currency(account['buying_power'])}")
        print(f"Account Status: {account['status']}")
        
        # Positions summary
        positions_data = portfolio_summary['positions']
        print_subsection("📈 POSITIONS OVERVIEW")
        print(f"Total Positions: {positions_data['count']}")
        print(f"Total Market Value: {format_currency(positions_data['total_market_value'])}")
        print(f"Total Cost Basis: {format_currency(positions_data['total_cost_basis'])}")
        print(f"Total Unrealized P&L: {format_currency(positions_data['total_unrealized_pl'])}")
        
        # Performance summary
        # Performance summary
        performance = portfolio_summary['performance']
        print_subsection("🎯 PERFORMANCE SUMMARY")
        print(f"Total Return: {format_percentage(performance['total_return_pct'])}")
        
        if performance['best_performer']:
            best = performance['best_performer']
            print(f"Best Performer: {best['symbol']} ({format_percentage(best['return_pct'])})")
            print(f"  P&L: {format_currency(best['unrealized_pl'])}")
        
        if performance['worst_performer']:
            worst = performance['worst_performer']
            print(f"Worst Performer: {worst['symbol']} ({format_percentage(worst['return_pct'])})")
            print(f"  P&L: {format_currency(worst['unrealized_pl'])}")
        
        # Individual position details
        if positions_data['by_symbol']:
            print_subsection("📋 POSITION BREAKDOWN")
            for symbol, pos_data in positions_data['by_symbol'].items():
                print(f"{symbol}:")
                print(f"  Market Value: {format_currency(pos_data['market_value'])}")
                print(f"  Return: {format_percentage(pos_data['return_pct'])}")
                print(f"  P&L: {format_currency(pos_data['unrealized_pl'])}")
                print(f"  Quantity: {pos_data['qty']:,.6f}")
        
        assert 'account' in portfolio_summary
        assert 'positions' in portfolio_summary
        
    except Exception as e:
        print(f"❌ Portfolio summary failed: {str(e)}")
        pytest.fail(f"Portfolio summary failed: {e}")

def test_risk_analysis(alpaca_client):
    """Test individual position risk analysis."""
    print_section_header("⚠️  POSITION RISK ANALYSIS")
    
    try:
        positions = alpaca_client.list_positions()
        if not positions or len(positions) == 0:
            print("📭 No positions available for risk analysis")
            pytest.skip("No positions available for risk analysis")
        
        print(f"🔍 Analyzing risk for {min(3, len(positions))} position(s):")
        
        for pos in positions[:3]:  # Analyze first 3 positions
            symbol = pos.get('symbol', 'N/A')
            risk_analysis = alpaca_client.analyze_position_risk(symbol)
            
            if 'error' in risk_analysis:
                print(f"\n❌ {symbol}: {risk_analysis['error']}")
                continue
            
            print(f"\n🔍 {symbol} Risk Analysis:")
            print(f"   Current Price: {format_currency(risk_analysis['current_price'])}")
            print(f"   Entry Price: {format_currency(risk_analysis['entry_price'])}")
            print(f"   Market Value: {format_currency(risk_analysis['market_value'])}")
            print(f"   Price Change: {format_percentage(risk_analysis['price_change_pct'])}")
            print(f"   Risk Level: {risk_analysis['risk_level']}")
            
            analysis = risk_analysis['analysis']
            print(f"   Status Flags:")
            print(f"     Profitable: {'✅' if analysis['is_profitable'] else '❌'}")
            print(f"     High Risk: {'⚠️ Yes' if analysis['is_high_risk'] else '✅ No'}")
            print(f"     Needs Attention: {'🚨 Yes' if analysis['needs_attention'] else '✅ No'}")
        
        assert len(positions) > 0
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {str(e)}")
        pytest.fail(f"Risk analysis failed: {e}")

def test_individual_positions(alpaca_client):
    """Test individual position lookup by symbol."""
    print_section_header("🔍 INDIVIDUAL POSITION LOOKUP")
    
    test_symbols = ["BTCUSD", "ETHUSD", "AAVEUSD", "DOGEUSD", "DOTUSD","LINKUSD","LTCUSD","XRPUSD","SUSHIUSD"]
    
    print(f"Testing position lookup for common crypto symbols:")
    
    found_positions = 0
    for symbol in test_symbols:
        try:
            position = alpaca_client.get_position(symbol)
            if position:
                print(f"✅ {symbol}: Found position")
                print(f"   Quantity: {position.get('qty', 'N/A')}")
                print(f"   Market Value: {format_currency(position.get('market_value', 'N/A'))}")
                print(f"   Unrealized P&L: {format_currency(position.get('unrealized_pl', 'N/A'))}")
                found_positions += 1
            else:
                print(f"📭 {symbol}: No position found")
        except Exception as e:
            print(f"📭 {symbol}: No position (API response: {str(e)[:50]}...)")
    
    print(f"\nSummary: Found {found_positions} out of {len(test_symbols)} tested symbols")
    # This test doesn't fail if no positions are found, as that's valid

def run_comprehensive_test():
    """Run the complete comprehensive test suite (for manual execution)."""
    print("=" * 80)
    print("🚀 COMPREHENSIVE ALPACA POSITIONS TEST SUITE")
    print("=" * 80)
    print("This script tests all aspects of Alpaca positions functionality:")
    print("• Configuration and authentication")
    print("• Account information retrieval")
    print("• Basic and detailed positions analysis") 
    print("• Portfolio summary and performance metrics")
    print("• Individual position risk analysis")
    print("• Position lookup by symbol")
    
    try:
        # Create client using the same logic as the fixture
        from config.config import AppConfig
        config = AppConfig()
        alpaca_config = config.get_alpaca_config()
        
        api_key = alpaca_config.get('api_key', '')
        api_secret = alpaca_config.get('api_secret', '')
        
        if not api_key or not api_secret:
            print("\n❌ Cannot continue without proper setup. Please fix configuration issues.")
            return
        
        from exchanges.alpaca_client import AlpacaCryptoTrading
        
        client = AlpacaCryptoTrading(
            api_key=api_key,
            api_secret=api_secret,
            base_url=alpaca_config['base_url']
        )
        
        # Run the test functions
        print("\n🔧 Testing account access...")
        test_account_access(client)
        
        print("\n📈 Testing basic positions...")
        try:
            test_positions_basic(client)
            test_positions_detailed(client)
            test_portfolio_summary(client)
            test_risk_analysis(client)
        except Exception as e:
            if "skip" in str(e).lower():
                print("Skipping position tests - no positions available")
        
        print("\n🔍 Testing individual positions...")
        test_individual_positions(client)
        
        print_section_header("✅ TEST SUITE COMPLETE")
        print("🎉 Comprehensive Alpaca positions test completed!")
        
    except Exception as e:
        print(f"\n❌ Unexpected error in test suite: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in test suite: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
