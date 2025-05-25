#!/usr/bin/env python3
"""
Focused test script for Alpaca positions functionality.
Tests position retrieval, portfolio summaries, and position analysis.
"""

import os
import sys

# Add parent directory to path so we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig
from exchanges.alpaca_client import AlpacaCryptoTrading

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
    
    print("=== Alpaca Positions Test ===\n")
    
    # Initialize configuration
    config = AppConfig()
    alpaca_config = config.get_alpaca_config()
    
    # Validate credentials
    if not alpaca_config['api_key'] or not alpaca_config['api_secret']:
        print("❌ Missing Alpaca API credentials!")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
        return
    
    print("✅ Alpaca credentials found")
    print(f"📊 Using base URL: {alpaca_config['base_url']}")
    print()
    
    # Initialize client
    try:
        client = AlpacaCryptoTrading(
            api_key=alpaca_config['api_key'],
            api_secret=alpaca_config['api_secret'],
            base_url=alpaca_config['base_url']
        )
        print("✅ Alpaca client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Alpaca client: {str(e)}")
        return
    
    # Test 1: Account Information
    print("\n" + "="*50)
    print("📋 ACCOUNT INFORMATION")
    print("="*50)
    
    try:
        account = client.get_account()
        
        print(f"Account ID: {account.get('id', 'N/A')}")
        print(f"Account Status: {account.get('status', 'N/A')}")
        print(f"Cash Available: {format_currency(account.get('cash', 'N/A'))}")
        print(f"Buying Power: {format_currency(account.get('buying_power', 'N/A'))}")
        print(f"Portfolio Value: {format_currency(account.get('portfolio_value', 'N/A'))}")
        print(f"Day Trade Count: {account.get('daytrade_count', 'N/A')}")
        print(f"Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Failed to get account information: {str(e)}")
        return
    
    # Test 2: Positions Information
    print("\n" + "="*50)
    print("📈 POSITIONS ANALYSIS")
    print("="*50)
    
    try:
        positions = client.list_positions()
        
        if not positions or len(positions) == 0:
            print("📭 No open positions found")
            print("\n💡 Tips for testing positions:")
            print("   • Place a small test order first")
            print("   • Check if you're using paper trading vs live")
            print("   • Verify your account has sufficient buying power")
        else:
            print(f"📊 Found {len(positions)} open position(s):\n")
            
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
            
    except Exception as e:
        print(f"❌ Failed to get positions: {str(e)}")
        print(f"Error details: {type(e).__name__}")
    
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
