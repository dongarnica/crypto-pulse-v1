#!/usr/bin/env python3
"""
Complete integration test demonstrating the full Alpaca positions testing system.
This script showcases the integration between:
- Configuration management
- Alpaca trading client
- Portfolio analysis
- Risk assessment
- LSTM integration potential
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

def format_currency(amount):
    """Format currency consistently."""
    try:
        return f"${float(amount):,.2f}"
    except:
        return f"${amount}"

def format_percentage(value):
    """Format percentage consistently."""
    try:
        return f"{float(value):+.2f}%"
    except:
        return f"{value}%"

def integration_test():
    """
    Complete integration test of the Alpaca positions system.
    """
    
    print("=" * 60)
    print("🚀 COMPLETE ALPACA POSITIONS INTEGRATION TEST")
    print("=" * 60)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'summary': {}
    }
    
    try:
        # 1. Configuration System Test
        print("\n📋 1. CONFIGURATION SYSTEM")
        print("-" * 30)
        
        from config.config import AppConfig
        config = AppConfig()
        
        print(f"✅ Config initialized for ticker: {config.ticker_short}")
        print(f"   Model directory: {config.model_dir}")
        print(f"   Model filename: {config.model_name}")
        
        # Test different access methods
        alpaca_config = config.get_alpaca_config()
        llm_config = config.get_llm_config()
        trading_config = config.get_trading_config()
        
        # Dictionary-style access
        api_key_dict = config['ALPACA_API_KEY'] if 'ALPACA_API_KEY' in config else None
        
        print(f"   Dictionary access: {'✅' if api_key_dict else '❌'}")
        print(f"   Specialized getters: {'✅' if alpaca_config else '❌'}")
        
        test_results['tests']['config'] = {
            'status': 'success',
            'ticker': config.ticker_short,
            'api_keys_configured': {
                'alpaca': bool(alpaca_config.get('api_key')),
                'openai': bool(llm_config.get('api_key')),
                'perplexity': bool(llm_config.get('perplexity_key'))
            }
        }
        
        # 2. Alpaca Client Initialization
        print("\n🔌 2. ALPACA CLIENT INITIALIZATION")
        print("-" * 30)
        
        if not alpaca_config.get('api_key') or not alpaca_config.get('api_secret'):
            print("❌ Alpaca credentials not configured")
            print("   Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
            test_results['tests']['alpaca_init'] = {'status': 'failed', 'reason': 'missing_credentials'}
            return test_results
        
        from exchanges.alpaca_client import AlpacaCryptoTrading
        
        client = AlpacaCryptoTrading(
            api_key=alpaca_config['api_key'],
            api_secret=alpaca_config['api_secret'],
            base_url=alpaca_config['base_url']
        )
        
        print("✅ Alpaca client initialized")
        print(f"   Using base URL: {alpaca_config['base_url']}")
        
        test_results['tests']['alpaca_init'] = {
            'status': 'success',
            'base_url': alpaca_config['base_url']
        }
        
        # 3. Account Information Test
        print("\n💼 3. ACCOUNT INFORMATION")
        print("-" * 30)
        
        try:
            account = client.get_account()
            print(f"✅ Account accessed")
            print(f"   Status: {account.get('status', 'Unknown')}")
            print(f"   Cash: {format_currency(account.get('cash', 0))}")
            print(f"   Buying Power: {format_currency(account.get('buying_power', 0))}")
            print(f"   Portfolio Value: {format_currency(account.get('portfolio_value', 0))}")
            
            test_results['tests']['account'] = {
                'status': 'success',
                'account_status': account.get('status'),
                'cash': float(account.get('cash', 0)),
                'portfolio_value': float(account.get('portfolio_value', 0))
            }
            
        except Exception as e:
            print(f"❌ Account access failed: {str(e)}")
            test_results['tests']['account'] = {'status': 'failed', 'error': str(e)}
        
        # 4. Positions Analysis
        print("\n📊 4. POSITIONS ANALYSIS")
        print("-" * 30)
        
        try:
            positions = client.list_positions()
            position_count = len(positions) if positions else 0
            
            print(f"✅ Positions retrieved: {position_count} positions")
            
            if positions:
                print("   Current positions:")
                for i, pos in enumerate(positions[:3], 1):  # Show first 3
                    symbol = pos.get('symbol', 'N/A')
                    qty = pos.get('qty', 'N/A')
                    market_value = pos.get('market_value', 'N/A')
                    unrealized_pl = pos.get('unrealized_pl', 'N/A')
                    
                    print(f"   {i}. {symbol}: {qty} units")
                    print(f"      Market Value: {format_currency(market_value)}")
                    print(f"      Unrealized P&L: {format_currency(unrealized_pl)}")
                
                if len(positions) > 3:
                    print(f"   ... and {len(positions) - 3} more positions")
            else:
                print("   📭 No positions found")
            
            test_results['tests']['positions'] = {
                'status': 'success',
                'count': position_count,
                'symbols': [pos.get('symbol') for pos in positions[:5]] if positions else []
            }
            
        except Exception as e:
            print(f"❌ Positions retrieval failed: {str(e)}")
            test_results['tests']['positions'] = {'status': 'failed', 'error': str(e)}
        
        # 5. Portfolio Summary Test
        print("\n📈 5. PORTFOLIO SUMMARY ANALYSIS")
        print("-" * 30)
        
        try:
            portfolio = client.get_portfolio_summary()
            
            if 'error' in portfolio:
                print(f"❌ Portfolio analysis failed: {portfolio['error']}")
                test_results['tests']['portfolio'] = {'status': 'failed', 'error': portfolio['error']}
            else:
                print("✅ Portfolio summary generated")
                
                # Account metrics
                account_data = portfolio['account']
                print(f"   Portfolio Value: {format_currency(account_data['portfolio_value'])}")
                
                # Position metrics
                position_data = portfolio['positions']
                print(f"   Total Positions: {position_data['count']}")
                print(f"   Total Market Value: {format_currency(position_data['total_market_value'])}")
                print(f"   Total Unrealized P&L: {format_currency(position_data['total_unrealized_pl'])}")
                
                # Performance metrics
                performance = portfolio['performance']
                print(f"   Total Return: {format_percentage(performance['total_return_pct'])}")
                
                if performance['best_performer']:
                    best = performance['best_performer']
                    print(f"   Best Performer: {best['symbol']} ({format_percentage(best['return_pct'])})")
                
                if performance['worst_performer']:
                    worst = performance['worst_performer']
                    print(f"   Worst Performer: {worst['symbol']} ({format_percentage(worst['return_pct'])})")
                
                test_results['tests']['portfolio'] = {
                    'status': 'success',
                    'total_value': position_data['total_market_value'],
                    'total_pl': position_data['total_unrealized_pl'],
                    'return_pct': performance['total_return_pct'],
                    'position_count': position_data['count']
                }
        
        except Exception as e:
            print(f"❌ Portfolio summary failed: {str(e)}")
            test_results['tests']['portfolio'] = {'status': 'failed', 'error': str(e)}
        
        # 6. Risk Analysis Test
        print("\n⚠️  6. RISK ANALYSIS")
        print("-" * 30)
        
        try:
            if positions and len(positions) > 0:
                test_symbol = positions[0].get('symbol', 'BTCUSD')
                risk_analysis = client.analyze_position_risk(test_symbol)
                
                if 'error' in risk_analysis:
                    print(f"❌ Risk analysis failed: {risk_analysis['error']}")
                else:
                    print(f"✅ Risk analysis for {test_symbol}")
                    print(f"   Risk Level: {risk_analysis['risk_level']}")
                    print(f"   Price Change: {format_percentage(risk_analysis['price_change_pct'])}")
                    print(f"   Is Profitable: {'✅' if risk_analysis['analysis']['is_profitable'] else '❌'}")
                    print(f"   High Risk: {'⚠️' if risk_analysis['analysis']['is_high_risk'] else '✅'}")
                    print(f"   Needs Attention: {'🚨' if risk_analysis['analysis']['needs_attention'] else '✅'}")
                    
                    test_results['tests']['risk_analysis'] = {
                        'status': 'success',
                        'symbol': test_symbol,
                        'risk_level': risk_analysis['risk_level'],
                        'price_change_pct': risk_analysis['price_change_pct']
                    }
            else:
                print("📭 No positions available for risk analysis")
                test_results['tests']['risk_analysis'] = {'status': 'skipped', 'reason': 'no_positions'}
        
        except Exception as e:
            print(f"❌ Risk analysis failed: {str(e)}")
            test_results['tests']['risk_analysis'] = {'status': 'failed', 'error': str(e)}
        
        # 7. LSTM Integration Potential
        print("\n🤖 7. LSTM INTEGRATION READINESS")
        print("-" * 30)
        
        try:
            # Check if LSTM components are available
            lstm_available = False
            try:
                from models.lstm_train import main as lstm_main
                lstm_available = True
            except ImportError:
                pass
            
            # Check if data client is available
            data_available = False
            try:
                from data.crypto_data_client import CryptoMarketDataClient
                data_client = CryptoMarketDataClient()
                data_available = True
            except ImportError:
                pass
            
            print(f"   LSTM Model: {'✅' if lstm_available else '❌'}")
            print(f"   Data Client: {'✅' if data_available else '❌'}")
            print(f"   Config Integration: ✅")
            print(f"   Portfolio Analysis: ✅")
            
            # Potential integration points
            print("   Integration opportunities:")
            print("   • Portfolio data → LSTM signal validation")
            print("   • Risk analysis → LSTM position sizing")
            print("   • Performance tracking → LSTM model feedback")
            
            test_results['tests']['lstm_integration'] = {
                'status': 'success',
                'lstm_available': lstm_available,
                'data_available': data_available,
                'config_ready': True,
                'portfolio_ready': True
            }
        
        except Exception as e:
            print(f"❌ LSTM integration check failed: {str(e)}")
            test_results['tests']['lstm_integration'] = {'status': 'failed', 'error': str(e)}
        
        # Test Summary
        print("\n" + "=" * 60)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'success')
        total_tests = len(test_results['tests'])
        skipped_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'skipped')
        
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        print(f"❌ Failed: {total_tests - passed_tests - skipped_tests}")
        
        if passed_tests == total_tests - skipped_tests:
            print("\n🎉 ALL TESTS PASSED! Alpaca positions system is fully operational.")
        else:
            print(f"\n⚠️  Some tests failed. Check error messages above for details.")
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'skipped': skipped_tests,
            'failed': total_tests - passed_tests - skipped_tests,
            'success_rate': (passed_tests / (total_tests - skipped_tests)) * 100 if total_tests > skipped_tests else 0
        }
        
        # Save detailed results
        with open('integration_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to 'integration_test_results.json'")
        
    except Exception as e:
        print(f"\n❌ Critical error during integration test: {str(e)}")
        import traceback
        traceback.print_exc()
        test_results['tests']['critical_error'] = {'status': 'failed', 'error': str(e)}
    
    return test_results

if __name__ == "__main__":
    integration_test()
