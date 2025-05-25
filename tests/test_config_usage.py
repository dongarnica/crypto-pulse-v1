#!/usr/bin/env python3
"""
Example script demonstrating how to use the AppConfig system with all components.
This script shows how to properly initialize and use the configuration across
different modules including Alpaca trading, LLM clients, and LSTM models.
"""

import os
import sys

# Add parent directory to path so we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig
from exchanges.alpaca_client import AlpacaCryptoTrading
from llm.llm_client import LLMClient
from data.crypto_data_client import CryptoMarketDataClient

def main():
    """Demonstrate config usage across all modules."""
    
    print("=== AppConfig Usage Example ===\n")
    
    # 1. Initialize configuration
    print("1. Initializing configuration...")
    config = AppConfig(ticker="BTC")  # Can specify ticker
    
    # 2. Display configuration status
    print(f"   Ticker: {config.ticker_short}")
    print(f"   Model directory: {config.model_dir}")
    print(f"   Model name: {config.model_name}")
    
    # Check which API keys are configured
    alpaca_config = config.get_alpaca_config()
    llm_config = config.get_llm_config()
    
    print(f"   Alpaca API configured: {'✅' if alpaca_config['api_key'] else '❌'}")
    print(f"   OpenAI API configured: {'✅' if llm_config['api_key'] else '❌'}")
    print(f"   Perplexity API configured: {'✅' if llm_config['perplexity_key'] else '❌'}")
    
    # 3. Alpaca Trading Client Example
    print("\n2. Alpaca Trading Client:")
    try:
        if alpaca_config['api_key'] and alpaca_config['api_secret']:
            alpaca_client = AlpacaCryptoTrading(
                api_key=alpaca_config['api_key'],
                api_secret=alpaca_config['api_secret'],
                base_url=alpaca_config['base_url']
            )
            print("   ✅ Alpaca client initialized successfully")
            
            # Example: Get account info (if credentials are valid)
            try:
                account = alpaca_client.get_account()
                print(f"   Account buying power: ${account.get('buying_power', 'N/A')}")
                print(f"   Account cash: ${account.get('cash', 'N/A')}")
                print(f"   Portfolio value: ${account.get('portfolio_value', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️  Could not fetch account info: {str(e)}")
            
            # Example: Get current positions
            try:
                positions = alpaca_client.list_positions()
                print(f"   Current positions: {len(positions) if positions else 0}")
                
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
                else:
                    print("   No open positions")
            except Exception as e:
                print(f"   ⚠️  Could not fetch positions: {str(e)}")
                
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
                    
                    if portfolio['performance']['best_performer']:
                        best = portfolio['performance']['best_performer']
                        print(f"   Best performer: {best['symbol']} ({best['return_pct']:+.2f}%)")
                else:
                    print(f"   ⚠️  Portfolio analysis failed: {portfolio['error']}")
            except Exception as e:
                print(f"   ⚠️  Portfolio analysis error: {str(e)}")
        else:
            print("   ❌ Alpaca credentials not configured")
    except Exception as e:
        print(f"   ❌ Error initializing Alpaca client: {str(e)}")
    
    # 4. LLM Client Example
    print("\n3. LLM Client:")
    try:
        llm_client = LLMClient(config=config)
        print("   ✅ LLM client initialized successfully")
        
        # Example query (only if API key is available)
        if llm_config['api_key']:
            try:
                response = llm_client.query(
                    "What is Bitcoin?", 
                    provider="openai", 
                    model="gpt-3.5-turbo"  # Use cheaper model for testing
                )
                print(f"   Sample response length: {len(response)} characters")
            except Exception as e:
                print(f"   ⚠️  LLM query failed: {str(e)}")
        else:
            print("   ⚠️  No OpenAI API key configured, skipping query test")
    except Exception as e:
        print(f"   ❌ Error initializing LLM client: {str(e)}")
    
    # 5. Crypto Data Client Example
    print("\n4. Crypto Data Client:")
    try:
        data_client = CryptoMarketDataClient()
        print("   ✅ Crypto data client initialized successfully")
        
        # Example: Get current price
        try:
            btc_data = data_client.get_realtime_price("BTC/USD")
            if btc_data:
                current_price = btc_data['price']
                change_24h = btc_data.get('24h_change', 0)
                print(f"   Current Bitcoin price: ${current_price:,.2f}")
                print(f"   24h change: {change_24h:+.2f}%")
            else:
                print("   ⚠️  No price data returned")
        except Exception as e:
            print(f"   ⚠️  Could not fetch price data: {str(e)}")
    except Exception as e:
        print(f"   ❌ Error initializing data client: {str(e)}")
    
    # 6. Configuration Access Examples
    print("\n5. Configuration Access Methods:")
    
    # Dictionary-style access
    print("   Dictionary-style access:")
    try:
        print(f"   config['ALPACA_API_KEY']: {'***configured***' if config['ALPACA_API_KEY'] else 'Not set'}")
        print(f"   config['TICKER_SHORT']: {config['TICKER_SHORT']}")
    except KeyError as e:
        print(f"   KeyError: {e}")
    
    # Direct attribute access
    print("   Direct attribute access:")
    print(f"   config.ticker_short: {config.ticker_short}")
    print(f"   config.model_name: {config.model_name}")
    
    # Specialized getters
    print("   Specialized getter methods:")
    trading_config = config.get_trading_config()
    print(f"   Trading config keys: {list(trading_config.keys())}")
    
    # 7. Environment Variables Check
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
    
    print("\n=== Configuration Complete ===")
    print("\nTo configure missing API keys:")
    print("1. Create a .env file in the project root")
    print("2. Add your API keys like:")
    print("   ALPACA_API_KEY=your_key_here")
    print("   ALPACA_SECRET_KEY=your_secret_here")
    print("   OPENAI_API_KEY=your_openai_key_here")
    print("   PERPLEXITY_API_KEY=your_perplexity_key_here")

if __name__ == "__main__":
    main()
