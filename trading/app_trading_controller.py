"""
Example usage of the enhanced trading system with position management and recommendations.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig
from trading.trading_controller import TradingController


def format_currency(amount):
    """Format currency values."""
    return f"${amount:,.2f}"


def format_percentage(value):
    """Format percentage values."""
    return f"{value:+.2f}%"


def print_section_header(title, char="=", width=60):
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def main():
    """Demonstrate the enhanced trading system."""
    
    print("🚀 Enhanced Trading System Demo")
    print("=" * 50)
    
    try:
        # Initialize trading controller
        print("Initializing trading controller...")
        config = AppConfig()
        controller = TradingController(config)
        
        # Test 1: Portfolio Dashboard
        print_section_header("📊 PORTFOLIO DASHBOARD")
        
        dashboard = controller.get_portfolio_dashboard()
        
        if 'error' in dashboard:
            print(f"❌ Dashboard Error: {dashboard['error']}")
        else:
            portfolio = dashboard['portfolio_overview']
            print(f"Portfolio Value: {format_currency(portfolio.get('total_market_value', 0))}")
            print(f"Unrealized P&L: {format_currency(portfolio.get('total_unrealized_pnl', 0))}")
            print(f"Total Positions: {portfolio.get('total_positions', 0)}")
            print(f"Average Return: {format_percentage(portfolio.get('average_return_pct', 0))}")
            
            # Risk distribution
            risk_analysis = dashboard['risk_analysis']
            print(f"\n🚨 Risk Distribution:")
            print(f"  Low Risk: {risk_analysis['low_risk_pct']:.1f}%")
            print(f"  Medium Risk: {risk_analysis['medium_risk_pct']:.1f}%")
            print(f"  High Risk: {risk_analysis['high_risk_pct']:.1f}%")
            print(f"  Critical Risk: {risk_analysis['critical_risk_pct']:.1f}%")
            
            # Position details
            if dashboard['position_metrics']:
                print(f"\n📈 Individual Positions:")
                for position in dashboard['position_metrics']:
                    symbol = position.get('symbol', 'Unknown')
                    size = position.get('size', 0)
                    market_value = position.get('market_value', 0)
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    return_pct = position.get('return_pct', 0)
                    
                    print(f"  {symbol}: {size} units | "
                          f"Value: {format_currency(market_value)} | "
                          f"P&L: {format_currency(unrealized_pnl)} "
                          f"({format_percentage(return_pct)})")

    except Exception as e:
        print(f"❌ Error in portfolio dashboard: {str(e)}")

if __name__ == "__main__":
    main()