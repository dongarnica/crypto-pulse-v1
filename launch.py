#!/usr/bin/env python3
"""
Quick Launch Script for Crypto Trading Bot
==========================================

This script provides quick access to common trading bot operations
with simplified command-line interface.
"""

import os
import sys
import subprocess
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print application banner."""
    print("=" * 80)
    print("🚀 CRYPTO TRADING BOT - QUICK LAUNCHER")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def show_menu():
    """Show main menu options."""
    print("📋 AVAILABLE MODES:")
    print()
    print("1. 🖥️  Interactive Dashboard     - Real-time portfolio monitoring")
    print("2. 🤖 Automated Trading         - LSTM-based trading (24h)")
    print("3. 📊 Performance Analysis      - Generate performance reports")
    print("4. 🔙 Strategy Backtesting      - Test strategies on historical data")
    print("5. 📈 Data Collection           - Collect market data")
    print("6. ⚙️  System Setup             - Initialize and configure system")
    print("7. 🧪 Run Tests                 - Execute test suite")
    print("8. 📖 View Documentation        - Show help and examples")
    print("9. 🔍 Check System Status       - Validate configuration")
    print("0. ❌ Exit")
    print()

def run_main_app(args):
    """Run the main application with arguments."""
    cmd = [sys.executable, "main_app.py"] + args
    return subprocess.run(cmd, cwd=os.path.dirname(__file__))

def run_tests():
    """Run the test suite."""
    print("🧪 Running comprehensive test suite...")
    cmd = [sys.executable, "tests/test_runner_enhanced.py"]
    return subprocess.run(cmd, cwd=os.path.dirname(__file__))

def show_documentation():
    """Show documentation and examples."""
    print("📖 CRYPTO TRADING BOT DOCUMENTATION")
    print("=" * 50)
    print()
    print("🎯 QUICK START:")
    print("  1. Set up environment variables in .env file:")
    print("     - ALPACA_API_KEY=your_key")
    print("     - ALPACA_SECRET_KEY=your_secret")
    print("     - OPENAI_API_KEY=your_key (optional)")
    print()
    print("  2. Run setup mode first: python launch.py -> option 6")
    print("  3. Start with dashboard mode: python launch.py -> option 1")
    print()
    print("📊 MODES EXPLAINED:")
    print()
    print("  🖥️  Dashboard Mode:")
    print("     - Real-time portfolio monitoring")
    print("     - Risk analysis and position tracking")
    print("     - Live recommendations")
    print("     - Updates every 30 seconds")
    print()
    print("  🤖 Automated Trading Mode:")
    print("     - LSTM-based signal generation")
    print("     - Automated trade execution")
    print("     - Risk management integration")
    print("     - LLM-enhanced decision making")
    print()
    print("  📊 Analysis Mode:")
    print("     - Comprehensive performance metrics")
    print("     - Risk assessment reports")
    print("     - Trading recommendations")
    print("     - Historical performance analysis")
    print()
    print("  🔙 Backtest Mode:")
    print("     - Historical strategy validation")
    print("     - Performance simulation")
    print("     - Risk/return analysis")
    print("     - Strategy optimization")
    print()
    print("🛠️  ADVANCED USAGE:")
    print("  python main_app.py --help                    # Full help")
    print("  python main_app.py --mode dashboard          # Dashboard")
    print("  python main_app.py --mode trading --duration 12  # 12h trading")
    print("  python main_app.py --mode analysis --period 7    # 7-day analysis")
    print("  python main_app.py --mode backtest --hours 168   # 1-week backtest")
    print()
    print("📁 IMPORTANT FILES:")
    print("  main_app.py              - Main application entry point")
    print("  config/config.py         - Configuration management")
    print("  trading/                 - Trading system components")
    print("  models/                  - LSTM models and backtesting")
    print("  data/                    - Market data and storage")
    print("  logs/                    - Application logs")
    print("  reports/                 - Generated reports")
    print()
    input("Press Enter to continue...")

def check_system_status():
    """Check system status and configuration."""
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 30)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check if main files exist
    files_to_check = [
        "main_app.py",
        "config/config.py",
        "trading/trading_controller.py",
        "data/crypto_data_client.py",
        "models/lstm_model_v2.py",
        "exchanges/alpaca_client.py"
    ]
    
    print("\n📁 FILE CHECK:")
    all_files_exist = True
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {file_path}")
        if not exists:
            all_files_exist = False
    
    # Check environment variables
    print("\n🔑 ENVIRONMENT VARIABLES:")
    env_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY", 
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "set"
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ⚠️  {var}: not set")
    
    # Check directories
    print("\n📂 DIRECTORIES:")
    dirs_to_check = ["logs", "data", "reports", "models/saved"]
    for dir_path in dirs_to_check:
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path}")
    
    # Try importing key modules
    print("\n🐍 MODULE IMPORTS:")
    modules_to_test = [
        "config.config",
        "trading.trading_controller", 
        "data.crypto_data_client",
        "exchanges.alpaca_client"
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
    
    print(f"\n📊 OVERALL STATUS: {'✅ System Ready' if all_files_exist else '⚠️  Issues Found'}")
    input("\nPress Enter to continue...")

def get_user_choice():
    """Get user menu choice."""
    while True:
        try:
            choice = input("👉 Select option (0-9): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return choice
            else:
                print("❌ Invalid choice. Please select 0-9.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return '0'

def main():
    """Main launcher function."""
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_banner()
        show_menu()
        
        choice = get_user_choice()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
            
        elif choice == '1':
            print("🖥️  Starting Interactive Dashboard...")
            run_main_app(["--mode", "dashboard"])
            
        elif choice == '2':
            print("🤖 Starting Automated Trading (24 hours)...")
            run_main_app(["--mode", "trading", "--duration", "24"])
            
        elif choice == '3':
            print("📊 Running Performance Analysis (30 days)...")
            run_main_app(["--mode", "analysis", "--period", "30"])
            
        elif choice == '4':
            print("🔙 Running Strategy Backtest (720 hours)...")
            run_main_app(["--mode", "backtest", "--hours", "720"])
            
        elif choice == '5':
            print("📈 Starting Data Collection (1 hour)...")
            run_main_app(["--mode", "data", "--duration", "1"])
            
        elif choice == '6':
            print("⚙️  Running System Setup...")
            run_main_app(["--mode", "setup"])
            
        elif choice == '7':
            run_tests()
            
        elif choice == '8':
            show_documentation()
            
        elif choice == '9':
            check_system_status()
        
        if choice != '0':
            input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
