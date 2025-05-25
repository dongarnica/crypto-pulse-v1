# 🚀 Crypto Trading Bot - Quick Start Guide

## Overview

This is a comprehensive crypto trading bot with LSTM-based signal generation, automated trading, portfolio analysis, and risk management features.

## Prerequisites

1. **Python 3.8+** with required dependencies
2. **Alpaca API Keys** (paper or live trading)
3. **OpenAI API Key** (optional, for LLM features)
4. **Environment Variables** configured

## Environment Setup

Create a `.env` file in the root directory:

```bash
# Alpaca Trading API
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# OpenAI for LLM features (optional)
OPENAI_API_KEY=your_openai_api_key

# Trading Parameters
POSITION_SIZE=0.02
STOP_LOSS_MULT=1.5
TAKE_PROFIT_MULT=3.0
MAX_POSITIONS=5

# LSTM Model Parameters
LSTM_EPOCHS=50
LSTM_LOOKBACK=60
```

## Quick Start

### Option 1: Interactive Launcher (Recommended)

```bash
python launch.py
```

This will show you a menu with all available options.

### Option 2: Direct Command Line

```bash
# System setup and validation
python main_app.py --mode setup

# Interactive dashboard
python main_app.py --mode dashboard

# Automated trading for 24 hours
python main_app.py --mode trading --duration 24

# Performance analysis
python main_app.py --mode analysis --period 30

# Strategy backtesting
python main_app.py --mode backtest --hours 720

# Data collection
python main_app.py --mode data --duration 1
```

## Available Modes

| Mode | Description | Example |
|------|-------------|---------|
| `setup` | System initialization and validation | `--mode setup` |
| `dashboard` | Real-time portfolio monitoring | `--mode dashboard` |
| `trading` | Automated LSTM-based trading | `--mode trading --duration 24` |
| `analysis` | Performance analytics and reports | `--mode analysis --period 30` |
| `backtest` | Historical strategy validation | `--mode backtest --hours 720` |
| `data` | Market data collection | `--mode data --duration 1` |

## Key Features

### 🤖 LSTM Signal Generation
- Advanced neural network model for price prediction
- Technical indicators and momentum analysis
- Aggressive trading optimization for short-term profits

### 📊 Portfolio Management
- Real-time position tracking
- Risk assessment and management
- Performance analytics and reporting

### 🔄 Automated Trading
- Signal-based order execution
- Stop-loss and take-profit management
- Position sizing based on risk parameters

### 📈 Analysis & Reporting
- Comprehensive performance metrics
- Trading recommendations
- Historical backtesting

### 🧠 AI Integration
- OpenAI-powered market analysis
- Enhanced signal validation
- Intelligent risk assessment

## Configuration Options

### Command Line Arguments

```bash
# Ticker symbol (default: BTC)
--ticker ETH

# Duration for trading/data modes (hours)
--duration 12

# Analysis period (days)
--period 7

# Backtest hours
--hours 168

# Initial balance for backtesting
--balance 50000

# Enable debug logging
--debug
```

### Examples

```bash
# Trade Ethereum for 12 hours with debug logging
python main_app.py --mode trading --ticker ETH --duration 12 --debug

# Backtest Dogecoin strategy for 1 week
python main_app.py --mode backtest --ticker DOGE --hours 168

# Analyze Bitcoin performance for last 7 days
python main_app.py --mode analysis --ticker BTC --period 7
```

## File Structure

```
📁 /workspaces/crypto-refactor/
├── 🚀 main_app.py              # Main application entry point
├── 🎮 launch.py                # Interactive launcher
├── ⚙️ config/                  # Configuration management
├── 📈 data/                    # Market data and storage
├── 🧠 models/                  # LSTM models and backtesting
├── 💰 trading/                 # Trading system components
├── 🔗 exchanges/               # Exchange API clients
├── 🤖 llm/                     # LLM integration
├── 📊 logs/                    # Application logs
├── 📋 reports/                 # Generated reports
└── 🧪 tests/                   # Test suite
```

## Safety Features

- **Paper Trading**: Default configuration uses Alpaca paper trading
- **Risk Limits**: Configurable position sizing and stop-losses
- **Data Validation**: Comprehensive input validation and error handling
- **Graceful Shutdown**: Proper signal handling for clean exits

## Troubleshooting

### Common Issues

1. **API Connection Errors**: Check your API keys and network connection
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Data Fetching Issues**: Verify your Alpaca account and market hours
4. **Model Training**: Initial LSTM training may take several minutes

### Logs

Check the logs directory for detailed error information:
- `logs/trading_YYYYMMDD.log` - Daily log files
- `logs/trading.log` - General log file

### Getting Help

1. Run system validation: `python main_app.py --mode setup`
2. Check logs for specific error messages
3. Verify environment variables are set correctly
4. Ensure all required files are present

## Next Steps

1. **Start with Setup**: Run `python main_app.py --mode setup` first
2. **Paper Trading**: Begin with dashboard mode to familiarize yourself
3. **Backtesting**: Test strategies before live trading
4. **Gradual Scaling**: Start with small position sizes

---

⚠️ **Disclaimer**: This is educational software. Cryptocurrency trading involves significant risk. Always test thoroughly with paper trading before using real funds.

🎯 **Happy Trading!** 🚀
