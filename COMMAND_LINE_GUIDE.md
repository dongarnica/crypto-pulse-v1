# 🚀 Crypto Trading Bot - Complete Command Line and Dashboard Guide

This comprehensive guide covers all available commands, modes, and features of the AI-assisted crypto trading bot system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Line Interface](#command-line-interface)
3. [Interactive Dashboard](#interactive-dashboard)
4. [Available Modes](#available-modes)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Quick Start

### Option 1: Interactive Launcher (Recommended for Beginners)

```bash
python launch.py
```

This launches an interactive menu with all available options:

```
📋 AVAILABLE MODES:

1. 🖥️  Interactive Dashboard     - Real-time portfolio monitoring
2. 🤖 Automated Trading         - LSTM-based trading (24h)
3. 📊 Performance Analysis      - Generate performance reports
4. 🔙 Strategy Backtesting      - Test strategies on historical data
5. 📈 Data Collection           - Collect market data
6. ⚙️  System Setup             - Initialize and configure system
7. 🧪 Run Tests                 - Execute test suite
8. 📖 View Documentation        - Show help and examples
9. 🔍 Check System Status       - Validate configuration
0. ❌ Exit
```

### Option 2: Direct Command Line

```bash
# System setup (run this first)
python main_app.py --mode setup

# Interactive dashboard
python main_app.py --mode dashboard

# Automated trading
python main_app.py --mode lstm-trading --duration 24
```

---

## Command Line Interface

The main application is accessed through `main_app.py` with various command-line arguments.

### Basic Syntax

```bash
python main_app.py [--mode MODE] [OPTIONS]
```

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | choice | `dashboard` | Execution mode (see [Available Modes](#available-modes)) |
| `--ticker` | string | `BTC` | Crypto ticker symbol |
| `--duration` | int | `24` | Duration in hours for trading/data modes |
| `--debug` | flag | False | Enable debug logging |

### Multi-Ticker Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--multi-ticker` | flag | False | Enable multi-ticker trading mode |
| `--max-tickers` | int | `3` | Maximum number of active tickers |
| `--ticker-allocation` | float | `0.33` | Portfolio allocation per ticker |

### LSTM Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-epochs` | int | `25` | Maximum training epochs for LSTM model |
| `--training-timeout` | int | `300` | Maximum training time in seconds |

### Analysis & Backtest Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--period` | int | `30` | Analysis period in days |
| `--hours` | int | `720` | Backtest hours |
| `--balance` | float | `100000` | Initial balance for backtest |

---

## Interactive Dashboard

The interactive dashboard provides real-time portfolio monitoring and updates every 30 seconds.

### Features

- **Portfolio Overview**: Total value, unrealized P&L, positions count
- **Risk Distribution**: Real-time risk analysis across positions
- **Individual Positions**: Detailed position breakdown
- **Recommendations**: Trading recommendations summary
- **Performance Metrics**: Live performance calculations

### Dashboard Layout

```
================================================================================
🚀 CRYPTO TRADING BOT - LIVE DASHBOARD
================================================================================
📅 2025-05-26 14:30:15
📊 Multi-Ticker Mode: 5 active tickers
🎯 Active: BTC/USD, ETH/USD, ADA/USD, DOT/USD, LINK/USD

💰 PORTFOLIO OVERVIEW
------------------------------
Total Value: $45,678.90
Unrealized P&L: $2,345.67
Total Positions: 8
Average Return: +5.23%

🚨 RISK DISTRIBUTION
------------------------------
Low Risk: 62.5%
Medium Risk: 25.0%
High Risk: 12.5%
Critical Risk: 0.0%

📈 INDIVIDUAL POSITIONS
------------------------------
BTC/USD: $15,234.56 | P&L: $1,234.56 (+8.83%) | Risk: low
ETH/USD: $12,456.78 | P&L: $567.89 (+4.77%) | Risk: low
...

⚡ RECOMMENDATIONS: 3 total
   Urgent: 0
   High: 1
   Medium: 2
   Low: 0

================================================================================
Press Ctrl+C to exit dashboard mode
```

### Navigation

- **Auto-refresh**: Dashboard updates every 30 seconds
- **Exit**: Press `Ctrl+C` to exit dashboard mode
- **Clear screen**: Dashboard clears screen on each update for clean display

---

## Available Modes

### 1. Setup Mode (`setup`)

Initialize and validate the system configuration.

```bash
python main_app.py --mode setup
```

**What it does:**
- Validates environment variables and API keys
- Tests connection to trading platforms
- Initializes required directories
- Checks system dependencies
- Validates model configurations

**Example output:**
```
⚙️  System Setup & Validation
✅ Environment variables loaded
✅ Alpaca connection successful
✅ Data client initialized
✅ Model directories created
✅ Logging system configured
```

### 2. Dashboard Mode (`dashboard`)

Real-time portfolio monitoring interface.

```bash
python main_app.py --mode dashboard
```

**Features:**
- Live portfolio updates every 30 seconds
- Risk analysis and position breakdown
- Performance metrics display
- Trading recommendations summary

### 3. LSTM Trading Mode (`lstm-trading`)

Advanced automated trading using LSTM models for all tickers.

```bash
# Basic LSTM trading
python main_app.py --mode lstm-trading --duration 24

# With custom training parameters
python main_app.py --mode lstm-trading --duration 48 --max-epochs 50 --training-timeout 600

# Multi-ticker LSTM trading
python main_app.py --mode lstm-trading --multi-ticker --max-tickers 5 --duration 24
```

**Features:**
- LSTM model training and signal generation
- Multi-ticker support with optimized data fetching
- Signal generator caching for performance
- Risk-adjusted position sizing
- Real-time portfolio monitoring

**Process Flow:**
1. Load tickers from `tickers.txt`
2. Initialize/train LSTM models for each ticker
3. Generate trading signals every 15 minutes
4. Execute trades based on confidence and risk assessment
5. Monitor portfolio performance

### 4. Legacy Trading Mode (`trading`)

Basic automated trading mode.

```bash
python main_app.py --mode trading --duration 24
```

### 5. Analysis Mode (`analysis`)

Generate comprehensive performance reports and analytics.

```bash
# 30-day analysis
python main_app.py --mode analysis --period 30

# 90-day analysis
python main_app.py --mode analysis --period 90
```

**Generates:**
- Performance metrics and returns analysis
- Risk assessment reports
- Trading recommendations
- Portfolio optimization suggestions
- Detailed JSON reports in `reports/` directory

### 6. Backtest Mode (`backtest`)

Historical strategy validation and testing.

```bash
# Basic backtest (720 hours = 30 days)
python main_app.py --mode backtest --hours 720

# Extended backtest with custom balance
python main_app.py --mode backtest --hours 2160 --balance 50000
```

**Metrics calculated:**
- Total return and profit/loss
- Win rate and trade statistics
- Maximum drawdown
- Sharpe ratio
- Risk-adjusted returns

### 7. Data Collection Mode (`data`)

Collect and store market data for analysis and training.

```bash
# 1-hour data collection
python main_app.py --mode data --duration 1

# Extended data collection
python main_app.py --mode data --duration 24
```

---

## Configuration

### Environment Setup

1. **Create `.env` file** with your API keys:

```bash
# Alpaca Trading API (Required)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# OpenAI API (Optional, for LLM enhancements)
OPENAI_API_KEY=your_openai_api_key

# Perplexity API (Optional, for market analysis)
PERPLEXITY_API_KEY=your_perplexity_api_key
```

2. **Configure tickers** in `tickers.txt`:

```
BTC/USD
ETH/USD
ADA/USD
DOT/USD
LINK/USD
LTC/USD
XRP/USD
SUSHI/USD
AAVE/USD
DOGE/USD
SOL/USD
MATIC/USD
AVAX/USD
```

### Validation

Test your configuration:

```bash
# Quick validation
python main_app.py --mode setup

# Comprehensive testing
python example_config_usage.py

# Alpaca-specific testing
python test_alpaca_positions.py
```

---

## Advanced Usage

### Multi-Ticker Trading

Enable trading across multiple cryptocurrency pairs simultaneously:

```bash
python main_app.py --mode lstm-trading \
  --multi-ticker \
  --max-tickers 5 \
  --ticker-allocation 0.2 \
  --duration 24
```

**Parameters:**
- `--multi-ticker`: Enable multi-ticker mode
- `--max-tickers`: Maximum number of active tickers (default: 3)
- `--ticker-allocation`: Portfolio allocation per ticker (default: 0.33)

### Custom Training Parameters

Fine-tune LSTM model training:

```bash
python main_app.py --mode lstm-trading \
  --max-epochs 100 \
  --training-timeout 900 \
  --duration 48
```

**Parameters:**
- `--max-epochs`: Maximum training epochs (default: 25)
- `--training-timeout`: Training timeout in seconds (default: 300)

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python main_app.py --mode dashboard --debug
```

### Extended Analysis

Perform comprehensive market analysis:

```bash
# Extended analysis with custom parameters
python main_app.py --mode analysis --period 90 --debug

# Backtest with extended timeframe
python main_app.py --mode backtest --hours 4320 --balance 100000 --debug
```

---

## Troubleshooting

### Common Issues

1. **"No API key found" Error**
   ```bash
   # Check environment variables
   python -c "import os; print('ALPACA_API_KEY' in os.environ)"
   
   # Validate configuration
   python main_app.py --mode setup
   ```

2. **Connection Errors**
   ```bash
   # Test Alpaca connection
   python test_alpaca_positions.py
   
   # Check system status
   python launch.py  # Choose option 9
   ```

3. **Model Training Issues**
   ```bash
   # Clear model cache and retrain
   rm -rf models/saved/*
   python main_app.py --mode lstm-trading --max-epochs 50
   ```

4. **No Tickers Found**
   ```bash
   # Check tickers.txt file exists and has content
   cat tickers.txt
   
   # Verify ticker format (should be "BTC/USD" not "BTCUSD")
   ```

### Log Files

Check log files for detailed error information:

```bash
# View latest log
tail -f logs/trading.log

# Check specific date
cat logs/trading_20250526.log
```

### Test Scripts

Run diagnostic tests:

```bash
# Comprehensive testing
python test_alpaca_comprehensive.py

# LSTM model testing
python test_lstm_model.py

# Configuration testing
python test_config_usage.py
```

---

## Examples

### Example 1: Basic Setup and Dashboard

```bash
# Step 1: System setup
python main_app.py --mode setup

# Step 2: Start dashboard
python main_app.py --mode dashboard
```

### Example 2: Automated Multi-Ticker Trading

```bash
# 24-hour automated trading across 5 tickers
python main_app.py --mode lstm-trading \
  --multi-ticker \
  --max-tickers 5 \
  --ticker-allocation 0.2 \
  --duration 24 \
  --max-epochs 30
```

### Example 3: Performance Analysis Workflow

```bash
# Step 1: Collect data
python main_app.py --mode data --duration 24

# Step 2: Run analysis
python main_app.py --mode analysis --period 30

# Step 3: Backtest strategy
python main_app.py --mode backtest --hours 720 --balance 50000
```

### Example 4: Development and Testing

```bash
# Step 1: Setup with debug logging
python main_app.py --mode setup --debug

# Step 2: Test single ticker
python main_app.py --mode lstm-trading --ticker BTC --duration 1 --debug

# Step 3: Validate with tests
python launch.py  # Choose option 7
```

### Example 5: Production Trading Setup

```bash
# Step 1: Validate configuration
python main_app.py --mode setup

# Step 2: Test with small duration first
python main_app.py --mode lstm-trading --duration 1

# Step 3: Run full production trading
python main_app.py --mode lstm-trading \
  --multi-ticker \
  --max-tickers 8 \
  --duration 168  # 1 week
```

---

## Command Reference Quick Card

### Essential Commands

```bash
# Setup
python main_app.py --mode setup

# Dashboard
python main_app.py --mode dashboard

# Trading (single ticker)
python main_app.py --mode lstm-trading --ticker BTC --duration 24

# Trading (multi-ticker)
python main_app.py --mode lstm-trading --multi-ticker --duration 24

# Analysis
python main_app.py --mode analysis --period 30

# Backtest
python main_app.py --mode backtest --hours 720

# Data collection
python main_app.py --mode data --duration 1
```

### Useful Test Commands

```bash
# Test configuration
python example_config_usage.py

# Test Alpaca connection
python test_alpaca_positions.py

# Interactive launcher
python launch.py

# Comprehensive testing
python test_alpaca_comprehensive.py
```

### File Locations

| Purpose | Location |
|---------|----------|
| Main application | `main_app.py` |
| Interactive launcher | `launch.py` |
| Configuration | `.env`, `config/config.py` |
| Tickers list | `tickers.txt` |
| Logs | `logs/` |
| Reports | `reports/` |
| Models | `models/saved/` |
| Market data | `data/market_data/` |

---

This guide covers all major features and use cases of the crypto trading bot. For additional help, use the interactive launcher (`python launch.py`) and choose option 8 for documentation, or option 9 for system status checks.
