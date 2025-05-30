# Crypto Refactor

## Configuration Setup

This project uses a centralized configuration system that manages API keys, trading parameters, and model settings. Before running any scripts, you need to set up your configuration.

### 1. Environment Variables Setup

Copy the template file and add your API keys:

```bash
# Copy the template
cp .env.template .env

# Edit the .env file with your actual API keys
nano .env  # or use your preferred editor
```

### 2. Required API Keys

- **Alpaca Trading API**: Get from [Alpaca Markets](https://alpaca.markets/)
  - `ALPACA_API_KEY`
  - `ALPACA_SECRET_KEY`
  
- **OpenAI API**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
  - `OPENAI_API_KEY`
  
- **Perplexity API** (optional): Get from [Perplexity AI](https://www.perplexity.ai/settings/api)
  - `PERPLEXITY_API_KEY`

### 3. Test Configuration

Run the configuration example to verify your setup:

```bash
python3 example_config_usage.py
```

This script will show you which APIs are configured and test basic connectivity.

**Alternative verification** (writes results to `verification_log.txt`):
```bash
python3 verify_implementation.py
```

---

## Quick Start Commands

### LSTM Signal Generation
```bash
# Basic signal generation for BTC/USD
# Trains an LSTM model and generates trading signals using 240 hours (10 days) of historical data.
python3 models/lstm_train.py --ticker BTC/USD --hours 240

# Advanced configuration with custom parameters
# Uses 720 hours (30 days) of data for ETH/USD, sets LSTM lookback window to 168 time steps (1 week if hourly), and retrains if the model is older than 12 hours.
python3 models/lstm_train.py --ticker ETH/USD --hours 720 --lookback 168 --retrain-threshold 12

# Generate signals for other cryptocurrencies
# Example: Generates signals for AAVE/USD using 480 hours (20 days) of data and a lookback window of 60 time steps.
python3 models/lstm_train.py --ticker AAVE/USD --hours 480 --lookback 60
```

### Backtesting
```bash
# Basic backtest for BTC/USD (30 days)
python3 models/app_backtester.py --ticker BTC/USD --hours 720

# Custom backtest with different parameters
python3 models/app_backtester.py --ticker ETH/USD --hours 1440 --initial-balance 50000 --fee 0.0015

# Long-term backtest (90 days) with higher initial balance
python3 models/app_backtester.py --ticker BTC/USD --hours 2160 --initial-balance 100000
```

### Alpaca Trading & Positions Testing
```bash
# Complete integration test with comprehensive analysis
python3 integration_test.py

# Test Alpaca client configuration and connectivity
python3 exchanges/test_alpaca_client.py

# COMPREHENSIVE positions testing with full portfolio analysis (RECOMMENDED)
python3 comprehensive_alpaca_test.py

# Comprehensive positions analysis and portfolio summary
python3 test_alpaca_positions.py

# Simple positions test with basic error handling
python3 simple_positions_test.py

# Test configuration system with all components
python3 example_config_usage.py

# Verification test (writes to verification_log.txt)
python3 verify_implementation.py
```

### Data Testing
```bash
# Test crypto data client
python3 models/data/app.py

# Test historical data formatting
python3 test_historical_data.py

# Test Mountain Time conversion
python3 test_mountain_time.py
```

---

## LSTM Functionality (`SignalGenerator`)

### Core Features

- **Deep Neural Network:** 3-layer LSTM architecture (256 → 128 → 64 units)
- **Advanced Feature Engineering:** 14 technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, etc.)
- **Smart Training:** Trains new models only; skips retraining if a model exists
- **Multi-Criteria Signals:** 10+ conditions for generating long/short signals
- **Dynamic Thresholds:** Volatility-adjusted decision boundaries

### Technical Highlights

- Batch normalization and dropout for model stability
- L2 regularization and gradient clipping
- Huber loss function for robustness
- Early stopping and learning rate scheduling
- Automatic model saving (persistence)

---

## Backtesting Framework (`Backtester`)

### Simulation Engine

- Realistic trading simulation with transaction costs
- Position management: Long, Short, Flat
- Trade-by-trade logging and analysis
- Performance metrics: returns, Sharpe ratio, win rate

### Analysis Features

- Debug statistics and signal counting
- Feature correlation analysis
- Real-time progress tracking
- Comprehensive trade logging

---

## Key Innovations

- **Intelligent Training:** Avoids unnecessary retraining by checking for existing models
- **Multi-Modal Signals:** Combines price prediction with technical analysis
- **Risk-Aware Thresholds:** Dynamic, volatility-based decision making
- **Production Ready:** Robust error handling, logging, and persistence

> The system is suitable for both research and production, featuring CLI applications for easy testing and comprehensive documentation to explain the underlying mechanics.

---

## Command Line Parameters

### LSTM Signal Generator (`lstm_train.py`)
- `--ticker`: Trading pair (default: BTC/USD) - Supports any crypto pair from tickers.txt
- `--hours`: Number of hours of historical data (default: 240) - More data = better training
- `--model`: Custom model name (optional) - Allows model versioning
- `--lookback`: LSTM sequence length (default: 60) - How many time steps the model considers
- `--retrain-threshold`: Hours before retraining (default: 24) - Model refresh frequency

### Backtester (`app_backtester.py`)
- `--ticker`: Trading pair to backtest
- `--hours`: Backtest period in hours (default: 720 = 30 days)
- `--initial-balance`: Starting capital (default: $100,000)
- `--fee`: Trading fee per trade (default: 0.001 = 0.1%)
- `--lookback`: LSTM lookback period (default: 60)

### Trading Scripts (`run_trading.py`, `run_multi_ticker.py`)
- `--ticker`: Cryptocurrency symbol (e.g., BTC/USD, ETH/USD)
- `--once`: Execute once and exit (vs continuous monitoring)
- `--demo`: Demo mode (paper trading, no real trades)
- `--mode`: Execution mode for multi-ticker (sequential/parallel/demo/once)
- `--max-workers`: Number of parallel workers (default: 4)

---

## Available Tickers

The system supports all major cryptocurrencies listed in `tickers.txt`:
```
BTC/USD, ETH/USD, ADA/USD, DOT/USD, LINK/USD, MATIC/USD, 
UNI/USD, AAVE/USD, ATOM/USD, XTZ/USD, ALGO/USD, SOL/USD, AVAX/USD
```

---

## Model Files

Trained models are saved in `models/data/` with naming convention:
- `btc_usd_lstm_v2.keras`
- `eth_usd_lstm_v2.keras` 
- `aave_usd_lstm_v2.keras`

Models are automatically loaded if they exist, skipping retraining for faster execution.
