# LSTM and Backtesting Functionality Documentation

## Overview

This cryptocurrency trading system implements a sophisticated LSTM (Long Short-Term Memory) neural network for price prediction and signal generation, combined with a comprehensive backtesting framework to evaluate strategy performance.

## LSTM Functionality (`lstm_model.py`)

### Core Architecture

The `SignalGenerator` class implements an advanced deep learning approach to cryptocurrency trading signal generation using a multi-layer LSTM neural network.

#### Key Components:

1. **Data Management**
   - Integrates with `CryptoMarketDataClient` to fetch real-time and historical data
   - Supports multiple trading pairs (e.g., BTC/USD, ETH/USD)
   - Configurable lookback periods (default: 60 time periods)

2. **Feature Engineering**
   The system calculates 14 sophisticated technical indicators:
   
   **Price-Based Indicators:**
   - RSI (Relative Strength Index) - momentum oscillator
   - Stochastic RSI - enhanced momentum indicator
   - MACD (Moving Average Convergence Divergence) - trend indicator
   - Bollinger Bands (Middle, Upper, Lower, Width) - volatility indicators
   - ATR (Average True Range) - volatility measure
   - ADX (Average Directional Index) - trend strength
   - EMA 20/50 (Exponential Moving Averages) - trend indicators

   **Volume-Based Indicators:**
   - OBV (On-Balance Volume) - volume momentum
   - VWAP (Volume Weighted Average Price) - price/volume relationship
   - Volume Delta and Volume Ratio - volume analysis

   **Advanced Features:**
   - Log Returns - normalized price changes
   - Realized Volatility - rolling volatility measure
   - Price ratios and trend cross signals

3. **Neural Network Architecture**
   
   **Model Structure:**
   ```
   Input Layer → BatchNormalization
   ↓
   LSTM Layer 1 (256 units) → Dropout (0.3) → BatchNorm
   ↓
   LSTM Layer 2 (128 units) → Dropout (0.3) → BatchNorm
   ↓
   LSTM Layer 3 (64 units) → Dropout (0.2) → BatchNorm
   ↓
   Dense Layer (64 units, Swish activation) → Dropout (0.2)
   ↓
   Dense Layer (32 units, Swish activation)
   ↓
   Output Layer (1 unit, Linear activation)
   ```

   **Advanced Features:**
   - L2 regularization on LSTM and Dense layers
   - Recurrent dropout for LSTM layers (0.1-0.2)
   - Batch normalization for training stability
   - Swish activation function for better gradient flow
   - Huber loss function for robust training

4. **Training Process**
   
   **Smart Training Logic:**
   - Only trains new models (skips training if model file exists)
   - Time-based retraining (configurable threshold, default: 24 hours)
   - Model persistence with automatic saving
   - Validation split (20%) for overfitting prevention

   **Training Configuration:**
   - Adam optimizer with learning rate 0.0005
   - Gradient clipping (value=0.5) for stability
   - Early stopping (patience=15) to prevent overfitting
   - Learning rate reduction on plateau
   - Model checkpointing for best weights

5. **Signal Generation**
   
   **Multi-Criteria Signal Logic:**
   
   **Long Signals Generated When:**
   - Predicted price > current price + dynamic threshold
   - RSI < 65 (not overbought)
   - MACD > signal line (bullish momentum)
   - OBV > 20-period average (volume confirmation)
   - Price > VWAP (institutional support)
   - Price > EMA 20 and EMA 20 > EMA 50 (uptrend)
   - ADX > 25 (strong trend)
   - Volatility not excessive
   - Volume above average

   **Short Signals Generated When:**
   - Predicted price < current price - dynamic threshold
   - RSI > 35 (not oversold)
   - MACD < signal line (bearish momentum)
   - OBV < 20-period average (volume confirmation)
   - Price < VWAP (institutional resistance)
   - Price < EMA 20 and EMA 20 < EMA 50 (downtrend)
   - ADX > 25 (strong trend)
   - Volatility not excessive
   - Volume below average

   **Dynamic Threshold Calculation:**
   ```
   Threshold = 0.5 × ATR + 0.3 × Realized_Volatility
   ```

6. **Output Format**
   
   The system provides comprehensive signal information:
   ```python
   {
       'timestamp': '2025-05-25T10:30:00',
       'ticker': 'BTC/USD',
       'signal': 1,  # 1=Long, -1=Short, 0=Hold
       'confidence': 0.85,
       'predicted_price': 67500.00,
       'current_price': 67000.00,
       'price_difference': 500.00,
       'threshold': 250.00,
       'indicators': {
           'rsi': 45.2,
           'macd_diff': 120.5,
           'atr': 1250.0,
           # ... all 14 indicators
       },
       'model_info': {
           'path': '/path/to/model.keras',
           'last_trained': '2025-05-25T08:00:00',
           'lookback_period': 60
       }
   }
   ```

## Backtesting Functionality (`lstm_backtest.py`)

### Core Framework

The `Backtester` class provides comprehensive strategy evaluation capabilities for the LSTM signal generator.

#### Key Features:

1. **Simulation Engine**
   - Realistic trading simulation with transaction costs
   - Position management (Long/Short/Flat)
   - Configurable trading fees (default: 0.1%)
   - Multiple position types and transitions

2. **Performance Metrics**
   
   **Financial Metrics:**
   - Total Return (%)
   - Final Balance
   - Number of Trades
   - Win Rate (%)
   - Sharpe Ratio
   - Individual trade P&L tracking

   **Risk Metrics:**
   - Volatility analysis
   - Drawdown calculations
   - Risk-adjusted returns

3. **Trading Logic Simulation**
   
   **Position Management:**
   ```
   Flat Position:
   - Long signal → Open Long
   - Short signal → Open Short
   
   Long Position:
   - Short signal → Close Long + Open Short
   - Neutral signal → Close Long
   
   Short Position:
   - Long signal → Close Short + Open Long
   - Neutral signal → Close Short
   ```

4. **Debug and Analysis Features**
   - Real-time progress tracking with tqdm
   - Detailed signal statistics
   - Feature correlation analysis
   - Prediction accuracy metrics
   - Trade-by-trade logging

5. **Historical Data Processing**
   - Fetches historical data for specified period
   - Recreates all LSTM features for consistency
   - Applies same signal logic as live trading
   - Handles data alignment and edge cases

### CLI Applications

#### 1. LSTM Signal Generator (`app_lstm.py`)
```bash
python3 models/app_lstm.py --ticker BTC/USD --hours 240 --lookback 60
```

**Parameters:**
- `--ticker`: Trading pair (default: BTC/USD)
- `--hours`: Historical data period (default: 240)
- `--model`: Custom model name (optional)
- `--lookback`: LSTM sequence length (default: 60)
- `--retrain-threshold`: Retraining interval in hours (default: 24)

#### 2. Backtest Runner (`app_backtester.py`)
```bash
python3 models/app_backtester.py --ticker ETH/USD --hours 720 --initial-balance 10000
```

**Parameters:**
- `--ticker`: Trading pair
- `--hours`: Backtest period (default: 720 = 30 days)
- `--initial-balance`: Starting capital (default: $100,000)
- `--fee`: Trading fee per trade (default: 0.001)
- `--lookback`: LSTM lookback period

## Advanced Features

### 1. Model Persistence
- Automatic model saving after training
- Smart loading of existing models
- Version control with timestamps
- Cross-session state preservation

### 2. Data Quality Management
- NaN/infinite value handling
- Forward/backward filling strategies
- Outlier detection and treatment
- Feature scaling with RobustScaler

### 3. Performance Optimization
- Batch prediction for efficiency
- GPU acceleration (if available)
- Memory-efficient data processing
- Configurable batch sizes

### 4. Risk Management
- Dynamic threshold adjustment
- Volatility-based position sizing
- Multiple confirmation signals
- Stop-loss logic integration

## Usage Examples

### Basic Signal Generation
```python
from lstm_model import SignalGenerator

sg = SignalGenerator(ticker='BTC/USD', lookback=60)
signals = sg.generate_signals(hours=240)
print(f"Signal: {signals['signal']}, Confidence: {signals['confidence']}")
```

### Comprehensive Backtesting
```python
from lstm_model import SignalGenerator
from lstm_backtest import Backtester

sg = SignalGenerator(ticker='ETH/USD')
backtester = Backtester(sg, hours=720, initial_balance=50000)
results = backtester.run(verbose=True)
print(f"Total Return: {results['total_return']*100:.2f}%")
```

### Multi-Timeframe Analysis
```python
# Short-term signals
short_signals = sg.generate_signals(hours=48, retrain_threshold=6)

# Medium-term signals  
medium_signals = sg.generate_signals(hours=240, retrain_threshold=24)

# Long-term backtest
long_backtest = backtester.run(hours=2160)  # 90 days
```

## Technical Considerations

### 1. Data Requirements
- Minimum 60+ hours of historical data for training
- Consistent data quality and availability
- Real-time data feed integration

### 2. Computational Requirements
- TensorFlow/Keras for LSTM implementation
- Pandas/NumPy for data processing
- Sufficient memory for sequence processing
- Optional GPU acceleration

### 3. Model Management
- Regular model retraining schedules
- Performance monitoring and validation
- Model version control and rollback
- A/B testing capabilities

This system represents a production-ready implementation of deep learning for cryptocurrency trading, combining sophisticated feature engineering, robust model architecture, and comprehensive backtesting capabilities for systematic strategy development and evaluation.
