
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from models.lstm_model import SignalGenerator



@pytest.fixture
def dummy_df():
    # Create a DataFrame with enough rows and all required columns for feature engineering and signal generation
    n = 120
    idx = pd.date_range(end=datetime.now(), periods=n, freq='H')
    df = pd.DataFrame({
        'close': np.linspace(30000, 31000, n) + np.random.normal(0, 50, n),
        'high': np.linspace(30100, 31100, n) + np.random.normal(0, 50, n),
        'low': np.linspace(29900, 30900, n) + np.random.normal(0, 50, n),
        'volume': np.abs(np.random.normal(100, 10, n)),
    }, index=idx)
    # Add columns that would be created by feature engineering
    df['RSI'] = np.random.uniform(30, 70, n)
    df['MACD'] = np.random.normal(0, 1, n)
    df['signal_line'] = np.random.normal(0, 1, n)
    df['OBV'] = np.cumsum(np.random.normal(0, 100, n))
    df['VWAP'] = df['close'] + np.random.normal(0, 10, n)
    df['EMA_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['EMA_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['ADX'] = np.random.uniform(20, 40, n)
    df['Stoch_RSI'] = np.random.uniform(0, 1, n)
    df['BB_Middle'] = df['close'].rolling(20, min_periods=1).mean()
    df['BB_Width'] = np.random.uniform(0.01, 0.05, n)
    df['ATR'] = np.random.uniform(50, 100, n)
    df['Realized_Vol'] = np.random.uniform(0.01, 0.05, n)
    df['Volume_Ratio'] = np.random.uniform(0.8, 1.2, n)
    return df

@pytest.fixture
def dummy_predictions(dummy_df):
    # Simulate model predictions (scaled values)
    return np.linspace(30500, 30800, min(100, len(dummy_df))).reshape(-1, 1)

@pytest.fixture
def dummy_scaler():
    # Dummy scaler that just returns the input (identity)
    class DummyScaler:
        def inverse_transform(self, arr):
            return arr
    return DummyScaler()

@pytest.fixture
def signal_generator(tmp_path):
    # Use a temporary directory for model files
    return SignalGenerator(ticker='BTC/USD', model_dir=tmp_path, lookback=60)

def test_generate_trading_signals_long(signal_generator, dummy_df, dummy_predictions, dummy_scaler):
    # Set up DataFrame so that long condition is met for the last row
    df = dummy_df.copy()
    idx = df.index[-len(dummy_predictions):]

    # Set all long conditions to be True for the last row only
    last_idx = df.index[-1]
    # Set all long conditions for the last row only
    df.at[last_idx, 'Prediction'] = df.at[last_idx, 'close'] + df.at[last_idx, 'ATR'] + df.at[last_idx, 'Realized_Vol'] + 1  # Prediction > close + Threshold
    df.at[last_idx, 'RSI'] = 50  # < 65
    df.at[last_idx, 'MACD'] = 2  # > signal_line
    df.at[last_idx, 'signal_line'] = 1
    obv_rolling = df['OBV'].rolling(20).mean().iloc[-1]
    df.at[last_idx, 'OBV'] = obv_rolling + 100  # OBV > rolling mean
    df.at[last_idx, 'VWAP'] = df.at[last_idx, 'close'] - 1  # close > VWAP
    df.at[last_idx, 'EMA_20'] = df.at[last_idx, 'close'] - 2  # close > EMA_20
    df.at[last_idx, 'EMA_50'] = df.at[last_idx, 'EMA_20'] - 1  # EMA_20 > EMA_50
    df.at[last_idx, 'ADX'] = 30  # > 25
    realized_vol_rolling = df['Realized_Vol'].rolling(20).mean().iloc[-1]
    df.at[last_idx, 'Realized_Vol'] = realized_vol_rolling * 0.8  # < rolling mean * 1.2
    df.at[last_idx, 'Volume_Ratio'] = 1.2  # > 1.1

    # Print debug info for the last row
    print("\n--- DEBUG: Last Row Values for Long Signal ---")
    print(df.loc[[last_idx]][['Prediction','close','ATR','Realized_Vol','RSI','MACD','signal_line','OBV','VWAP','EMA_20','EMA_50','ADX','Volume_Ratio']])
    print(f"OBV rolling mean: {obv_rolling}")
    print(f"Realized_Vol rolling mean: {realized_vol_rolling}")

    # Evaluate each long condition for the last row
    row = df.loc[last_idx]
    threshold = 0.5 * row['ATR'] + 0.3 * row['Realized_Vol']
    obv_rolling_mean = df['OBV'].rolling(20).mean().iloc[-1]
    realized_vol_rolling_mean = df['Realized_Vol'].rolling(20).mean().iloc[-1]
    long_conditions = {
        'Prediction > close + Threshold': row['Prediction'] > row['close'] + threshold,
        'RSI < 65': row['RSI'] < 65,
        'MACD > signal_line': row['MACD'] > row['signal_line'],
        'OBV > OBV_rolling_mean': row['OBV'] > obv_rolling_mean,
        'close > VWAP': row['close'] > row['VWAP'],
        'close > EMA_20': row['close'] > row['EMA_20'],
        'EMA_20 > EMA_50': row['EMA_20'] > row['EMA_50'],
        'ADX > 25': row['ADX'] > 25,
        'Realized_Vol < Realized_Vol_rolling_mean * 1.2': row['Realized_Vol'] < realized_vol_rolling_mean * 1.2,
        'Volume_Ratio > 1.1': row['Volume_Ratio'] > 1.1
    }
    print("\n--- DEBUG: Long Condition Evaluations for Last Row ---")
    for cond, val in long_conditions.items():
        print(f"{cond}: {val}")

    # Print the actual long_cond mask for the last 5 rows
    # Reproduce the long_cond mask as in the model
    threshold_col = 0.5 * df['ATR'] + 0.3 * df['Realized_Vol']
    obv_rolling_mean_col = df['OBV'].rolling(20).mean()
    realized_vol_rolling_col = df['Realized_Vol'].rolling(20).mean()
    long_cond = (
        (df['Prediction'] > df['close'] + threshold_col) &
        (df['RSI'] < 65) &
        (df['MACD'] > df['signal_line']) &
        (df['OBV'] > obv_rolling_mean_col) &
        (df['close'] > df['VWAP']) &
        (df['close'] > df['EMA_20']) &
        (df['EMA_20'] > df['EMA_50']) &
        (df['ADX'] > 25) &
        (df['Realized_Vol'] < realized_vol_rolling_col * 1.2) &
        (df['Volume_Ratio'] > 1.1)
    )
    print("\n--- DEBUG: long_cond mask for last 5 rows ---")
    print(long_cond.tail())
    print(f"long_cond for last row: {long_cond.iloc[-1]}")

    # Reset index to ensure alignment for mask assignment in model
    df = df.reset_index(drop=True)
    last_idx = df.index[-1]

    result = signal_generator._generate_trading_signals(df, dummy_predictions, dummy_scaler)
    assert isinstance(result, dict)
    assert result['signal'] == 1
    assert result['confidence'] >= 0
    assert 'predicted_price' in result
    assert 'current_price' in result

def test_generate_trading_signals_short(signal_generator, dummy_df, dummy_predictions, dummy_scaler):
    # Set up DataFrame so that short condition is met for the last row
    df = dummy_df.copy()
    idx = df.index[-len(dummy_predictions):]
    df.loc[idx, 'Prediction'] = df.loc[idx, 'close'] - df.loc[idx, 'ATR'] - 10  # Ensure Prediction < close - Threshold
    df.loc[idx, 'RSI'] = 50
    df.loc[idx, 'MACD'] = -2
    df.loc[idx, 'signal_line'] = 1
    df.loc[idx, 'OBV'] = df['OBV'].min() - 100
    df.loc[idx, 'VWAP'] = df.loc[idx, 'close'] + 10
    df.loc[idx, 'EMA_20'] = df.loc[idx, 'close'] - 5
    df.loc[idx, 'EMA_50'] = df.loc[idx, 'close']