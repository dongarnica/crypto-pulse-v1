import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import Huber, MeanSquaredError
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
import io
from contextlib import contextmanager

# Custom callback for detailed epoch progress tracking
class DetailedEpochProgressCallback(Callback):
    """Custom callback to display detailed progress for each epoch"""
    def __init__(self, total_epochs):
        super(DetailedEpochProgressCallback, self).__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print(f"🏁 Starting training with {self.total_epochs} epochs")
        print("=" * 60)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        sys.stdout.write(f"\r📊 Epoch {epoch+1}/{self.total_epochs} - Starting... ")
        sys.stdout.flush()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.start_time
        
        # Calculate ETA for remaining epochs
        avg_epoch_time = total_time / (epoch + 1)
        eta = avg_epoch_time * (self.total_epochs - epoch - 1)
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        
        # Format metrics for display
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        
        # Print epoch summary
        print(f"\r✅ Epoch {epoch+1}/{self.total_epochs} - {metrics_str} - {epoch_time:.2f}s - ETA: {eta_min:02d}:{eta_sec:02d}")
        sys.stdout.flush()

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.crypto_data_client import CryptoMarketDataClient
from utils.safe_math import safe_div, safe_ratio, safe_percentage, safe_risk_reward_ratio, clean_numerical_data

# Enable TensorFlow optimizations
tf.config.experimental.enable_tensor_float_32_execution(True)  # Updated API call
tf.get_logger().setLevel('ERROR')  # Reduce TF logging noise

class TimeoutException(Exception):
    """Exception raised when training times out"""
    pass

@contextmanager
def timeout_handler(seconds):
    """Context manager for handling training timeouts"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutException(f"Training timed out after {seconds} seconds")
    
    # Set the signal handler and a {seconds}-second alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class EnhancedProgressCallback(Callback):
    """Enhanced callback for detailed epoch progress monitoring"""
    
    def __init__(self, update_freq=1):
        super().__init__()
        self.update_freq = update_freq
        self.epoch_start_time = None
        self.training_start_time = None
        
    def on_train_begin(self, logs=None):
        self.training_start_time = time.time()
        print("\n🚀 ══════════════════════════════════════════════════════════")
        print("   📊 LSTM MODEL TRAINING STARTED")
        print("🚀 ══════════════════════════════════════════════════════════")
        print(f"⏰ Training started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🎯 Target epochs: {self.params.get('epochs', 'Unknown')}")
        print(f"📦 Batch size: {self.params.get('batch_size', 'Unknown')}")
        print("═══════════════════════════════════════════════════════════\n")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        total_epochs = self.params.get('epochs', 0)
        progress_pct = ((epoch + 1) / total_epochs) * 100 if total_epochs > 0 else 0
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // total_epochs) if total_epochs > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\n📈 Epoch {epoch + 1:3d}/{total_epochs} [{bar}] {progress_pct:5.1f}%")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
        
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            total_elapsed = time.time() - self.training_start_time
            
            # Extract metrics
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            mae = logs.get('mae', 0)
            val_mae = logs.get('val_mae', 0)
            lr = logs.get('lr', self.model.optimizer.learning_rate.numpy() if hasattr(self.model.optimizer.learning_rate, 'numpy') else 0.002)
            
            # Calculate ETA
            avg_epoch_time = total_elapsed / (epoch + 1)
            remaining_epochs = self.params.get('epochs', 0) - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            print(f"✅ Completed in {epoch_duration:.2f}s")
            print(f"📊 Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"📏 MAE: {mae:.6f} | Val MAE: {val_mae:.6f}")
            print(f"🎛️  Learning Rate: {lr:.2e}")
            print(f"⏱️  Total Time: {str(timedelta(seconds=int(total_elapsed)))}")
            print(f"⏳ ETA: {eta_str}")
            print("─" * 60)
            
    def on_train_end(self, logs=None):
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print("\n🎉 ══════════════════════════════════════════════════════════")
            print("   ✅ LSTM MODEL TRAINING COMPLETED!")
            print("🎉 ══════════════════════════════════════════════════════════")
            print(f"⏰ Training completed at: {datetime.now().strftime('%H:%M:%S')}")
            print(f"⏱️  Total training time: {str(timedelta(seconds=int(total_time)))}")
            if logs:
                final_loss = logs.get('loss', logs.get('val_loss', 'N/A'))
                print(f"📊 Final loss: {final_loss}")
            print("═══════════════════════════════════════════════════════════\n")

class AggressiveCryptoSignalGenerator:
    """
    Ultra-aggressive LSTM trading signal generator optimized for maximum short-term crypto profits.
    Features: High-frequency signals, volatility exploitation, momentum capture, and risk amplification.
    """
    def __init__(self, ticker='BTC/USD', model_dir=None, model_name=None, lookback=30):
        self.ticker = ticker
        self.lookback = lookback  # Reduced for faster reactions
        
        # Aggressive trading parameters
        self.min_confidence_threshold = 0.3  # Lower threshold for more trades
        self.volatility_multiplier = 2.0     # Amplify volatility-based signals
        self.momentum_weight = 1.5           # Increase momentum factor
        self.risk_amplification = 1.8        # Amplify position sizing
        
        # Portfolio management parameters
        self.max_position_size = 0.25        # Max 25% per position
        self.max_total_exposure = 0.75       # Max 75% total exposure
        self.position_correlation_limit = 0.8 # Limit correlated positions
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing Aggressive LSTM Signal Generator for {ticker}")
        
        # Enhanced feature set optimized for short-term momentum
        self.feature_columns = [
            'close', 'RSI_Fast', 'RSI_Slow', 'MACD_Aggressive', 'MACD_Signal', 
            'BB_Position', 'ATR_Normalized', 'Volume_Surge', 'Price_Velocity',
            'Momentum_Score', 'Volatility_Breakout', 'Support_Resistance',
            'EMA_5', 'EMA_13', 'EMA_21', 'Stoch_RSI_Fast', 'Williams_R',
            'CCI', 'MFI', 'Squeeze_Momentum', 'Trend_Strength', 'Volume_Profile'
        ]
        
        self.logger.debug(f"Using {len(self.feature_columns)} aggressive features")
        
        # Use MinMaxScaler for better gradient flow in aggressive scenarios
        self.scalers = {
            'price': MinMaxScaler(feature_range=(-1, 1)),
            'features': RobustScaler()
        }
        
        self.data_client = CryptoMarketDataClient()
        
        # Model setup
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.abspath(model_dir or os.path.join(base_dir, 'models'))
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_name = model_name or f'{self.ticker.replace("/", "_").lower()}_aggressive_lstm.keras'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model = None
        self.last_training_time = None
        
        # Performance tracking
        self.prediction_cache = {}
        self.signal_history = []
        
        print(f"🚀 Initialized Aggressive Crypto Signal Generator for {ticker}")
        print(f"📊 Model path: {self.model_path}")
        print(f"⚡ Optimized for maximum short-term profit extraction")

    def get_model_path(self) -> str:
        """Return the path where the model is saved/will be saved"""
        return self.model_path

    def get_historical_data(self, hours=24):  # Reduced timeframe for aggressiveness
        """Fetch high-frequency data optimized for short-term trading"""
        print(f"📈 Fetching {hours} hours of high-frequency data for {self.ticker}...")
        try:
            data = self.data_client.get_historical_bars(self.ticker, hours=hours)
            if data is None or data.empty:
                raise ValueError(f"No data available for {self.ticker}")
            
            print(f"✅ Retrieved {len(data)} data points for {self.ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.ticker}: {e}")
            return pd.DataFrame()

    def _clean_aggressive_data(self, df):
        """Aggressive data cleaning to remove noise and outliers"""
        df = df.copy()
        
        # Remove extreme price spikes (potential data errors)
        price_change = df['close'].pct_change().abs()
        extreme_threshold = price_change.quantile(0.99)
        df = df[price_change <= extreme_threshold]
        
        # Remove zero volume periods (inactive markets)
        df = df[df['volume'] > 0]
        
        return df

    def _calculate_aggressive_features(self, df):
        """Calculate features optimized for aggressive short-term trading"""
        print("⚡ Calculating aggressive technical features...")
        df = df.copy()
        
        try:
            # Basic price features
            df['Price_Change'] = df['close'].pct_change()
            df['Price_Velocity'] = df['Price_Change'].rolling(3).mean()
            
            # Aggressive RSI (faster periods)
            df['RSI_Fast'] = self._calculate_rsi(df['close'], period=7)
            df['RSI_Slow'] = self._calculate_rsi(df['close'], period=14)
            
            # Aggressive MACD
            ema_fast = df['close'].ewm(span=5).mean()
            ema_slow = df['close'].ewm(span=13).mean()
            df['MACD_Aggressive'] = ema_fast - ema_slow
            df['MACD_Signal'] = df['MACD_Aggressive'].ewm(span=3).mean()
            
            # Bollinger Bands position
            bb_period = 10
            bb_std = 1.5
            bb_middle = df['close'].rolling(bb_period).mean()
            bb_std_dev = df['close'].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            df['BB_Position'] = safe_div(df['close'] - bb_lower, bb_upper - bb_lower, 0.5)
            
            # ATR normalized
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = tr.rolling(10).mean()
            df['ATR_Normalized'] = safe_div(atr, df['close'], 0.01)
            
            # Volume features
            vol_sma = df['volume'].rolling(20).mean()
            df['Volume_Surge'] = safe_div(df['volume'], vol_sma, 1.0)
            
            # Momentum features
            momentum_3 = safe_div(df['close'], df['close'].shift(3), 1.0) - 1
            momentum_7 = safe_div(df['close'], df['close'].shift(7), 1.0) - 1
            df['Momentum_Score'] = (momentum_3 * 0.7) + (momentum_7 * 0.3)
            
            # Volatility breakout
            vol_threshold = df['ATR_Normalized'].rolling(20).quantile(0.8)
            df['Volatility_Breakout'] = (df['ATR_Normalized'] > vol_threshold).astype(int)
            
            # Support/Resistance
            df['Support_Resistance'] = 0  # Placeholder - implement your logic
            
            # EMAs
            df['EMA_5'] = df['close'].ewm(span=5).mean()
            df['EMA_13'] = df['close'].ewm(span=13).mean()
            df['EMA_21'] = df['close'].ewm(span=21).mean()
            
            # Additional indicators
            df['Stoch_RSI_Fast'] = df['RSI_Fast']  # Simplified
            high_14_max = df['high'].rolling(14).max()
            low_14_min = df['low'].rolling(14).min()
            df['Williams_R'] = safe_div(high_14_max - df['close'], high_14_max - low_14_min, 0.5) * -100
            
            df['CCI'] = 0  # Placeholder
            df['MFI'] = 50  # Placeholder
            df['Squeeze_Momentum'] = df['Momentum_Score']  # Simplified
            df['Trend_Strength'] = abs(df['Momentum_Score'])
            df['Volume_Profile'] = df['Volume_Surge']
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            # Fill with defaults if calculation fails
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
        
        # Aggressive NaN handling with safe cleaning
        df = clean_numerical_data(df, fill_method='forward')
        
        print(f"✅ Aggressive features calculated. Shape: {df.shape}")
        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator with safe division"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = safe_div(gain, loss, 1.0)
        rsi = 100 - safe_div(100, 1 + rs, 50.0)
        return rsi.fillna(50)

    def create_aggressive_model(self, input_shape):
        """Create an aggressive model architecture optimized for fast learning and high sensitivity"""
        print(f"🧠 Creating aggressive model with input shape: {input_shape}")
        
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=128, kernel_size=3, activation='swish', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv1D(filters=64, kernel_size=3, activation='swish', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            BatchNormalization(),
            
            GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
            BatchNormalization(),
            
            GRU(32, return_sequences=False, dropout=0.1),
            BatchNormalization(),
            
            Dense(64, activation='swish'),
            Dropout(0.3),
            Dense(32, activation='swish'),
            Dropout(0.2),
            Dense(16, activation='swish'),
            Dense(1, activation='linear')
        ])
        
        model.summary()
        return model

    def get_or_create_model(self, input_shape):
        """Load or create model with aggressive settings"""
        print(f"🔄 Getting or creating aggressive model at {self.model_path}")
        
        try:
            if os.path.exists(self.model_path):
                print("📂 Loading existing aggressive model...")
                self.model = load_model(self.model_path)
                print("✅ Aggressive model loaded successfully")
            else:
                print("🆕 Creating new aggressive model...")
                self.model = self.create_aggressive_model(input_shape)
        except Exception as e:
            print(f"⚠️ Error loading model, creating new one: {e}")
            self.model = self.create_aggressive_model(input_shape)
        
        return self.model

    def train_aggressive_model(self, X, y, validation_split=0.15, max_epochs=50, timeout_seconds=600):
        """Aggressive training with high learning rate and fast convergence
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Portion of data to use for validation (default: 0.15)
            max_epochs: Maximum number of epochs to train (default: 50)
            timeout_seconds: Maximum training time in seconds (default: 600 - 10 minutes)
            
        Returns:
            training history or None if timeout occurred
        """
        print("🚀 Starting aggressive model training...")
        print(f"Training data - X: {X.shape}, y: {y.shape}")
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No training data available")
        
        # Aggressive callbacks with detailed progress tracking
        progress_callback = DetailedEpochProgressCallback(total_epochs=max_epochs)
        callbacks = [
            progress_callback,
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
            ModelCheckpoint(self.model_path, save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1)
        ]

        # Aggressive optimizer settings
        optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss=Huber(delta=0.5), metrics=['mae'])

        print("\n⚡ Training with aggressive parameters...")
        print(f"⏱️ Max epochs: {max_epochs}, Batch size: 32, Timeout: {timeout_seconds}s")
        
        try:
            # Use the timeout_handler context manager to prevent hanging
            with timeout_handler(timeout_seconds):
                history = self.model.fit(
                    X, y,
                    epochs=max_epochs,
                    batch_size=32,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=0,  # Set to 0 as we're using our custom progress callback
                    shuffle=True
                )
            
            print("✅ Aggressive training completed")
            self.last_training_time = datetime.now()
            return history
            
        except TimeoutException as e:
            print(f"\n⚠️ {str(e)}")
            print("💾 Saving current model state...")
            self.model.save(self.model_path)
            print("✅ Model saved despite timeout")
            self.last_training_time = datetime.now()
            return None
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clear GPU memory
            tf.keras.backend.clear_session()

    def _create_sequences(self, data):
        """Create sequences optimized for aggressive trading"""
        if len(data) < self.lookback:
            return np.array([]), np.array([])
        
        X = np.array([data[i-self.lookback:i] for i in range(self.lookback, len(data))])
        y = data[self.lookback:, 0]  # Target is close price
        
        return X, y

    def generate_aggressive_signals(self, hours=12, retrain_threshold=6):
        """Generate ultra-aggressive trading signals with high frequency"""
        print(f"\n{'🚀'*20}\nGENERATING AGGRESSIVE SIGNALS FOR {self.ticker}\n{'🚀'*20}")
        
        # Step 1: Get fresh data
        print("\n[1/4] 📊 Fetching fresh market data...")
        raw_data = self.get_historical_data(hours=hours)
        if raw_data.empty:
            return {"error": "No data available", "signal": 0}

        # Step 2: Prepare aggressive features
        print("\n[2/4] ⚡ Preparing aggressive features...")
        try:
            df = self._clean_aggressive_data(raw_data)
            df = self._calculate_aggressive_features(df)
            
            if len(df) < self.lookback + 10:
                return {"error": "Insufficient data after processing", "signal": 0}
                
        except Exception as e:
            return {"error": f"Feature calculation failed: {e}", "signal": 0}

        # Step 3: Model handling with aggressive retraining
        print("\n[3/4] 🧠 Handling aggressive model...")
        try:
            # Prepare features
            feature_data = df[self.feature_columns].values
            
            # Scale features
            scaled_features = self.scalers['features'].fit_transform(feature_data)
            scaled_prices = self.scalers['price'].fit_transform(df[['close']].values)
            
            # Create sequences
            X, y = self._create_sequences(scaled_features)
            
            if len(X) == 0:
                return {"error": "Could not create training sequences", "signal": 0}
            
            # Get or create model
            self.get_or_create_model((self.lookback, len(self.feature_columns)))
            
            # Check if retraining is needed
            need_retrain = (
                self.last_training_time is None or
                (datetime.now() - self.last_training_time).total_seconds() > retrain_threshold * 3600
            )
            
            if need_retrain:
                print("🔄 Retraining aggressive model...")
                # Use faster training settings in production
                max_epochs = 25  # Reduced from 50 to make training faster
                timeout = 300    # 5-minute timeout
                self.train_aggressive_model(
                    X, 
                    y[:len(X)], 
                    max_epochs=max_epochs,
                    timeout_seconds=timeout
                )
                    
        except Exception as e:
            return {"error": f"Model preparation failed: {e}", "signal": 0}

        # Step 4: Generate aggressive predictions
        print("\n[4/4] 🎯 Generating aggressive predictions...")
        try:
            # Make prediction on latest data
            latest_sequence = scaled_features[-self.lookback:].reshape(1, self.lookback, -1)
            prediction = self.model.predict(latest_sequence, verbose=0)
            
            # Generate trading signals
            signals_df = df.copy()
            predictions = np.full(len(signals_df), np.nan)
            predictions[-1] = prediction[0, 0]  # Only latest prediction
            
            result = self._generate_aggressive_trading_signals(
                signals_df, predictions[-1:], self.scalers['price']
            )
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction generation failed: {e}", "signal": 0}

    def _generate_aggressive_trading_signals(self, df, predictions, scaler):
        """Generate ultra-aggressive trading signals with amplified risk/reward"""
        print("🎯 Generating aggressive trading signals...")
        
        df = df.copy()
        
        # Process predictions
        pred_reshaped = predictions.reshape(-1, 1)
        pred_reshaped = np.nan_to_num(pred_reshaped, nan=0, posinf=0, neginf=0)
        inv_preds = scaler.inverse_transform(pred_reshaped).flatten()
        
        # Store predictions
        df.loc[df.index[-len(inv_preds):], 'Prediction'] = inv_preds
        
        # Aggressive dynamic thresholds
        volatility = df['ATR_Normalized'].rolling(10).mean().fillna(0)
        df['Aggressive_Threshold'] = (
            volatility * self.volatility_multiplier + 
            df['Price_Change'].abs().rolling(5).mean() * 1.5
        )
        
        # Initialize aggressive signals
        df['Signal'] = 0
        df['Confidence'] = 0.0
        df['Risk_Multiplier'] = 1.0
        
        # Ultra-aggressive long conditions
        aggressive_long = (
            (df['Prediction'] > df['close'] * (1 + df['Aggressive_Threshold'])) &
            (df['RSI_Fast'] < 70) &
            (df['Momentum_Score'] > 0.2) &
            (df['Volume_Surge'] > 1.2) &
            (df['Volatility_Breakout'] == 1) &
            (df['EMA_5'] > df['EMA_13']) &
            (df['MACD_Aggressive'] > 0) &
            (df['Price_Velocity'] > 0) &
            (df['Trend_Strength'] > df['Trend_Strength'].rolling(10).quantile(0.6))
        )
        
        # Ultra-aggressive short conditions
        aggressive_short = (
            (df['Prediction'] < df['close'] * (1 - df['Aggressive_Threshold'])) &
            (df['RSI_Fast'] > 30) &
            (df['Momentum_Score'] < -0.2) &
            (df['Volume_Surge'] > 1.2) &
            (df['Volatility_Breakout'] == 1) &
            (df['EMA_5'] < df['EMA_13']) &
            (df['MACD_Aggressive'] < 0) &
            (df['Price_Velocity'] < 0) &
            (df['Trend_Strength'] > df['Trend_Strength'].rolling(10).quantile(0.6))
        )
        
        # Apply aggressive signals
        df.loc[aggressive_long, 'Signal'] = 1
        df.loc[aggressive_short, 'Signal'] = -1
        
        # Aggressive confidence scoring with safe division
        price_diff_pct = safe_div(abs(df['Prediction'] - df['close']), df['close'], 0.0)
        momentum_factor = abs(df['Momentum_Score'])
        volatility_factor = df['Volatility_Breakout']
        volume_factor = np.clip(df['Volume_Surge'] - 1, 0, 2)
        
        df['Confidence'] = np.clip(
            (price_diff_pct * 2 + momentum_factor + volatility_factor + volume_factor * 0.5) * 
            self.risk_amplification, 0, 5
        )
        
        # Risk multiplier for position sizing
        df['Risk_Multiplier'] = np.clip(
            1 + (df['Confidence'] - 1) * self.risk_amplification, 0.5, 3.0
        )
        
        latest = df.iloc[-1]
        
        # Enhanced signal output
        signal_result = {
            'signal': int(latest['Signal']),
            'confidence': float(latest['Confidence']),
            'current_price': float(latest['close']),
            'predicted_price': float(latest['Prediction']),
            'risk_multiplier': float(latest['Risk_Multiplier']),
            'momentum_score': float(latest['Momentum_Score']),
            'volatility_breakout': bool(latest['Volatility_Breakout']),
            'volume_surge': float(latest['Volume_Surge']),
            'rsi_fast': float(latest['RSI_Fast']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate trade recommendation
        signal_result['trade_recommendation'] = self._get_trade_recommendation(latest)
        
        return signal_result

    def _get_trade_recommendation(self, latest_data):
        """Generate specific trade recommendations based on aggressive signals and current positions"""
        signal = int(latest_data['Signal'])
        confidence = float(latest_data['Confidence'])
        risk_mult = float(latest_data['Risk_Multiplier'])
        current_price = float(latest_data['close'])
        atr = float(latest_data['ATR_Normalized']) * current_price
        
        # Get current position and portfolio summary
        current_position = self.data_client.get_position(self.ticker)
        portfolio_summary = self.data_client.get_position_summary()
        
        # Base recommendation structure
        recommendation = {
            'action': 'HOLD',
            'position_size_pct': 0,
            'stop_loss': None,
            'take_profit': None,
            'reasoning': 'No strong signal detected',
            'current_position': current_position,
            'portfolio_impact': {},
            'risk_management': {}
        }
        
        if signal == 0:
            # Check if we should close existing position
            if current_position:
                recommendation.update(self._evaluate_position_exit(current_position, latest_data))
            return recommendation
        
        # Calculate position sizing based on portfolio constraints
        position_sizing = self._calculate_position_sizing(
            signal, confidence, risk_mult, portfolio_summary, current_position
        )
        
        if position_sizing['action'] == 'NO_ACTION':
            recommendation.update(position_sizing)
            return recommendation
        
        # Generate trade levels
        trade_levels = self._calculate_trade_levels(current_price, atr, signal, confidence)
        
        # Determine action based on current position
        if current_position is None:
            # New position
            action = 'BUY_AGGRESSIVE' if signal == 1 else 'SELL_AGGRESSIVE'
            reasoning = f"Opening new {action} position with {confidence:.2f} confidence"
        else:
            # Position modification logic
            current_side = current_position.get('side', 'long')
            
            if (signal == 1 and current_side == 'long') or (signal == -1 and current_side == 'short'):
                # Same direction - consider adding to position
                if position_sizing['can_add_to_position']:
                    action = f"ADD_TO_{current_side.upper()}"
                    reasoning = f"Adding to existing {current_side} position"
                else:
                    action = 'HOLD'
                    reasoning = f"Signal confirms {current_side} bias but position limits reached"
            else:
                # Opposite direction - consider closing and reversing
                action = f"REVERSE_TO_{'LONG' if signal == 1 else 'SHORT'}"
                reasoning = f"Reversing from {current_side} to {'long' if signal == 1 else 'short'}"
        
        recommendation.update({
            'action': action,
            'position_size_pct': position_sizing['position_size_pct'],
            'stop_loss': trade_levels['stop_loss'],
            'take_profit': trade_levels['take_profit'],
            'reasoning': reasoning,
            'risk_reward_ratio': trade_levels['risk_reward_ratio'],
            'portfolio_impact': position_sizing['portfolio_impact'],
            'risk_management': {
                'max_risk_per_trade': position_sizing['max_risk_amount'],
                'portfolio_heat': portfolio_summary.get('total_exposure', 0),
                'correlation_check': position_sizing.get('correlation_warning', False),
                'position_limit_reached': not position_sizing['can_add_to_position']
            }
        })
        
        return recommendation
    
    def _calculate_position_sizing(self, signal, confidence, risk_mult, portfolio_summary, current_position):
        """Calculate optimal position sizing considering portfolio constraints"""
        base_position = 0.1  # 10% base position
        
        # Adjust for confidence and risk multiplier
        confidence_multiplier = min(confidence * 1.5, 1.2)  # Max 1.2x multiplier
        aggressive_position = base_position * confidence_multiplier * risk_mult
        
        # Portfolio constraint checks
        total_exposure = portfolio_summary.get('total_exposure', 0)
        current_positions_count = portfolio_summary.get('total_positions', 0)
        
        # Check maximum exposure limit
        if total_exposure >= self.max_total_exposure:
            return {
                'action': 'NO_ACTION',
                'position_size_pct': 0,
                'reasoning': f'Portfolio exposure limit reached ({total_exposure:.1%})',
                'can_add_to_position': False,
                'portfolio_impact': {'exposure_limit_reached': True}
            }
        
        # Adjust position size based on current exposure
        available_exposure = self.max_total_exposure - total_exposure
        max_position_size = min(self.max_position_size, available_exposure)
        
        # Final position size
        final_position_size = min(aggressive_position, max_position_size)
        
        # Check if we can add to existing position
        can_add_to_position = True
        if current_position:
            current_position_value = current_position.get('position_value', 0)
            current_position_pct = current_position_value / 100000  # Assuming $100k portfolio
            
            if current_position_pct >= self.max_position_size:
                can_add_to_position = False
        
        # Calculate risk amount
        max_risk_amount = final_position_size * 0.02  # Risk 2% per position
        
        return {
            'action': 'POSITION' if final_position_size > 0.01 else 'NO_ACTION',
            'position_size_pct': round(final_position_size * 100, 1),
            'max_risk_amount': max_risk_amount,
            'can_add_to_position': can_add_to_position,
            'portfolio_impact': {
                'current_exposure': total_exposure,
                'new_exposure': total_exposure + final_position_size,
                'available_exposure': available_exposure,
                'positions_count': current_positions_count
            }
        }
    
    def _calculate_trade_levels(self, current_price, atr, signal, confidence):
        """Calculate stop loss and take profit levels with safe division"""
        # More aggressive levels for higher confidence
        stop_multiplier = max(1.0, 2.0 - confidence)  # Tighter stops for high confidence
        profit_multiplier = min(4.0, 2.0 + confidence * 2)  # Bigger targets for high confidence
        
        if signal == 1:  # LONG
            stop_loss = current_price - (atr * stop_multiplier)
            take_profit = current_price + (atr * profit_multiplier)
        else:  # SHORT
            stop_loss = current_price + (atr * stop_multiplier)
            take_profit = current_price - (atr * profit_multiplier)
        
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = safe_risk_reward_ratio(risk_amount, reward_amount, 0.0)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2)
        }
    
    def _evaluate_position_exit(self, current_position, latest_data):
        """Evaluate whether to exit current position"""
        current_price = float(latest_data['close'])
        entry_price = current_position.get('entry_price', current_price)
        side = current_position.get('side', 'long')
        stop_loss = current_position.get('stop_loss')
        take_profit = current_position.get('take_profit')
        
        # Check stop loss and take profit
        if stop_loss and ((side == 'long' and current_price <= stop_loss) or 
                         (side == 'short' and current_price >= stop_loss)):
            return {
                'action': 'CLOSE_STOP_LOSS',
                'reasoning': f'Stop loss triggered at {current_price}',
                'exit_price': current_price
            }
        
        if take_profit and ((side == 'long' and current_price >= take_profit) or 
                           (side == 'short' and current_price <= take_profit)):
            return {
                'action': 'CLOSE_TAKE_PROFIT',
                'reasoning': f'Take profit triggered at {current_price}',
                'exit_price': current_price
            }
        
        # Check for trend reversal indicators
        momentum_score = float(latest_data['Momentum_Score'])
        rsi = float(latest_data['RSI_Fast'])
        
        reversal_signals = 0
        
        if side == 'long':
            if momentum_score < -0.3:
                reversal_signals += 1
            if rsi > 80:  # Overbought
                reversal_signals += 1
        else:  # short
            if momentum_score > 0.3:
                reversal_signals += 1
            if rsi < 20:  # Oversold
                reversal_signals += 1
        
        if reversal_signals >= 2:
            return {
                'action': 'CLOSE_REVERSAL',
                'reasoning': f'Multiple reversal signals detected ({reversal_signals})',
                'exit_price': current_price
            }
        
        return {
            'action': 'HOLD',
            'reasoning': 'Position maintained - no exit criteria met'
        }
    
    def execute_trade_recommendation(self, recommendation: Dict) -> Dict:
        """Execute a trade recommendation and update positions"""
        action = recommendation['action']
        current_price = recommendation.get('current_price', 0)
        
        try:
            if action.startswith('BUY_') or action.startswith('SELL_'):
                # Opening new position
                side = 'long' if action.startswith('BUY_') else 'short'
                position_size_pct = recommendation['position_size_pct']
                
                position_data = {
                    'side': side,
                    'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(),
                    'quantity': position_size_pct / 100,  # Convert to decimal
                    'stop_loss': recommendation.get('stop_loss'),
                    'take_profit': recommendation.get('take_profit'),
                    'confidence': recommendation.get('confidence', 0),
                    'risk_amount': recommendation.get('risk_management', {}).get('max_risk_per_trade', 0),
                    'status': 'active'
                }
                
                self.data_client.update_position(self.ticker, position_data)
                return {'status': 'success', 'action': 'position_opened', 'data': position_data}
            
            elif action.startswith('CLOSE_'):
                # Closing position
                exit_reason = action.replace('CLOSE_', '').lower()
                closed_position = self.data_client.close_position(
                    self.ticker, current_price, exit_reason
                )
                return {'status': 'success', 'action': 'position_closed', 'data': closed_position}
            
            elif action.startswith('ADD_TO_'):
                # Adding to existing position
                current_position = self.data_client.get_position(self.ticker)
                if current_position:
                    additional_quantity = recommendation['position_size_pct'] / 100
                    current_position['quantity'] += additional_quantity
                    current_position['last_modified'] = datetime.now().isoformat()
                    
                    self.data_client.update_position(self.ticker, current_position)
                    return {'status': 'success', 'action': 'position_increased', 'data': current_position}
            
            elif action.startswith('REVERSE_'):
                # Close current and open opposite
                self.data_client.close_position(self.ticker, current_price, 'reversal')
                
                new_side = 'long' if 'LONG' in action else 'short'
                position_data = {
                    'side': new_side,
                    'entry_price': current_price,
                    'entry_time': datetime.now().isoformat(),
                    'quantity': recommendation['position_size_pct'] / 100,
                    'stop_loss': recommendation.get('stop_loss'),
                    'take_profit': recommendation.get('take_profit'),
                    'status': 'active'
                }
                
                self.data_client.update_position(self.ticker, position_data)
                return {'status': 'success', 'action': 'position_reversed', 'data': position_data}
            
            else:
                return {'status': 'no_action', 'action': action}
                
        except Exception as e:
            self.logger.error(f"Error executing trade recommendation: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_performance_metrics(self):
        """Get performance metrics for the model"""
        return {
            'avg_confidence': 0.5,
            'signal_count': len(self.signal_history),
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None
        }
