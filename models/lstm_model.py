import os
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import sys
import tensorflow as tf

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.crypto_data_client import CryptoMarketDataClient

# Enable TensorFlow logging
tf.get_logger().setLevel('INFO')

class SignalGenerator:
    """
    Optimized trading signal generator using LSTM with improved features, model architecture, and training process.
    """
    def __init__(self, ticker='BTC/USD', model_dir=None, model_name=None, lookback=60):
        self.ticker = ticker
        self.lookback = lookback
        
        # Enhanced feature set
        self.feature_columns = [
            'close', 'RSI', 'MACD', 'BB_Middle', 'ATR', 'OBV',
            'VWAP', 'Log_Returns', 'Realized_Vol', 'Volume_Delta',
            'ADX', 'Stoch_RSI', 'EMA_20', 'EMA_50'
        ]
        
        self.scalers = {
            'price': RobustScaler(),
            'features': RobustScaler()
        }
        
        self.data_client = CryptoMarketDataClient()
        
        # Use absolute path for model directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.abspath(model_dir or os.path.join(base_dir, 'data'))
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_name = model_name or f'{self.ticker.replace("/", "_").lower()}_lstm_v2.keras'
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.model = None
        self.last_training_time = None
        
        print(f"Initialized SignalGenerator for {ticker}")
        print(f"Model will be saved to: {self.model_path}")

    def get_historical_data(self, hours=48):
        """Fetch historical data with error handling and caching"""
        print(f"Fetching {hours} hours of historical data for {self.ticker}...")
        try:
            bars = self.data_client.get_historical_bars(self.ticker, hours=hours)
            if bars.empty:
                print("Warning: Received empty DataFrame from data client")
                return pd.DataFrame()
                
            bars = bars.ffill().bfill()
            print(f"Retrieved data with shape: {bars.shape}")
            return self._calculate_features(bars)
        except Exception as e:
            print(f"[Error] Failed to get historical data: {str(e)}")
            return pd.DataFrame()

    def _calculate_features(self, df):
        """Enhanced feature engineering with more technical indicators"""
        print("Calculating features...")
        df = df.copy()
        
        def safe_div(a, b, fill_val=0):
            return np.where(b != 0, a / b, fill_val)
            
        try:
            # Price-based indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
            
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd_diff()
            df['macd_line'] = macd.macd()
            df['signal_line'] = macd.macd_signal()
            
            boll = ta.volatility.BollingerBands(df['close'])
            df['BB_Middle'] = boll.bollinger_mavg()
            df['BB_Upper'] = boll.bollinger_hband()
            df['BB_Lower'] = boll.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            
            # Volume-based indicators
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['OBV'] = obv.on_balance_volume()
            
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = safe_div((df['volume'] * typical_price).cumsum(), df['volume'].cumsum())
            
            # Moving averages
            df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            # Price transformations
            df['price_ratio'] = df['close'] / df['close'].shift(1)
            df['Log_Returns'] = np.log(df['price_ratio'].clip(lower=1e-7))
            df['Realized_Vol'] = df['Log_Returns'].rolling(24, min_periods=1).std() * np.sqrt(365)
            
            # Volume features
            df['Volume_Delta'] = df['volume'].diff()
            df['Volume_MA'] = df['volume'].rolling(10).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
            
            # Trend features
            df['EMA_20_50_Cross'] = np.where(df['EMA_20'] > df['EMA_50'], 1, -1)
            
        except Exception as e:
            print(f"[Feature Engineering Error] {str(e)}")
            
        # Handle infinite/NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df.isna().any().any():
            print(f"Found {df.isna().sum().sum()} NaN values after feature engineering")
            df = df.ffill().bfill()
            if df.isna().any().any():
                print(f"Still {df.isna().sum().sum()} NaN values after filling")
            
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def create_model(self, input_shape):
        """Enhanced model architecture with more sophisticated layers"""
        print(f"Creating new model with input shape: {input_shape}")
        model = Sequential([
            Input(shape=input_shape),
            BatchNormalization(),
            
            # First LSTM block
            LSTM(256, return_sequences=True, recurrent_dropout=0.2, 
                kernel_regularizer='l2', recurrent_regularizer='l2'),
            Dropout(0.3),
            BatchNormalization(),
            
            # Second LSTM block
            LSTM(128, return_sequences=True, recurrent_dropout=0.2,
                kernel_regularizer='l2', recurrent_regularizer='l2'),
            Dropout(0.3),
            BatchNormalization(),
            
            # Third LSTM block
            LSTM(64, return_sequences=False, recurrent_dropout=0.1),
            Dropout(0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(64, activation='swish', kernel_regularizer='l2'),
            Dropout(0.2),
            Dense(32, activation='swish'),
            Dense(1, activation='linear')
        ])
        
        model.summary()
        return model

    def get_or_create_model(self, input_shape):
        """Load or create model with better error handling"""
        print(f"\nAttempting to get or create model at {self.model_path}")
        
        try:
            if os.path.exists(self.model_path):
                try:
                    print("Found existing model file. Attempting to load...")
                    model = load_model(self.model_path)
                    print(f"Successfully loaded existing model from {self.model_path}")
                    model.summary()
                    
                    # Set last_training_time to file modification time for existing models
                    model_mtime = os.path.getmtime(self.model_path)
                    self.last_training_time = datetime.fromtimestamp(model_mtime)
                    print(f"Model last modified: {self.last_training_time}")
                    
                    return model
                except Exception as e:
                    print(f"Failed to load existing model: {str(e)}")
                    print("Creating new model instead...")
                    return self.create_model(input_shape)
            else:
                print("No existing model found. Creating new one...")
                return self.create_model(input_shape)
        except Exception as e:
            print(f"Critical error in model initialization: {str(e)}")
            raise

    def _create_sequences(self, data):
        """Create sequences with validation"""
        print("\nCreating sequences...")
        if len(data) < self.lookback:
            raise ValueError(f"Not enough data points. Need at least {self.lookback}, got {len(data)}")
        
        X = np.array([data[i-self.lookback:i] for i in range(self.lookback, len(data))])
        y = data[self.lookback:, 0]  # First column is the target (close price)
        
        print(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train_model(self, X, y, validation_split=0.2):
        """Enhanced training process with more debugging"""
        print("\nStarting model training...")
        print(f"Training data shapes - X: {X.shape}, y: {y.shape}")
        
        # Verify data isn't empty
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data received")
        
        # Verify model exists
        if self.model is None:
            raise ValueError("Model not initialized before training")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(self.model_path, save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        optimizer = Adam(learning_rate=0.0005, clipvalue=0.5)
        self.model.compile(optimizer=optimizer, loss=Huber())

        print("\nStarting training process...")
        try:
            history = self.model.fit(
                X, y,
                epochs=100,
                batch_size=64,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            print("\nTraining completed successfully")
            self.last_training_time = datetime.now()
            return history
        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            raise
        finally:
            # Always save the model after training
            try:
                print(f"Attempting to save model to: {self.model_path}")
                print(f"Model directory exists: {os.path.exists(self.model_dir)}")
                print(f"Model object exists: {self.model is not None}")
                
                self.model.save(self.model_path)
                print(f"Model saved to {self.model_path}")
                
                # Verify the file was actually created
                if os.path.exists(self.model_path):
                    file_size = os.path.getsize(self.model_path)
                    print(f"✓ Model file created successfully - Size: {file_size} bytes")
                else:
                    print("✗ Model file was not created!")
                    
            except Exception as e:
                print(f"Failed to save model: {str(e)}")
                import traceback
                traceback.print_exc()

    def generate_signals(self, hours=48, retrain_threshold=24):
        """Enhanced signal generation with more debugging"""
        print(f"\n{'='*50}\nStarting signal generation for {self.ticker}\n{'='*50}")
        
        # 1. Get data
        print("\n[Step 1/4] Fetching historical data...")
        raw_data = self.get_historical_data(hours=hours)
        if raw_data.empty:
            print("Error: Received empty DataFrame from get_historical_data()")
            return {"error": "Failed to fetch historical data"}
        print(f"Received data with shape: {raw_data.shape}")

        # 2. Prepare data
        print("\n[Step 2/4] Preprocessing data...")
        try:
            scaled_close = self.scalers['price'].fit_transform(raw_data[['close']])
            scaled_features = self.scalers['features'].fit_transform(raw_data[self.feature_columns[1:]])
            combined = np.concatenate([scaled_close, scaled_features], axis=1)
            X, y = self._create_sequences(combined)
            print(f"Created training sequences - X: {X.shape}, y: {y.shape}")
        except Exception as e:
            print(f"Data preparation failed: {str(e)}")
            return {"error": "Data preparation failed"}

        # 3. Model handling
        print("\n[Step 3/4] Handling model...")
        try:
            if self.model is None:
                input_shape = (X.shape[1], X.shape[2])
                print(f"Initializing model with input shape: {input_shape}")
                model_existed = os.path.exists(self.model_path)
                self.model = self.get_or_create_model(input_shape)
                
                # Only train if no model file existed
                if not model_existed:
                    print("\nNo existing model found - starting training...")
                    history = self.train_model(X, y)
                    print("Training completed")
                else:
                    print("Using existing trained model - skipping training")
            else:
                # Check if we should retrain (only for time-based retraining)
                should_retrain = (
                    self.last_training_time is not None and 
                    (datetime.now() - self.last_training_time) > timedelta(hours=retrain_threshold))
                    
                if should_retrain:
                    print("\nModel is older than threshold - starting retraining...")
                    history = self.train_model(X, y)
                    print("Retraining completed")
                else:
                    print("Using existing model without retraining")
                
        except Exception as e:
            print(f"Model handling failed: {str(e)}")
            return {"error": "Model initialization failed"}

        # 4. Generate predictions
        print("\n[Step 4/4] Generating predictions...")
        try:
            predictions = self.model.predict(X[-100:], batch_size=32)
            print("Predictions generated successfully")
            return self._generate_trading_signals(raw_data, predictions, self.scalers['price'])
        except Exception as e:
            print(f"Prediction generation failed: {str(e)}")
            return {"error": "Prediction failed"}

    def _generate_trading_signals(self, df, predictions, scaler):
        """Enhanced signal generation with more sophisticated rules"""
        print("\nGenerating trading signals...")
        df = df.copy()
        pred_reshaped = predictions.reshape(-1, 1)
        pred_reshaped = np.nan_to_num(pred_reshaped, nan=0, posinf=0, neginf=0)
        inv_preds = scaler.inverse_transform(pred_reshaped).flatten()
        
        # Store predictions
        df.loc[df.index[-len(inv_preds):], 'Prediction'] = inv_preds
        
        # Dynamic threshold based on volatility
        df['Threshold'] = 0.5 * df['ATR'] + 0.3 * df['Realized_Vol']
        
        # Initialize signals
        df['Signal'] = 0
        df['Confidence'] = 0.0
        
        # Enhanced long conditions
        long_cond = (
            (df['Prediction'] > df['close'] + df['Threshold']) &
            (df['RSI'] < 65) &
            (df['MACD'] > df['signal_line']) &
            (df['OBV'] > df['OBV'].rolling(20).mean()) &
            (df['close'] > df['VWAP']) &
            (df['close'] > df['EMA_20']) &
            (df['EMA_20'] > df['EMA_50']) &
            (df['ADX'] > 25) &
            (df['Realized_Vol'] < df['Realized_Vol'].rolling(20).mean() * 1.2) &
            (df['Volume_Ratio'] > 1.1)
        )
        
        # Enhanced short conditions
        short_cond = (
            (df['Prediction'] < df['close'] - df['Threshold']) &
            (df['RSI'] > 35) &
            (df['MACD'] < df['signal_line']) &
            (df['OBV'] < df['OBV'].rolling(20).mean()) &
            (df['close'] < df['VWAP']) &
            (df['close'] < df['EMA_20']) &
            (df['EMA_20'] < df['EMA_50']) &
            (df['ADX'] > 25) &
            (df['Realized_Vol'] < df['Realized_Vol'].rolling(20).mean() * 1.2) &
            (df['Volume_Ratio'] < 0.9)
        )
        
        # Apply signals with confidence scores
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1
        df['Confidence'] = abs(df['Prediction'] - df['close']) / df['Threshold']
        
        latest = df.iloc[-1]
        
        # Prepare comprehensive output
        signal_info = {
            'timestamp': latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name),
            'ticker': self.ticker,
            'signal': int(latest['Signal']),
            'confidence': float(latest['Confidence']),
            'predicted_price': float(latest['Prediction']),
            'current_price': float(latest['close']),
            'price_difference': float(latest['Prediction'] - latest['close']),
            'threshold': float(latest['Threshold']),
            'indicators': {
                'rsi': float(latest['RSI']),
                'macd_diff': float(latest['MACD']),
                'atr': float(latest['ATR']),
                'obv': float(latest['OBV']),
                'vwap': float(latest['VWAP']),
                'adx': float(latest['ADX']),
                'stoch_rsi': float(latest['Stoch_RSI']),
                'ema_20': float(latest['EMA_20']),
                'ema_50': float(latest['EMA_50']),
                'bb_width': float(latest['BB_Width']),
                'realized_vol': float(latest['Realized_Vol']),
                'volume_ratio': float(latest['Volume_Ratio'])
            },
            'model_info': {
                'path': self.model_path,
                'last_trained': self.last_training_time.isoformat() if self.last_training_time else None,
                'lookback_period': self.lookback
            }
        }
        
        print("\nSignal generation complete")
        print(f"Final signal: {signal_info['signal']} (confidence: {signal_info['confidence']:.2f})")
        return signal_info