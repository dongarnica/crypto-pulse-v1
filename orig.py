import argparse
import json
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import pytz
import re
import ta
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(description="Crypto signal generator")
parser.add_argument("--ticker", "-t", default="BTC", help="Short ticker symbol, e.g., BTC, ETH")
args = parser.parse_args()
TICKER_SHORT = args.ticker.upper()
TICKER_SLASH = f'{TICKER_SHORT}/USD'
TICKER_STRIP = f'{TICKER_SHORT}USD'
TICKER_REALTIME = f'{TICKER_SHORT}-USD'
MODELTICKER = TICKER_SHORT.lower()
MODEL_NAME = f'ticker_model_{MODELTICKER}.keras'
MODEL_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(MODEL_DIR, exist_ok=True)

load_dotenv()
AI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_LLM = os.getenv("MODEL_LLM")
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

trading_prompt = PromptTemplate.from_file("trading_prompt.txt")

def get_crypto_sentiment(crypto_name: str, api_key: str) -> dict:
    prompt = (
        f"Analyze the current market sentiment for {crypto_name}. "
        "Return a sentiment score between -1 (very negative) and 1 (very positive), "
        "and provide a brief rationale for the score. "
        "Respond in JSON format: {\"sentiment_score\": float, \"rationale\": str}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    content = response.choices[0].message.content
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if match:
        json_str = match.group(1)
    else:
        json_str = content

    try:
        result = json.loads(json_str)
    except Exception:
        result = {
            "sentiment_score": None,
            "rationale": content
        }
    return result

class SignalGenerator:
    """Pure signal generation system with enhanced feature engineering"""

    def __init__(self, ticker=TICKER_SLASH):
        print(f"[SignalGenerator] Initializing for {ticker}")
        self.ticker = ticker
        self.model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        self.model = None
        self.scalers = {
            'price': RobustScaler(),
            'features': RobustScaler()
        }
        self.lookback = 60
        self.feature_columns = ['close', 'RSI', 'MACD', 'BB_Middle', 'ATR', 'OBV']
        self.data_client = CryptoHistoricalDataClient()
        print("[SignalGenerator] Data client initialized | Scalers created")

    def get_historical_data(self, days=365) -> pd.DataFrame:
        print(f"\n[SignalGenerator] Fetching {days} days of historical data")
        timeframe = TimeFrame.Minute
        start = datetime.now(pytz.UTC) - timedelta(days=days)
        print("[SignalGenerator] Requesting data from Alpaca API...")
        bars = self.data_client.get_crypto_bars(
            CryptoBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=timeframe,
                start=start,
                exchanges=["CBSE"]
            )
        ).df.droplevel(0)
        print(f"[SignalGenerator] Received {len(bars)} raw bars | Columns: {list(bars.columns)}")
        if bars.isna().any().any():
            nan_count = bars.isna().sum().sum()
            print(f"[SignalGenerator] Found {nan_count} NaN values in raw data | Forward filling...")
            bars = bars.ffill().bfill()
            print(f"[SignalGenerator] Remaining NaNs after cleaning: {bars.isna().sum().sum()}")
        return self._calculate_features(bars)

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[SignalGenerator] Calculating technical indicators")
        df = df.copy()
        def safe_div(a, b, fill_val=0):
            return np.where(b != 0, a / b, fill_val)
        try:
            print("[SignalGenerator] Calculating RSI...")
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            print("[SignalGenerator] Calculating MACD...")
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd_diff()
            df['macd_line'] = macd.macd()
            df['signal_line'] = macd.macd_signal()
            print("[SignalGenerator] Calculating Bollinger Bands...")
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Std'] = bollinger.bollinger_hband() - bollinger.bollinger_mavg()
            print("[SignalGenerator] Calculating ATR...")
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            print("[SignalGenerator] Calculating OBV...")
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['OBV'] = obv.on_balance_volume()
            print("[SignalGenerator] Calculating VWAP...")
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = safe_div(
                (df['volume'] * typical_price).cumsum(),
                df['volume'].cumsum()
            )
            print("[SignalGenerator] Calculating volatility features...")
            df['price_ratio'] = df['close'] / df['close'].shift(1)
            df['Log_Returns'] = np.log(df['price_ratio'].clip(lower=0.0000001))
            df['Realized_Vol'] = df['Log_Returns'].rolling(24, min_periods=1).std() * np.sqrt(365)
            df['Volume_Delta'] = df['volume'].diff()
        except Exception as e:
            print(f"[SignalGenerator] Feature calculation error: {str(e)}")
        print("[SignalGenerator] Cleaning final NaNs...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_nans = df.isna().sum().sum()
        df = df.ffill().bfill()
        final_nans = df.isna().sum().sum()
        print(f"[SignalGenerator] NaNs removed: {initial_nans - final_nans} | Remaining: {final_nans}")
        return df

    def create_model(self, input_shape: tuple) -> Sequential:
        print("\n[SignalGenerator] Building LSTM model")
        model = Sequential([
            Input(shape=input_shape),
            BatchNormalization(),
            LSTM(128, return_sequences=True, recurrent_dropout=0.1),
            Dropout(0.2),
            LSTM(64, return_sequences=False, recurrent_dropout=0.1),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        print("[SignalGenerator] Model architecture created")
        return model

    def get_or_create_model(self, input_shape):
        if os.path.exists(self.model_path):
            print(f"[SignalGenerator] Loading existing model from {self.model_path}")
            model = load_model(self.model_path)
        else:
            print("[SignalGenerator] No saved model found. Building a new model.")
            model = self.create_model(input_shape)
        return model

    def generate_signals(self, days=365) -> dict:
        print("\n[SignalGenerator] Starting signal generation pipeline")
        try:
            raw_data = self.get_historical_data(days)
            print(f"[SignalGenerator] Data shape after feature engineering: {raw_data.shape}")
            print("\n[SignalGenerator] Normalizing data")
            scaled_close = self.scalers['price'].fit_transform(raw_data[['close']])
            scaled_features = self.scalers['features'].fit_transform(raw_data[self.feature_columns[1:]])
            combined = np.concatenate([scaled_close, scaled_features], axis=1)
            print(f"[SignalGenerator] Combined data shape: {combined.shape}")
            print("\n[SignalGenerator] Creating training sequences")
            X, y = self._create_sequences(combined)
            input_shape = (X.shape[1], X.shape[2])
            print(f"[SignalGenerator] Training data shape - X: {X.shape}, y: {y.shape}")
            if not self.model:
                self.model = self.get_or_create_model(input_shape)
                if not os.path.exists(self.model_path):
                    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
                    self.model.compile(optimizer=optimizer, loss='mse')
                    print("\n[SignalGenerator] Training model")
                    history = self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
                    print(f"[SignalGenerator] Final training loss: {history.history['loss'][-1]:.4f}")
                    self.model.save(self.model_path)
                    print(f"[SignalGenerator] Model saved to {self.model_path}")
            print("\n[SignalGenerator] Making predictions")
            predictions = self.model.predict(X[-100:])
            print(f"[SignalGenerator] Prediction shape: {predictions.shape}")
            print("\n[SignalGenerator] Generating trading signals")
            signals = self._generate_trading_signals(raw_data, predictions, self.scalers['price'])
            print("[SignalGenerator] Signal generation complete")
            print(f"[SignalGenerator] Signals: {signals}")
            return signals
        except Exception as e:
            print(f"[SignalGenerator] Pipeline error: {str(e)}")
            raise

    def _create_sequences(self, data: np.ndarray) -> tuple:
        print(f"[SignalGenerator] Creating sequences with lookback={self.lookback}")
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i, 0])
        print(f"[SignalGenerator] Created {len(X)} sequences")
        return np.array(X), np.array(y)

    def _generate_trading_signals(self, df: pd.DataFrame, predictions: np.ndarray, scaler: RobustScaler) -> dict:
        print("[SignalGenerator] Starting signal generation process...")

        df = df.copy()
        pred_reshaped = predictions.reshape(-1, 1)
        if np.isnan(pred_reshaped).any() or np.isinf(pred_reshaped).any():
            bad_values = np.isnan(pred_reshaped).sum() + np.isinf(pred_reshaped).sum()
            print(f"[SignalGenerator] Cleaning {bad_values} invalid prediction values")
            pred_reshaped = np.nan_to_num(pred_reshaped, nan=0, posinf=0, neginf=0)
        inv_preds = scaler.inverse_transform(pred_reshaped).flatten()
        try:
            df.loc[df.index[-len(inv_preds):], 'Prediction'] = inv_preds
            print(f"[SignalGenerator] Last prediction value after inversion: {inv_preds[-1]:.2f}")
        except Exception as e:
            print(f"[SignalGenerator] Prediction assignment error: {str(e)}")

        print("[SignalGenerator] Calculating dynamic thresholds (0.4*ATR + 0.2*BB_Std)...")
        df['Threshold'] = 0.4 * df['ATR'] + 0.2 * df['BB_Std']
        print(f"[SignalGenerator] Latest threshold value: {df['Threshold'].iloc[-1]:.4f}")

        df['Signal'] = 0

        try:
            print("[SignalGenerator] Evaluating long conditions...")
            latest = df.iloc[-1]
            print(f"  [Long] Prediction: {latest['Prediction']:.2f}, Close: {latest['close']:.2f}, Threshold: {latest['Threshold']:.2f}")
            print(f"  [Long] RSI: {latest['RSI']:.2f} (must be < 70)")
            print(f"  [Long] MACD: {latest['MACD']:.4f} (must be > 0)")
            print(f"  [Long] OBV: {latest['OBV']:.2f} vs OBV_MA10: {df['OBV'].rolling(10).mean().iloc[-1]:.2f} (must be >)")
            print(f"  [Long] Close: {latest['close']:.2f} vs VWAP: {latest['VWAP']:.2f} (must be >)")
            print(f"  [Long] Close: {latest['close']:.2f} vs BB_Middle: {latest['BB_Middle']:.2f} (must be >)")
            print(f"  [Long] Realized_Vol: {latest['Realized_Vol']:.4f} vs Vol_MA10*1.5: {df['Realized_Vol'].rolling(10).mean().iloc[-1]*1.5:.4f} (must be <)")
            print(f"  [Long] Volume_Delta: {latest['Volume_Delta']:.2f} (must be > 0)")

            long_cond = (
                (df['Prediction'] > df['close'] + df['Threshold']) &
                (df['RSI'] < 70) &
                (df['MACD'] > 0) &
                (df['OBV'] > df['OBV'].rolling(10).mean()) &
                (df['close'] > df['VWAP']) &
                (df['close'] > df['BB_Middle']) &
                (df['Realized_Vol'] < df['Realized_Vol'].rolling(10).mean() * 1.5) &
                (df['Volume_Delta'] > 0)
            )
            df.loc[long_cond, 'Signal'] = 1
            print(f"[SignalGenerator] Long signals found: {long_cond.sum()}")
        except Exception as e:
            print(f"[SignalGenerator] Long signal error: {str(e)}")

        try:
            print("[SignalGenerator] Evaluating short conditions...")
            print(f"  [Short] Prediction: {latest['Prediction']:.2f}, Close: {latest['close']:.2f}, Threshold: {latest['Threshold']:.2f}")
            print(f"  [Short] RSI: {latest['RSI']:.2f} (must be > 30)")
            print(f"  [Short] MACD: {latest['MACD']:.4f} (must be < 0)")
            print(f"  [Short] OBV: {latest['OBV']:.2f} vs OBV_MA10: {df['OBV'].rolling(10).mean().iloc[-1]:.2f} (must be <)")
            print(f"  [Short] Close: {latest['close']:.2f} vs VWAP: {latest['VWAP']:.2f} (must be <)")
            print(f"  [Short] Close: {latest['close']:.2f} vs BB_Middle: {latest['BB_Middle']:.2f} (must be <)")
            print(f"  [Short] Realized_Vol: {latest['Realized_Vol']:.4f} vs Vol_MA10*1.5: {df['Realized_Vol'].rolling(10).mean().iloc[-1]*1.5:.4f} (must be <)")
            print(f"  [Short] Volume_Delta: {latest['Volume_Delta']:.2f} (must be < 0)")

            short_cond = (
                (df['Prediction'] < df['close'] - df['Threshold']) &
                (df['RSI'] > 30) &
                (df['MACD'] < 0) &
                (df['OBV'] < df['OBV'].rolling(10).mean()) &
                (df['close'] < df['VWAP']) &
                (df['close'] < df['BB_Middle']) &
                (df['Realized_Vol'] < df['Realized_Vol'].rolling(10).mean() * 1.5) &
                (df['Volume_Delta'] < 0)
            )
            df.loc[short_cond, 'Signal'] = -1
            print(f"[SignalGenerator] Short signals found: {short_cond.sum()}")
        except Exception as e:
            print(f"[SignalGenerator] Short signal error: {str(e)}")

        try:
            latest = df.iloc[-1]
            print("\n[SignalGenerator] === Final Signal Summary ===")
            print(f"  Current Price: {latest['close']:.2f}")
            print(f"  Predicted Price: {latest['Prediction']:.2f}")
            print(f"  Signal: {latest['Signal']}")
            print(f"  RSI: {latest['RSI']:.2f}, ATR: {latest['ATR']:.2f}")
            print(f"  MACD: {latest['macd_line']:.4f}, Signal Line: {latest['signal_line']:.4f}")
            print(f"  BB_Middle: {latest['BB_Middle']:.2f}, BB_Std: {latest['BB_Std']:.2f}")
            print(f"  OBV: {latest['OBV']:.2f}, VWAP: {latest['VWAP']:.2f}")
            print(f"  Price Ratio: {latest['price_ratio']:.4f}")
            print(f"  Log Returns: {latest['Log_Returns']:.4f}")
            print(f"  Realized Vol: {latest['Realized_Vol']:.4f}")
            print(f"  Volume Delta: {latest['Volume_Delta']:.2f}")

            return {
                'timestamp': latest.name.isoformat(),
                'ticker': self.ticker,
                'signal': int(latest['Signal']),
                'predicted_price': float(latest['Prediction']),
                'current_price': float(latest['close']),
                'confidence': abs(float(latest['Prediction'] - latest['close'])),
                'rsi': float(latest['RSI']),
                'atr': float(latest['ATR']),
                'macd_line': float(latest['macd_line']),
                'signal_line': float(latest['signal_line']),
                'bb_middle': float(latest['BB_Middle']),
                'bb_std': float(latest['BB_Std']),
                'obv': float(latest['OBV']),
                'vwap': float(latest['VWAP']),
                'price_ratio': float(latest['price_ratio']),
                'log_returns': float(latest['Log_Returns']),
                'realized_vol': float(latest['Realized_Vol']),
                'volume_delta': float(latest['Volume_Delta'])
            }
        except Exception as e:
            print(f"[SignalGenerator] Signal compilation error: {str(e)}")
            return {
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'ticker': self.ticker,
                'signal': 0,
                'error': str(e)
            }

def execute_trade(signal: dict, llm_response: dict):
    print("\n[Trade] === Executing Trade ===")
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = trading_client.get_account()
    cash_available = float(account.cash)
    print(f"[Trade] Cash available: {cash_available}")
    action = llm_response.get("action", "hold").lower()
    print(f"[Trade] LLM action: {action}")
    print(f"[Trade] LLM response: {llm_response}")

    position_size = cash_available * 0.02
    current_price = signal["current_price"]
    desired_quantity = round(position_size / current_price, 4)
    
    print("\n[Trade] ===== Trade Execution Variables =====")
    print(f"[Trade] Account cash: {cash_available}")
    print(f"[Trade] Current price: {current_price}")
    print(f"[Trade] Position size (2%): {position_size}")
    print(f"[Trade] Action: {action}")

    if action == "hold":
        print("[Trade] No trade execution - hold position")
        return

    if action == "sell":
        try:
            position = trading_client.get_open_position(TICKER_STRIP)
            ticker_available = float(position.qty)
        except Exception:
            ticker_available = 0.0

        quantity = min(desired_quantity, ticker_available)
        if quantity < 0.0001:
            print(f"[Trade] Not enough {TICKER_SHORT} to sell. Skipping order.")
            return
    else:
        max_affordable_qty = round(cash_available / current_price, 4)
        quantity = min(desired_quantity, max_affordable_qty)
        if quantity < 0.0001:
            print(f"[Trade] Not enough cash to buy {TICKER_SHORT}. Skipping order.")
            return

    order_params = {
        "symbol": TICKER_STRIP,
        "qty": quantity,
        "side": OrderSide.BUY if action == "buy" else OrderSide.SELL,
        "time_in_force": TimeInForce.GTC,
        "order_class": "simple",
    }
    print(f"[Trade] Order parameters: {order_params}")
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(**order_params)
        )
        print(f"[Trade] Order submitted successfully: {order.id}")
        print(f"[Trade] Details: {quantity} {TICKER_SHORT} @ {current_price}")
    except Exception as e:
        if "insufficient balance" in str(e).lower():
            print(f"[Trade] Order failed: Insufficient balance. Available: ${cash_available:.2f}, Needed: ${quantity * current_price:.2f}")
        else:
            print(f"[Trade] Order failed: {str(e)}")

def main():
    print("\n[Main] ===== Starting Signal Generation Process =====")
    generator = SignalGenerator()
    try:
        signals = generator.generate_signals()
        print(f"[Main] Signals generated: {signals}")

        sentiment = get_crypto_sentiment(TICKER_SLASH, PERPLEXITY_API_KEY)
        print(f"[Main] Crypto sentiment: {sentiment}")

        stop_loss = round(signals['atr'] * 1.5, 2)
        take_profit = round(signals['atr'] * 3, 2)
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        cash_available = float(account.cash)
        position_size = f"{round(cash_available * 0.02, 2)} USD"
        sentiment_score = sentiment.get("sentiment_score")
        sentiment_rationale = sentiment.get("rationale")

        # Pass all required prompt variables, including all technicals and quantitative metrics
        inputs = {
            "ticker": TICKER_SLASH,
            "current_price": signals['current_price'],
            "predicted_price": signals['predicted_price'],
            "rsi": signals['rsi'],
            "macd_line": signals['macd_line'],
            "signal_line": signals['signal_line'],
            "bb_middle": round(signals['bb_middle'], 2),
            "bb_std": round(signals['bb_std'], 2),
            "atr": signals['atr'],
            "obv": signals['obv'],
            "vwap": signals['vwap'],
            "price_ratio": signals['price_ratio'],
            "volume_delta": signals['volume_delta'],
            "log_returns": signals['log_returns'],
            "realized_vol": signals['realized_vol'],
            "signal": signals['signal'],
            "confidence": signals['confidence'],
            "timestamp": signals['timestamp'],
            "sentiment_score": sentiment_score,
            "sentiment_rationale": sentiment_rationale,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        formatted_prompt = trading_prompt.format(**inputs)
        print("\n[Main] ===== Final Trading Decision Prompt =====")
        print(formatted_prompt)
        print(f"MACD: {signals['macd_line']:.4f} vs {signals['signal_line']:.4f}")

        llm = ChatOpenAI(
            openai_api_key=AI_API_KEY,
            temperature=0,
            model_name=MODEL_LLM
        )
        trading_chain = LLMChain(
            llm=llm,
            prompt=trading_prompt,
            verbose=True
        )
        print("[Main] Invoking LLM for trading decision...")
        response = trading_chain.invoke(input=inputs)

        print("\n[Main] === Trading Decision (LLM Output) ===")
        print("[Main] LLM raw output:", response['text'])

        mt_tz = pytz.timezone('America/Denver')
        now_mt = datetime.now(mt_tz)
        print("[Main] Current Mountain Time:", now_mt.strftime("%Y-%m-%d %H:%M:%S %Z %z"))
        print(TICKER_SLASH)

        if response['text']:
            try:
                llm_decision = json.loads(response['text'])
                print("\n[Main] ===== Trade Decision Variables =====")
                for k, v in llm_decision.items():
                    print(f"[Main] {k:>15}: {v}")
                execute_trade(signals, llm_decision)
            except json.JSONDecodeError:
                print("[Main] Failed to parse LLM response - using local calculations")
                execute_trade(signals, {
                    "action": "hold",
                    "rationale": "Fallback: Invalid LLM response",
                    "risk_parameters": {
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "position_size": position_size
                    }
                })
    except Exception as e:
        print(f"[Main] Main process failed: {str(e)}")

if __name__ == "__main__":
    print("[Main] Starting.")
    main()