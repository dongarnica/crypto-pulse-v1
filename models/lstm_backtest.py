# backtest.py: Standalone backtesting module for LSTM-based crypto signal generator (for use with lstm_model.py)
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

class Backtester:
    """
    Backtests a signal generator or model on historical data and outputs performance metrics.
    """
    def __init__(self, signal_generator, hours=720, initial_balance=100000, fee=0.001):
        """
        signal_generator: instance of SignalGenerator (from lstm_model.py)
        hours: number of hours to backtest (default: 30 days)
        initial_balance: starting capital for simulation
        fee: trading fee per trade (fraction, e.g., 0.001 = 0.1%)
        """
        self.signal_generator = signal_generator
        self.hours = hours
        self.initial_balance = initial_balance
        self.fee = fee
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Backtester initialized: {hours}h, balance=${initial_balance:,.2f}, fee={fee*100:.3f}%")

    def run(self, verbose=True):
        self.logger.info(f"Starting backtest for {self.signal_generator.ticker}")
        
        # Fetch historical data and features
        self.logger.info(f"Fetching {self.hours} hours of historical data...")
        raw_data = self.signal_generator.get_historical_data(hours=self.hours)
        
        if raw_data.empty or len(raw_data) < self.signal_generator.lookback + 2:
            self.logger.error("Not enough data for backtest")
            print("[Backtester] Not enough data for backtest.")
            return None
            
        self.logger.info(f"Retrieved {len(raw_data)} data points for backtesting")

        # Prepare features for LSTM
        self.logger.debug("Preparing features for LSTM model...")
        scaled_close = self.signal_generator.scalers['price'].fit_transform(raw_data[['close']])
        scaled_features = self.signal_generator.scalers['features'].fit_transform(
            raw_data[self.signal_generator.feature_columns[1:]]
        )
        combined = np.concatenate([scaled_close, scaled_features], axis=1)
        X, y = self.signal_generator._create_sequences(combined)
        
        self.logger.info(f"Created {len(X)} sequences for prediction")

        # Load model (do not retrain)
        input_shape = (X.shape[1], X.shape[2])
        if self.signal_generator.model is None:
            self.logger.info("Loading LSTM model...")
            self.signal_generator.model = self.signal_generator.get_or_create_model(input_shape)

        # Predict for all available windows
        self.logger.info("Generating predictions...")
        preds = self.signal_generator.model.predict(X, batch_size=32)
        preds = preds.reshape(-1, 1)
        preds = np.nan_to_num(preds, nan=0, posinf=0, neginf=0)
        inv_preds = self.signal_generator.scalers['price'].inverse_transform(preds).flatten()

        # Align predictions with raw_data
        df = raw_data.iloc[-len(inv_preds):].copy()
        df['Prediction'] = inv_preds

        # Recompute all features needed for signal logic
        df['Threshold'] = 0.5 * df['ATR'] + 0.3 * df['Realized_Vol']
        df['Signal'] = 0
        df['Confidence'] = 0.0


        # Debug: print feature and prediction stats
        print("\n[DEBUG] Feature and Prediction Stats (last 10 rows):")
        print(df[['Prediction', 'close', 'Threshold']].tail(10))
        print("Prediction min/max/mean:", df['Prediction'].min(), df['Prediction'].max(), df['Prediction'].mean())
        print("Close min/max/mean:", df['close'].min(), df['close'].max(), df['close'].mean())
        print("Threshold min/max/mean:", df['Threshold'].min(), df['Threshold'].max(), df['Threshold'].mean())

        # Relaxed signal logic for debugging
        long_cond = (df['Prediction'] > df['close'] + df['Threshold'])
        short_cond = (df['Prediction'] < df['close'] - df['Threshold'])
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1
        df['Confidence'] = abs(df['Prediction'] - df['close']) / df['Threshold']

        # Print signal counts
        print('[DEBUG] Long signals:', (df['Signal'] == 1).sum())
        print('[DEBUG] Short signals:', (df['Signal'] == -1).sum())

        # Simulate trading
        balance = self.initial_balance
        position = 0  # +1 for long, -1 for short, 0 for flat
        entry_price = 0
        returns = []
        trade_log = []

        for i, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
            signal = int(row['Signal'])
            price = float(row['close'])
            if position == 0:
                if signal == 1:
                    position = 1
                    entry_price = price
                    trade_log.append({'type': 'long', 'entry': price, 'timestamp': row.name})
                elif signal == -1:
                    position = -1
                    entry_price = price
                    trade_log.append({'type': 'short', 'entry': price, 'timestamp': row.name})
            elif position == 1:
                if signal == -1:
                    # Close long, open short
                    pnl = (price - entry_price) * (1 - self.fee)
                    balance += pnl
                    returns.append(pnl / entry_price)
                    trade_log.append({'type': 'close_long', 'exit': price, 'timestamp': row.name, 'pnl': pnl})
                    position = -1
                    entry_price = price
                    trade_log.append({'type': 'short', 'entry': price, 'timestamp': row.name})
                elif signal == 0:
                    # Close long
                    pnl = (price - entry_price) * (1 - self.fee)
                    balance += pnl
                    returns.append(pnl / entry_price)
                    trade_log.append({'type': 'close_long', 'exit': price, 'timestamp': row.name, 'pnl': pnl})
                    position = 0
            elif position == -1:
                if signal == 1:
                    # Close short, open long
                    pnl = (entry_price - price) * (1 - self.fee)
                    balance += pnl
                    returns.append(pnl / entry_price)
                    trade_log.append({'type': 'close_short', 'exit': price, 'timestamp': row.name, 'pnl': pnl})
                    position = 1
                    entry_price = price
                    trade_log.append({'type': 'long', 'entry': price, 'timestamp': row.name})
                elif signal == 0:
                    # Close short
                    pnl = (entry_price - price) * (1 - self.fee)
                    balance += pnl
                    returns.append(pnl / entry_price)
                    trade_log.append({'type': 'close_short', 'exit': price, 'timestamp': row.name, 'pnl': pnl})
                    position = 0

        # If still in a position at the end, close it
        if position != 0:
            final_price = float(df.iloc[-1]['close'])
            if position == 1:
                pnl = (final_price - entry_price) * (1 - self.fee)
                balance += pnl
                returns.append(pnl / entry_price)
                trade_log.append({'type': 'close_long', 'exit': final_price, 'timestamp': df.index[-1], 'pnl': pnl})
            elif position == -1:
                pnl = (entry_price - final_price) * (1 - self.fee)
                balance += pnl
                returns.append(pnl / entry_price)
                trade_log.append({'type': 'close_short', 'exit': final_price, 'timestamp': df.index[-1], 'pnl': pnl})

        # Compute metrics
        total_return = (balance - self.initial_balance) / self.initial_balance
        num_trades = len([t for t in trade_log if t['type'] in ['close_long', 'close_short']])
        win_trades = [t for t in trade_log if t['type'] in ['close_long', 'close_short'] and t['pnl'] > 0]
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

        if verbose:
            print("\n=== Backtest Results ===")
            print(f"Final Balance: {balance:.2f}")
            print(f"Total Return: {total_return*100:.2f}%")
            print(f"Number of Trades: {num_trades}")
            print(f"Win Rate: {win_rate*100:.2f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")

        return {
            'final_balance': balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'trade_log': trade_log,
            'returns': returns,
            'df': df
        }
