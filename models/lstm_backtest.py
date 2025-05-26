# backtest.py: Standalone backtesting module for LSTM-based crypto signal generator (for use with lstm_model_v2.py)
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

class Backtester:
    """
    Backtests a signal generator or model on historical data and outputs performance metrics.
    """
    def __init__(self, signal_generator=None, hours=720, initial_balance=100000, fee=0.001):
        """
        signal_generator: instance of AggressiveCryptoSignalGenerator (from lstm_model_v2.py)
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

        # Check if we have the required feature columns - if not, calculate them
        missing_features = [col for col in self.signal_generator.feature_columns if col not in raw_data.columns]
        if missing_features:
            self.logger.info(f"Missing features detected: {len(missing_features)} features. Calculating...")
            
            # For AggressiveCryptoSignalGenerator, we need to calculate features
            if hasattr(self.signal_generator, '_calculate_aggressive_features'):
                raw_data = self.signal_generator._calculate_aggressive_features(raw_data)
            elif hasattr(self.signal_generator, '_calculate_features'):
                raw_data = self.signal_generator._calculate_features(raw_data)
            else:
                self.logger.error("Signal generator doesn't have feature calculation method")
                return None
                
            # Check again for missing features
            missing_features = [col for col in self.signal_generator.feature_columns if col not in raw_data.columns]
            if missing_features:
                self.logger.error(f"Still missing features after calculation: {missing_features}")
                return None

        # Prepare features for LSTM
        self.logger.debug("Preparing features for LSTM model...")
        
        # Ensure all required columns exist before accessing them
        available_features = [col for col in self.signal_generator.feature_columns[1:] if col in raw_data.columns]
        if len(available_features) != len(self.signal_generator.feature_columns[1:]):
            missing_cols = [col for col in self.signal_generator.feature_columns[1:] if col not in raw_data.columns]
            self.logger.error(f"Missing feature columns: {missing_cols}")
            return None
            
        scaled_close = self.signal_generator.scalers['price'].fit_transform(raw_data[['close']])
        scaled_features = self.signal_generator.scalers['features'].fit_transform(
            raw_data[available_features]
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

        # Recompute threshold based on available features
        # Try to use ATR and volatility measures if available
        if 'ATR' in df.columns and 'Realized_Vol' in df.columns:
            # Original SignalGenerator features
            df['Threshold'] = 0.5 * df['ATR'] + 0.3 * df['Realized_Vol']
        elif 'ATR_Normalized' in df.columns:
            # AggressiveCryptoSignalGenerator features
            current_price = df['close'].mean()
            atr_value = df['ATR_Normalized'] * current_price
            # Use price volatility as a fallback
            price_volatility = df['close'].rolling(20).std().fillna(df['close'].std())
            df['Threshold'] = 0.5 * atr_value + 0.3 * price_volatility
        else:
            # Fallback: use price volatility
            price_volatility = df['close'].rolling(20).std().fillna(df['close'].std())
            df['Threshold'] = 0.01 * df['close'] + 0.3 * price_volatility  # 1% of price + volatility
        
        df['Signal'] = 0
        df['Confidence'] = 0.0

        # Ensure threshold is valid (no NaN or zero values)
        df['Threshold'] = df['Threshold'].fillna(df['close'] * 0.01)  # 1% of price as fallback
        df['Threshold'] = np.where(df['Threshold'] <= 0, df['close'] * 0.01, df['Threshold'])


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

        # Simple and correct trading simulation
        cash = self.initial_balance
        position = 0  # 0 = no position, 1 = long, -1 = short
        position_size = 0  # Amount of crypto held (positive for long, negative for short)
        entry_price = 0
        balance_history = [self.initial_balance]
        trade_log = []

        for i, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
            signal = int(row['Signal'])
            price = float(row['close'])
            
            # Calculate current portfolio value
            if position == 1:  # Long position
                portfolio_value = cash + (position_size * price)
            elif position == -1:  # Short position  
                # Short P&L = position_size * (entry_price - current_price)
                portfolio_value = cash + (position_size * (entry_price - price))
            else:
                portfolio_value = cash
            
            previous_position = position
            
            # Trading logic - only enter/exit on signal changes
            if position == 0 and signal != 0:
                # Enter position
                trade_amount = portfolio_value * 0.95  # Use 95% of portfolio
                fees = trade_amount * self.fee
                
                if signal == 1:  # Go long
                    position_size = (trade_amount - fees) / price
                    cash = portfolio_value - trade_amount
                    position = 1
                    entry_price = price
                    trade_log.append({
                        'type': 'enter_long', 
                        'price': price, 
                        'size': position_size,
                        'cash_after': cash,
                        'timestamp': row.name
                    })
                    
                elif signal == -1:  # Go short
                    position_size = (trade_amount - fees) / price
                    cash = portfolio_value - fees  # Only pay fees for short entry
                    position = -1
                    entry_price = price
                    trade_log.append({
                        'type': 'enter_short', 
                        'price': price, 
                        'size': position_size,
                        'cash_after': cash,
                        'timestamp': row.name
                    })
            
            elif position != 0 and (signal != previous_position or signal == 0):
                # Exit current position
                if position == 1:  # Close long
                    exit_value = position_size * price
                    fees = exit_value * self.fee
                    pnl = exit_value - fees - (position_size * entry_price)
                    cash += exit_value - fees
                    
                    trade_log.append({
                        'type': 'exit_long',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'size': position_size,
                        'pnl': pnl,
                        'cash_after': cash,
                        'timestamp': row.name
                    })
                    
                elif position == -1:  # Close short
                    # Short profit = size * (entry_price - exit_price)
                    pnl = position_size * (entry_price - price)
                    fees = abs(pnl) * self.fee if pnl > 0 else position_size * price * self.fee
                    pnl -= fees
                    cash += pnl
                    
                    trade_log.append({
                        'type': 'exit_short',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'size': position_size,
                        'pnl': pnl,
                        'cash_after': cash,
                        'timestamp': row.name
                    })
                
                position = 0
                position_size = 0
                entry_price = 0
                
                # Enter new position if signal indicates
                if signal != 0:
                    trade_amount = cash * 0.95
                    fees = trade_amount * self.fee
                    
                    if signal == 1:  # Go long
                        position_size = (trade_amount - fees) / price
                        cash -= trade_amount
                        position = 1
                        entry_price = price
                        trade_log.append({
                            'type': 'enter_long', 
                            'price': price, 
                            'size': position_size,
                            'cash_after': cash,
                            'timestamp': row.name
                        })
                        
                    elif signal == -1:  # Go short
                        position_size = (trade_amount - fees) / price
                        cash -= fees
                        position = -1
                        entry_price = price
                        trade_log.append({
                            'type': 'enter_short', 
                            'price': price, 
                            'size': position_size,
                            'cash_after': cash,
                            'timestamp': row.name
                        })
            
            # Update portfolio value for history
            if position == 1:
                current_value = cash + (position_size * price)
            elif position == -1:
                current_value = cash + (position_size * (entry_price - price))
            else:
                current_value = cash
                
            balance_history.append(current_value)

        # Close any remaining position at the end
        final_price = float(df.iloc[-1]['close'])
        if position == 1:
            exit_value = position_size * final_price
            fees = exit_value * self.fee
            pnl = exit_value - fees - (position_size * entry_price)
            cash += exit_value - fees
            trade_log.append({
                'type': 'exit_long',
                'entry_price': entry_price,
                'exit_price': final_price,
                'size': position_size,
                'pnl': pnl,
                'cash_after': cash,
                'timestamp': df.index[-1]
            })
        elif position == -1:
            pnl = position_size * (entry_price - final_price)
            fees = abs(pnl) * self.fee if pnl > 0 else position_size * final_price * self.fee
            pnl -= fees
            cash += pnl
            trade_log.append({
                'type': 'exit_short',
                'entry_price': entry_price,
                'exit_price': final_price,
                'size': position_size,
                'pnl': pnl,
                'cash_after': cash,
                'timestamp': df.index[-1]
            })

        balance = cash

        # Compute metrics with corrected calculations
        total_return = (balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        # Count completed trades (exit trades)
        exit_trades = [t for t in trade_log if t['type'] in ['exit_long', 'exit_short']]
        num_trades = len(exit_trades)
        
        if num_trades > 0:
            win_trades = [t for t in exit_trades if t['pnl'] > 0]
            lose_trades = [t for t in exit_trades if t['pnl'] <= 0]
            win_rate = len(win_trades) / num_trades
            
            # Calculate returns for Sharpe ratio
            returns = [t['pnl'] / self.initial_balance for t in exit_trades]
            
            # Calculate max drawdown from balance history
            balance_series = np.array(balance_history)
            peak = np.maximum.accumulate(balance_series)
            drawdown = (peak - balance_series) / peak
            max_drawdown = np.max(drawdown)
            
            # Sharpe ratio
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0
                
        else:
            win_trades = []
            lose_trades = []
            win_rate = 0
            max_drawdown = 0
            sharpe = 0
            returns = []

        # Add debugging info
        print(f"\n[DEBUG] Balance tracking:")
        print(f"Initial balance: ${self.initial_balance:,.2f}")
        print(f"Final balance: ${balance:,.2f}")
        print(f"Balance difference: ${balance - self.initial_balance:,.2f}")
        print(f"Total return calculation: ({balance:.2f} - {self.initial_balance}) / {self.initial_balance} = {total_return:.4f}")
        print(f"Number of trade_log entries: {len(trade_log)}")
        print(f"Exit trades count: {num_trades}")
        print(f"Win trades: {len(win_trades)}, Lose trades: {len(lose_trades)}")
        
        # Show sample trades for verification
        if len(trade_log) > 0:
            print("\n[DEBUG] Sample trades:")
            for i, trade in enumerate(trade_log[:4]):  # Show first 4 trades
                print(f"  Trade {i+1}: {trade}")

        if verbose:
            print("\n=== Backtest Results ===")
            print(f"Initial Balance: ${self.initial_balance:,.2f}")
            print(f"Final Balance: ${balance:,.2f}")
            print(f"Total Return: {total_return*100:.2f}%")
            print(f"Total Trades: {num_trades}")
            print(f"Winning Trades: {len(win_trades)}")
            print(f"Losing Trades: {len(lose_trades)}")
            print(f"Win Rate: {win_rate*100:.2f}%")
            print(f"Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            
            if num_trades > 0:
                avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
                avg_loss = np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
                if avg_loss != 0:
                    profit_factor = abs(avg_win * len(win_trades)) / abs(avg_loss * len(lose_trades))
                    print(f"Profit Factor: {profit_factor:.2f}")

        return {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'num_trades': num_trades,
            'winning_trades': len(win_trades),
            'losing_trades': len(lose_trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trade_log': trade_log,
            'returns': returns,
            'df': df
        }
