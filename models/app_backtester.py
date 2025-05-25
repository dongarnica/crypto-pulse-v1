# app_backtester.py: CLI entry point for LSTM backtesting using Backtester and SignalGenerator
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm_model import SignalGenerator
from lstm_backtest import Backtester

def main():
    parser = argparse.ArgumentParser(description="Backtest LSTM-based crypto signal generator.")
    parser.add_argument('--ticker', type=str, default='BTC/USD', help='Crypto ticker (default: BTC/USD)')
    parser.add_argument('--hours', type=int, default=720, help='Number of hours to backtest (default: 720)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window for LSTM (default: 60)')
    parser.add_argument('--fee', type=float, default=0.001, help='Trading fee per trade (default: 0.001)')
    parser.add_argument('--initial-balance', type=float, default=100000, help='Initial balance (default: 100000)')
    args = parser.parse_args()

    sg = SignalGenerator(
        ticker=args.ticker,
        model_name=args.model,
        lookback=args.lookback
    )
    backtester = Backtester(
        signal_generator=sg,
        hours=args.hours,
        initial_balance=args.initial_balance,
        fee=args.fee
    )
    results = backtester.run(verbose=True)

if __name__ == "__main__":
    main()
