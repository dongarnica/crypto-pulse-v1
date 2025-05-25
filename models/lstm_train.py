# lstm_train.py: CLI entry point for LSTM-based advanced signal generator
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from lstm_model import SignalGenerator

def main():
    parser = argparse.ArgumentParser(description="Run advanced LSTM-based crypto signal generator.")
    parser.add_argument('--ticker', type=str, default='BTC/USD', help='Crypto ticker (default: BTC/USD)')
    parser.add_argument('--hours', type=int, default=240, help='Number of hours of data to use (default: 240)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window for LSTM (default: 60)')
    parser.add_argument('--retrain-threshold', type=int, default=24, help='Hours before retraining (default: 24)')
    args = parser.parse_args()

    sg = SignalGenerator(
        ticker=args.ticker,
        model_name=args.model,
        lookback=args.lookback
    )
    result = sg.generate_signals(hours=args.hours, retrain_threshold=args.retrain_threshold)
    print("\n=== LSTM Signal Output ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

    # Example command to run this script with DOT/USD ticker:
    # python3 /workspaces/crypto-refactor/models/lstm_train.py --ticker ETH/USD --hours 720 --lookback 168 --retrain-threshold 12