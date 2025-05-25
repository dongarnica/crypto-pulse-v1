# lstm_train.py: CLI entry point for LSTM-based advanced signal generator
import sys
import os
import logging
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from lstm_model import SignalGenerator

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Run advanced LSTM-based crypto signal generator.")
    parser.add_argument('--ticker', type=str, default='BTC/USD', help='Crypto ticker (default: BTC/USD)')
    parser.add_argument('--hours', type=int, default=240, help='Number of hours of data to use (default: 240)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window for LSTM (default: 60)')
    parser.add_argument('--retrain-threshold', type=int, default=24, help='Hours before retraining (default: 24)')
    args = parser.parse_args()

    logger.info(f"Starting LSTM signal generation for {args.ticker}")
    logger.info(f"Parameters: hours={args.hours}, lookback={args.lookback}, retrain_threshold={args.retrain_threshold}")

    try:
        sg = SignalGenerator(
            ticker=args.ticker,
            model_name=args.model,
            lookback=args.lookback
        )
        
        logger.info("Generating signals...")
        result = sg.generate_signals(hours=args.hours, retrain_threshold=args.retrain_threshold)
        
        logger.info("Signal generation completed successfully")
        print("\n=== LSTM Signal Output ===")
        for k, v in result.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        logger.error(f"Signal generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

    # Example command to run this script with DOT/USD ticker:
    # python3 /workspaces/crypto-refactor/models/lstm_train.py --ticker ETH/USD --hours 720 --lookback 168 --retrain-threshold 12