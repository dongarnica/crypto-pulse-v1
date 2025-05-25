# app_backtester.py: CLI entry point for LSTM backtesting using Backtester and SignalGenerator
import sys
import os
import logging
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lstm_model import SignalGenerator
from lstm_backtest import Backtester

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Backtest LSTM-based crypto signal generator.")
    parser.add_argument('--ticker', type=str, default='BTC/USD', help='Crypto ticker (default: BTC/USD)')
    parser.add_argument('--hours', type=int, default=720, help='Number of hours to backtest (default: 720)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback window for LSTM (default: 60)')
    parser.add_argument('--fee', type=float, default=0.001, help='Trading fee per trade (default: 0.001)')
    parser.add_argument('--initial-balance', type=float, default=100000, help='Initial balance (default: 100000)')
    args = parser.parse_args()

    logger.info(f"Starting backtest for {args.ticker} with {args.hours} hours of data")
    logger.info(f"Parameters: lookback={args.lookback}, fee={args.fee}, initial_balance=${args.initial_balance:,.2f}")

    try:
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
        
        logger.info("Running backtest...")
        results = backtester.run(verbose=True)
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
