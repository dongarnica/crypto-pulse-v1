#!/usr/bin/env python3
# filepath: /workspaces/crypto-refactor/examples/lstm_v2_example.py
"""
Example script for using the LSTM V2 model.
Shows how to use the AggressiveCryptoSignalGenerator class directly.
"""
import os
import sys
import json
from datetime import datetime

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the model
from models.lstm_model_v2 import AggressiveCryptoSignalGenerator

def main():
    """Main function to demonstrate LSTM v2 model usage."""
    
    print(f"{'='*80}")
    print(f"LSTM V2 MODEL - EXAMPLE USAGE")
    print(f"{'='*80}")
    
    # Create the model
    model = AggressiveCryptoSignalGenerator(
        ticker="BTC/USD",
        lookback=20
    )
    
    # Custom parameters
    model.volatility_multiplier = 1.8  # Less aggressive
    model.momentum_weight = 1.2       # Less momentum-focused
    model.risk_amplification = 1.5    # Less risk
    
    print(f"\n1. Generating signals for BTC/USD...")
    signal_data = model.generate_aggressive_signals(hours=24)
    
    # Print the signal results
    if 'error' in signal_data:
        print(f"❌ Error: {signal_data['error']}")
    else:
        print(f"\n📊 SIGNAL RESULTS:")
        print(f"Signal: {'🟢 LONG' if signal_data['signal'] == 1 else '🔴 SHORT' if signal_data['signal'] == -1 else '⚪ NEUTRAL'} ({signal_data['signal']})")
        print(f"Confidence: {signal_data['confidence']:.2f}")
        print(f"Current Price: ${signal_data['current_price']:.2f}")
        print(f"Predicted Price: ${signal_data['predicted_price']:.2f}")
        
        # Print trade recommendation
        if signal_data['trade_recommendation']:
            rec = signal_data['trade_recommendation']
            print(f"\nTRADE RECOMMENDATION:")
            print(f"Action: {rec['action']}")
            print(f"Position Size: {rec.get('position_size_pct', 0)}%")
            
            if rec.get('stop_loss'):
                print(f"Stop Loss: ${rec['stop_loss']:.2f}")
            
            if rec.get('take_profit'):
                print(f"Take Profit: ${rec['take_profit']:.2f}")
    
    # Save the signal to a file
    output_file = f"btc_usd_signal_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    output_path = os.path.join(project_root, "outputs", output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(signal_data, f, indent=2)
        print(f"\n✅ Signal saved to {output_path}")
    except Exception as e:
        print(f"⚠️ Error saving signal: {e}")
    
    print(f"\n2. Getting performance metrics...")
    metrics = model.get_performance_metrics()
    print(f"\nPERFORMANCE METRICS:")
    if 'message' in metrics:
        print(f"⚠️ {metrics['message']}")
    else:
        print(f"Total Signals: {metrics['total_signals']}")
        print(f"Long Signals: {metrics['long_signals']}")
        print(f"Short Signals: {metrics['short_signals']}")
        print(f"Average Confidence: {metrics['avg_confidence']:.2f}")
    
    print(f"\n{'='*80}")
    print(f"Example completed successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
