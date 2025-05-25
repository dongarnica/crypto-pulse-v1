import asyncio
from data.crypto_data_client import CryptoMarketDataClient

def main():
    print("--- CryptoMarketDataClient Demo ---")
    client = CryptoMarketDataClient()

    # 1. Fetch historical bars for BTC/USD
    print("\n[1] Fetching historical bars for BTC/USD (1 day, 1-min bars)...")
    bars = client.get_historical_bars('BTC/USD', days=1)
    print(bars.head())

    # 2. Fetch CoinGecko data for BTC and ETH
    print("\n[2] Fetching CoinGecko data for BTC and ETH...")
    data = CryptoMarketDataClient.get_crypto_data()
    if data:
        for coin, info in data.items():
            print(f"\n{coin.capitalize()}:")
            print(f"  Current Price: ${info.get('usd', 'N/A'):,}")
            print(f"  24h Change: {info.get('usd_24h_change', 'N/A')}%")
            print(f"  Market Cap: ${info.get('usd_market_cap', 'N/A'):,}")
    else:
        print("[ERROR] No data returned from CoinGecko API.")

    # 3. (Optional) Subscribe to real-time trades for BTC-USD (Coinbase)
    print("\n[3] Subscribing to real-time trades for BTC-USD (Coinbase, 1 trade)...")
    async def run_realtime():
        event = asyncio.Event()
        task = None
        async def on_trade(trade):
            print(f"[REAL-TIME TRADE] {trade}")
            event.set()
        try:
            task = asyncio.create_task(client.get_crypto_data(BTC='BTC-USD', on_trade=on_trade, limit=1))
            await asyncio.wait_for(event.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("[INFO] No real-time trades received in 10 seconds.")
        finally:
            if task is not None:
                task.cancel()
                try:
                    await task
                except Exception:
                    pass

    # Comment out the next line if you do not want to run the real-time demo
    # asyncio.run(run_realtime())

if __name__ == "__main__":
    main()
