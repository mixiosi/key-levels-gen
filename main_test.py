# main_test.py

import asyncio
from ib_connector import IBConnector
from config import DEFAULT_HIST_DURATION, DEFAULT_HIST_BARSIZE

async def run_test():
    """Connects to TWS, fetches historical data for a sample stock, and prints it."""

    connector = IBConnector() # Uses settings from config.py

    # Use async context manager for automatic connection/disconnection
    async with connector:
        if not connector.is_connected:
            print("Failed to connect to TWS/Gateway. Please ensure it's running and API is configured.")
            return

        # 1. Define the stock symbol to test
        test_symbol = 'AAPL' # Change to another symbol if needed

        # 2. Qualify the contract
        print(f"\nAttempting to qualify contract for {test_symbol}...")
        contract = await connector.qualify_contract(test_symbol)

        if not contract:
            print(f"Could not qualify contract for {test_symbol}. Exiting.")
            return

        print(f"Contract Qualified: {contract}")

        # 3. Fetch Historical Data
        print(f"\nFetching historical data for {test_symbol}...")
        hist_df = await connector.fetch_historical_data(
            contract=contract,
            duration_str=DEFAULT_HIST_DURATION, # e.g., '1 M'
            bar_size_setting=DEFAULT_HIST_BARSIZE # e.g., '1 day'
        )

        # 4. Display Results
        if hist_df is not None and not hist_df.empty:
            print(f"\nSuccessfully fetched {len(hist_df)} bars.")
            print("DataFrame Info:")
            hist_df.info()
            print("\nDataFrame Head:")
            print(hist_df.head())
            print("\nDataFrame Tail:")
            print(hist_df.tail())
        elif hist_df is not None and hist_df.empty:
             print("\nFetched data but the DataFrame is empty (no data for period or symbol issue?).")
        else:
            print("\nFailed to fetch historical data.")

    print("\nTest finished. Disconnected.")

if __name__ == "__main__":
    # Ensure TWS or IB Gateway is running and configured for API connections
    # before running this script.
    print("Starting connection test...")
    try:
        asyncio.run(run_test())
    except (KeyboardInterrupt, SystemExit):
        print("Test interrupted.")
    except Exception as e:
        print(f"An error occurred during the test: {e}")