# ib_connector.py

import asyncio
import logging
from datetime import datetime
from typing import Optional, List

from ib_insync import IB, Contract, BarData, util
import pandas as pd

from config import TWS_HOST, USE_PORT, CLIENT_ID, DEFAULT_STOCK_EXCHANGE, DEFAULT_CURRENCY

# Configure logging
util.logToConsole(logging.INFO) # Change to DEBUG for more verbose logs
logger = logging.getLogger(__name__)

class IBConnector:
    """Handles connection and data requests to Interactive Brokers TWS/Gateway."""

    def __init__(self, host: str = TWS_HOST, port: int = USE_PORT, client_id: int = CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the IB client is currently connected."""
        return self._is_connected and self.ib.isConnected()

    async def connect(self) -> bool:
        """Establish connection to TWS/Gateway."""
        if self.is_connected:
            logger.info("Already connected.")
            return True
        try:
            logger.info(f"Connecting to TWS/Gateway at {self.host}:{self.port} with ClientID {self.client_id}...")
            await self.ib.connectAsync(self.host, self.port, self.client_id, timeout=10)
            if self.ib.isConnected():
                self._is_connected = True
                logger.info("Successfully connected to TWS/Gateway.")
                # Request market data type (1=live, 2=frozen, 3=delayed, 4=delayed frozen)
                # Use 3 for delayed data if not subscribed to market data
                self.ib.reqMarketDataType(3)
                return True
            else:
                logger.error("Connection attempt finished, but not connected.")
                self._is_connected = False
                return False
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            logger.error(f"Connection failed: {e}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during connection: {e}")
            self._is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self.is_connected:
            logger.info("Disconnecting from TWS/Gateway...")
            self.ib.disconnect()
            self._is_connected = False
            logger.info("Disconnected.")
        else:
            logger.info("Already disconnected.")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def qualify_contract(self, symbol: str, sec_type: str = 'STK',
                               exchange: str = DEFAULT_STOCK_EXCHANGE, currency: str = DEFAULT_CURRENCY) -> Optional[Contract]:
        """
        Qualifies a contract to get full details from IB.

        Args:
            symbol: The stock symbol (e.g., 'AAPL').
            sec_type: Security type (default: 'STK').
            exchange: Destination exchange (default: 'SMART').
            currency: Currency (default: 'USD').

        Returns:
            The qualified IB Contract object, or None if not found or error.
        """
        if not self.is_connected:
            logger.error("Not connected. Cannot qualify contract.")
            return None

        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        logger.info(f"Qualifying contract: {symbol}")
        try:
            qualified_contracts: List[Contract] = await self.ib.qualifyContractsAsync(contract)
            if qualified_contracts:
                logger.info(f"Contract qualified: {qualified_contracts[0]}")
                return qualified_contracts[0]
            else:
                logger.warning(f"Could not qualify contract for symbol: {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error qualifying contract {symbol}: {e}")
            return None

    async def fetch_historical_data(self, contract: Contract, duration_str: str,
                                    bar_size_setting: str, what_to_show: str = 'TRADES',
                                    use_rth: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetches historical bar data for a given contract.

        Args:
            contract: The qualified IB Contract object.
            duration_str: Duration string (e.g., '1 Y', '6 M', '30 D', '1 W').
            bar_size_setting: Bar size string (e.g., '1 min', '5 mins', '1 hour', '1 day').
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK', etc.).
            use_rth: Whether to fetch data during Regular Trading Hours only.

        Returns:
            A pandas DataFrame with OHLCV data, indexed by DateTime, or None on error.
        """
        if not self.is_connected:
            logger.error("Not connected. Cannot fetch historical data.")
            return None
        if not contract or not contract.conId:
             logger.error("Invalid or unqualified contract provided.")
             return None

        logger.info(f"Fetching historical data for {contract.symbol}: "
                    f"Duration='{duration_str}', BarSize='{bar_size_setting}', UseRTH={use_rth}")

        try:
            # Ensure event loop is running if called outside main async flow (less common now)
            # util.patchAsyncio() # Usually not needed if run via asyncio.run

            bars: List[BarData] = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime='',  # Empty string for current time
                durationStr=duration_str,
                barSizeSetting=bar_size_setting,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,  # 1 for datetime objects, 2 for strings
                keepUpToDate=False # Set to True later for live updates integrated with history
            )

            if not bars:
                logger.warning(f"No historical data returned for {contract.symbol} with the given parameters.")
                return pd.DataFrame() # Return empty DataFrame

            # Convert to pandas DataFrame
            df = util.df(bars)
            if df is None or df.empty:
                 logger.warning(f"Historical data conversion to DataFrame failed for {contract.symbol}.")
                 return pd.DataFrame()

            # --- Data Cleaning and Formatting ---
            # Set 'date' column to DateTime objects if not already (should be due to formatDate=1)
            # df['date'] = pd.to_datetime(df['date']) # Already datetime object

            # Rename 'date' column to 'DateTime' for clarity
            df.rename(columns={'date': 'DateTime'}, inplace=True)

            # Set DateTime as the index
            df.set_index('DateTime', inplace=True)

            # Ensure the index is timezone-aware (IB usually returns UTC)
            if df.index.tz is None:
                logger.warning("Timezone information missing from historical data. Assuming UTC.")
                df.index = df.index.tz_localize('UTC')
            else:
                # Optional: Convert to a specific timezone if needed, e.g., US/Eastern
                # df.index = df.index.tz_convert('America/New_York')
                pass # Keep original timezone (likely UTC)

            # Ensure correct data types
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            # 'average' and 'barCount' might also be present

            logger.info(f"Successfully fetched {len(df)} bars for {contract.symbol}.")
            return df[['open', 'high', 'low', 'close', 'volume']] # Return standard OHLCV

        except Exception as e:
            logger.error(f"Error fetching historical data for {contract.symbol}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return None

    # --- Placeholder Methods for Live Data (Phase 5) ---

    async def subscribe_live_data(self, contract: Contract, callback: callable):
        """Placeholder for subscribing to live bar updates."""
        if not self.is_connected:
            logger.error("Not connected. Cannot subscribe to live data.")
            return
        logger.warning("Live data subscription not yet implemented.")
        # Example structure (implement later):
        # self.ib.reqRealTimeBars(contract, 5, 'TRADES', False) # Request 5-sec bars
        # self.ib.barUpdateEvent += callback # Add callback to event

    async def unsubscribe_live_data(self, contract: Contract):
        """Placeholder for unsubscribing from live bar updates."""
        if not self.is_connected:
            logger.error("Not connected. Cannot unsubscribe from live data.")
            return
        logger.warning("Live data unsubscription not yet implemented.")
        # Example structure (implement later):
        # self.ib.cancelRealTimeBars(req) # Need to store the request object/ticker