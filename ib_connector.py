# ib_connector.py

import asyncio
import logging
from typing import Optional, List, Callable, Dict, Any

from ib_insync import IB, Contract, BarData, util, Ticker
import pandas as pd

from config import TWS_HOST, USE_PORT, CLIENT_ID, DEFAULT_STOCK_EXCHANGE, DEFAULT_CURRENCY

# Configure logging
# Use basicConfig in app.py for global settings, or configure here
# util.logToConsole(logging.DEBUG) # Use DEBUG for very detailed logs
util.logToConsole(logging.INFO)
logger = logging.getLogger(__name__)

class IBConnector:
    """Handles connection and data requests to Interactive Brokers TWS/Gateway."""

    def __init__(self, host: str = TWS_HOST, port: int = USE_PORT, client_id: int = CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._is_connected = False
        self._live_bars_subscriptions: Dict[str, Any] = {} # Store live bar subscriptions {symbol: bars_object}

        # Event Handlers Setup during __init__
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Centralized setup for event handlers."""
        # Clear existing handlers first to prevent duplicates if re-initialized
        self.ib.connectedEvent.clear()
        self.ib.disconnectedEvent.clear()
        self.ib.errorEvent.clear()
        # Add our handlers
        self.ib.connectedEvent += self._on_connect
        self.ib.disconnectedEvent += self._on_disconnect
        self.ib.errorEvent += self._on_error

    # --- Connection Management ---

    @property
    def is_connected(self) -> bool:
        """Check if the IB client is currently connected."""
        return self.ib.isConnected()

    async def connect(self) -> bool:
        """Establish connection to TWS/Gateway."""
        if self.is_connected:
            logger.info("Already connected.")
            return True
        try:
            logger.info(f"Attempting to connect to TWS/Gateway at {self.host}:{self.port} with ClientID {self.client_id}...")
            await self.ib.connectAsync(self.host, self.port, self.client_id, timeout=15)
            return self.is_connected
        except ConnectionRefusedError:
            logger.error(f"Connection refused. Is TWS/Gateway running and API configured on port {self.port}?")
            self._is_connected = False
            return False
        except asyncio.TimeoutError:
            logger.error(f"Connection attempt timed out after 15 seconds.")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during connection: {e}", exc_info=True)
            self._is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from TWS/Gateway."""
        # Cancel active subscriptions before disconnecting
        symbols_to_unsubscribe = list(self._live_bars_subscriptions.keys())
        for symbol in symbols_to_unsubscribe:
            self.unsubscribe_live_updates(symbol) # Use the proper method

        if self.is_connected:
            logger.info("Disconnecting from TWS/Gateway...")
            self.ib.disconnect()
            # Allow a moment for disconnection
            await asyncio.sleep(0.5)
        else:
            logger.info("Already disconnected or not connected.")
        self._is_connected = False # Explicitly set after disconnect call

    def _on_connect(self):
        """Callback executed when connection is established."""
        self._is_connected = True
        logger.info("Successfully connected to TWS/Gateway.")
        # Request market data type (1=live, 2=frozen, 3=delayed, 4=delayed frozen)
        try:
            self.ib.reqMarketDataType(3) # Defaulting to delayed. Change to 1 if needed.
            logger.info("Requested market data type 3 (Delayed).")
        except Exception as e:
             logger.error(f"Failed to request market data type: {e}")

    def _on_disconnect(self):
        """Callback executed when connection is lost."""
        if self._is_connected: # Only log if we thought we were connected
            logger.warning("Disconnected from TWS/Gateway (event received).")
        self._is_connected = False

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Optional[Contract] = None):
        """Callback for API errors."""
        # Filter specific informational codes if desired
        ignore_codes = {2104, 2106, 2108, 2158, 2109} # Market data farm connection messages etc.
        if errorCode in ignore_codes:
            logger.debug(f"IB Info (ReqId: {reqId}, Code: {errorCode}): {errorString}")
        elif errorCode < 1000:
             logger.warning(f"IB API Warning (ReqId: {reqId}, Code: {errorCode}): {errorString}")
        else:
            logger.error(f"IB API Error (ReqId: {reqId}, Code: {errorCode}): {errorString}")
            if errorCode == 1100 and contract is None: # Connectivity lost between API client and TWS/Gateway
                 logger.error("Connectivity to TWS/Gateway lost!")
                 self._is_connected = False # Ensure status reflects reality

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    # --- Contract Qualification ---

    async def qualify_contract(self, symbol: str, sec_type: str = 'STK',
                               exchange: str = DEFAULT_STOCK_EXCHANGE, currency: str = DEFAULT_CURRENCY) -> Optional[Contract]:
        """Qualifies a contract to get full details from IB."""
        if not self.is_connected:
            logger.error("Not connected. Cannot qualify contract.")
            return None

        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        logger.info(f"Qualifying contract: {symbol} ({sec_type} on {exchange})")
        try:
            qualified_contracts = await asyncio.wait_for(self.ib.qualifyContractsAsync(contract), timeout=10)
            if qualified_contracts:
                if len(qualified_contracts) > 1:
                    logger.warning(f"Multiple contracts qualified for {symbol}. Using the first one: {qualified_contracts[0]}")
                qualified_contract = qualified_contracts[0]
                logger.info(f"Contract qualified: {qualified_contract.localSymbol}, ConId: {qualified_contract.conId}")
                return qualified_contract
            else:
                logger.warning(f"Could not qualify contract for symbol: {symbol}")
                return None
        except asyncio.TimeoutError:
             logger.error(f"Timeout occurred while qualifying contract for {symbol}.")
             return None
        except Exception as e:
            logger.error(f"Error qualifying contract {symbol}: {e}", exc_info=True)
            return None

    # --- Data Fetching ---

    async def fetch_historical_data(self, contract: Contract, duration_str: str,
                                    bar_size_setting: str, what_to_show: str = 'TRADES',
                                    use_rth: bool = True) -> Optional[pd.DataFrame]:
        """Fetches historical bar data for a given contract (one-time request)."""
        if not self.is_connected:
            logger.error("Not connected. Cannot fetch historical data.")
            return None
        if not contract or not contract.conId:
             logger.error("Invalid or unqualified contract provided.")
             return None

        logger.info(f"Fetching historical data for {contract.localSymbol}: "
                    f"Duration='{duration_str}', BarSize='{bar_size_setting}', UseRTH={use_rth}, KeepUpToDate=False")

        try:
            # Added timeout for the request itself
            bars: List[BarData] = await asyncio.wait_for(
                self.ib.reqHistoricalDataAsync(
                    contract=contract,
                    endDateTime='',
                    durationStr=duration_str,
                    barSizeSetting=bar_size_setting,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False
                ),
                timeout=60 # 60 second timeout for historical data request
            )

            if not bars:
                logger.warning(f"No historical data returned for {contract.localSymbol} with the given parameters.")
                return pd.DataFrame() # Return empty DataFrame

            df = self._format_bars_to_dataframe(bars)
            if df is None: # Check for None explicitly, empty DataFrame is okay
                 logger.error(f"Historical data conversion to DataFrame failed for {contract.localSymbol}.")
                 return None # Indicate failure clearly

            logger.info(f"Successfully fetched {len(df)} bars for {contract.localSymbol}.")
            return df

        except asyncio.TimeoutError:
            logger.error(f"Timeout occurred while fetching historical data for {contract.localSymbol}.")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {contract.localSymbol}: {e}", exc_info=True)
            return None


    async def subscribe_live_updates(self, contract: Contract, bar_size_setting: str,
                                     update_callback: Callable[[List[BarData], bool], None],
                                     what_to_show: str = 'TRADES', use_rth: bool = True):
        """
        Subscribes to live bar updates (using reqRealTimeBars).
        Note: reqRealTimeBars only provides 5-second bars. Aggregation needs to happen externally.

        Args:
            contract: The qualified IB Contract object.
            bar_size_setting: *IGNORED* by reqRealTimeBars (always 5 seconds). Kept for potential future use.
            update_callback: Function to call when a new 5-second bar arrives.
                             Receives (list_containing_single_bar, has_new_bar_flag=True).
            what_to_show: Data type ('TRADES', 'MIDPOINT', etc.).
            use_rth: Whether to request data outside RTH.
        """
        if not self.is_connected:
            logger.error(f"Not connected. Cannot subscribe to live updates for {contract.localSymbol}.")
            return None
        if not contract or not contract.conId:
             logger.error(f"Invalid or unqualified contract provided for live updates.")
             return None
        if contract.localSymbol in self._live_bars_subscriptions:
            logger.warning(f"Already subscribed to live updates for {contract.localSymbol}. Re-subscribing.")
            self.unsubscribe_live_updates(contract.localSymbol) # Unsubscribe first

        logger.info(f"Subscribing to live 5-sec bars for {contract.localSymbol}: "
                    f"UseRTH={use_rth}")

        try:
            # Request real-time bars (5-second bars)
            # The returned 'ticker' object handles the subscription updates
            ticker = self.ib.reqRealTimeBars(
                contract=contract,
                barSize=5, # This is fixed for reqRealTimeBars
                whatToShow=what_to_show,
                useRTH=use_rth
                # realTimeBarsOptions=[] # Optional list
            )

            if not ticker:
                 logger.error(f"reqRealTimeBars call failed for {contract.localSymbol}.")
                 return None

            logger.info(f"Successfully initiated real-time bar subscription for {contract.localSymbol}.")

            # Store the 'ticker' object which manages the subscription
            self._live_bars_subscriptions[contract.localSymbol] = ticker

            # Attach the external callback to the update event of *this specific* subscription
            # The event fires for each new 5-second bar
            ticker.updateEvent += lambda t, hasNewBar: update_callback(t, hasNewBar) # Pass ticker (list of bars) and flag

            return ticker # Return the subscription object

        except Exception as e:
            logger.error(f"Error subscribing to live updates for {contract.localSymbol}: {e}", exc_info=True)
            # Clean up if subscription object was partially created
            if contract.localSymbol in self._live_bars_subscriptions:
                 del self._live_bars_subscriptions[contract.localSymbol]
            return None

    def unsubscribe_live_updates(self, symbol_or_localSymbol: str):
        """Unsubscribes from live bar updates for a symbol/localSymbol."""
        # Normalize symbol (though localSymbol might be more reliable if available)
        key = symbol_or_localSymbol.upper()
        # Find the correct key if localSymbol was used
        actual_key = None
        for k in self._live_bars_subscriptions.keys():
             if k.upper() == key:
                  actual_key = k
                  break
             # Attempt match on symbol part if localSymbol format like 'AAPL.ISLAND'
             if '.' in k and k.split('.')[0].upper() == key:
                  actual_key = k
                  break

        if actual_key and actual_key in self._live_bars_subscriptions:
            ticker_object = self._live_bars_subscriptions[actual_key]
            logger.info(f"Unsubscribing from live updates for {actual_key}...")
            try:
                # It's good practice to remove the event handler before cancelling
                # This is tricky as the lambda might not be easily removable by reference.
                # Relying on cancelRealTimeBars might be sufficient.
                # ticker_object.updateEvent.clear() # Or specific removal if possible

                self.ib.cancelRealTimeBars(ticker_object)
                del self._live_bars_subscriptions[actual_key]
                logger.info(f"Successfully unsubscribed from {actual_key}.")

            except Exception as e:
                 logger.error(f"Error during unsubscription for {actual_key}: {e}", exc_info=True)
        else:
            logger.warning(f"No active live subscription found for symbol/localSymbol: {key}")


    # --- Helper Methods ---

    def _format_bars_to_dataframe(self, bars: List[BarData]) -> Optional[pd.DataFrame]:
        """Converts a list of BarData objects to a pandas DataFrame."""
        if not bars:
            return pd.DataFrame()

        try:
            df = util.df(bars)
            if df is None or df.empty:
                 logger.warning("util.df returned None or empty DataFrame during conversion.")
                 return pd.DataFrame() # Return empty but valid DF

            df.rename(columns={'date': 'DateTime', 'open': 'Open', 'high': 'High',
                               'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                               'average': 'Average', 'barCount': 'BarCount'}, inplace=True)

            if 'DateTime' in df.columns:
                 # Ensure 'DateTime' is converted to datetime objects
                 df['DateTime'] = pd.to_datetime(df['DateTime'])
                 df.set_index('DateTime', inplace=True)

                 if df.index.tz is None:
                     logger.debug("Localizing naive DateTimeIndex to UTC.")
                     try:
                         df.index = df.index.tz_localize('UTC')
                     except TypeError: # Already localized
                          pass
                     except Exception as tz_err:
                         logger.error(f"Failed to localize timezone: {tz_err}. Index might be naive.")
                 # else: # Already tz-aware, maybe convert?
                 #     df.index = df.index.tz_convert('America/New_York') # Example conversion

            else:
                 logger.error("Critical: 'DateTime' column not found after util.df conversion.")
                 return None # Indicate critical failure

            cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
            existing_cols = [col for col in cols_to_keep if col in df.columns]
            if not all(col in existing_cols for col in ['Open', 'High', 'Low', 'Close']):
                logger.error("Missing core OHLC columns after conversion.")
                # Still return what we have, but log error
                # return None
                pass # Allow partial data if volume is missing, for example

            df = df[existing_cols]

            for col in existing_cols:
                if col != 'DateTime': # Index already handled
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Optional: Drop rows where core OHLC data is missing after coercion
            df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error formatting bars to DataFrame: {e}", exc_info=True)
            return None # Indicate failure