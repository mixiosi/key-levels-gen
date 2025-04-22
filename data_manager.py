# data_manager.py

import pandas as pd
from typing import List, Optional, Tuple
from ib_insync import BarData, util, Ticker # Added Ticker
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages historical and live price data for a single financial instrument.
    Handles incoming 5-second bars and aggregates them if needed (basic implementation).
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        # Main DataFrame holding the bars (e.g., 1-min, 5-min, etc.)
        self.df: pd.DataFrame = self._create_empty_df()
        # Temporary storage for incoming 5-second bars to build larger bars
        self._current_agg_bar_data: Dict[str, any] = {}
        self._target_bar_size_seconds: Optional[int] = None # e.g., 60 for 1-min bars

        logger.info(f"DataManager initialized for symbol: {self.symbol}")

    def _create_empty_df(self) -> pd.DataFrame:
        """Creates a standardized empty DataFrame."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df.index = pd.to_datetime([]).tz_localize('UTC') # Ensure UTC DateTimeIndex
        df.index.name = 'DateTime'
        return df

    def set_aggregation_period(self, bar_size_setting: str):
        """
        Sets the target aggregation period (e.g., '1 min', '5 mins').
        If set, incoming 5-sec bars will be aggregated.
        If None or invalid, 5-sec bars might be stored directly (or ignored).
        """
        try:
            if 'min' in bar_size_setting:
                minutes = int(bar_size_setting.split()[0])
                self._target_bar_size_seconds = minutes * 60
            elif 'hour' in bar_size_setting:
                 hours = int(bar_size_setting.split()[0])
                 self._target_bar_size_seconds = hours * 3600
            elif 'day' in bar_size_setting:
                 # Daily aggregation from 5-sec bars is complex due to sessions;
                 # Best handled by requesting daily bars directly.
                 logger.warning("Daily aggregation from 5-sec bars not directly supported. Request daily bars instead.")
                 self._target_bar_size_seconds = None
            elif 'secs' in bar_size_setting:
                 self._target_bar_size_seconds = int(bar_size_setting.split()[0])
            else:
                 self._target_bar_size_seconds = None
                 logger.warning(f"Unsupported aggregation period: {bar_size_setting}. Disabling aggregation.")

            if self._target_bar_size_seconds:
                 logger.info(f"Aggregation period for {self.symbol} set to {self._target_bar_size_seconds} seconds.")
            self._current_agg_bar_data = {} # Reset aggregation on period change

        except Exception as e:
             logger.error(f"Error setting aggregation period '{bar_size_setting}': {e}")
             self._target_bar_size_seconds = None
             self._current_agg_bar_data = {}

    def load_historical_data(self, historical_df: Optional[pd.DataFrame]):
        """Loads the initial set of historical data."""
        if historical_df is None or historical_df.empty:
            logger.warning(f"Attempted to load empty or None historical data for {self.symbol}.")
            self.df = self._create_empty_df()
            return

        if not isinstance(historical_df.index, pd.DatetimeIndex):
            logger.error(f"Historical data for {self.symbol} must have a DatetimeIndex.")
            return
        if not historical_df.index.tz:
             logger.warning(f"Historical data for {self.symbol} lacks timezone. Assuming UTC.")
             try:
                 historical_df.index = historical_df.index.tz_localize('UTC')
             except Exception as e:
                  logger.error(f"Failed to localize historical data index to UTC: {e}")
                  return

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in historical_df.columns for col in required_cols):
            logger.error(f"Historical data for {self.symbol} missing required columns: {required_cols}. Has: {historical_df.columns}")
            return

        self.df = historical_df[required_cols].copy()
        self.df.sort_index(inplace=True)

        logger.info(f"Loaded {len(self.df)} historical bars for {self.symbol}. "
                    f"Range: {self.df.index.min()} to {self.df.index.max()}")
        # Clear aggregation state when new history is loaded
        self._current_agg_bar_data = {}


    def update_live_data(self, ticker: Ticker, has_new_bar: bool):
        """
        Processes incoming live 5-second bar data from a Ticker object.
        Aggregates bars if _target_bar_size_seconds is set.

        Args:
            ticker (Ticker): The Ticker object containing the live bar data (ticker.realtimeBars).
            has_new_bar (bool): Flag indicating if this is a new 5-second bar.
        """
        if not ticker or not ticker.realtimeBars:
            # logger.debug(f"No realtimeBars in ticker update for {self.symbol}.")
            return

        last_bar = ticker.realtimeBars[-1]
        # Ensure the bar's date is timezone-aware (should be UTC from IB)
        if isinstance(last_bar.date, datetime) and last_bar.date.tzinfo is None:
             bar_time = last_bar.date.replace(tzinfo='UTC') # Assume UTC if naive
        else:
             bar_time = pd.to_datetime(last_bar.date) # Convert if needed, keep timezone

        # --- Option 1: No Aggregation ---
        if self._target_bar_size_seconds is None or self._target_bar_size_seconds <= 5:
             self._append_bar_to_df(bar_time, last_bar)
             return

        # --- Option 2: Aggregation ---
        if not self._current_agg_bar_data:
            # Start of a new aggregation period
            self._start_new_aggregation_bar(bar_time, last_bar)
        else:
            # Continue existing aggregation bar
            current_bar_start_time = self._current_agg_bar_data['timestamp']
            time_diff = (bar_time - current_bar_start_time).total_seconds()

            if time_diff < self._target_bar_size_seconds:
                # Update the current aggregation bar
                self._update_aggregation_bar(last_bar)
            else:
                # Current aggregation period ended, finalize and append the bar
                self._finalize_and_append_aggregation_bar()
                # Start the next aggregation period with the current 5-sec bar
                self._start_new_aggregation_bar(bar_time, last_bar)

    def _append_bar_to_df(self, timestamp: pd.Timestamp, bar_data: BarData):
        """Appends or updates a single bar in the main DataFrame."""
        new_row = pd.DataFrame({
            'Open': [bar_data.open],
            'High': [bar_data.high],
            'Low': [bar_data.low],
            'Close': [bar_data.close],
            'Volume': [bar_data.volume]
        }, index=[timestamp])
        new_row.index.name = 'DateTime'

        # Ensure consistent timezone (UTC)
        if new_row.index.tz != self.df.index.tz:
            try:
                 new_row.index = new_row.index.tz_convert('UTC')
            except Exception as e:
                 logger.error(f"Error converting new row timezone to UTC: {e}")
                 return

        # Check if this timestamp already exists (update) or is new (append)
        if timestamp in self.df.index:
            self.df.loc[timestamp] = new_row.iloc[0]
            # logger.debug(f"Updated bar at {timestamp} for {self.symbol}")
        else:
            self.df = pd.concat([self.df, new_row])
            # Optional: Resort if appending out of order, but IB should send sequentially
            # self.df.sort_index(inplace=True)
            # logger.debug(f"Appended new bar at {timestamp} for {self.symbol}")

        # Optional: Limit DataFrame size
        # max_rows = 5000
        # if len(self.df) > max_rows:
        #     self.df = self.df.iloc[-max_rows:]


    def _start_new_aggregation_bar(self, bar_time: pd.Timestamp, first_bar: BarData):
        """Initializes the aggregation data for a new target bar."""
        # Align timestamp to the start of the aggregation interval
        interval_seconds = self._target_bar_size_seconds
        aligned_timestamp_unix = (bar_time.timestamp() // interval_seconds) * interval_seconds
        aligned_timestamp = pd.Timestamp.fromtimestamp(aligned_timestamp_unix, tz='UTC')

        self._current_agg_bar_data = {
            'timestamp': aligned_timestamp,
            'open': first_bar.open,
            'high': first_bar.high,
            'low': first_bar.low,
            'close': first_bar.close,
            'volume': first_bar.volume,
            'count': 1 # Number of 5-sec bars included
        }
        # logger.debug(f"Started new aggregation bar for {self.symbol} at {aligned_timestamp}")

    def _update_aggregation_bar(self, new_bar: BarData):
        """Updates the currently aggregating bar with a new 5-sec bar."""
        self._current_agg_bar_data['high'] = max(self._current_agg_bar_data['high'], new_bar.high)
        self._current_agg_bar_data['low'] = min(self._current_agg_bar_data['low'], new_bar.low)
        self._current_agg_bar_data['close'] = new_bar.close # Last close price
        self._current_agg_bar_data['volume'] += new_bar.volume
        self._current_agg_bar_data['count'] += 1

    def _finalize_and_append_aggregation_bar(self):
        """Finalizes the aggregated bar and appends it to the main DataFrame."""
        if not self._current_agg_bar_data:
            return

        agg_timestamp = self._current_agg_bar_data['timestamp']
        # Create a BarData-like structure (or directly a DataFrame row)
        final_bar_row = pd.DataFrame({
            'Open': [self._current_agg_bar_data['open']],
            'High': [self._current_agg_bar_data['high']],
            'Low': [self._current_agg_bar_data['low']],
            'Close': [self._current_agg_bar_data['close']],
            'Volume': [self._current_agg_bar_data['volume']]
        }, index=[agg_timestamp])
        final_bar_row.index.name = 'DateTime'

        # Append/update this finalized bar in the main DataFrame
        self._append_bar_to_df(agg_timestamp, final_bar_row.iloc[0]) # Pass the Series

        # logger.debug(f"Finalized aggregation bar for {self.symbol} at {agg_timestamp}")
        # Reset for the next bar (will be started by the next incoming 5-sec bar)
        self._current_agg_bar_data = {}

    def get_data(self) -> pd.DataFrame:
        """Returns a copy of the current aggregated DataFrame."""
        # If aggregation is active, ensure the potentially ongoing bar is handled
        # (This might be complex - for simplicity, maybe only return finalized bars,
        # or add the current partial bar if needed by the chart)
        # For now, just return the current state of df
        return self.df.copy()