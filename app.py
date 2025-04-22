# app.py

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import asyncio
import logging
from datetime import datetime, timezone # Added timezone
from typing import Dict, Any # Added typing hints

# Import our modules
from config import APP_TITLE, DEFAULT_SYMBOL, DEFAULT_HIST_DURATION, DEFAULT_HIST_BARSIZE, LIVE_UPDATE_INTERVAL_MS
from ib_connector import IBConnector
from data_manager import DataManager
from charting import create_candlestick_chart, add_sr_lines_to_chart
from sr_calculator import calculate_sr_levels_peaks

# --- Global Variables & Setup ---
# Setup basic logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' # Changed date format
)
logger = logging.getLogger(__name__)

# Initialize Dash App with Bootstrap theme and suppress callback exceptions initially
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True # Useful if callbacks are created dynamically or target initially non-existent elements
)
app.title = APP_TITLE

# --- Define Dropdown Options ---
duration_options = [
    {'label': '5 Days', 'value': '5 D'},
    {'label': '10 Days', 'value': '10 D'},
    {'label': '1 Month', 'value': '1 M'},
    {'label': '3 Months', 'value': '3 M'},
    {'label': '6 Months', 'value': '6 M'},
    {'label': '1 Year', 'value': '1 Y'},
    {'label': '2 Years', 'value': '2 Y'},
    {'label': '5 Years', 'value': '5 Y'},
    {'label': '10 Years', 'value': '10 Y'},
]

barsize_options = [
    {'label': '1 min', 'value': '1 min'},
    {'label': '5 mins', 'value': '5 mins'},
    {'label': '15 mins', 'value': '15 mins'},
    {'label': '30 mins', 'value': '30 mins'},
    {'label': '1 hour', 'value': '1 hour'},
    {'label': '4 hours', 'value': '4 hours'},
    {'label': '1 day', 'value': '1 day'},
    {'label': '1 week', 'value': '1 week'},
    # Note: IB has limitations on combining very small barsizes with long durations
    # e.g., '1 min' bars have limited history depth (often < 1 week)
]

# --- Dash App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12)),

    dbc.Row([
        dbc.Col([
            dbc.Label("Stock Symbol:", html_for='symbol-input'),
            dbc.Input(id='symbol-input', type='text', value=DEFAULT_SYMBOL, debounce=True),
        ], width=2),
        dbc.Col([
            dbc.Label("History Duration:", html_for='duration-input'),
            dcc.Dropdown(
                id='duration-input', options=duration_options,
                value=DEFAULT_HIST_DURATION, clearable=False, searchable=False
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("Bar Size:", html_for='barsize-input'),
            dcc.Dropdown(
                id='barsize-input', options=barsize_options,
                value=DEFAULT_HIST_BARSIZE, clearable=False, searchable=False
            ),
        ], width=2),
        dbc.Col([
            # Add a Div for spacing alignment with labels above
            html.Div(style={'marginTop': '31px'}), # Adjust px value for alignment
            dbc.Button("Fetch & Draw", id='fetch-button', n_clicks=0, color="primary"),
        ], width=2),
    ], className="mb-3 align-items-start"), # Use margin bottom and align items

    dbc.Row(dbc.Col(dcc.Loading(
        id="loading-chart",
        children=[dcc.Graph(id='price-chart')],
        type="circle",
    ), width=12)),

    dbc.Row(dbc.Col(html.Div(id='status-output', style={'marginTop': '15px'}), width=12)),

    # --- Hidden Stores for State ---
    # Stores the currently displayed symbol
    dcc.Store(id='current-symbol-store'),
    # Stores the fetched data as JSON (use with caution for very large datasets)
    dcc.Store(id='price-data-store'),

    # Interval component for future live updates
    dcc.Interval(
        id='interval-component',
        interval=LIVE_UPDATE_INTERVAL_MS,
        n_intervals=0,
        disabled=True # Keep disabled until live functionality is implemented
    )
], fluid=True)

# --- Dash Callbacks ---

@app.callback(
    [Output('price-chart', 'figure'),
     Output('status-output', 'children'),
     Output('current-symbol-store', 'data'),
     Output('price-data-store', 'data'),
     Output('interval-component', 'disabled')], # Output to control interval timer
    [Input('fetch-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('duration-input', 'value'),
     State('barsize-input', 'value')],
    prevent_initial_call=True # Don't run this callback when the app loads initially
)
def update_chart_on_fetch(n_clicks: int, symbol: str, duration: str, bar_size: str) -> tuple:
    """
    Callback triggered when the fetch button is clicked.
    Connects to IB, fetches historical data (including extended hours),
    calculates S/R, creates chart, and stores data.
    """
    triggered_id = dash.callback_context.triggered_id
    if not triggered_id or triggered_id != 'fetch-button':
        return no_update # Should not happen with prevent_initial_call=True

    if not symbol:
        # Return an empty chart and a user-friendly message
        empty_fig = create_candlestick_chart(pd.DataFrame(), "No Symbol", title="Enter a Symbol")
        return empty_fig, "Please enter a stock symbol.", no_update, no_update, True

    symbol = symbol.strip().upper()
    start_time = datetime.now(timezone.utc) # Use timezone-aware start time
    status_messages = [f"Request received for {symbol} at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}..."]
    logger.info(f"Fetching data for {symbol}, Duration: {duration}, Bar Size: {bar_size}, Including Extended Hours")

    # Initialize return values for error cases
    error_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error Loading {symbol}")
    error_status = ""

    try:
        # --- Connect and Fetch Data ---
        # Create connector instance within the callback context for simplicity now.
        # For live updates, this needs to be persistent.
        connector = IBConnector()

        # Run async connection and data fetching within the sync callback context
        # This blocks the Dash worker until complete! Consider background tasks for complex apps.
        async def fetch_async():
            # Ensure previous connection is closed if any (unlikely here, but good practice)
            # await connector.disconnect() # Avoid if connector is truly per-callback instance
            if not await connector.connect():
                raise ConnectionError("Failed to connect to TWS/Gateway. Check connection and API settings.")

            contract = await connector.qualify_contract(symbol)
            if not contract:
                 raise ValueError(f"Could not qualify contract for {symbol}. Is it a valid symbol?")

            # Fetch historical data including extended hours (use_rth=False)
            df = await connector.fetch_historical_data(
                contract=contract,
                duration_str=duration,
                bar_size_setting=bar_size,
                use_rth=False # Include pre/post market data
            )
            # Disconnect after fetching
            await connector.disconnect()
            return df

        # Execute the async fetching function
        df_hist = asyncio.run(fetch_async())

        # --- Process Data & Create Chart ---
        if df_hist is None:
            # This indicates an error during fetch or formatting (already logged)
            raise ConnectionError(f"Failed to retrieve or format historical data for {symbol}. Check logs.")
        elif df_hist.empty:
            status_messages.append(f"No historical data returned for {symbol} with specified parameters (Duration: {duration}, Bar Size: {bar_size}, Extended Hours: Yes).")
            empty_fig = create_candlestick_chart(df_hist, symbol) # Let charting handle empty message
            return empty_fig, html.Ul([html.Li(msg) for msg in status_messages]), symbol, None, True # Disable interval
        else:
            tz_info = f"(UTC)" if df_hist.index.tz == timezone.utc else f" ({df_hist.index.tz})"
            status_messages.append(f"Successfully fetched {len(df_hist)} bars {tz_info}.")

            # Use DataManager (though simple fetch doesn't strictly need it yet)
            dm = DataManager(symbol)
            dm.load_historical_data(df_hist)
            chart_df = dm.get_data() # Get a copy

            status_messages.append("Calculating Support/Resistance...")
            # Use S/R calculator
            sr_levels = calculate_sr_levels_peaks(chart_df, lookback=252) # Default lookback ~1 year daily
            status_messages.append(f"Found {len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance levels.")

            status_messages.append("Generating chart...")
            fig = create_candlestick_chart(chart_df, symbol, title=f"{symbol} - {bar_size} bars") # Add bar size to title
            add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'))

            end_time = datetime.now(timezone.utc)
            duration_secs = (end_time - start_time).total_seconds()
            status_messages.append(f"Chart ready. Total time: {duration_secs:.2f} seconds.")

            # Store data as JSON (consider performance for large data)
            data_json = chart_df.reset_index().to_json(date_format='iso', orient='split')

            # Decide whether to enable interval timer (keep disabled for now)
            enable_interval = True # Set to True ONLY when live updates are implemented

            return fig, html.Ul([html.Li(msg) for msg in status_messages]), symbol, data_json, enable_interval

    except (ConnectionError, ValueError, asyncio.TimeoutError) as e:
        logger.error(f"Error during chart update for {symbol}: {e}", exc_info=False) # Log concise error
        error_status = f"Error processing {symbol}: {e}"
        return error_fig, error_status, symbol, None, True # Keep interval disabled
    except Exception as e:
        logger.exception(f"An unexpected error occurred for {symbol}: {e}") # Log full traceback
        error_status = f"An unexpected error occurred for {symbol}. Please check server logs."
        return error_fig, error_status, symbol, None, True


# --- Placeholder Callback for Live Updates (Phase 5) ---
# This callback requires significant changes for actual live updates.
# It would need access to a persistent IBConnector and DataManager,
# likely running in a background thread/process.
# @app.callback(
#     Output('price-chart', 'figure', allow_duplicate=True),
#     Input('interval-component', 'n_intervals'),
#     State('current-symbol-store', 'data'),
#     State('price-data-store', 'data'), # Get potentially stored data
#     prevent_initial_call=True
# )
# def update_live_chart(n_intervals, current_symbol, stored_data_json):
#     if not current_symbol or not stored_data_json:
#         return no_update
#
#     logger.info(f"Interval {n_intervals}: Checking for live updates for {current_symbol}...")
#     # ---- Replace with actual live data retrieval ----
#     # Option A: Retrieve from a persistent DataManager updated by background task
#     # Option B: Re-fetch latest data slice (less efficient)
#     # Option C: Use WebSockets or other push mechanism if implemented
#     # -------------------------------------------------
#     try:
#         # Example: reloading stored data (NOT LIVE)
#         df = pd.read_json(stored_data_json, orient='split', date_unit='iso')
#         if 'DateTime' in df.columns:
#             df.set_index('DateTime', inplace=True)
#         if not isinstance(df.index, pd.DatetimeIndex):
#             raise ValueError("Stored data index is not DatetimeIndex")
#         if df.index.tz is None: df.index = df.index.tz_localize('UTC')
#         else: df.index = df.index.tz_convert('UTC')
#
#         # --- TODO: Add logic here to get *actual* live updates ---
#         # e.g., get latest bar from shared DataManager
#         # latest_bar = shared_data_manager_dict[current_symbol].get_latest_bar()
#         # append/update df...
#
#         sr_levels = calculate_sr_levels_peaks(df, lookback=200)
#         fig = create_candlestick_chart(df, current_symbol)
#         add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'))
#         return fig
#
#     except Exception as e:
#          logger.error(f"Error during live update attempt: {e}")
#          return no_update # Avoid breaking the chart on interval error


# --- Run the App ---
if __name__ == '__main__':
    logger.info(f"Starting {APP_TITLE}...")
    logger.info("Ensure TWS/Gateway is running and API connections are enabled.")
    # Set debug=False for more stable operation, True for development (auto-reload)
    app.run(debug=False, host='127.0.0.1', port=8050)