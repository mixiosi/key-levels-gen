# app.py

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc # Optional: for better styling
import plotly.graph_objects as go
import pandas as pd
import asyncio # Required for running async ib_connector methods
import logging
from datetime import datetime

# Import our modules
from config import APP_TITLE, DEFAULT_SYMBOL, DEFAULT_HIST_DURATION, DEFAULT_HIST_BARSIZE, LIVE_UPDATE_INTERVAL_MS
from ib_connector import IBConnector
from data_manager import DataManager
from charting import create_candlestick_chart, add_sr_lines_to_chart # Import S/R adder
from sr_calculator import calculate_sr_levels_peaks # Import S/R calculator

# --- Global Variables & Setup ---
# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize IBConnector and DataManager instances globally?
# Pros: Maintain connection across callbacks.
# Cons: Global state can be tricky; need careful management.
# Alternative: Create/destroy within callbacks (simpler for now, less efficient).
# Let's start with creating within callbacks for simplicity.

# Initialize Dash App
# Use Bootstrap themes for better appearance
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = APP_TITLE

# --- Dash App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12)),

    dbc.Row([
        dbc.Col([
            dbc.Label("Stock Symbol:"),
            dbc.Input(id='symbol-input', type='text', value=DEFAULT_SYMBOL, debounce=True), # Debounce waits for user to stop typing
        ], width=2),
         dbc.Col([
            dbc.Label("History Duration:"),
            dbc.Input(id='duration-input', type='text', value=DEFAULT_HIST_DURATION),
        ], width=2),
        dbc.Col([
            dbc.Label("Bar Size:"),
            # Consider using dcc.Dropdown for predefined choices later
            dbc.Input(id='barsize-input', type='text', value=DEFAULT_HIST_BARSIZE),
        ], width=2),
         dbc.Col([
            dbc.Button("Fetch Data & Draw Chart", id='fetch-button', n_clicks=0, color="primary"),
        ], width=2, align="end"), # Align button to bottom of its column
    ], className="mb-3 align-items-end"), # Use margin bottom and align items

    dbc.Row(dbc.Col(dcc.Loading( # Wrap Graph in Loading component
        id="loading-chart",
        children=[dcc.Graph(id='price-chart')],
        type="circle", # or "graph", "cube", "dot"
    ), width=12)),

    dbc.Row(dbc.Col(html.Div(id='status-output', style={'marginTop': '15px'}), width=12)),

    # Hidden div to store the current symbol being displayed (for potential refresh logic)
    dcc.Store(id='current-symbol-store'),
    # Hidden div to store the loaded data as JSON (alternative to global DataManager)
    # Be cautious with large data amounts here.
    dcc.Store(id='price-data-store'),

    # Interval component for future live updates (initially disabled or long interval)
    dcc.Interval(
        id='interval-component',
        interval=LIVE_UPDATE_INTERVAL_MS, # Configurable interval
        n_intervals=0,
        disabled=True # Disable initially, enable after first fetch if live needed
    )
], fluid=True) # Use fluid container for full width

# --- Dash Callbacks ---

@app.callback(
    [Output('price-chart', 'figure'),
     Output('status-output', 'children'),
     Output('current-symbol-store', 'data'),
     Output('price-data-store', 'data'),
     Output('interval-component', 'disabled')], # Control interval timer
    [Input('fetch-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('duration-input', 'value'),
     State('barsize-input', 'value')],
    prevent_initial_call=True # Don't run on page load
)
def update_chart_on_fetch(n_clicks, symbol, duration, bar_size):
    """
    Callback triggered when the fetch button is clicked.
    Connects to IB, fetches historical data, calculates S/R, creates chart.
    """
    if not symbol:
        return no_update, "Please enter a stock symbol.", no_update, no_update, True

    symbol = symbol.upper()
    start_time = datetime.now()
    status_messages = [f"Request received for {symbol} at {start_time.strftime('%H:%M:%S')}..."]
    logger.info(f"Fetching data for {symbol}, Duration: {duration}, Bar Size: {bar_size}")

    # Placeholder figure while loading
    loading_fig = go.Figure()
    loading_fig.update_layout(title=f"Loading {symbol}...", xaxis={'visible': False}, yaxis={'visible': False})

    try:
        # --- Connect and Fetch Data ---
        # Create connector instance within the callback context
        # NOTE: This connects/disconnects on every click - inefficient but avoids complex state management for now.
        connector = IBConnector() # Uses config settings

        # Run async connection and data fetching within the sync callback context
        # IMPORTANT: This blocks the Dash worker until complete!
        async def fetch_async():
            await connector.connect()
            if not connector.is_connected:
                raise ConnectionError("Failed to connect to TWS/Gateway.")

            contract = await connector.qualify_contract(symbol)
            if not contract:
                 raise ValueError(f"Could not qualify contract for {symbol}.")

            # Fetch historical data
            df = await connector.fetch_historical_data(
                contract=contract,
                duration_str=duration,
                bar_size_setting=bar_size
                # Use RTH=True is default in connector method
            )
            # Disconnect when done with this fetch
            await connector.disconnect()
            return df

        # Run the async part
        df_hist = asyncio.run(fetch_async())

        if df_hist is None:
            # Error occurred during fetch (logged in connector)
            raise ConnectionError(f"Failed to fetch historical data for {symbol}. Check logs.")
        elif df_hist.empty:
            status_messages.append(f"No historical data found for {symbol} with specified parameters.")
            empty_fig = create_candlestick_chart(df_hist, symbol) # Will show "No Data" message
            return empty_fig, html.Div(status_messages), symbol, None, True # Disable interval
        else:
            status_messages.append(f"Successfully fetched {len(df_hist)} bars.")

            # --- Process Data & Create Chart ---
            # Load into a temporary DataManager instance (or just use df_hist directly)
            dm = DataManager(symbol)
            dm.load_historical_data(df_hist)
            chart_df = dm.get_data() # Get a copy

            # Calculate S/R Levels (Phase 3 integration)
            status_messages.append("Calculating Support/Resistance...")
            sr_levels = calculate_sr_levels_peaks(chart_df, lookback=200) # Lookback last 200 bars
            status_messages.append(f"Found {len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance levels.")

            # Create the chart
            status_messages.append("Generating chart...")
            fig = create_candlestick_chart(chart_df, symbol)

            # Add S/R lines (Phase 4 integration)
            add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'))

            end_time = datetime.now()
            duration_secs = (end_time - start_time).total_seconds()
            status_messages.append(f"Chart ready. Total time: {duration_secs:.2f} seconds.")

            # Store data (optional, consider size limits)
            # Convert df to JSON for storage
            data_json = chart_df.reset_index().to_json(date_format='iso', orient='split')

            # Enable interval timer for potential live updates later?
            enable_interval = False # Keep disabled for now

            return fig, html.Ul([html.Li(msg) for msg in status_messages]), symbol, data_json, enable_interval

    except (ConnectionError, ValueError, asyncio.TimeoutError) as e:
        logger.error(f"Error during chart update for {symbol}: {e}", exc_info=True)
        error_message = f"Error processing {symbol}: {e}"
        # Return an empty chart and the error message
        empty_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error loading {symbol}")
        return empty_fig, error_message, symbol, None, True # Keep interval disabled on error
    except Exception as e:
        logger.exception(f"An unexpected error occurred for {symbol}: {e}") # Log full traceback
        error_message = f"An unexpected error occurred for {symbol}. Check logs."
        empty_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error loading {symbol}")
        return empty_fig, error_message, symbol, None, True


# --- Placeholder Callback for Live Updates (Phase 5) ---
# @app.callback(
#     Output('price-chart', 'figure', allow_duplicate=True), # allow_duplicate needed if modifying same output
#     Input('interval-component', 'n_intervals'),
#     State('current-symbol-store', 'data'),
#     State('price-data-store', 'data'), # Get potentially stored data
#     prevent_initial_call=True
# )
# def update_live_chart(n_intervals, current_symbol, stored_data_json):
#     if not current_symbol or stored_data_json is None:
#         return no_update
#
#     logger.info(f"Interval {n_intervals}: Checking for live updates for {current_symbol}...")
#     # This needs significant rework for Phase 5:
#     # 1. Need a persistent IBConnector instance (maybe in a separate thread/process).
#     # 2. Need DataManager instance updated by the connector's callback.
#     # 3. Retrieve updated data from DataManager (or reconstruct from stored + new).
#     # 4. Recalculate S/R.
#     # 5. Re-create and return the chart figure.
#
#     # --- Dummy example: Reload from stored data and maybe add a fake bar ---
#     try:
#         df = pd.read_json(stored_data_json, orient='split', date_unit='iso')
#         df.set_index('DateTime', inplace=True)
#         # Ensure timezone
#         if df.index.tz is None:
#              df.index = df.index.tz_localize('UTC')
#         else:
#              df.index = df.index.tz_convert('UTC')
#
#         # --- Add logic here to get *actual* live updates ---
#         # (This part is complex and requires the background connector task)
#
#         # Recalculate S/R and redraw
#         sr_levels = calculate_sr_levels_peaks(df, lookback=200)
#         fig = create_candlestick_chart(df, current_symbol)
#         add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'))
#         return fig
#
#     except Exception as e:
#          logger.error(f"Error during live update: {e}")
#          return no_update # Avoid breaking the chart on interval error


# --- Run the App ---
if __name__ == '__main__':
    # IMPORTANT: Ensure TWS/Gateway is running and API connection is enabled!
    logger.info(f"Starting {APP_TITLE}...")
    # Set debug=False for production/stable use
    # Set debug=True for development (enables hot-reloading, but runs things twice sometimes)
    app.run_server(debug=False, host='127.0.0.1', port=8050)