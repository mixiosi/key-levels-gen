# app.py

import dash
# Make sure dcc is imported directly if not already
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import asyncio
import logging
from datetime import datetime

# Import our modules
from config import APP_TITLE, DEFAULT_SYMBOL, DEFAULT_HIST_DURATION, DEFAULT_HIST_BARSIZE, LIVE_UPDATE_INTERVAL_MS
from ib_connector import IBConnector
from data_manager import DataManager
from charting import create_candlestick_chart, add_sr_lines_to_chart
from sr_calculator import calculate_sr_levels_peaks

# --- Global Variables & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
]

barsize_options = [
    {'label': '1 Minute', 'value': '1 min'},
    {'label': '5 Minutes', 'value': '5 mins'},
    {'label': '15 Minutes', 'value': '15 mins'},
    {'label': '30 Minutes', 'value': '30 mins'},
    {'label': '1 Hour', 'value': '1 hour'},
    {'label': '4 Hours', 'value': '4 hours'},
    {'label': '1 Day', 'value': '1 day'},
    {'label': '1 Week', 'value': '1 week'},
    # Note: IB has limitations on combining very small barsizes with long durations
]

# --- Dash App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12)),

    dbc.Row([
        # Symbol Input (remains dbc.Input)
        dbc.Col([
            dbc.Label("Stock Symbol:"),
            dbc.Input(id='symbol-input', type='text', value=DEFAULT_SYMBOL, debounce=True),
        ], width=2),

        # History Duration Dropdown
        dbc.Col([
            dbc.Label("History Duration:"),
            dcc.Dropdown(
                id='duration-input', # Keep the same ID
                options=duration_options,
                value=DEFAULT_HIST_DURATION, # Default value from config
                clearable=False, # Prevent user from clearing selection
                searchable=False # Not really needed for this short list
            ),
        ], width=2),

        # Bar Size Dropdown
        dbc.Col([
            dbc.Label("Bar Size:"),
            dcc.Dropdown(
                id='barsize-input', # Keep the same ID
                options=barsize_options,
                value=DEFAULT_HIST_BARSIZE, # Default value from config
                clearable=False,
                searchable=False
            ),
        ], width=2),

        # Fetch Button
        dbc.Col([
            # Add a Div to push the button down if Label height differs from Dropdown height
            html.Div(style={'marginTop': '28px'}), # Adjust px value as needed for alignment
            dbc.Button("Fetch Data & Draw Chart", id='fetch-button', n_clicks=0, color="primary"),
        ], width=2), # Removed align="end" as Dropdowns handle vertical space differently

    ], className="mb-3 align-items-start"), # Align items to start might look better with dropdowns

    dbc.Row(dbc.Col(dcc.Loading(
        id="loading-chart",
        children=[dcc.Graph(id='price-chart')],
        type="circle",
    ), width=12)),

    dbc.Row(dbc.Col(html.Div(id='status-output', style={'marginTop': '15px'}), width=12)),

    dcc.Store(id='current-symbol-store'),
    dcc.Store(id='price-data-store'),
    dcc.Interval(
        id='interval-component',
        interval=LIVE_UPDATE_INTERVAL_MS,
        n_intervals=0,
        disabled=True
    )
], fluid=True)

# --- Dash Callbacks ---

@app.callback(
    [Output('price-chart', 'figure'),
     Output('status-output', 'children'),
     Output('current-symbol-store', 'data'),
     Output('price-data-store', 'data'),
     Output('interval-component', 'disabled')],
    [Input('fetch-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('duration-input', 'value'), # Reads value from the dropdown
     State('barsize-input', 'value')],  # Reads value from the dropdown
    prevent_initial_call=True
)
def update_chart_on_fetch(n_clicks, symbol, duration, bar_size):
    """
    Callback triggered when the fetch button is clicked.
    Connects to IB, fetches historical data, calculates S/R, creates chart.
    (No changes needed inside the function itself)
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
        connector = IBConnector()

        async def fetch_async():
            await connector.connect()
            if not connector.is_connected:
                raise ConnectionError("Failed to connect to TWS/Gateway.")

            contract = await connector.qualify_contract(symbol)
            if not contract:
                 raise ValueError(f"Could not qualify contract for {symbol}.")

            df = await connector.fetch_historical_data(
                contract=contract,
                duration_str=duration, # Value comes directly from dropdown state
                bar_size_setting=bar_size # Value comes directly from dropdown state
            )
            await connector.disconnect()
            return df

        df_hist = asyncio.run(fetch_async())

        if df_hist is None:
            raise ConnectionError(f"Failed to fetch historical data for {symbol}. Check logs.")
        elif df_hist.empty:
            status_messages.append(f"No historical data found for {symbol} with specified parameters.")
            empty_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"{symbol} - No Data")
            return empty_fig, html.Ul([html.Li(msg) for msg in status_messages]), symbol, None, True
        else:
            status_messages.append(f"Successfully fetched {len(df_hist)} bars.")

            # --- Process Data & Create Chart ---
            dm = DataManager(symbol)
            dm.load_historical_data(df_hist)
            chart_df = dm.get_data()

            status_messages.append("Calculating Support/Resistance...")
            sr_levels = calculate_sr_levels_peaks(chart_df, lookback=200)
            status_messages.append(f"Found {len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance levels.")

            status_messages.append("Generating chart...")
            fig = create_candlestick_chart(chart_df, symbol)
            add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'))

            end_time = datetime.now()
            duration_secs = (end_time - start_time).total_seconds()
            status_messages.append(f"Chart ready. Total time: {duration_secs:.2f} seconds.")

            data_json = chart_df.reset_index().to_json(date_format='iso', orient='split')
            enable_interval = False

            return fig, html.Ul([html.Li(msg) for msg in status_messages]), symbol, data_json, enable_interval

    except (ConnectionError, ValueError, asyncio.TimeoutError) as e:
        logger.error(f"Error during chart update for {symbol}: {e}", exc_info=False) # Show concise error in UI
        error_message = f"Error processing {symbol}: {e}"
        empty_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error loading {symbol}")
        return empty_fig, error_message, symbol, None, True
    except Exception as e:
        logger.exception(f"An unexpected error occurred for {symbol}: {e}")
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
    logger.info(f"Starting {APP_TITLE}...")
    app.run(debug=False, host='127.0.0.1', port=8050)