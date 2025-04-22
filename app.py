# app.py

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple # Added Tuple

# Import our modules
from config import APP_TITLE, DEFAULT_SYMBOL, DEFAULT_HIST_DURATION, DEFAULT_HIST_BARSIZE, LIVE_UPDATE_INTERVAL_MS
from ib_connector import IBConnector
from data_manager import DataManager
from charting import create_candlestick_chart, add_sr_lines_to_chart
# Import the NEW ML-filtered function
from sr_calculator import calculate_sr_levels_ml_filtered

# --- Global Variables & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = APP_TITLE

# --- Dropdown Options ---
duration_options = [
    {'label': '5 Days', 'value': '5 D'}, {'label': '10 Days', 'value': '10 D'},
    {'label': '1 Month', 'value': '1 M'}, {'label': '3 Months', 'value': '3 M'},
    {'label': '6 Months', 'value': '6 M'}, {'label': '1 Year', 'value': '1 Y'},
    {'label': '2 Years', 'value': '2 Y'}, {'label': '5 Years', 'value': '5 Y'},
    {'label': '10 Years', 'value': '10 Y'},
]
barsize_options = [
    {'label': '1 min', 'value': '1 min'}, {'label': '5 mins', 'value': '5 mins'},
    {'label': '15 mins', 'value': '15 mins'}, {'label': '30 mins', 'value': '30 mins'},
    {'label': '1 hour', 'value': '1 hour'}, {'label': '4 hours', 'value': '4 hours'},
    {'label': '1 day', 'value': '1 day'}, {'label': '1 week', 'value': '1 week'},
]

# --- Dash App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12)),

    # --- Input Controls Row ---
    dbc.Row([
        dbc.Col([
            dbc.Label("Symbol:", html_for='symbol-input'),
            dbc.Input(id='symbol-input', type='text', value=DEFAULT_SYMBOL, debounce=True),
        ], width=2),
        dbc.Col([
            dbc.Label("Duration:", html_for='duration-input'),
            dcc.Dropdown(id='duration-input', options=duration_options, value=DEFAULT_HIST_DURATION, clearable=False, searchable=False),
        ], width=2),
        dbc.Col([
            dbc.Label("Bar Size:", html_for='barsize-input'),
            dcc.Dropdown(id='barsize-input', options=barsize_options, value=DEFAULT_HIST_BARSIZE, clearable=False, searchable=False),
        ], width=2),
        # --- Add Confidence Threshold Slider ---
        dbc.Col([
            dbc.Label("S/R Confidence Threshold:", html_for='sr-confidence-slider'),
            dcc.Slider(id='sr-confidence-slider', min=0.1, max=0.9, step=0.05, value=0.5, # Start at 0.5
                       marks={i/10: f'{i/10:.1f}' for i in range(1, 10)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], width=3), # Increase width to accommodate slider
        # --- Fetch Button ---
        dbc.Col([
            html.Div(style={'marginTop': '31px'}),
            dbc.Button("Fetch & Draw", id='fetch-button', n_clicks=0, color="primary"),
        ], width=2), # Adjust width if needed
    ], className="mb-3 align-items-start"),

    # --- Chart and Status Row ---
    dbc.Row(dbc.Col(dcc.Loading(id="loading-chart", children=[dcc.Graph(id='price-chart')], type="circle"), width=12)),
    dbc.Row(dbc.Col(html.Div(id='status-output', style={'marginTop': '15px', 'whiteSpace': 'pre-line'}), width=12)), # Use pre-line for newlines

    # --- Hidden Stores ---
    dcc.Store(id='current-symbol-store'),
    dcc.Store(id='price-data-store'),
    dcc.Interval(id='interval-component', interval=LIVE_UPDATE_INTERVAL_MS, n_intervals=0, disabled=True)
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
     State('duration-input', 'value'),
     State('barsize-input', 'value'),
     State('sr-confidence-slider', 'value')],
    prevent_initial_call=True
)
def update_chart_on_fetch(n_clicks: int, symbol: str, duration: str, bar_size: str, sr_threshold: float) -> Tuple:
    """
    Callback triggered when the fetch button is clicked.
    Fetches data, calculates ML-filtered S/R, creates chart with non-overlapping labels.
    """
    triggered_id = dash.callback_context.triggered_id
    if not triggered_id or triggered_id != 'fetch-button':
        return no_update

    if not symbol:
        empty_fig = create_candlestick_chart(pd.DataFrame(), "No Symbol", title="Enter a Symbol")
        return empty_fig, "Please enter a stock symbol.", no_update, no_update, True

    symbol = symbol.strip().upper()
    start_time = datetime.now(timezone.utc)
    status_messages = [f"Request received for {symbol} at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}..."]
    logger.info(f"Fetching data for {symbol}, Duration: {duration}, Bar Size: {bar_size}, SR Threshold: {sr_threshold}, Incl Extended Hours")

    error_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error Loading {symbol}")

    try:
        connector = IBConnector()

        async def fetch_async():
            # ... (connection and fetch logic remains the same) ...
            if not await connector.connect():
                raise ConnectionError("Failed to connect to TWS/Gateway. Check connection and API settings.")
            contract = await connector.qualify_contract(symbol)
            if not contract:
                 raise ValueError(f"Could not qualify contract for {symbol}. Is it valid?")
            df = await connector.fetch_historical_data(
                contract=contract, duration_str=duration,
                bar_size_setting=bar_size, use_rth=False
            )
            await connector.disconnect()
            return df

        df_hist = asyncio.run(fetch_async())

        if df_hist is None:
            raise ConnectionError(f"Failed to retrieve/format historical data for {symbol}. Check logs.")
        elif df_hist.empty:
            status_messages.append(f"No historical data returned for {symbol} (Duration: {duration}, Bar Size: {bar_size}, ExtHrs: Yes).")
            empty_fig = create_candlestick_chart(df_hist, symbol)
            return empty_fig, "\n".join(status_messages), symbol, None, True
        else:
            tz_info = f"(UTC)" if df_hist.index.tz == timezone.utc else f" ({df_hist.index.tz})"
            status_messages.append(f"Successfully fetched {len(df_hist)} bars {tz_info}.")

            dm = DataManager(symbol)
            dm.load_historical_data(df_hist)
            chart_df = dm.get_data()

            # --- Calculate Y-axis range for label overlap prevention ---
            y_min = chart_df['Low'].min()
            y_max = chart_df['High'].max()
            y_range_tuple = (y_min, y_max) if not pd.isna(y_min) and not pd.isna(y_max) else None
            # -----------------------------------------------------------

            status_messages.append(f"Calculating S/R (Confidence >= {sr_threshold:.2f})...")
            sr_levels = calculate_sr_levels_ml_filtered(
                chart_df, confidence_threshold=sr_threshold, lookback=252
            )
            status_messages.append(f"Found {len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance levels passing threshold.")

            status_messages.append("Generating chart...")
            chart_title = f"{symbol} - {bar_size} bars (S/R Confidence >= {sr_threshold:.2f})"
            fig = create_candlestick_chart(chart_df, symbol, title=chart_title)

            # --- Pass y_range_tuple to the plotting function ---
            add_sr_lines_to_chart(fig, sr_levels.get('support'), sr_levels.get('resistance'), y_range=y_range_tuple)
            # ----------------------------------------------------

            end_time = datetime.now(timezone.utc)
            duration_secs = (end_time - start_time).total_seconds()
            status_messages.append(f"Chart ready. Total time: {duration_secs:.2f} seconds.")

            data_json = chart_df.reset_index().to_json(date_format='iso', orient='split')
            enable_interval = True # Keep disabled until live is needed/implemented

            return fig, "\n".join(status_messages), symbol, data_json, enable_interval

    except (ConnectionError, ValueError, asyncio.TimeoutError) as e:
        logger.error(f"Error during chart update for {symbol}: {e}", exc_info=False)
        error_status = f"Error processing {symbol}: {e}"
        return error_fig, error_status, symbol, None, True
    except Exception as e:
        logger.exception(f"An unexpected error occurred for {symbol}: {e}")
        error_status = f"An unexpected error occurred: {e}. Check server logs."
        return error_fig, error_status, symbol, None, True

# --- Live Update Callback Placeholder ---
# ... remains the same ...

# --- Run the App ---
if __name__ == '__main__':
    logger.info(f"Starting {APP_TITLE}...")
    logger.info("Ensure TWS/Gateway is running and API connections are enabled.")
    # load_sr_model_and_scaler() # Optional: Load model at startup if persistent
    app.run(debug=False, host='127.0.0.1', port=8050)