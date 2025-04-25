# app.py

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List # Added List

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
            html.Div(style={'marginTop': '31px'}), # Add margin to align button vertically
            dbc.Button("Fetch & Draw", id='fetch-button', n_clicks=0, color="primary"),
        ], width=2), # Adjust width if needed
    ], className="mb-3 align-items-end"), # Use align-items-end for better vertical alignment

    # --- Chart and Status Row ---
    dbc.Row(dbc.Col(dcc.Loading(id="loading-chart", children=[dcc.Graph(id='price-chart')], type="circle"), width=12)),
    dbc.Row(dbc.Col(html.Div(id='status-output', style={'marginTop': '15px', 'whiteSpace': 'pre-line'}), width=12)), # Use pre-line for newlines

    # --- NEW: Export Areas ---
    dbc.Row([
        dbc.Col([
            dbc.Label("Support Levels (Copy for TradingView):", html_for='support-export-area'),
            dcc.Textarea(
                id='support-export-area',
                readOnly=True,
                style={'width': '100%', 'height': 50, 'fontFamily': 'monospace', 'fontSize': '0.9em'},
                placeholder="Calculated support levels will appear here..."
            )
        ], width=6),
        dbc.Col([
            dbc.Label("Resistance Levels (Copy for TradingView):", html_for='resistance-export-area'),
            dcc.Textarea(
                id='resistance-export-area',
                readOnly=True,
                style={'width': '100%', 'height': 50, 'fontFamily': 'monospace', 'fontSize': '0.9em'},
                placeholder="Calculated resistance levels will appear here..."
            )
        ], width=6),
    ], className="mb-3 align-items-start"),
    # --- End NEW Export Areas ---

    # --- Hidden Stores ---
    dcc.Store(id='current-symbol-store'),
    dcc.Store(id='price-data-store'),
    dcc.Interval(id='interval-component', interval=LIVE_UPDATE_INTERVAL_MS, n_intervals=0, disabled=True) # Start disabled
], fluid=True)

# --- Dash Callbacks ---

@app.callback(
    [Output('price-chart', 'figure'),
     Output('status-output', 'children'),
     Output('current-symbol-store', 'data'),
     Output('price-data-store', 'data'),
     Output('interval-component', 'disabled'),
     Output('support-export-area', 'value'),   # NEW Output for support levels string
     Output('resistance-export-area', 'value')], # NEW Output for resistance levels string
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
    Fetches data, calculates ML-filtered S/R, creates chart with non-overlapping labels,
    and prepares S/R levels for export.
    """
    triggered_id = dash.callback_context.triggered_id
    # Check if the button was actually clicked (or if it's the initial load)
    if not triggered_id or triggered_id != 'fetch-button':
        return no_update # Important: return no_update for all outputs if not triggered

    # Initialize export strings - they will be returned even on error or no symbol
    support_export_str = ""
    resistance_export_str = ""

    if not symbol:
        empty_fig = create_candlestick_chart(pd.DataFrame(), "No Symbol", title="Enter a Symbol")
        # Return default values for all outputs, including the empty export strings
        return empty_fig, "Please enter a stock symbol.", no_update, no_update, True, support_export_str, resistance_export_str

    symbol = symbol.strip().upper()
    start_time = datetime.now(timezone.utc)
    status_messages = [f"Request received for {symbol} at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}..."]
    logger.info(f"Fetching data for {symbol}, Duration: {duration}, Bar Size: {bar_size}, SR Threshold: {sr_threshold}, Incl Extended Hours")

    # Create a default error figure in case of issues
    error_fig = create_candlestick_chart(pd.DataFrame(), symbol, title=f"Error Loading {symbol}")

    try:
        connector = IBConnector()

        async def fetch_async():
            logger.info("Attempting to connect to IBKR...")
            if not await connector.connect():
                logger.error("Failed to connect to TWS/Gateway.")
                raise ConnectionError("Failed to connect to TWS/Gateway. Check connection and API settings.")
            logger.info("Connected to IBKR. Qualifying contract...")
            contract = await connector.qualify_contract(symbol)
            if not contract:
                 logger.warning(f"Could not qualify contract for {symbol}.")
                 raise ValueError(f"Could not qualify contract for {symbol}. Is it valid?")
            logger.info(f"Contract qualified: {contract.localSymbol}. Fetching historical data...")
            df = await connector.fetch_historical_data(
                contract=contract, duration_str=duration,
                bar_size_setting=bar_size, use_rth=False # Fetch including outside RTH
            )
            logger.info("Disconnecting from IBKR...")
            await connector.disconnect()
            logger.info("Disconnected.")
            return df

        # Run the asynchronous fetching function
        df_hist = asyncio.run(fetch_async())

        if df_hist is None:
            # This case might be handled by exceptions within fetch_async, but check just in case
            raise ConnectionError(f"Failed to retrieve/format historical data for {symbol}. Check logs.")
        elif df_hist.empty:
            status_messages.append(f"No historical data returned for {symbol} (Duration: {duration}, Bar Size: {bar_size}, ExtHrs: Yes). Is the symbol correct and data available for this period?")
            empty_fig = create_candlestick_chart(df_hist, symbol, title=f"No Data for {symbol}")
             # Return empty figure and export strings
            return empty_fig, "\n".join(status_messages), symbol, None, True, support_export_str, resistance_export_str
        else:
            # Data received successfully
            tz_info = f"(UTC)" if df_hist.index.tz == timezone.utc else f" ({df_hist.index.tz})" if df_hist.index.tz else ""
            status_messages.append(f"Successfully fetched {len(df_hist)} bars {tz_info}.")

            dm = DataManager(symbol)
            dm.load_historical_data(df_hist)
            chart_df = dm.get_data() # Get the potentially cleaned/processed data

            # --- Calculate Y-axis range for label overlap prevention ---
            y_min = chart_df['Low'].min()
            y_max = chart_df['High'].max()
            y_range_tuple = (y_min, y_max) if not pd.isna(y_min) and not pd.isna(y_max) else None
            # -----------------------------------------------------------

            status_messages.append(f"Calculating S/R levels (Confidence >= {sr_threshold:.2f})...")
            sr_levels = calculate_sr_levels_ml_filtered(
                chart_df, confidence_threshold=sr_threshold, lookback=252 # Adjust lookback as needed
            )

            support_levels: List[float] = sr_levels.get('support', [])
            resistance_levels: List[float] = sr_levels.get('resistance', [])
            status_messages.append(f"Found {len(support_levels)} support, {len(resistance_levels)} resistance levels passing threshold.")

            # --- Format S/R levels for export ---
            # Sort levels for consistency before creating the string
            support_export_str = ",".join(map(str, sorted(support_levels))) if support_levels else ""
            resistance_export_str = ",".join(map(str, sorted(resistance_levels))) if resistance_levels else ""
            # --- End Formatting ---

            status_messages.append("Generating chart...")
            chart_title = f"{symbol} - {bar_size} bars (S/R Confidence >= {sr_threshold:.2f})"
            fig = create_candlestick_chart(chart_df, symbol, title=chart_title)

            # --- Pass y_range_tuple to the plotting function ---
            add_sr_lines_to_chart(fig, support_levels, resistance_levels, y_range=y_range_tuple)
            # ----------------------------------------------------

            end_time = datetime.now(timezone.utc)
            duration_secs = (end_time - start_time).total_seconds()
            status_messages.append(f"Chart ready. Total time: {duration_secs:.2f} seconds.")

            # Serialize DataFrame for potential future use (e.g., live updates)
            # Using reset_index() to include the datetime index in the JSON
            data_json = chart_df.reset_index().to_json(date_format='iso', orient='split')

            # Disable interval timer for now, enable if live updates are implemented
            enable_interval = True # Set to False if you don't want auto-updates

            # Return all outputs, including the chart, status, stored data, and export strings
            return fig, "\n".join(status_messages), symbol, data_json, enable_interval, support_export_str, resistance_export_str

    except (ConnectionError, ValueError, asyncio.TimeoutError) as e:
        logger.error(f"Handled Error during chart update for {symbol}: {e}", exc_info=False) # Log specific handled errors
        error_status = f"Error processing {symbol}: {e}"
        status_messages.append(error_status)
        # Return error figure and empty export strings
        return error_fig, "\n".join(status_messages), symbol, None, True, support_export_str, resistance_export_str
    except Exception as e:
        logger.exception(f"An unexpected error occurred for {symbol}: {e}") # Log the full traceback for unexpected errors
        error_status = f"An unexpected error occurred: {e}. Check server logs for details."
        status_messages.append(error_status)
         # Return error figure and empty export strings
        return error_fig, "\n".join(status_messages), symbol, None, True, support_export_str, resistance_export_str

# --- Live Update Callback Placeholder ---
@app.callback(
    Output('price-chart', 'figure', allow_duplicate=True), # Use allow_duplicate
    Input('interval-component', 'n_intervals'),
    State('current-symbol-store', 'data'),
    State('price-data-store', 'data'),
    State('barsize-input', 'value'),
    State('sr-confidence-slider', 'value'), # Need threshold for recalculation
    State('price-chart', 'figure'), # Get current figure to update
    prevent_initial_call=True
)
def update_chart_live(n_intervals: int, symbol: str, price_data_json: str, bar_size: str, sr_threshold: float, current_fig_dict: Dict) -> Any:
    """
    Placeholder for live data updates.
    This would involve fetching live data, appending it, recalculating S/R,
    and updating the chart figure.
    """
    if not symbol or not price_data_json:
        return no_update # Don't update if no symbol or data loaded

    logger.info(f"Live update triggered for {symbol} (Interval: {n_intervals})")

    # --- Implementation Needed ---
    # 1. Connect to IBKR (briefly, or maintain connection if feasible)
    # 2. Fetch latest tick or bar data for the symbol.
    # 3. Load the historical data from price_data_json back into a DataFrame.
    # 4. Append/update the DataFrame with the new live data.
    # 5. Potentially resample if live data doesn't match the chart's bar size.
    # 6. Recalculate S/R levels using calculate_sr_levels_ml_filtered on the updated DataFrame.
    # 7. Update the existing Plotly figure (current_fig_dict) rather than creating a new one.
    #    - Update the candlestick trace data.
    #    - Remove old S/R line shapes.
    #    - Add new S/R line shapes.
    #    - Update layout ranges if necessary.
    # 8. Store the *updated* DataFrame back into price_data_json (optional, depends on strategy).
    # 9. Return the updated figure dictionary.

    # Example of how to update (conceptual):
    # fig = go.Figure(current_fig_dict)
    # # ... update fig.data, fig.layout.shapes ...
    # return fig

    # For now, just log and do nothing:
    logger.warning("Live update functionality is not fully implemented yet.")
    return no_update # Return no_update until implemented

# --- Run the App ---
if __name__ == '__main__':
    logger.info(f"Starting {APP_TITLE}...")
    logger.info("Ensure TWS/Gateway is running and API connections are enabled.")
    # Consider pre-loading ML model/scaler here if they are static and large
    # load_sr_model_and_scaler() # Optional: Load model at startup if persistent
    app.run(debug=False, host='127.0.0.1', port=8050) # Set debug=False for production/regular use