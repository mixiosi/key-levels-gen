# config.py

# Interactive Brokers TWS/Gateway connection settings
TWS_HOST = '127.0.0.1'  # Standard loopback address
TWS_PORT_PAPER = 7497   # Default paper trading port
TWS_PORT_LIVE = 7496    # Default live trading port
CLIENT_ID = 2           # Unique client ID for this connection (change if needed)

# Select which port to use (change if using live account)
USE_PORT = TWS_PORT_PAPER

# --- Data Request Defaults ---
DEFAULT_HIST_DURATION = '3 M' # Default duration for historical data (e.g., '1 Y', '6 M', '10 D')
DEFAULT_HIST_BARSIZE = '1 day' # Default bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
DEFAULT_STOCK_EXCHANGE = 'SMART' # Use SMART routing
DEFAULT_CURRENCY = 'USD'

# --- Charting ---
DEFAULT_SYMBOL = 'AAPL' # Default symbol to load on startup

# --- App Settings ---
APP_TITLE = "IB Charting App with S/R Lines"
# Use Dash interval for future live updates (milliseconds)
# Set to a large value initially if not using live updates yet
LIVE_UPDATE_INTERVAL_MS = 60 * 60 * 1000 # e.g., 1 hour (will reduce later)