# IB Charting App with Automatic S/R Lines

A Python application using Dash and the Interactive Brokers TWS API (via `ib_insync`) to display interactive stock charts with automatically calculated and confidence-filtered Support and Resistance (S/R) levels.

## Demo

![Demo GIF](images/demo.gif)

## Features

*   Connects to Interactive Brokers TWS or Gateway.
*   Fetches historical OHLCV data for stocks, including extended trading hours (pre/post-market).
*   Displays data on an interactive Plotly candlestick chart.
*   Automatically calculates potential Support and Resistance levels using peak/trough detection (`scipy.signal.find_peaks`).
*   Filters calculated S/R levels based on a confidence score (currently using a heuristic placeholder based on peak prominence and estimated touches).
*   Draws the filtered S/R lines on the chart.
*   Adds price labels for S/R lines next to the Y-axis, preventing overlap by alternating placement to the right side if necessary.
*   Visually removes weekend gaps from the chart's X-axis for continuous display of trading days.
*   Web-based UI built with Dash and Dash Bootstrap Components.
*   User Controls:
    *   Stock Symbol Input
    *   History Duration Dropdown
    *   Bar Size Dropdown
    *   S/R Confidence Threshold Slider

## File Structure & Script Functions

*   **`app.py`**:
    *   The main Dash application file.
    *   Defines the web UI layout (input fields, dropdowns, slider, chart area, status messages).
    *   Contains the core callback logic triggered by user interactions (e.g., "Fetch & Draw" button).
    *   Orchestrates the process: connects to IB, fetches data, calculates S/R, generates the Plotly figure, and updates the UI.
*   **`ib_connector.py`**:
    *   Handles all communication with the Interactive Brokers TWS/Gateway using the `ib_insync` library.
    *   Manages connection/disconnection.
    *   Qualifies stock contracts to get unique identifiers.
    *   Provides functions to fetch historical bar data (`fetch_historical_data`) with options for regular or extended hours.
    *   Includes placeholders for subscribing to live data updates (`subscribe_live_updates`, `unsubscribe_live_updates` - currently using `reqRealTimeBars`).
*   **`data_manager.py`**:
    *   Manages the price data (primarily historical) for a given symbol, stored in a `pandas` DataFrame.
    *   Provides methods to load historical data (`load_historical_data`) and update with live data (`update_live_data` - currently designed for 5-sec bars and includes basic aggregation logic).
*   **`charting.py`**:
    *   Contains functions related to creating the visual chart using Plotly.
    *   `create_candlestick_chart`: Generates the main candlestick chart with volume and configures layout options (including removing weekend gaps).
    *   `add_sr_lines_to_chart`: Takes a Plotly figure and S/R levels, adds the horizontal lines, and adds the non-overlapping price annotations to the Y-axis (alternating sides if needed).
*   **`sr_calculator.py`**:
    *   Implements the logic for identifying and filtering Support and Resistance levels.
    *   Uses `scipy.signal.find_peaks` to find initial candidates.
    *   `calculate_features_for_level`: Calculates characteristics for each candidate level (e.g., prominence, touches).
    *   `predict_level_strength`: **Placeholder function** using heuristics to assign a confidence score (0-1) based on features. *This needs to be replaced with a real ML model for true ML-based filtering.*
    *   `calculate_sr_levels_ml_filtered`: The main function called by `app.py`. It orchestrates peak finding, feature calculation, confidence scoring (via the placeholder), filtering based on the UI threshold, and merging of close levels.
    *   `merge_close_levels`: Helper function to consolidate S/R levels that are very close together.
*   **`config.py`**:
    *   Stores configuration settings like TWS host/port/client ID, default UI values (symbol, duration, bar size), and the application title. Makes settings easy to change in one place.
*   **`requirements.txt`**:
    *   Lists all the necessary Python package dependencies for the project.

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   Interactive Brokers account (Paper or Live)
    *   Interactive Brokers Trader Workstation (TWS) or IB Gateway installed and running.
2.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure TWS/Gateway API:**
    *   Ensure TWS/Gateway is running.
    *   Go to `File -> Global Configuration -> API -> Settings`.
    *   Check **"Enable ActiveX and Socket Clients"**.
    *   Note the **"Socket port"** (e.g., 7497 for paper, 7496 for live). Ensure it matches `USE_PORT` in `config.py`.
    *   Under **"Trusted IP Addresses"**, add `127.0.0.1` or ensure "Allow connections from localhost only" is enabled if running locally.
5.  **(Optional) Adjust `config.py`:** Modify port, client ID, or default UI values if needed.

## Usage

1.  Make sure TWS or IB Gateway is running and configured correctly.
2.  Run the Dash application from your terminal:
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:8050`.
4.  Enter a stock symbol, select duration/bar size, adjust the S/R confidence slider, and click "Fetch & Draw".