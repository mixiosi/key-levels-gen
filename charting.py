# charting.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Optional, List, Tuple # Added Tuple

logger = logging.getLogger(__name__)

# create_candlestick_chart function remains the same...
# ... (previous code for create_candlestick_chart) ...
def create_candlestick_chart(df: pd.DataFrame, symbol: str, title: Optional[str] = None) -> go.Figure:
    """
    Creates a Plotly Candlestick chart.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
                           columns and a DatetimeIndex.
        symbol (str): The symbol name for labeling.
        title (str, optional): Chart title. Defaults to symbol name.

    Returns:
        go.Figure: Plotly figure object.
    """
    if df is None or df.empty:
        logger.warning(f"Cannot create chart for {symbol}: DataFrame is empty or None.")
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title=f"{symbol} - No Data Available",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No data loaded. Please select symbol, duration, and bar size.',
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]
        )
        return fig

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
         logger.error(f"DataFrame for {symbol} is missing essential columns: {[c for c in required_cols if c not in df.columns]}. Cannot plot.")
         fig = go.Figure()
         fig.update_layout(title=f"{symbol} - Error: Missing OHLC Data")
         return fig


    if title is None:
        title = f"{symbol} Candlestick Chart"

    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, # Adjusted spacing
                        row_heights=[0.75, 0.25]) # Give more space to price chart

    # --- Candlestick Chart ---
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    # --- Volume Chart ---
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color='rgba(100, 149, 237, 0.6)'), # Cornflower blue
                      row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
         logger.warning(f"Volume data missing for {symbol}, skipping volume plot.")
         # Consider hiding the second row explicitly if needed
         # fig.update_layout(yaxis2_visible=False)

    # --- Layout ---
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date/Time',
        xaxis_rangeslider_visible=False, # Hide range slider for cleaner look
        hovermode='x unified', # Improved hover experience (shows all data for that x point)
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1), # Legend at top
        template='plotly_white', # Use a clean template
        margin=dict(l=50, r=50, t=50, b=50), # Adjust margins
        height=700 # Adjust overall height as needed
    )

    # Improve X axis tick formatting potentially
    # fig.update_xaxes(tickformat="%Y-%m-%d\n%H:%M") # Example format

    logger.info(f"Created chart for {symbol} with {len(df)} bars.")
    return fig


# --- Modified add_sr_lines_to_chart ---
def add_sr_lines_to_chart(
    fig: go.Figure,
    support_levels: Optional[List[float]],
    resistance_levels: Optional[List[float]],
    y_range: Optional[Tuple[float, float]] = None # Add y_range parameter
    ):
    """
    Adds horizontal S/R lines and non-overlapping price labels to the y-axis.

    Args:
        fig: The Plotly figure object.
        support_levels: List of support price levels.
        resistance_levels: List of resistance price levels.
        y_range: Tuple containing (min_y, max_y) of the main price data area.
                 Used to calculate relative spacing for labels.
    """
    if not fig or not hasattr(fig, 'data') or not fig.data:
        logger.warning("Cannot add S/R lines/labels: Figure object is invalid or has no data.")
        return

    if not (support_levels or resistance_levels):
        logger.debug("No support or resistance levels provided to add.")
        return

    # --- Configuration for Labels ---
    label_x_position = -0.01 # Position just left of the y-axis (adjust as needed)
    label_font_size = 9
    label_min_separation_fraction = 0.025 # Minimum separation as fraction of y-range (e.g., 2.5%)

    # --- Determine X and Y range ---
    min_x, max_x = None, None
    # Get x-range from the first trace (candlestick)
    if fig.data[0].x is not None and len(fig.data[0].x) > 0:
        try:
            x_data = pd.Series(fig.data[0].x)
            min_x, max_x = x_data.min(), x_data.max()
        except Exception as e:
            logger.warning(f"Could not determine x-axis range from fig.data[0].x: {e}.")

    if min_x is None or max_x is None:
        logger.warning("Cannot determine x-axis range for S/R lines. Lines will not be drawn.")
        return

    # Calculate minimum separation in data units if y_range is provided
    min_y_separation = None
    if y_range and y_range[1] > y_range[0]:
        min_y_separation = (y_range[1] - y_range[0]) * label_min_separation_fraction
        logger.debug(f"Minimum Y separation for labels: {min_y_separation:.4f}")
    else:
        logger.warning("Y-axis range not provided or invalid; cannot perform overlap prevention for labels.")
        # Labels might overlap if y_range is None


    # --- Prepare Lists for Shapes and Annotations ---
    shapes = list(fig.layout.shapes or [])
    annotations = list(fig.layout.annotations or [])

    line_width = 1.5
    opacity = 0.6

    # --- Process Support Levels (Ascending) ---
    last_placed_y_support = -float('inf') # Initialize tracker for support labels
    if support_levels:
        sorted_support = sorted(support_levels) # Ensure sorted low to high
        for level in sorted_support:
            # Add the line shape
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y", x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color=f"rgba(34, 139, 34, {opacity})", width=line_width, dash="dash"),
                name=f"Support {level:.2f}"
            ))

            # Check if we should add an annotation (prevent overlap)
            can_add_annotation = True
            if min_y_separation is not None:
                if level - last_placed_y_support < min_y_separation:
                    can_add_annotation = False
                    logger.debug(f"Skipping support label for {level:.2f} (too close to {last_placed_y_support:.2f})")

            if can_add_annotation:
                annotations.append(go.layout.Annotation(
                    x=label_x_position, y=level, xref="paper", yref="y",
                    text=f"<b>{level:.2f}</b>", # Bold text
                    showarrow=False, align='right', xanchor='right', yanchor='middle',
                    font=dict(size=label_font_size, color="rgba(34, 139, 34, 0.9)") # Slightly darker than line
                ))
                last_placed_y_support = level # Update last placed position

    # --- Process Resistance Levels (Descending) ---
    last_placed_y_resistance = float('inf') # Initialize tracker for resistance labels
    if resistance_levels:
        sorted_resistance = sorted(resistance_levels, reverse=True) # Ensure sorted high to low
        for level in sorted_resistance:
            # Add the line shape
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y", x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color=f"rgba(220, 20, 60, {opacity})", width=line_width, dash="dash"),
                name=f"Resistance {level:.2f}"
            ))

            # Check if we should add an annotation (prevent overlap)
            can_add_annotation = True
            if min_y_separation is not None:
                 # Compare difference with last placed *resistance* level
                if last_placed_y_resistance - level < min_y_separation:
                    can_add_annotation = False
                    logger.debug(f"Skipping resistance label for {level:.2f} (too close to {last_placed_y_resistance:.2f})")

            if can_add_annotation:
                annotations.append(go.layout.Annotation(
                     x=label_x_position, y=level, xref="paper", yref="y",
                    text=f"<b>{level:.2f}</b>", # Bold text
                    showarrow=False, align='right', xanchor='right', yanchor='middle',
                    font=dict(size=label_font_size, color="rgba(220, 20, 60, 0.9)") # Slightly darker than line
                ))
                last_placed_y_resistance = level # Update last placed position

    # Update the figure layout with the combined lists of shapes and annotations
    fig.update_layout(shapes=shapes, annotations=annotations)
    logger.info(f"Added {len(support_levels or [])} support, {len(resistance_levels or [])} resistance lines/labels.")