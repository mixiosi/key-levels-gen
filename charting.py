# charting.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# create_candlestick_chart function remains the same...
def create_candlestick_chart(df: pd.DataFrame, symbol: str, title: Optional[str] = None) -> go.Figure:
    """
    Creates a Plotly Candlestick chart, removing weekend gaps from the x-axis.

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
        fig = go.Figure()
        fig.update_layout(
            title=f"{symbol} - No Data Available",
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{
                'text': 'No data loaded. Select symbol, duration, and bar size.',
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]
        )
        return fig

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
         logger.error(f"DataFrame for {symbol} missing essential columns: {[c for c in required_cols if c not in df.columns]}. Cannot plot.")
         fig = go.Figure()
         fig.update_layout(title=f"{symbol} - Error: Missing OHLC Data")
         return fig

    if title is None:
        title = f"{symbol} Candlestick Chart"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'),
                  row=1, col=1)

    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                             marker_color='rgba(100, 149, 237, 0.6)'),
                      row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
         logger.warning(f"Volume data missing for {symbol}, skipping volume plot.")

    fig.update_layout(
        title=title, yaxis_title='Price', xaxis_title='Date/Time',
        xaxis_rangeslider_visible=False, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        template='plotly_white', margin=dict(l=50, r=50, t=50, b=50), height=700
    )

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    logger.info(f"Created chart for {symbol} with {len(df)} bars, removing weekend gaps.")
    return fig


# --- Modified add_sr_lines_to_chart with Alternating Labels ---
def add_sr_lines_to_chart(
    fig: go.Figure,
    support_levels: Optional[List[float]],
    resistance_levels: Optional[List[float]],
    y_range: Optional[Tuple[float, float]] = None
    ):
    """
    Adds S/R lines and price labels, placing labels on the right side
    if they would overlap on the default left side.

    Args:
        fig: The Plotly figure object.
        support_levels: List of support price levels.
        resistance_levels: List of resistance price levels.
        y_range: Tuple (min_y, max_y) of the main price data area.
    """
    if not fig or not hasattr(fig, 'data') or not fig.data:
        logger.warning("Cannot add S/R lines/labels: Figure object is invalid or has no data.")
        return

    if not (support_levels or resistance_levels):
        logger.debug("No support or resistance levels provided to add.")
        return

    # --- Configuration for Labels ---
    label_x_left = -0.01   # Position just left of y-axis
    label_x_right = 1.01  # Position just right of y-axis (plot area ends at 1.0)
    label_font_size = 9
    label_min_separation_fraction = 0.025 # Minimum separation as fraction of y-range

    # --- Determine X and Y range ---
    min_x, max_x = None, None
    if fig.data[0].x is not None and len(fig.data[0].x) > 0:
        try:
            x_data = pd.Series(fig.data[0].x)
            min_x, max_x = x_data.min(), x_data.max()
        except Exception as e:
            logger.warning(f"Could not determine x-axis range from fig.data[0].x: {e}.")

    if min_x is None or max_x is None:
        logger.warning("Cannot determine x-axis range for S/R lines. Lines will not be drawn.")
        return

    min_y_separation = None
    if y_range and y_range[1] > y_range[0]:
        min_y_separation = (y_range[1] - y_range[0]) * label_min_separation_fraction
        logger.debug(f"Minimum Y separation for labels: {min_y_separation:.4f}")
    else:
        logger.warning("Y-axis range not provided or invalid; overlap prevention might be inaccurate.")
        # Set a small default if needed, or proceed without overlap check
        min_y_separation = 0 # Effectively disables overlap check based on separation

    # --- Prepare Lists ---
    shapes = list(fig.layout.shapes or [])
    annotations = list(fig.layout.annotations or [])
    line_width = 1.5
    opacity = 0.6

    # --- Track last placed Y position ON EACH SIDE ---
    last_placed_y_left_support = -float('inf')
    last_placed_y_right_support = -float('inf')
    last_placed_y_left_resistance = float('inf')
    last_placed_y_right_resistance = float('inf')

    # --- Function to add annotation ---
    def _add_annotation(level, x_pos, x_anchor, color):
         annotations.append(go.layout.Annotation(
                x=x_pos, y=level, xref="paper", yref="y",
                text=f"<b>{level:.2f}</b>", showarrow=False,
                align='left' if x_anchor == 'left' else 'right', # Match alignment to anchor
                xanchor=x_anchor, yanchor='middle',
                font=dict(size=label_font_size, color=color)
            ))

    # --- Process Support Levels (Ascending) ---
    if support_levels:
        sorted_support = sorted(support_levels)
        for level in sorted_support:
            # Add the line shape (always added)
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y", x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color=f"rgba(34, 139, 34, {opacity})", width=line_width, dash="dash"),
                name=f"Support {level:.2f}"
            ))

            # Decide label placement
            overlaps_left = (level - last_placed_y_left_support < min_y_separation) if min_y_separation is not None else False

            if not overlaps_left:
                # Place on left (default)
                _add_annotation(level, label_x_left, 'right', f"rgba(34, 139, 34, 0.9)")
                last_placed_y_left_support = level
                logger.debug(f"Placed support label LEFT: {level:.2f}")
            else:
                # Try placing on right
                overlaps_right = (level - last_placed_y_right_support < min_y_separation) if min_y_separation is not None else False
                _add_annotation(level, label_x_right, 'left', f"rgba(34, 139, 34, 0.9)")
                last_placed_y_right_support = level
                if overlaps_right:
                     logger.debug(f"Placed support label RIGHT (potential overlap): {level:.2f}")
                else:
                     logger.debug(f"Placed support label RIGHT (due to left overlap): {level:.2f}")


    # --- Process Resistance Levels (Descending) ---
    if resistance_levels:
        sorted_resistance = sorted(resistance_levels, reverse=True)
        for level in sorted_resistance:
            # Add the line shape (always added)
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y", x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color=f"rgba(220, 20, 60, {opacity})", width=line_width, dash="dash"),
                name=f"Resistance {level:.2f}"
            ))

            # Decide label placement
            overlaps_left = (last_placed_y_left_resistance - level < min_y_separation) if min_y_separation is not None else False

            if not overlaps_left:
                # Place on left (default)
                 _add_annotation(level, label_x_left, 'right', f"rgba(220, 20, 60, 0.9)")
                 last_placed_y_left_resistance = level
                 logger.debug(f"Placed resistance label LEFT: {level:.2f}")
            else:
                # Try placing on right
                 overlaps_right = (last_placed_y_right_resistance - level < min_y_separation) if min_y_separation is not None else False
                 _add_annotation(level, label_x_right, 'left', f"rgba(220, 20, 60, 0.9)")
                 last_placed_y_right_resistance = level
                 if overlaps_right:
                     logger.debug(f"Placed resistance label RIGHT (potential overlap): {level:.2f}")
                 else:
                     logger.debug(f"Placed resistance label RIGHT (due to left overlap): {level:.2f}")


    # Update the figure layout
    # Ensure layout updates don't remove unrelated annotations if any existed before
    fig.update_layout(shapes=shapes, annotations=annotations)
    logger.info(f"Added {len(support_levels or [])} support, {len(resistance_levels or [])} resistance lines/labels (alternating placement).")