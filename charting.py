# charting.py

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

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
                'text': 'No data loaded. Please select a symbol and fetch data.',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return fig

    if title is None:
        title = f"{symbol} Candlestick Chart"

    # Create figure with secondary y-axis for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, # Reduced spacing
                        row_heights=[0.7, 0.3]) # Give more space to price chart

    # --- Candlestick Chart ---
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'),
                  row=1, col=1)

    # --- Volume Chart ---
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(100, 100, 200, 0.6)'),
                  row=2, col=1)

    # --- Layout ---
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date/Time',
        xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
        hovermode='x unified', # Improved hover experience
        legend_title_text='Legend',
        template='plotly_white', # Use a clean template
        height=600 # Adjust height as needed
    )

    # Update y-axis for volume
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    # Ensure date/time axis formats nicely
    # fig.update_xaxes(type='category') # Sometimes helps with gaps, but messes up zooming

    logger.info(f"Created chart for {symbol} with {len(df)} bars.")
    return fig

# --- Placeholder for S/R lines (Phase 4) ---
def add_sr_lines_to_chart(fig: go.Figure, support_levels: Optional[List[float]], resistance_levels: Optional[List[float]]):
    """Adds horizontal support and resistance lines to the figure."""
    if not fig or not (support_levels or resistance_levels):
        return # Nothing to add or no figure

    min_x = fig.data[0].x[0] if fig.data and len(fig.data[0].x) > 0 else 0
    max_x = fig.data[0].x[-1] if fig.data and len(fig.data[0].x) > 0 else 1

    shapes = []
    if support_levels:
        for level in support_levels:
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y",
                x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color="green", width=1, dash="dash"),
                name=f"Support {level:.2f}" # Name might not show directly
            ))
    if resistance_levels:
         for level in resistance_levels:
            shapes.append(go.layout.Shape(
                type="line", xref="x", yref="y",
                x0=min_x, y0=level, x1=max_x, y1=level,
                line=dict(color="red", width=1, dash="dash"),
                name=f"Resistance {level:.2f}"
            ))

    fig.update_layout(shapes=shapes)
    logger.info(f"Added {len(support_levels or [])} support and {len(resistance_levels or [])} resistance lines.")