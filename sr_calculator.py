# sr_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_sr_levels_peaks(df: pd.DataFrame, prominence_factor: float = 0.01, distance_bars: int = 5, lookback: Optional[int] = None) -> Dict[str, List[float]]:
    """
    Calculates Support and Resistance levels using peak/trough analysis.

    Args:
        df (pd.DataFrame): Price DataFrame with 'High' and 'Low' columns.
        prominence_factor (float): Required prominence relative to price range for peak detection.
        distance_bars (int): Minimum horizontal distance between peaks/troughs.
        lookback (int, optional): Number of recent bars to consider. If None, uses all data.

    Returns:
        Dict[str, List[float]]: Dictionary with 'support' and 'resistance' lists.
    """
    if df is None or df.empty or not all(c in df.columns for c in ['High', 'Low']):
        logger.warning("Insufficient data for S/R calculation.")
        return {'support': [], 'resistance': []}

    data_to_analyze = df.iloc[-lookback:] if lookback else df
    if data_to_analyze.empty:
         logger.warning("Lookback resulted in empty data for S/R calculation.")
         return {'support': [], 'resistance': []}

    price_range = data_to_analyze['High'].max() - data_to_analyze['Low'].min()
    # Avoid division by zero if price hasn't moved
    min_prominence = price_range * prominence_factor if price_range > 0 else 0.01

    # Resistance from High peaks
    resistance_indices, _ = find_peaks(data_to_analyze['High'], prominence=min_prominence, distance=distance_bars)
    resistance_levels = data_to_analyze['High'].iloc[resistance_indices].tolist()

    # Support from Low troughs (find peaks on inverted Lows)
    support_indices, _ = find_peaks(-data_to_analyze['Low'], prominence=min_prominence, distance=distance_bars)
    support_levels = data_to_analyze['Low'].iloc[support_indices].tolist()

    # Basic cleanup: remove duplicates and sort
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    support_levels = sorted(list(set(support_levels)))

    # Optional: Merge levels that are very close? (e.g., within 0.1%)

    logger.info(f"Calculated S/R: {len(support_levels)} support, {len(resistance_levels)} resistance levels.")
    return {'support': support_levels, 'resistance': resistance_levels}

# --- Add other S/R methods here later if needed ---