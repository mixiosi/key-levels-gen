# sr_calculator.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple, Any # Added Tuple, Any
import logging
# import joblib # Uncomment when using a real saved model
# from sklearn.preprocessing import StandardScaler # Example import for scaling features

logger = logging.getLogger(__name__)

# --- Feature Calculation ---

def count_touches(level: float, series: pd.Series, tolerance_percent: float = 0.1) -> int:
    """Counts how many times price 'touches' a level within a tolerance."""
    tolerance_abs = level * (tolerance_percent / 100.0)
    lower_bound = level - tolerance_abs
    upper_bound = level + tolerance_abs
    # Count bars where the low is below the upper bound AND the high is above the lower bound
    touches = series[(series >= lower_bound) & (series <= upper_bound)]
    # More robust: count where price *crosses* the zone
    # For simplicity, just count bars within the zone for now
    return len(touches)

def calculate_features_for_level(level_price: float, level_index: int, level_type: str,
                                df: pd.DataFrame, properties: Dict[str, np.ndarray],
                                index_in_properties: int, lookback_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates features for a given potential S/R level.

    Args:
        level_price: The price of the support/resistance level.
        level_index: The bar index (position in lookback_data) where the peak/trough occurred.
        level_type: 'support' or 'resistance'.
        df: The full original DataFrame (needed for potential wider context).
        properties: The 'properties' dictionary returned by find_peaks.
        index_in_properties: The index corresponding to this level within the properties arrays.
        lookback_data: The slice of the DataFrame used for peak finding.

    Returns:
        A dictionary of calculated features.
    """
    features = {}
    touch_tolerance = 0.1 # 0.1% tolerance for counting touches

    # 1. Prominence (already calculated by find_peaks)
    #    Normalize prominence relative to price? Or relative to ATR?
    prominence = properties['prominences'][index_in_properties]
    features['prominence'] = prominence
    # features['prominence_norm'] = prominence / level_price if level_price > 0 else 0

    # 2. Width of the peak/trough (from find_peaks)
    # features['peak_width'] = properties.get('widths', [0])[index_in_properties] # Requires width calculation in find_peaks

    # 3. Touch Count (within the lookback period)
    if level_type == 'support':
        # Count touches on the 'Low' series around the support level
        features['touch_count'] = count_touches(level_price, lookback_data['Low'], touch_tolerance)
    else: # resistance
        # Count touches on the 'High' series around the resistance level
        features['touch_count'] = count_touches(level_price, lookback_data['High'], touch_tolerance)

    # 4. Volume near the level index? (More complex, requires careful indexing)
    # volume_at_peak = lookback_data['Volume'].iloc[level_index]
    # avg_volume = lookback_data['Volume'].mean()
    # features['volume_ratio'] = volume_at_peak / avg_volume if avg_volume > 0 else 0

    # Add more features here: ATR at peak, time since peak, etc.

    return features

# --- ML Model Simulation ---

# Global placeholder for the model and scaler (load once)
# ML_MODEL = None
# SCALER = None

def load_sr_model_and_scaler(model_path="sr_model.joblib", scaler_path="sr_scaler.joblib"):
    """Placeholder: Loads the trained model and scaler."""
    global ML_MODEL, SCALER
    # if ML_MODEL is None:
    #     try:
    #         ML_MODEL = joblib.load(model_path)
    #         SCALER = joblib.load(scaler_path)
    #         logger.info(f"Successfully loaded ML model from {model_path} and scaler from {scaler_path}")
    #     except FileNotFoundError:
    #         logger.error(f"ML model or scaler not found at {model_path}/{scaler_path}. Using heuristic fallback.")
    #         ML_MODEL = None # Ensure it's None if loading fails
    #         SCALER = None
    #     except Exception as e:
    #         logger.error(f"Error loading ML model/scaler: {e}", exc_info=True)
    #         ML_MODEL = None
    #         SCALER = None
    pass # Keep placeholder active for now

def predict_level_strength(features: Dict[str, Any]) -> float:
    """
    Placeholder: Predicts the strength ('confidence') of an S/R level.
    Uses heuristics based on features if no real model is loaded.
    """
    # --- Heuristic Score Calculation (Placeholder) ---
    # This needs significant tuning or replacement with a real model.
    # We use prominence and touch count as simple examples. Normalize them crudely.

    # Max expected values (rough estimates, should ideally come from training data stats)
    max_prominence = features.get('prominence', 0) * 5 # Assume current prominence is maybe 1/5th of max possible
    max_touches = 10 # Assume max 10 touches is high confidence

    norm_prominence = min(1.0, features.get('prominence', 0) / max_prominence if max_prominence > 0 else 0)
    norm_touches = min(1.0, features.get('touch_count', 0) / max_touches if max_touches > 0 else 0)

    # Combine scores (example weights)
    heuristic_score = 0.6 * norm_prominence + 0.4 * norm_touches
    heuristic_score = max(0.0, min(1.0, heuristic_score)) # Clamp between 0 and 1

    # --- Real Model Prediction (when implemented) ---
    # global ML_MODEL, SCALER
    # if ML_MODEL is not None and SCALER is not None:
    #     try:
    #         # Ensure features are in the correct order expected by the model
    #         feature_names = ['prominence', 'touch_count'] # Example order
    #         feature_vector = np.array([[features.get(f, 0) for f in feature_names]])
    #
    #         # Scale features using the loaded scaler
    #         scaled_features = SCALER.transform(feature_vector)
    #
    #         # Predict probability (assuming binary classification: 0=weak, 1=strong)
    #         # Predict proba of the "strong" class (usually class 1)
    #         probability_strong = ML_MODEL.predict_proba(scaled_features)[0, 1]
    #         logger.debug(f"ML Model Prediction: {probability_strong:.3f}")
    #         return probability_strong
    #     except Exception as e:
    #         logger.error(f"Error during ML prediction: {e}. Falling back to heuristic.")
    #         # Fallback to heuristic if prediction fails
    #         return heuristic_score
    # else:
        # If no model loaded, return the heuristic score
    logger.debug(f"Using Heuristic Score: {heuristic_score:.3f} (Prominence: {features.get('prominence', 0):.2f}, Touches: {features.get('touch_count', 0)})")
    return heuristic_score


# --- Main S/R Calculation Functions ---

def merge_close_levels(levels: List[float], tolerance_percent: float = 0.1) -> List[float]:
    """Merges levels that are close to each other based on a percentage tolerance."""
    if not levels or len(levels) < 2:
        return levels

    levels = sorted(levels)
    merged_levels = [levels[0]]
    current_cluster = [levels[0]] # Keep track of levels in the current cluster

    for i in range(1, len(levels)):
        # Check percentage difference with the *average* of the current cluster
        cluster_avg = np.mean(current_cluster)
        if levels[i] <= cluster_avg * (1 + tolerance_percent / 100.0):
            # Add to the current cluster
            current_cluster.append(levels[i])
        else:
            # Finalize the previous cluster by taking the average
            merged_levels[-1] = np.mean(current_cluster)
            # Start a new cluster
            current_cluster = [levels[i]]
            merged_levels.append(levels[i])

    # Finalize the last cluster
    if current_cluster:
        merged_levels[-1] = np.mean(current_cluster)

    return sorted(merged_levels) # Return sorted list


def calculate_sr_levels_ml_filtered(
    df: pd.DataFrame,
    confidence_threshold: float = 0.5, # User-defined threshold
    prominence_factor: float = 0.015,
    distance_bars: int = 8,
    lookback: Optional[int] = 252,
    merge_tolerance_percent: float = 0.1
) -> Dict[str, List[float]]:
    """
    Calculates S/R levels using peak/trough analysis and filters them
    based on a simulated ML confidence score.
    """
    if df is None or df.empty or not all(c in df.columns for c in ['High', 'Low', 'Close']): # Added Close check
        logger.warning("Insufficient data for S/R calculation (missing H/L/C or empty DF).")
        return {'support': [], 'resistance': []}

    # --- 1. Find Initial Candidates ---
    data_to_analyze = df.iloc[-lookback:] if lookback and lookback < len(df) else df
    if data_to_analyze.empty:
        logger.warning("Lookback resulted in empty data for S/R calculation.")
        return {'support': [], 'resistance': []}

    price_range = data_to_analyze['High'].max() - data_to_analyze['Low'].min()
    min_prominence = (price_range * prominence_factor) if price_range > 0 else (data_to_analyze['Close'].iloc[-1] * 0.001)
    logger.debug(f"Using prominence: {min_prominence:.4f} for peak finding.")

    try:
        res_indices, res_props = find_peaks(data_to_analyze['High'], prominence=min_prominence, distance=distance_bars)
        raw_resistance_levels = data_to_analyze['High'].iloc[res_indices]
    except Exception as e:
        logger.error(f"Error finding resistance peaks: {e}", exc_info=True)
        res_indices, res_props, raw_resistance_levels = [], {}, pd.Series(dtype=float)

    try:
        sup_indices, sup_props = find_peaks(-data_to_analyze['Low'], prominence=min_prominence, distance=distance_bars)
        raw_support_levels = data_to_analyze['Low'].iloc[sup_indices]
    except Exception as e:
        logger.error(f"Error finding support troughs: {e}", exc_info=True)
        sup_indices, sup_props, raw_support_levels = [], {}, pd.Series(dtype=float)

    # --- 2. Filter Based on Confidence ---
    filtered_resistance = []
    for i, idx in enumerate(res_indices):
        level_price = raw_resistance_levels.iloc[i]
        features = calculate_features_for_level(level_price, idx, 'resistance', df, res_props, i, data_to_analyze)
        confidence = predict_level_strength(features)
        if confidence >= confidence_threshold:
            filtered_resistance.append(level_price)
            logger.debug(f"Keeping Resistance Level: {level_price:.2f} (Confidence: {confidence:.3f})")
        else:
             logger.debug(f"Filtering Resistance Level: {level_price:.2f} (Confidence: {confidence:.3f})")


    filtered_support = []
    for i, idx in enumerate(sup_indices):
        level_price = raw_support_levels.iloc[i]
        features = calculate_features_for_level(level_price, idx, 'support', df, sup_props, i, data_to_analyze)
        confidence = predict_level_strength(features)
        if confidence >= confidence_threshold:
            filtered_support.append(level_price)
            logger.debug(f"Keeping Support Level: {level_price:.2f} (Confidence: {confidence:.3f})")
        else:
            logger.debug(f"Filtering Support Level: {level_price:.2f} (Confidence: {confidence:.3f})")


    # --- 3. Merge and Sort Final Levels ---
    if merge_tolerance_percent > 0:
        merged_resistance = merge_close_levels(filtered_resistance, merge_tolerance_percent)
        merged_support = merge_close_levels(filtered_support, merge_tolerance_percent)
    else:
        merged_resistance = sorted(list(set(filtered_resistance)), reverse=True)
        merged_support = sorted(list(set(filtered_support)))

    logger.info(f"ML Filtered S/R for {len(data_to_analyze)} bars: "
                f"{len(merged_support)} support, {len(merged_resistance)} resistance levels "
                f"(Threshold: {confidence_threshold}, Merge: {merge_tolerance_percent}%)")

    return {'support': merged_support, 'resistance': merged_resistance}

# --- Keep original peak-based function for reference/fallback ---
def calculate_sr_levels_peaks(df: pd.DataFrame, prominence_factor: float = 0.015, distance_bars: int = 8, lookback: Optional[int] = 252, merge_tolerance_percent: float = 0.1) -> Dict[str, List[float]]:
    """Original peak/trough S/R calculation without ML filtering."""
    # This function remains mostly the same as before, just calls merge_close_levels
    if df is None or df.empty or not all(c in df.columns for c in ['High', 'Low', 'Close']):
        logger.warning("Insufficient data for S/R calculation (peaks).")
        return {'support': [], 'resistance': []}

    data_to_analyze = df.iloc[-lookback:] if lookback and lookback < len(df) else df
    if data_to_analyze.empty:
         logger.warning("Lookback resulted in empty data for S/R calculation (peaks).")
         return {'support': [], 'resistance': []}

    price_range = data_to_analyze['High'].max() - data_to_analyze['Low'].min()
    min_prominence = (price_range * prominence_factor) if price_range > 0 else (data_to_analyze['Close'].iloc[-1] * 0.001)

    try:
        res_indices, _ = find_peaks(data_to_analyze['High'], prominence=min_prominence, distance=distance_bars)
        resistance_levels = data_to_analyze['High'].iloc[res_indices].unique().tolist()
    except Exception: resistance_levels = [] # Simplified error handling

    try:
        sup_indices, _ = find_peaks(-data_to_analyze['Low'], prominence=min_prominence, distance=distance_bars)
        support_levels = data_to_analyze['Low'].iloc[sup_indices].unique().tolist()
    except Exception: support_levels = [] # Simplified error handling

    if merge_tolerance_percent > 0:
        merged_resistance = merge_close_levels(resistance_levels, merge_tolerance_percent)
        merged_support = merge_close_levels(support_levels, merge_tolerance_percent)
    else:
        merged_resistance = sorted(resistance_levels, reverse=True)
        merged_support = sorted(support_levels)

    logger.info(f"Peak-Based S/R: {len(merged_support)} support, {len(merged_resistance)} resistance (Merge: {merge_tolerance_percent}%)")
    return {'support': merged_support, 'resistance': merged_resistance}