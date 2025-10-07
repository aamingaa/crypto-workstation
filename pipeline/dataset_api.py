"""
Dataset API for exporting microstructure features for a single family.

This module exposes a simple function `build_microstructure_dataset` that
uses `TradingPipeline` to construct bars, extract features, and return a
single-family feature matrix X with an aligned target y.
Optionally applies a "gp-like" normalization: sign·log1p(|x|) followed by
rolling standard deviation scaling (causal, right-aligned).
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .trading_pipeline import TradingPipeline


def _family_features_config(family: str, bucketed_flow_enabled: bool = False) -> Dict:
    """
    Build a features_config dict that enables only the requested family.
    Supported families: 'orderflow', 'impact', 'volatility', 'momentum', 'tail', 'path_shape', 'basic'
    """
    family = (family or '').lower()
    cfg = {
        'basic': False,
        'volatility': False,
        'momentum': False,
        'orderflow': False,
        'impact': False,
        'tail': False,
        'path_shape': False,
        'bucketed_flow': {
            'enabled': False,
            'low_q': 0.2,
            'high_q': 0.8,
            'lag': 1,
            'vpin_bins': 10,
            'min_trades_alpha': 50,
        },
    }
    if family in cfg:
        cfg[family] = True
    # Orderflow optional sub-module
    if family == 'orderflow':
        cfg['bucketed_flow']['enabled'] = bool(bucketed_flow_enabled)
    return cfg


def _gp_like_normalize(df: pd.DataFrame, window: int = 2000) -> pd.DataFrame:
    """
    Apply sign·log1p(|x|) then rolling std scaling per column (causal).
    NaN/Inf are replaced with 0.0. Non-numeric columns are ignored.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df

    x = df[numeric_cols].astype(float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        mean_abs = x.abs().mean(axis=0).replace(0.0, np.nan)
        scaled = np.sign(x) * np.log1p(x.abs()) / np.log1p(mean_abs)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Rolling std, right-aligned (causal)
    std = scaled.rolling(window=window, min_periods=1).std()
    std = std.replace(0.0, np.nan)
    x_norm = scaled.divide(std)
    x_norm = x_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df[numeric_cols] = x_norm
    return df


def build_microstructure_dataset(
    start_date: str,
    end_date: str,
    *,
    bar_type: str = 'time',
    time_interval: str = '1h',
    family: str = 'orderflow',
    bucketed_flow_enabled: bool = False,
    feature_window_bars: int = 10,
    horizon: int = 5,
    normalize: str = 'none',  # 'none' | 'gp_like'
    data_config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, Dict]:
    """
    Build and return (X, y, idx, meta) for a single feature family.

    - X: feature matrix for the selected family
    - y: target series log_return_{horizon}
    - idx: datetime index aligned with X/y
    - meta: dict with {family, horizon, features}
    """

    # Initialize pipeline and override features
    pipeline = TradingPipeline()
    features_config = _family_features_config(family, bucketed_flow_enabled)

    # Prepare run config for pipeline
    run_config = {
        'data_config': data_config or {},
        'bar_type': bar_type,
        'time_freq': time_interval if bar_type == 'time' else None,
        'dollar_threshold': 60000000,  # ignored when bar_type == 'time'
        'bar_cache_template': None,
        'feature_window_bars': feature_window_bars,
        'model_type': 'linear',  # placeholder, we won't use model here
        'target_horizon': horizon,
        'n_splits': 3,
        'embargo_bars': 0,
        'save_plots': False,
    }

    # Inject features_config override
    run_config['features_config'] = features_config

    # Run partial pipeline steps to obtain data, bars, features, labels
    trades_df = pipeline.load_data(**(data_config or {}), date_range=(start_date, end_date))
    bars = pipeline.build_bars(
        trades_df,
        run_config['dollar_threshold'],
        bar_cache_template=run_config.get('bar_cache_template'),
        bar_type=bar_type,
        time_freq=run_config.get('time_freq'),
    )
    X_all, y_all = pipeline.extract_features(trades_df, bars, feature_window_bars)

    # Select target horizon
    target_col = f'log_return_{horizon}'
    if target_col not in y_all.columns:
        raise ValueError(f"Target column {target_col} not found in labels")
    y = y_all[target_col].copy()

    # Keep only selected family's columns
    # Heuristic: extractor produces names per sub-extractor; here we include all columns since
    # the extractor instance was created with only one family enabled.
    X = X_all.copy()

    # Optional gp-like normalization
    if normalize == 'gp_like':
        X = _gp_like_normalize(X)

    idx = X.index
    meta = {
        'family': family,
        'horizon': horizon,
        'features': list(X.columns),
    }
    return X, y, idx, meta


