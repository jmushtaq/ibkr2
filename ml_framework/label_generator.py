import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import timedelta

class LabelGenerator:
    """
    Generate labels for supervised learning with various strategies
    """

    def __init__(self, data: pd.DataFrame, frequency: str = '1D'):
        """
        Initialize label generator

        Args:
            data: DataFrame with OHLCV data (must have 'close' column)
            frequency: Data frequency
        """
        self.data = data.copy()
        self.frequency = frequency
        self._set_period_mappings()

    def _set_period_mappings(self):
        """Map periods to number of data points based on frequency"""
        period_mapping = {
            '1min': {'1d': 390, '1w': 1950, '1m': 7800, '3m': 23400},
            '5min': {'1d': 78, '1w': 390, '1m': 1560, '3m': 4680},
            '15min': {'1d': 26, '1w': 130, '1m': 520, '3m': 1560},
            '1H': {'1d': 7, '1w': 35, '1m': 140, '3m': 420},
            '4H': {'1d': 2, '1w': 10, '1m': 40, '3m': 120},
            '1D': {'1d': 1, '1w': 5, '1m': 21, '3m': 63, '6m': 126, '1y': 252}
        }
        self.period_mapping = period_mapping.get(self.frequency, {'1d': 1, '1w': 5, '1m': 21})

    def _get_lookahead_periods(self, lookahead: str) -> int:
        """Convert lookahead string to number of periods"""
        mapping = {
            '1d': 1, '2d': 2, '3d': 3, '5d': 5, '1w': 5, '2w': 10,
            '1m': 21, '2m': 42, '3m': 63, '6m': 126, '1y': 252
        }

        if lookahead in mapping:
            return mapping[lookahead]

        # Try to get from period mapping
        if lookahead in self.period_mapping.get(self.frequency, {}):
            return self.period_mapping[self.frequency][lookahead]

        # Default to 1 period
        return 1

    def generate_binary_labels(self, lookahead: str = '1d',
                              threshold: float = 0.0,
                              upside_only: bool = False) -> pd.DataFrame:
        """
        Generate binary labels (1 for up, 0 for down)

        Args:
            lookahead: Lookahead period (e.g., '1d', '1w', '1m')
            threshold: Minimum return to consider as positive
            upside_only: If True, only label upside moves (ignore downs)
        """
        df = self.data.copy()
        periods = self._get_lookahead_periods(lookahead)

        # Calculate future return
        df['future_return'] = df['close'].shift(-periods) / df['close'] - 1

        # Generate labels based on threshold
        if upside_only:
            df['label'] = (df['future_return'] > threshold).astype(int)
        else:
            df['label'] = np.where(df['future_return'] > threshold, 1,
                                  np.where(df['future_return'] < -threshold, 0, np.nan))

        # Drop NaN labels (where future data is not available)
        df = df.dropna(subset=['label'])

        return df[['label']]

    def generate_multi_class_labels(self, lookahead: str = '1d',
                                   bins: List[float] = [-0.02, -0.01, 0.01, 0.02]) -> pd.DataFrame:
        """
        Generate multi-class labels for returns

        Args:
            lookahead: Lookahead period
            bins: Breakpoints for classification (excluding -inf and inf)
        """
        df = self.data.copy()
        periods = self._get_lookahead_periods(lookahead)

        # Calculate future return
        df['future_return'] = df['close'].shift(-periods) / df['close'] - 1

        # Create bins
        all_bins = [-np.inf] + bins + [np.inf]
        df['label'] = pd.cut(df['future_return'], bins=all_bins, labels=False)

        # Drop NaN
        df = df.dropna(subset=['label'])

        return df[['label']]

    def generate_regression_labels(self, lookahead: str = '1d') -> pd.DataFrame:
        """Generate regression labels (future returns)"""
        df = self.data.copy()
        periods = self._get_lookahead_periods(lookahead)

        # Calculate future return
        df['label'] = df['close'].shift(-periods) / df['close'] - 1

        # Drop NaN
        df = df.dropna(subset=['label'])

        return df[['label']]

    def generate_risk_reward_labels(self, lookahead: str = '1d',
                                   stop_loss: float = 0.02,
                                   take_profit: float = 0.04) -> pd.DataFrame:
        """
        Generate labels based on risk-reward ratio
        Returns: 1 if take profit hit first, 0 if stop loss hit first
        """
        df = self.data.copy()
        periods = self._get_lookahead_periods(lookahead)

        # Initialize labels
        labels = np.full(len(df), np.nan)

        for i in range(len(df) - periods):
            entry_price = df['close'].iloc[i]
            stop_price = entry_price * (1 - stop_loss)
            take_profit_price = entry_price * (1 + take_profit)

            # Look at future prices within the lookahead period
            future_prices = df['close'].iloc[i+1:i+periods+1]

            if len(future_prices) > 0:
                # Check if stop loss or take profit is hit
                stop_hit = np.any(future_prices <= stop_price)
                profit_hit = np.any(future_prices >= take_profit_price)

                if profit_hit and not stop_hit:
                    labels[i] = 1
                elif stop_hit and not profit_hit:
                    labels[i] = 0
                elif profit_hit and stop_hit:
                    # Check which happened first
                    first_stop = np.argmax(future_prices <= stop_price) if stop_hit else np.inf
                    first_profit = np.argmax(future_prices >= take_profit_price) if profit_hit else np.inf
                    labels[i] = 1 if first_profit < first_stop else 0

        df['label'] = labels
        df = df.dropna(subset=['label'])

        return df[['label']]

    def generate_triple_barrier_labels(self, lookahead: str = '1m',
                                      stop_loss: float = 0.02,
                                      take_profit: float = 0.04,
                                      time_barrier: int = None) -> pd.DataFrame:
        """
        Triple barrier labeling method (from Advances in Financial Machine Learning)

        Args:
            lookahead: Maximum lookahead period
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            time_barrier: Time barrier (if None, use lookahead periods)
        """
        df = self.data.copy()
        periods = self._get_lookahead_periods(lookahead)
        time_barrier = time_barrier or periods

        labels = np.full(len(df), np.nan)

        for i in range(len(df) - time_barrier):
            entry_price = df['close'].iloc[i]
            stop_price = entry_price * (1 - stop_loss)
            take_profit_price = entry_price * (1 + take_profit)

            # Look at future prices up to time barrier
            future_prices = df['close'].iloc[i+1:i+time_barrier+1]

            if len(future_prices) > 0:
                # Check barriers
                stop_hit = np.any(future_prices <= stop_price)
                profit_hit = np.any(future_prices >= take_profit_price)

                if profit_hit and not stop_hit:
                    labels[i] = 1
                elif stop_hit and not profit_hit:
                    labels[i] = 0
                elif profit_hit and stop_hit:
                    # Check which happened first
                    first_stop = np.argmax(future_prices <= stop_price) if stop_hit else np.inf
                    first_profit = np.argmax(future_prices >= take_profit_price) if profit_hit else np.inf
                    labels[i] = 1 if first_profit < first_stop else 0
                else:
                    # Time barrier hit - label based on final return
                    final_return = (future_prices.iloc[-1] / entry_price - 1)
                    labels[i] = 1 if final_return > 0 else 0

        df['label'] = labels
        df = df.dropna(subset=['label'])

        return df[['label']]
