import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrendFollowingLabelGenerator:
    """
    Generate labels for trend following with different risk:reward ratios
    """

    def __init__(self, data: pd.DataFrame, stop_loss_pct: float = 0.02):
        """
        Initialize label generator

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close
            stop_loss_pct: Stop loss percentage (default 2%)
        """
        self.data = data.copy()
        self.stop_loss_pct = stop_loss_pct

    def generate_labels_for_rr_ratio(self, rr_ratio: float, lookahead_periods: int = 21) -> pd.DataFrame:
        """
        Generate labels for a specific risk:reward ratio

        Args:
            rr_ratio: Risk:Reward ratio (e.g., 1.5, 2.0, 3.0)
            lookahead_periods: Maximum lookahead periods

        Returns:
            DataFrame with labels:
            1 = Take profit hit first
            0 = Stop loss hit first
            -1 = Neither hit (neutral/exit)
        """
        take_profit_pct = self.stop_loss_pct * rr_ratio

        labels = np.full(len(self.data), -1)

        # Only process if we have enough data
        if len(self.data) <= lookahead_periods:
            logger.warning(f"Not enough data: {len(self.data)} rows, need at least {lookahead_periods + 1}")
            result_df = pd.DataFrame({
                'label': labels,
                'entry_price': self.data['close'],
                'stop_loss': self.data['close'] * (1 - self.stop_loss_pct),
                'take_profit': self.data['close'] * (1 + take_profit_pct),
                'rr_ratio': rr_ratio,
                'lookahead_periods': lookahead_periods
            }, index=self.data.index)
            return result_df

        for i in range(len(self.data) - lookahead_periods):
            entry_price = self.data['close'].iloc[i]
            stop_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            # Look ahead up to lookahead_periods
            future_highs = self.data['high'].iloc[i+1:i+lookahead_periods+1]
            future_lows = self.data['low'].iloc[i+1:i+lookahead_periods+1]

            if len(future_highs) > 0:
                # Check if stop loss or take profit is hit
                stop_hit = np.any(future_lows <= stop_price)
                profit_hit = np.any(future_highs >= take_profit_price)

                if profit_hit and not stop_hit:
                    labels[i] = 1  # Take profit hit
                elif stop_hit and not profit_hit:
                    labels[i] = 0  # Stop loss hit
                elif profit_hit and stop_hit:
                    # Check which happened first
                    first_stop_idx = np.argmax(future_lows <= stop_price) if stop_hit else np.inf
                    first_profit_idx = np.argmax(future_highs >= take_profit_price) if profit_hit else np.inf
                    labels[i] = 1 if first_profit_idx < first_stop_idx else 0

        result_df = pd.DataFrame({
            'label': labels,
            'entry_price': self.data['close'],
            'stop_loss': self.data['close'] * (1 - self.stop_loss_pct),
            'take_profit': self.data['close'] * (1 + take_profit_pct),
            'rr_ratio': rr_ratio,
            'lookahead_periods': lookahead_periods
        }, index=self.data.index)

        return result_df

    def generate_all_rr_labels(self, rr_ratios: List[float] = [1.5, 2.0, 3.0]) -> Dict[float, pd.DataFrame]:
        """
        Generate labels for multiple risk:reward ratios
        """
        results = {}
        for rr in rr_ratios:
            logger.info(f"Generating labels for RR {rr}:1")
            results[rr] = self.generate_labels_for_rr_ratio(rr)

            # Print class distribution
            labels = results[rr]['label']
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                tp_count = (valid_labels == 1).sum()
                sl_count = (valid_labels == 0).sum()
                logger.info(f"  TP hits: {tp_count}, SL hits: {sl_count}, Ratio: {tp_count/sl_count:.2f}")

        return results

    def get_best_rr_ratio(self, rr_ratios: List[float] = [1.5, 2.0, 3.0]) -> Tuple[float, pd.DataFrame]:
        """
        Find which RR ratio gives the best risk-adjusted returns
        """
        results = self.generate_all_rr_labels(rr_ratios)

        best_ratio = None
        best_score = -np.inf

        for rr, df in results.items():
            valid = df[df['label'] >= 0]
            if len(valid) > 0:
                # Score = (win_rate * rr) - (loss_rate * 1)
                win_rate = (valid['label'] == 1).sum() / len(valid)
                score = (win_rate * rr) - ((1 - win_rate) * 1)

                logger.info(f"RR {rr}:1 - Win Rate: {win_rate:.2%}, Score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_ratio = rr

        return best_ratio, results.get(best_ratio)

