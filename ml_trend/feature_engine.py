import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrendFollowingFeatures:
    """
    Generate features for trend following strategy
    """

    def __init__(self, data: pd.DataFrame, config):
        """
        Initialize feature generator

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume
            config: TrendConfig instance with strategy parameters
        """
        self.data = data.copy()
        self.config = config

        # Ensure data is sorted
        self.data = self.data.sort_index()

        # Calculate required indicators
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Calculate all technical indicators"""
        # Use pandas Series for operations that need shift()
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        open_prices = self.data['open']
        volume = self.data['volume']

        # Convert to numpy arrays for TA-Lib
        close_array = close.values.astype(float)
        high_array = high.values.astype(float)
        low_array = low.values.astype(float)
        open_array = open_prices.values.astype(float)
        volume_array = volume.values.astype(float)

        # 1. Moving Averages for trend identification
        self.data['sma_fast'] = talib.SMA(close_array, timeperiod=self.config.fast_sma)
        self.data['sma_slow'] = talib.SMA(close_array, timeperiod=self.config.slow_sma)
        self.data['trend_strength'] = (self.data['sma_fast'] / self.data['sma_slow'] - 1) * 100
        self.data['is_uptrend'] = (self.data['sma_fast'] > self.data['sma_slow']).astype(int)

        # 2. EMA for pullback zones
        self.data['ema_fast'] = talib.EMA(close_array, timeperiod=self.config.pullback_ema_fast)
        self.data['ema_slow'] = talib.EMA(close_array, timeperiod=self.config.pullback_ema_slow)
        self.data['price_to_ema_fast'] = (close / self.data['ema_fast'] - 1) * 100
        self.data['price_to_ema_slow'] = (close / self.data['ema_slow'] - 1) * 100

        # 3. Stochastic oscillator
        stoch_k, stoch_d = talib.STOCH(
            high_array, low_array, close_array,
            fastk_period=self.config.stochastic_period,
            slowk_period=3,
            slowd_period=3
        )
        self.data['stoch_k'] = stoch_k
        self.data['stoch_d'] = stoch_d

        # Stochastic conditions (using pandas Series for shift)
        self.data['stoch_oversold'] = (self.data['stoch_k'] < self.config.stochastic_oversold).astype(int)
        self.data['stoch_overbought'] = (self.data['stoch_k'] > self.config.stochastic_overbought).astype(int)

        # Stochastic cross signals - ensure we have valid values
        self.data['stoch_cross_above'] = 0
        self.data['stoch_cross_below'] = 0

        # Only calculate cross signals where we have previous values
        for i in range(1, len(self.data)):
            if not pd.isna(self.data['stoch_k'].iloc[i]) and not pd.isna(self.data['stoch_k'].iloc[i-1]):
                # Cross above oversold
                if (self.data['stoch_k'].iloc[i] > self.config.stochastic_oversold and
                    self.data['stoch_k'].iloc[i-1] <= self.config.stochastic_oversold):
                    self.data.loc[self.data.index[i], 'stoch_cross_above'] = 1

                # Cross below overbought
                if (self.data['stoch_k'].iloc[i] < self.config.stochastic_overbought and
                    self.data['stoch_k'].iloc[i-1] >= self.config.stochastic_overbought):
                    self.data.loc[self.data.index[i], 'stoch_cross_below'] = 1

        # 4. VWAP calculation
        self.data['vwap'] = self._calculate_vwap()
        self.data['vwap_deviation'] = ((close - self.data['vwap']) / self.data['vwap']) * 100
        self.data['vwap_deviation_normalized'] = np.clip(self.data['vwap_deviation'] / 5, -1, 1)

        # VWAP conditions
        self.data['below_vwap'] = (close < self.data['vwap']).astype(int)
        self.data['above_vwap'] = (close > self.data['vwap']).astype(int)

        # 5. Linear regression trend
        self.data['regression_slope'] = self._calculate_regression_slope('close', self.config.regression_period)
        self.data['regression_signal'] = np.where(
            self.data['regression_slope'] > self.config.regression_threshold, 1,
            np.where(self.data['regression_slope'] < -self.config.regression_threshold, -1, 0)
        )

        # Regression cross signals
        self.data['regression_cross_up'] = 0
        self.data['regression_cross_down'] = 0

        for i in range(1, len(self.data)):
            if not pd.isna(self.data['regression_signal'].iloc[i]) and not pd.isna(self.data['regression_signal'].iloc[i-1]):
                # Cross up (from -1 or 0 to 1)
                if (self.data['regression_signal'].iloc[i] > 0 and
                    self.data['regression_signal'].iloc[i-1] <= 0):
                    self.data.loc[self.data.index[i], 'regression_cross_up'] = 1

                # Cross down (from 1 or 0 to -1)
                if (self.data['regression_signal'].iloc[i] < 0 and
                    self.data['regression_signal'].iloc[i-1] >= 0):
                    self.data.loc[self.data.index[i], 'regression_cross_down'] = 1

        # 6. Additional features
        # ATR for volatility
        self.data['atr'] = talib.ATR(high_array, low_array, close_array, timeperiod=14)
        self.data['atr_pct'] = (self.data['atr'] / close) * 100

        # Volume features
        self.data['volume_sma'] = talib.SMA(volume_array, timeperiod=20)
        self.data['volume_ratio'] = volume / self.data['volume_sma']

        # Price features (using pandas Series for shift)
        self.data['return_1d'] = close / close.shift(1) - 1
        self.data['return_5d'] = close / close.shift(5) - 1
        self.data['return_10d'] = close / close.shift(10) - 1

        # Range features
        self.data['range'] = high - low
        self.data['range_pct'] = (high - low) / close * 100

        # Fill NaN values
        self.data = self.data.fillna(method='ffill').fillna(0)

        # Log some statistics
        logger.info(f"Trend following features calculated")
        logger.info(f"  Uptrend periods: {self.data['is_uptrend'].sum()}")
        logger.info(f"  Stochastic cross above: {self.data['stoch_cross_above'].sum()}")
        logger.info(f"  Regression cross up: {self.data['regression_cross_up'].sum()}")
        logger.info(f"  Below VWAP: {self.data['below_vwap'].sum()}")

    def _calculate_vwap(self) -> pd.Series:
        """Calculate VWAP based on anchor"""
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3

        if self.config.vwap_anchor == 'session':
            # Daily VWAP - cumulative
            cumulative_tp_volume = (typical_price * self.data['volume']).cumsum()
            cumulative_volume = self.data['volume'].cumsum()
            return cumulative_tp_volume / cumulative_volume
        elif self.config.vwap_anchor == 'week':
            # Weekly VWAP - rolling 5-day VWAP
            return (typical_price * self.data['volume']).rolling(5).sum() / self.data['volume'].rolling(5).sum()
        else:
            # Monthly VWAP - rolling 21-day VWAP
            return (typical_price * self.data['volume']).rolling(21).sum() / self.data['volume'].rolling(21).sum()

    def _calculate_regression_slope(self, column: str, period: int) -> pd.Series:
        """Calculate rolling linear regression slope"""
        slopes = []
        for i in range(len(self.data)):
            if i >= period - 1:
                y = self.data[column].iloc[i-period+1:i+1].values
                x = np.arange(len(y))
                if len(y) > 1 and not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(0)
            else:
                slopes.append(0)
        return pd.Series(slopes, index=self.data.index)

    def get_entry_features(self) -> pd.DataFrame:
        """
        Generate features for entry signal detection
        """
        features = pd.DataFrame(index=self.data.index)

        # Trend features
        features['trend_strength'] = self.data['trend_strength']
        features['is_uptrend'] = self.data['is_uptrend']

        # Pullback features
        features['price_to_ema_fast'] = self.data['price_to_ema_fast']
        features['price_to_ema_slow'] = self.data['price_to_ema_slow']

        # Stochastic features
        features['stoch_k'] = self.data['stoch_k']
        features['stoch_d'] = self.data['stoch_d']
        features['stoch_oversold'] = self.data['stoch_oversold']
        features['stoch_overbought'] = self.data['stoch_overbought']
        features['stoch_cross_above'] = self.data['stoch_cross_above']
        features['stoch_cross_below'] = self.data['stoch_cross_below']

        # VWAP features
        features['vwap_deviation'] = self.data['vwap_deviation']
        features['vwap_deviation_normalized'] = self.data['vwap_deviation_normalized']
        features['below_vwap'] = self.data['below_vwap']
        features['above_vwap'] = self.data['above_vwap']

        # Regression features
        features['regression_slope'] = self.data['regression_slope']
        features['regression_signal'] = self.data['regression_signal']
        features['regression_cross_up'] = self.data['regression_cross_up']
        features['regression_cross_down'] = self.data['regression_cross_down']

        # Volume features
        features['volume_ratio'] = self.data['volume_ratio']

        # Volatility features
        features['atr_pct'] = self.data['atr_pct']
        features['range_pct'] = self.data['range_pct']

        # Return features (momentum)
        features['return_1d'] = self.data['return_1d']
        features['return_5d'] = self.data['return_5d']

        return features

    def get_signal_conditions(self) -> pd.DataFrame:
        """
        Generate signal conditions for long and short entries
        """
        signals = pd.DataFrame(index=self.data.index)

        # LONG SIGNAL CONDITIONS
        long_conditions = pd.Series(True, index=self.data.index)

        if self.config.use_ema_pullback:
            # Price near EMA zone (within 2% of fast EMA and above slow EMA for uptrend)
            ema_condition = (
                (abs(self.data['price_to_ema_fast']) < 2) &  # Near fast EMA
                (self.data['price_to_ema_slow'] > -2)  # Above slow EMA
            )
            long_conditions &= ema_condition

        if self.config.use_stochastic:
            # Stochastic oversold crossover
            stoch_condition = (self.data['stoch_cross_above'] == 1)
            long_conditions &= stoch_condition

        if self.config.use_vwap_deviation:
            # Pullback below VWAP
            vwap_condition = self.data['below_vwap'] == 1
            long_conditions &= vwap_condition

        if self.config.use_regression_trend:
            # Regression turns positive
            reg_condition = (self.data['regression_cross_up'] == 1)
            long_conditions &= reg_condition

        # Also require uptrend
        long_conditions &= (self.data['is_uptrend'] == 1)

        signals['long_signal'] = long_conditions.astype(int)

        # SHORT SIGNAL CONDITIONS
        short_conditions = pd.Series(True, index=self.data.index)

        if self.config.use_ema_pullback:
            # Price near EMA zone for downtrend
            ema_condition_short = (
                (abs(self.data['price_to_ema_fast']) < 2) &  # Near fast EMA
                (self.data['price_to_ema_slow'] < 2)  # Below slow EMA
            )
            short_conditions &= ema_condition_short

        if self.config.use_stochastic:
            # Stochastic overbought crossover
            stoch_condition_short = (self.data['stoch_cross_below'] == 1)
            short_conditions &= stoch_condition_short

        if self.config.use_vwap_deviation:
            # Pullback above VWAP
            vwap_condition_short = self.data['above_vwap'] == 1
            short_conditions &= vwap_condition_short

        if self.config.use_regression_trend:
            # Regression turns negative
            reg_condition_short = (self.data['regression_cross_down'] == 1)
            short_conditions &= reg_condition_short

        # Also require downtrend
        short_conditions &= (self.data['is_uptrend'] == 0)

        signals['short_signal'] = short_conditions.astype(int)

        # Log signal statistics
        logger.info(f"Long signals generated: {signals['long_signal'].sum()}")
        logger.info(f"Short signals generated: {signals['short_signal'].sum()}")

        return signals
