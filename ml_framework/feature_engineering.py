import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering for financial time series data
    """

    def __init__(self, data: pd.DataFrame, frequency: str = '1D'):
        """
        Initialize feature engineer

        Args:
            data: DataFrame with OHLCV data (must have columns: open, high, low, close, volume)
            frequency: Data frequency (1min, 5min, 15min, 1H, 4H, 1D)
        """
        self.data = data.copy()
        self.frequency = frequency

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Data must contain column: {col}")

        # Convert to numeric
        for col in required_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

    def add_price_features(self) -> pd.DataFrame:
        """Add price-based features"""
        df = self.data.copy()

        # Returns at various lags
        for period in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

        # Price differences
        for period in [1, 5, 10, 20, 50]:
            df[f'price_diff_{period}d'] = df['close'] - df['close'].shift(period)

        # Price relative to various moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            ma = df['close'].rolling(window=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / ma - 1

        # High-low range features
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['high_close_ratio'] = df['high'] / df['close'] - 1
        df['low_close_ratio'] = df['low'] / df['close'] - 1
        df['close_open_ratio'] = df['close'] / df['open'] - 1

        # Rolling statistics
        for window in [5, 10, 20, 50, 100]:
            df[f'close_std_{window}d'] = df['close'].rolling(window).std()
            df[f'close_skew_{window}d'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}d'] = df['close'].rolling(window).kurt()

        return df

    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators using TA-Lib where available"""
        df = self.data.copy()

        try:
            # Ensure we have numpy arrays for TA-Lib
            open_prices = df['open'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            volume = df['volume'].values.astype(float)

            # Moving Averages
            df['sma_5'] = talib.SMA(close_prices, timeperiod=5)
            df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['sma_100'] = talib.SMA(close_prices, timeperiod=100)
            df['sma_200'] = talib.SMA(close_prices, timeperiod=200)

            df['ema_5'] = talib.EMA(close_prices, timeperiod=5)
            df['ema_10'] = talib.EMA(close_prices, timeperiod=10)
            df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
            df['ema_50'] = talib.EMA(close_prices, timeperiod=50)

            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                close_prices, fastperiod=12, slowperiod=26, signalperiod=9
            )

            # RSI
            df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            df['rsi_7'] = talib.RSI(close_prices, timeperiod=7)
            df['rsi_21'] = talib.RSI(close_prices, timeperiod=21)

            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=14, slowk_period=3, slowd_period=3
            )

            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close_prices, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (close_prices - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # ATR
            df['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df['atr_20'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)

            # ADX
            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)

            # Volume indicators
            df['obv'] = talib.OBV(close_prices, volume)
            df['obv_ema'] = talib.EMA(df['obv'].values, timeperiod=20)

            # Money Flow Index
            df['mfi_14'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)

            # Williams %R
            df['willr_14'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

            # CCI
            df['cci_20'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=20)

            # Parabolic SAR
            df['sar'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)

        except Exception as e:
            print(f"Warning: TA-Lib calculation error: {e}")

        return df

    def add_volatility_features(self) -> pd.DataFrame:
        """Add volatility-based features"""
        df = self.data.copy()

        # Returns for volatility calculation
        returns = df['close'].pct_change()

        # Realized volatility
        for window in [5, 10, 20, 50, 100]:
            df[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        # Parkinson volatility (high-low)
        for window in [20, 50, 100]:
            hl_vol = (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
            df[f'parkinson_vol_{window}d'] = hl_vol.rolling(window).mean().apply(np.sqrt) * np.sqrt(252)

        # Garman-Klass volatility (open, high, low, close)
        for window in [20, 50, 100]:
            gk_vol = (0.5 * (np.log(df['high'] / df['low']) ** 2) -
                      (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2))
            df[f'garman_klass_vol_{window}d'] = gk_vol.rolling(window).mean().apply(np.sqrt) * np.sqrt(252)

        # Volatility regime - use numeric codes instead of strings
        vol_20d = df['volatility_20d'].dropna()
        if len(vol_20d) > 0:
            low_vol = np.percentile(vol_20d, 25)
            high_vol = np.percentile(vol_20d, 75)
            df['volatility_regime'] = pd.cut(df['volatility_20d'],
                                             bins=[-np.inf, low_vol, high_vol, np.inf],
                                             labels=[1, 2, 3])
            df['volatility_regime'] = df['volatility_regime'].astype(float)

        return df

    def add_momentum_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Add momentum-based features"""
        if df is None:
            df = self.data.copy()
        else:
            # Make sure we're working with a copy and only have the close column
            df = df.copy()
            # Ensure we have the close column from the original data
            if 'close' not in df.columns:
                df['close'] = self.data['close']

        # Price momentum
        for period in [10, 20, 50, 100, 200]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1

        # Rate of change
        for period in [10, 20, 50, 100]:
            df[f'roc_{period}d'] = (df['close'] / df['close'].shift(period) - 1) * 100

        # Moving average convergence (check if SMA columns exist)
        for fast, slow in [(5, 20), (10, 50), (20, 100)]:
            if f'sma_{fast}' in df.columns and f'sma_{slow}' in df.columns:
                df[f'ma_cross_{fast}_{slow}'] = (df[f'sma_{fast}'] / df[f'sma_{slow}'] - 1)

        # Ultimate Oscillator
        try:
            df['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                               timeperiod1=7, timeperiod2=14, timeperiod3=28)
        except:
            pass

        return df

    def add_volume_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Add volume-based features"""
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
            # Ensure we have required columns
            if 'close' not in df.columns:
                df['close'] = self.data['close']
            if 'volume' not in df.columns:
                df['volume'] = self.data['volume']
            if 'high' not in df.columns:
                df['high'] = self.data['high']
            if 'low' not in df.columns:
                df['low'] = self.data['low']

        # Volume moving averages
        for period in [10, 20, 50]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']

        # Volume indicators
        try:
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['obv_ratio'] = df['obv'] / df['obv'].rolling(100).mean()
            df['vpt'] = talib.VPT(df['close'], df['volume'])
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            df['cmf_20'] = talib.CMF(df['high'], df['low'], df['close'], df['volume'], timeperiod=20)
        except:
            pass

        return df

    def add_pattern_recognition(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()
            # Ensure we have required columns
            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    df[col] = self.data[col]

        try:
            open_prices = df['open'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)

            # Bullish patterns
            df['pattern_hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            df['pattern_engulfing_bullish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            df['pattern_morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
            df['pattern_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)

            # Bearish patterns
            df['pattern_hanging_man'] = talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)
            df['pattern_engulfing_bearish'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices) * -1
            df['pattern_evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            df['pattern_three_black_crows'] = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)

            # Neutral patterns
            df['pattern_doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        except:
            pass

        return df

    def add_cycle_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Add cyclical/time-based features"""
        if df is None:
            df = self.data.copy()
        else:
            df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def generate_all_features(self) -> pd.DataFrame:
        """Generate all features in the correct order"""
        print("Generating price features...")
        df = self.add_price_features()

        print("Generating technical indicators...")
        df = self.add_technical_indicators()

        print("Generating volatility features...")
        df = self.add_volatility_features()

        print("Generating momentum features...")
        df = self.add_momentum_features(df)

        print("Generating volume features...")
        df = self.add_volume_features(df)

        print("Generating pattern recognition...")
        df = self.add_pattern_recognition(df)

        print("Generating cycle features...")
        df = self.add_cycle_features(df)

        # Drop rows with NaN values
        print(f"Before dropping NaN: {len(df)} rows")
        df = df.dropna()
        print(f"After dropping NaN: {len(df)} rows")

        return df
