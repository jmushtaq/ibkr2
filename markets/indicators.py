"""
Technical indicators module using TA-Lib
"""
import talib
import numpy as np
import pandas as pd
import math

class TechnicalIndicators:
    """Class to calculate various technical indicators"""

    @staticmethod
    def _clean_nan(data_list):
        """Convert NaN values to None (null in JSON)"""
        return [None if math.isnan(x) else x for x in data_list]

    @staticmethod
    def get_available_indicators():
        """Return list of available indicators with their parameters"""
        return [
            # Overlay indicators (plotted on main chart)
            {
                'id': 'sma',
                'name': 'Simple Moving Average',
                'category': 'Overlay',
                'type': 'overlay',
                'params': [{'name': 'period', 'default': 20, 'min': 5, 'max': 200}]
            },
            {
                'id': 'ema',
                'name': 'Exponential Moving Average',
                'category': 'Overlay',
                'type': 'overlay',
                'params': [{'name': 'period', 'default': 20, 'min': 5, 'max': 200}]
            },
            {
                'id': 'bbands',
                'name': 'Bollinger Bands',
                'category': 'Overlay',
                'type': 'overlay',
                'params': [
                    {'name': 'period', 'default': 20, 'min': 5, 'max': 200},
                    {'name': 'nbdevup', 'default': 2, 'min': 1, 'max': 4},
                    {'name': 'nbdevdn', 'default': 2, 'min': 1, 'max': 4}
                ]
            },
            {
                'id': 'atr',
                'name': 'Average True Range',
                'category': 'Overlay',
                'type': 'overlay',
                'params': [{'name': 'period', 'default': 14, 'min': 5, 'max': 50}]
            },
            {
                'id': 'psar',
                'name': 'Parabolic SAR',
                'category': 'Overlay',
                'type': 'overlay',
                'params': [
                    {'name': 'acceleration', 'default': 0.02, 'min': 0.01, 'max': 0.1},
                    {'name': 'maximum', 'default': 0.2, 'min': 0.1, 'max': 0.5}
                ]
            },

            # Oscillator indicators (go in subplots)
            {
                'id': 'rsi',
                'name': 'Relative Strength Index',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [{'name': 'period', 'default': 14, 'min': 5, 'max': 50}]
            },
            {
                'id': 'macd',
                'name': 'MACD',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [
                    {'name': 'fastperiod', 'default': 12, 'min': 5, 'max': 50},
                    {'name': 'slowperiod', 'default': 26, 'min': 10, 'max': 100},
                    {'name': 'signalperiod', 'default': 9, 'min': 5, 'max': 50}
                ]
            },
            {
                'id': 'stoch',
                'name': 'Stochastic Oscillator',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [
                    {'name': 'fastk_period', 'default': 14, 'min': 5, 'max': 50},
                    {'name': 'slowk_period', 'default': 3, 'min': 1, 'max': 10},
                    {'name': 'slowd_period', 'default': 3, 'min': 1, 'max': 10}
                ]
            },
            {
                'id': 'cci',
                'name': 'Commodity Channel Index',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [{'name': 'period', 'default': 20, 'min': 5, 'max': 100}]
            },
            {
                'id': 'willr',
                'name': 'Williams %R',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [{'name': 'period', 'default': 14, 'min': 5, 'max': 50}]
            },
            {
                'id': 'mfi',
                'name': 'Money Flow Index',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [{'name': 'period', 'default': 14, 'min': 5, 'max': 50}]
            },
            {
                'id': 'adx',
                'name': 'Average Directional Index',
                'category': 'Oscillator',
                'type': 'oscillator',
                'params': [{'name': 'period', 'default': 14, 'min': 5, 'max': 50}]
            },

            # Volume indicators
            {
                'id': 'obv',
                'name': 'On-Balance Volume',
                'category': 'Volume',
                'type': 'volume',
                'params': []
            }
        ]

    @staticmethod
    def calculate_indicators(df, indicators):
        """
        Calculate selected indicators for the given DataFrame

        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicator IDs to calculate

        Returns:
            Dictionary of indicator results
        """
        results = {}

        # Extract price arrays
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        for indicator_id in indicators:
            try:
                if indicator_id == 'sma':
                    results['sma'] = {
                        'data': TechnicalIndicators._clean_nan(talib.SMA(close_prices, timeperiod=20).tolist()),
                        'name': 'SMA 20',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }

                elif indicator_id == 'ema':
                    results['ema'] = {
                        'data': TechnicalIndicators._clean_nan(talib.EMA(close_prices, timeperiod=20).tolist()),
                        'name': 'EMA 20',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }

                elif indicator_id == 'bbands':
                    upper, middle, lower = talib.BBANDS(
                        close_prices,
                        timeperiod=20,
                        nbdevup=2,
                        nbdevdn=2
                    )
                    results['bbands_upper'] = {
                        'data': TechnicalIndicators._clean_nan(upper.tolist()),
                        'name': 'BB Upper',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }
                    results['bbands_middle'] = {
                        'data': TechnicalIndicators._clean_nan(middle.tolist()),
                        'name': 'BB Middle',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }
                    results['bbands_lower'] = {
                        'data': TechnicalIndicators._clean_nan(lower.tolist()),
                        'name': 'BB Lower',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }

                elif indicator_id == 'rsi':
                    results['rsi'] = {
                        'data': TechnicalIndicators._clean_nan(talib.RSI(close_prices, timeperiod=14).tolist()),
                        'name': 'RSI 14',
                        'yaxis': 'y3',  # Will go to first subplot
                        'type': 'oscillator',
                        'range': [0, 100]
                    }

                elif indicator_id == 'macd':
                    macd, signal, hist = talib.MACD(
                        close_prices,
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                    results['macd'] = {
                        'data': TechnicalIndicators._clean_nan(macd.tolist()),
                        'name': 'MACD',
                        'yaxis': 'y3',  # Will go to first subplot
                        'type': 'oscillator'
                    }
                    results['macd_signal'] = {
                        'data': TechnicalIndicators._clean_nan(signal.tolist()),
                        'name': 'Signal',
                        'yaxis': 'y3',
                        'type': 'oscillator'
                    }
                    results['macd_hist'] = {
                        'data': TechnicalIndicators._clean_nan(hist.tolist()),
                        'name': 'Histogram',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'subtype': 'bar'
                    }

                elif indicator_id == 'stoch':
                    slowk, slowd = talib.STOCH(
                        high_prices,
                        low_prices,
                        close_prices,
                        fastk_period=14,
                        slowk_period=3,
                        slowk_matype=0,
                        slowd_period=3,
                        slowd_matype=0
                    )
                    results['stoch_k'] = {
                        'data': TechnicalIndicators._clean_nan(slowk.tolist()),
                        'name': 'Stoch %K',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'range': [0, 100]
                    }
                    results['stoch_d'] = {
                        'data': TechnicalIndicators._clean_nan(slowd.tolist()),
                        'name': 'Stoch %D',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'range': [0, 100]
                    }

                elif indicator_id == 'atr':
                    results['atr'] = {
                        'data': TechnicalIndicators._clean_nan(talib.ATR(high_prices, low_prices, close_prices, timeperiod=14).tolist()),
                        'name': 'ATR 14',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }

                elif indicator_id == 'obv':
                    results['obv'] = {
                        'data': TechnicalIndicators._clean_nan(talib.OBV(close_prices, volume).tolist()),
                        'name': 'OBV',
                        'yaxis': 'y4',  # Different subplot for volume indicators
                        'type': 'volume'
                    }

                elif indicator_id == 'adx':
                    results['adx'] = {
                        'data': TechnicalIndicators._clean_nan(talib.ADX(high_prices, low_prices, close_prices, timeperiod=14).tolist()),
                        'name': 'ADX 14',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'range': [0, 100]
                    }

                elif indicator_id == 'cci':
                    results['cci'] = {
                        'data': TechnicalIndicators._clean_nan(talib.CCI(high_prices, low_prices, close_prices, timeperiod=20).tolist()),
                        'name': 'CCI 20',
                        'yaxis': 'y3',
                        'type': 'oscillator'
                    }

                elif indicator_id == 'willr':
                    results['willr'] = {
                        'data': TechnicalIndicators._clean_nan(talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14).tolist()),
                        'name': 'Williams %R 14',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'range': [-100, 0]
                    }

                elif indicator_id == 'mfi':
                    results['mfi'] = {
                        'data': TechnicalIndicators._clean_nan(talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14).tolist()),
                        'name': 'MFI 14',
                        'yaxis': 'y3',
                        'type': 'oscillator',
                        'range': [0, 100]
                    }

                elif indicator_id == 'psar':
                    results['psar'] = {
                        'data': TechnicalIndicators._clean_nan(talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2).tolist()),
                        'name': 'Parabolic SAR',
                        'yaxis': 'y',
                        'type': 'overlay'
                    }

            except Exception as e:
                print(f"Error calculating {indicator_id}: {e}")

        return results
