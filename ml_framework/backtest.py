import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

class Backtester:
    """
    Backtest trading strategies
    """

    def __init__(self, data: pd.DataFrame, model: object, feature_names: List[str]):
        """
        Initialize backtester

        Args:
            data: OHLCV data for backtesting
            model: Trained ML model
            feature_names: List of feature names
        """
        self.data = data.copy()
        self.model = model
        self.feature_names = feature_names

    def run_backtest(self,
                    initial_capital: float = 100000,
                    position_size: float = 0.1,
                    stop_loss: float = None,
                    take_profit: float = None,
                    commission: float = 0.001,
                    min_confidence: float = 0.6) -> Dict:
        """
        Run backtest using model predictions
        """
        # Make predictions
        X = self.data[self.feature_names].values
        predictions = self.model.predict(X)

        # For classifiers, get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = np.ones(len(predictions)) * 0.5

        # Initialize tracking variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []

        for i in range(len(self.data)):
            date = self.data.index[i]
            price = self.data['close'].iloc[i]
            pred = predictions[i]
            prob = probabilities[i] if i < len(probabilities) else 0.5

            # Check stop loss and take profit if position is open
            if position != 0:
                # Calculate returns
                if position > 0:  # Long position
                    price_change = (price - entry_price) / entry_price

                    # Check stop loss
                    if stop_loss and price_change <= -stop_loss:
                        capital = position * price * (1 - commission)
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'direction': 'long',
                            'return': price_change,
                            'reason': 'stop_loss'
                        })
                        position = 0

                    # Check take profit
                    elif take_profit and price_change >= take_profit:
                        capital = position * price * (1 - commission)
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'direction': 'long',
                            'return': price_change,
                            'reason': 'take_profit'
                        })
                        position = 0

                elif position < 0:  # Short position
                    price_change = (entry_price - price) / entry_price

                    # Check stop loss
                    if stop_loss and price_change <= -stop_loss:
                        capital = -position * (2 * entry_price - price) * (1 - commission)
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'direction': 'short',
                            'return': price_change,
                            'reason': 'stop_loss'
                        })
                        position = 0

                    # Check take profit
                    elif take_profit and price_change >= take_profit:
                        capital = -position * (2 * entry_price - price) * (1 - commission)
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'direction': 'short',
                            'return': price_change,
                            'reason': 'take_profit'
                        })
                        position = 0

            # Enter new position if no position open
            if position == 0 and prob >= min_confidence:
                # Determine direction based on prediction
                if pred > 0.5:  # Long signal
                    position_size_value = capital * position_size
                    position = position_size_value / price
                    entry_price = price
                    entry_date = date
                elif pred < 0.5:  # Short signal
                    position_size_value = capital * position_size
                    position = -position_size_value / price
                    entry_price = price
                    entry_date = date

            # Update equity
            if position != 0:
                if position > 0:
                    equity = capital + position * price
                else:
                    equity = capital - position * price
            else:
                equity = capital

            equity_curve.append({
                'date': date,
                'equity': equity,
                'price': price
            })

        # Close any remaining position at the end
        if position != 0:
            last_price = self.data['close'].iloc[-1]
            if position > 0:
                final_return = (last_price - entry_price) / entry_price
                capital = position * last_price * (1 - commission)
            else:
                final_return = (entry_price - last_price) / entry_price
                capital = -position * (2 * entry_price - last_price) * (1 - commission)

            trades.append({
                'entry_date': entry_date,
                'exit_date': self.data.index[-1],
                'entry_price': entry_price,
                'exit_price': last_price,
                'direction': 'long' if position > 0 else 'short',
                'return': final_return,
                'reason': 'end_of_period'
            })

        # Calculate performance metrics
        metrics = self._calculate_metrics(trades, equity_curve, initial_capital)

        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict], initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        # Calculate returns
        final_equity = equity_curve[-1]['equity']
        total_return = (final_equity - initial_capital) / initial_capital

        # Annualized return (assuming daily data)
        days = len(equity_curve)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # Calculate returns for Sharpe ratio
        equity_values = [point['equity'] for point in equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

        # Calculate maximum drawdown
        equity_series = pd.Series([point['equity'] for point in equity_curve])
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trade statistics
        winning_trades = [t for t in trades if t['return'] > 0]
        losing_trades = [t for t in trades if t['return'] <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum([t['return'] for t in winning_trades])
        gross_loss = abs(sum([t['return'] for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
