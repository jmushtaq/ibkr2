import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import logging

from .feature_engine import TrendFollowingFeatures
from .label_generator import TrendFollowingLabelGenerator
from .models import MLTrendDataset, TrendConfig

logger = logging.getLogger(__name__)


class TrendMLDatasetBuilder:
    """
    Build ML datasets for trend following strategy
    """

    def __init__(self, data: pd.DataFrame, config: TrendConfig, symbol=None):
        """
        Initialize dataset builder

        Args:
            data: OHLCV DataFrame
            config: TrendConfig instance
            symbol: Optional symbol object
        """
        self.data = data.copy()
        self.config = config
        self.symbol = symbol
        self.features = None
        self.labels = None

    def build_dataset(self, rr_ratio: float = 2.0, lookahead_periods: int = 21) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Build complete dataset with features and labels

        Args:
            rr_ratio: Risk:Reward ratio for labeling
            lookahead_periods: Maximum lookahead periods
        """
        logger.info(f"Building dataset with RR {rr_ratio}:1")

        # Generate features
        feature_engine = TrendFollowingFeatures(self.data, self.config)
        self.features = feature_engine.get_entry_features()
        logger.info(f"Generated {len(self.features.columns)} features")
        logger.info(f"Feature names: {list(self.features.columns)[:10]}...")

        # Generate signals (for filtering)
        signals = feature_engine.get_signal_conditions()

        # Log signal statistics
        long_signals = signals['long_signal'].sum()
        short_signals = signals['short_signal'].sum()
        logger.info(f"Long signals: {long_signals}, Short signals: {short_signals}")

        # Generate labels
        label_gen = TrendFollowingLabelGenerator(self.data, self.config.default_stop_loss)
        labels = label_gen.generate_labels_for_rr_ratio(rr_ratio, lookahead_periods)

        # Combine features and labels
        dataset = pd.concat([self.features, labels['label']], axis=1)
        dataset.columns = list(self.features.columns) + ['label']

        # Only keep rows where we have valid labels
        dataset = dataset[dataset['label'] >= 0]
        logger.info(f"After label filtering: {len(dataset)} samples")

        # Filter by signal conditions (only where our strategy would have entered)
        # Get the indices where we have long signals
        long_signal_indices = signals[signals['long_signal'] == 1].index
        dataset = dataset.loc[dataset.index.intersection(long_signal_indices)]
        logger.info(f"After long signal filtering: {len(dataset)} samples")

        if len(dataset) == 0:
            logger.warning("No samples after strict filtering. Trying with relaxed conditions...")

            # Get the features DataFrame with individual signal components
            # We need to use the original features which contain the individual signal components
            relaxed_conditions = pd.Series(False, index=self.features.index)

            # Use stochastic cross above OR regression cross up, AND uptrend
            if 'stoch_cross_above' in self.features.columns:
                relaxed_conditions |= (self.features['stoch_cross_above'] == 1)
                logger.info("  Using stochastic cross above condition")

            if 'regression_cross_up' in self.features.columns:
                relaxed_conditions |= (self.features['regression_cross_up'] == 1)
                logger.info("  Using regression cross up condition")

            # Also require uptrend
            if 'is_uptrend' in self.features.columns:
                relaxed_conditions &= (self.features['is_uptrend'] == 1)
                logger.info("  Requiring uptrend")

            # Create dataset with relaxed conditions
            dataset = pd.concat([self.features, labels['label']], axis=1)
            dataset = dataset[dataset['label'] >= 0]
            dataset = dataset.loc[dataset.index.intersection(self.features.index[relaxed_conditions])]
            logger.info(f"After relaxed filtering: {len(dataset)} samples")

        if len(dataset) == 0:
            logger.error("Still no samples after relaxed filtering")
            # Return empty dataset with a warning
            empty_df = pd.DataFrame(columns=list(self.features.columns) + ['label'])
            return empty_df, StandardScaler()

        logger.info(f"Final dataset size: {len(dataset)} samples")

        # Log class distribution
        class_dist = dataset['label'].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        # Check class balance
        if len(class_dist) < 2:
            logger.warning(f"Only one class present: {class_dist}")
            return dataset, StandardScaler()

        # Scale features
        feature_cols = [c for c in dataset.columns if c != 'label']
        scaler = StandardScaler()

        # Handle any remaining NaN values
        dataset = dataset.dropna()

        if len(dataset) == 0:
            logger.error("No samples after dropping NaN")
            return pd.DataFrame(columns=feature_cols + ['label']), StandardScaler()

        dataset_scaled = dataset.copy()
        dataset_scaled[feature_cols] = scaler.fit_transform(dataset[feature_cols])

        return dataset_scaled, scaler

    def save_dataset(self, dataset: pd.DataFrame, scaler, name: str, description: str):
        """
        Save dataset to database
        """
        if len(dataset) == 0:
            logger.error("Cannot save empty dataset")
            return None

        # Prepare data for JSON storage
        dataset_dict = {
            'samples': dataset.reset_index().to_dict('records'),
            'feature_names': [c for c in dataset.columns if c != 'label'],
            'scaler_type': type(scaler).__name__,
        }

        # Create MLTrendDataset record
        try:
            mldataset = MLTrendDataset.objects.create(
                name=name,
                description=description,
                config=self.config,
                data=dataset_dict,
                symbol=self.symbol,
                frequency='1D',
                start_date=dataset.index.min().date() if hasattr(dataset.index, 'min') else None,
                end_date=dataset.index.max().date() if hasattr(dataset.index, 'max') else None,
                total_samples=len(dataset),
                feature_names=dataset_dict['feature_names']
            )

            # Save scaler
            scaler_path = f"ml_models/trend_scalers/{name}_scaler.pkl"
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

            logger.info(f"Dataset saved: {name} with {len(dataset)} samples")

            return mldataset

        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return None

