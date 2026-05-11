import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
import os

from .models import MLDataset, FeatureSet, LabeledDataset
from .feature_engineering import FeatureEngineer
from .label_generator import LabelGenerator

class MLDatasetManager:
    """
    Manages creation and storage of ML datasets
    """

    def __init__(self, data: pd.DataFrame, symbol: str = None, frequency: str = '1D'):
        """
        Initialize dataset manager

        Args:
            data: OHLCV DataFrame
            symbol: Symbol ticker
            frequency: Data frequency
        """
        self.raw_data = data.copy()
        self.data = self._clean_data()
        self.symbol = symbol
        self.frequency = frequency
        self.features = None
        self.labels = None

        print(f"Data cleaned: {len(self.raw_data)} -> {len(self.data)} rows")

    def _clean_data(self) -> pd.DataFrame:
        """Clean the data before feature generation"""
        df = self.raw_data.copy()

        # Remove any rows with zero or negative prices
        df = df[(df['close'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['open'] > 0)]

        # Remove any rows with zero volume
        df = df[df['volume'] > 0]

        # Forward fill any missing values
        df = df.fillna(method='ffill')

        # Sort by index
        df = df.sort_index()

        return df

    def create_dataset(self,
                      feature_config: Dict,
                      label_config: Dict,
                      name: str,
                      description: str = "",
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Create a complete dataset with features and labels

        Args:
            feature_config: Configuration for feature engineering
            label_config: Configuration for label generation
            name: Dataset name
            description: Dataset description
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')

        Returns:
            DataFrame with features and labels
        """
        print(f"Creating dataset: {name}")

        try:
            # Generate features
            print("Generating features...")
            self.features = self._generate_features(feature_config)
            print(f"  Generated {len(self.features.columns)} features")

            # Generate labels
            print("Generating labels...")
            self.labels = self._generate_labels(label_config)
            print(f"  Generated {len(self.labels)} labels")

            # Merge features and labels
            dataset = pd.merge(self.features, self.labels,
                              left_index=True, right_index=True, how='inner')

            # Remove rows with NaN values
            initial_len = len(dataset)
            dataset = dataset.dropna()
            print(f"Dropped {initial_len - len(dataset)} rows with NaN values")

            if len(dataset) == 0:
                raise ValueError("No valid samples after dropping NaN values")

            # Scale features
            print(f"Scaling features using {scaler_type} scaler...")
            dataset, scaler = self._scale_features(dataset, scaler_type)

            # Store dataset in database
            self._save_to_database(dataset, name, description, feature_config, label_config, scaler)

            return dataset

        except Exception as e:
            print(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
            raise


    def _generate_features(self, config: Dict) -> pd.DataFrame:
        """Generate features based on configuration"""
        engineer = FeatureEngineer(self.data, self.frequency)

        # Generate all features if 'all' is specified
        if config.get('all'):
            return engineer.generate_all_features()

        # Otherwise, build features incrementally
        # Start with a base DataFrame that has only the index
        base_df = pd.DataFrame(index=self.data.index)

        # Add close price as a reference for later calculations
        base_df['close'] = self.data['close']
        base_df['open'] = self.data['open']
        base_df['high'] = self.data['high']
        base_df['low'] = self.data['low']
        base_df['volume'] = self.data['volume']

        feature_df = base_df.copy()

        if config.get('price'):
            price_df = engineer.add_price_features()
            # Add only the new columns (not the base OHLCV columns)
            new_cols = [col for col in price_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, price_df[new_cols]], axis=1)

        if config.get('technical'):
            tech_df = engineer.add_technical_indicators()
            new_cols = [col for col in tech_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, tech_df[new_cols]], axis=1)

        if config.get('volatility'):
            vol_df = engineer.add_volatility_features()
            new_cols = [col for col in vol_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, vol_df[new_cols]], axis=1)

        if config.get('momentum'):
            # Pass a DataFrame with at least close column
            temp_df = feature_df[['close']].copy()
            mom_df = engineer.add_momentum_features(temp_df)
            new_cols = [col for col in mom_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, mom_df[new_cols]], axis=1)

        if config.get('volume'):
            # Pass a DataFrame with required columns
            temp_df = feature_df[['close', 'volume', 'high', 'low']].copy()
            vol_df = engineer.add_volume_features(temp_df)
            new_cols = [col for col in vol_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, vol_df[new_cols]], axis=1)

        if config.get('pattern'):
            # Pass a DataFrame with OHLC columns
            temp_df = feature_df[['open', 'high', 'low', 'close']].copy()
            pattern_df = engineer.add_pattern_recognition(temp_df)
            new_cols = [col for col in pattern_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, pattern_df[new_cols]], axis=1)

        if config.get('cycle'):
            cycle_df = engineer.add_cycle_features(feature_df)
            new_cols = [col for col in cycle_df.columns if col not in feature_df.columns]
            feature_df = pd.concat([feature_df, cycle_df[new_cols]], axis=1)

        # Remove the base columns if they shouldn't be features
        # Keep them for now as they might be useful

        return feature_df

    def _generate_labels(self, config: Dict) -> pd.DataFrame:
        """Generate labels based on configuration"""
        label_gen = LabelGenerator(self.data, self.frequency)

        label_type = config.get('type', 'binary')

        if label_type == 'binary':
            return label_gen.generate_binary_labels(
                lookahead=config.get('lookahead', '1d'),
                threshold=config.get('threshold', 0.0),
                upside_only=config.get('upside_only', False)
            )
        elif label_type == 'multi_class':
            return label_gen.generate_multi_class_labels(
                lookahead=config.get('lookahead', '1d'),
                bins=config.get('bins', [-0.02, -0.01, 0.01, 0.02])
            )
        elif label_type == 'regression':
            return label_gen.generate_regression_labels(
                lookahead=config.get('lookahead', '1d')
            )
        elif label_type == 'risk_reward':
            return label_gen.generate_risk_reward_labels(
                lookahead=config.get('lookahead', '1d'),
                stop_loss=config.get('stop_loss', 0.02),
                take_profit=config.get('take_profit', 0.04)
            )
        elif label_type == 'triple_barrier':
            return label_gen.generate_triple_barrier_labels(
                lookahead=config.get('lookahead', '1m'),
                stop_loss=config.get('stop_loss', 0.02),
                take_profit=config.get('take_profit', 0.04),
                time_barrier=config.get('time_barrier', None)
            )
        else:
            raise ValueError(f"Unknown label type: {label_type}")

    def _scale_features(self, dataset: pd.DataFrame, scaler_type: str) -> Tuple[pd.DataFrame, object]:
        """Scale features - handle categorical columns appropriately"""
        # Separate features and labels
        feature_cols = [col for col in dataset.columns if col != 'label']

        # Identify numeric and categorical columns
        numeric_cols = []
        categorical_cols = []

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        print(f"  Numeric features: {len(numeric_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")

        # Handle categorical columns - convert to numeric using one-hot encoding
        if categorical_cols:
            print(f"  Encoding categorical columns: {categorical_cols}")
            dummies = pd.get_dummies(dataset[categorical_cols], prefix=categorical_cols, drop_first=True)
            dataset = pd.concat([dataset, dummies], axis=1)
            dataset = dataset.drop(columns=categorical_cols)
            feature_cols = [col for col in dataset.columns if col != 'label']
            numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(dataset[col])]

        # Prepare X matrix with only numeric columns
        X = dataset[numeric_cols].values.astype(float)
        y = dataset['label'].values if 'label' in dataset.columns else None

        # Handle infinite or missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Choose scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Fit and transform
        X_scaled = scaler.fit_transform(X)

        # Create scaled DataFrame
        scaled_df = pd.DataFrame(X_scaled, index=dataset.index, columns=numeric_cols)
        if y is not None:
            scaled_df['label'] = y

        return scaled_df, scaler

    def _save_to_database(self, dataset: pd.DataFrame, name: str, description: str,
                        feature_config: Dict, label_config: Dict, scaler: object):
        """Save dataset to database"""
        from markets.models import Symbol
        from django.db import IntegrityError

        # Create or get FeatureSet
        try:
            feature_set, created = FeatureSet.objects.get_or_create(
                name=f"{name}_features",
                defaults={
                    'description': f"Features for {name}",
                    'feature_type': 'custom',
                    'features_json': feature_config
                }
            )
            if created:
                print(f"  Created new FeatureSet: {feature_set.name}")
            else:
                print(f"  Using existing FeatureSet: {feature_set.name}")
        except IntegrityError as e:
            print(f"  Error creating FeatureSet: {e}")
            # Try to get existing one
            feature_set = FeatureSet.objects.get(name=f"{name}_features")
            print(f"  Using existing FeatureSet: {feature_set.name}")

        # Create or get LabeledDataset
        try:
            labeled_dataset, created = LabeledDataset.objects.get_or_create(
                name=f"{name}_labels",
                defaults={
                    'description': f"Labels for {name}",
                    'label_type': label_config.get('type', 'binary'),
                    'config': label_config
                }
            )
            if created:
                print(f"  Created new LabeledDataset: {labeled_dataset.name}")
            else:
                print(f"  Using existing LabeledDataset: {labeled_dataset.name}")
        except IntegrityError as e:
            print(f"  Error creating LabeledDataset: {e}")
            labeled_dataset = LabeledDataset.objects.get(name=f"{name}_labels")
            print(f"  Using existing LabeledDataset: {labeled_dataset.name}")

        # Prepare data for JSON storage
        # Convert datetime index to string for JSON serialization
        dataset_copy = dataset.copy()

        # Handle the index properly for JSON serialization
        if isinstance(dataset_copy.index, pd.DatetimeIndex):
            dataset_copy['timestamp'] = dataset_copy.index.astype(str)
        else:
            dataset_copy['timestamp'] = dataset_copy.index

        # Replace any NaN values with None for JSON serialization
        dataset_copy = dataset_copy.replace({np.nan: None})

        dataset_dict = {
            'samples': dataset_copy.reset_index(drop=True).to_dict('records'),
            'feature_names': [col for col in dataset.columns if col != 'label'],
            'scaler_type': type(scaler).__name__,
        }

        # Get the Symbol object if symbol is provided
        symbol_obj = None
        if self.symbol:
            try:
                # If self.symbol is a string, get the Symbol object
                if isinstance(self.symbol, str):
                    symbol_obj = Symbol.objects.get(ticker=self.symbol)
                else:
                    symbol_obj = self.symbol
                print(f"  Using symbol: {symbol_obj.ticker}")
            except Symbol.DoesNotExist:
                print(f"Warning: Symbol {self.symbol} not found, saving without symbol")
                symbol_obj = None

        # Create or get MLDataset
        try:
            mldataset, created = MLDataset.objects.get_or_create(
                name=name,
                defaults={
                    'description': description,
                    'feature_set': feature_set,
                    'labeled_dataset': labeled_dataset,
                    'data': dataset_dict,
                    'symbol': symbol_obj,
                    'frequency': self.frequency,
                    'start_date': dataset.index.min().date() if hasattr(dataset.index, 'min') else None,
                    'end_date': dataset.index.max().date() if hasattr(dataset.index, 'max') else None,
                    'total_samples': len(dataset),
                    'feature_names': dataset_dict['feature_names'],
                    'version': '1.0.0'
                }
            )

            if created:
                print(f"  Created new MLDataset: {mldataset.name}")
            else:
                print(f"  Updating existing MLDataset: {mldataset.name}")
                # Update the existing dataset with new data
                mldataset.data = dataset_dict
                mldataset.total_samples = len(dataset)
                mldataset.feature_names = dataset_dict['feature_names']
                mldataset.start_date = dataset.index.min().date() if hasattr(dataset.index, 'min') else None
                mldataset.end_date = dataset.index.max().date() if hasattr(dataset.index, 'max') else None
                mldataset.save()

        except IntegrityError as e:
            print(f"  Error creating MLDataset: {e}")
            mldataset = MLDataset.objects.get(name=name)
            # Update the existing dataset
            mldataset.data = dataset_dict
            mldataset.total_samples = len(dataset)
            mldataset.feature_names = dataset_dict['feature_names']
            mldataset.start_date = dataset.index.min().date() if hasattr(dataset.index, 'min') else None
            mldataset.end_date = dataset.index.max().date() if hasattr(dataset.index, 'max') else None
            mldataset.save()
            print(f"  Updated existing MLDataset: {mldataset.name}")

        # Save scaler to file
        scaler_path = f"ml_models/scalers/{name}_scaler.pkl"
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to: {scaler_path}")

        print(f"Dataset saved: {name} with {len(dataset)} samples")

        return mldataset

    def split_data(self, dataset: pd.DataFrame,
                  test_size: float = 0.2,
                  validation_size: float = 0.1,
                  time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets

        Args:
            dataset: DataFrame with features and labels
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            time_based: If True, use time-based split (no future leakage)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if time_based:
            # Time-based split (preserve order)
            total_len = len(dataset)
            test_start = int(total_len * (1 - test_size))
            val_start = int(test_start * (1 - validation_size / (1 - test_size)))

            train_df = dataset.iloc[:val_start]
            val_df = dataset.iloc[val_start:test_start]
            test_df = dataset.iloc[test_start:]
        else:
            # Random split
            X = dataset.drop('label', axis=1)
            y = dataset['label']

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(test_size + validation_size), random_state=42
            )
            val_size = validation_size / (test_size + validation_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_size), random_state=42
            )

            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

        print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df
