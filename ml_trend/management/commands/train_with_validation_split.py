from django.core.management.base import BaseCommand
from markets.models import Symbol, OHLCVData
from ml_trend.models import TrendConfig, MLTrendDataset, MLTrendModel
from ml_trend.feature_engine import TrendFollowingFeatures
from ml_trend.dataset_builder import TrendMLDatasetBuilder
from ml_trend.model_trainer import TrendModelTrainer
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import json
import math
from imblearn.over_sampling import SMOTE
from django.db import IntegrityError

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train model with explicit train/test date split'

    def add_arguments(self, parser):
        parser.add_argument('--symbols-file', type=str, required=True, help='File containing list of symbols')
        parser.add_argument('--train-end-date', type=str, required=True, help='Training data end date (YYYY-MM-DD)')
        parser.add_argument('--test-years', type=int, default=2, help='Years for out-of-sample test')
        parser.add_argument('--rr-ratio', type=float, default=2.0, help='Risk:Reward ratio')
        parser.add_argument('--sampling-method', type=str, default='smote', help='Sampling method')
        parser.add_argument('--output-model', type=str, default='validated_model', help='Output model name')
        parser.add_argument('--frequency', type=str, default='1D', help='OHLCV frequency(1min, 5min, 15min, 1H, 4H, 1D)')

    def handle(self, *args, **options):
        symbols_file = options['symbols_file']
        train_end_date_str = options['train_end_date']
        test_years = options['test_years']
        rr_ratio = options['rr_ratio']
        sampling_method = options['sampling_method']
        output_model_name = options['output_model']
        frequency = options['frequency']

        # Create timezone-aware training end date
        train_end_date = pd.Timestamp(train_end_date_str).tz_localize('UTC')

        self.stdout.write(f"Training on symbols from: {symbols_file}")
        self.stdout.write(f"Training data end date: {train_end_date.date()}")
        self.stdout.write(f"Test period: {test_years} years after {train_end_date.date()}")

        # Load symbols
        if not os.path.exists(symbols_file):
            self.stdout.write(self.style.ERROR(f"Symbols file not found: {symbols_file}"))
            return

        with open(symbols_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]

        self.stdout.write(f"Loaded {len(symbols)} symbols")

        # Get config
        config, created = TrendConfig.objects.get_or_create(
            name="Default Trend Following",
            defaults={
                'description': "Default trend following configuration",
                'fast_sma': 100,
                'slow_sma': 200,
                'pullback_ema_fast': 50,
                'pullback_ema_slow': 100,
                'stochastic_period': 14,
                'stochastic_overbought': 75,
                'stochastic_oversold': 25,
                'vwap_anchor': 'session',
                'regression_period': 20,
                'regression_threshold': 0.001,
                'default_stop_loss': 0.02,
                'use_ema_pullback': True,
                'use_stochastic': True,
                'use_vwap_deviation': True,
                'use_regression_trend': True,
            }
        )

        # Collect training and test data
        train_datasets = []
        test_datasets = []
        successful_symbols = []

        for sym in symbols:
            self.stdout.write(f"\nProcessing symbol: {sym}")

            try:
                symbol_obj = Symbol.objects.get(ticker=sym)
            except Symbol.DoesNotExist:
                self.stdout.write(f"  Symbol not found, skipping...")
                continue

            # Get all OHLCV data
            ohlcv_records = OHLCVData.objects.filter(
                symbol=symbol_obj,
                frequency=frequency
            ).order_by('year')

            if not ohlcv_records:
                self.stdout.write(f"  No data found, skipping...")
                continue

            # Combine data
            all_data = []
            for record in ohlcv_records:
                try:
                    df = record.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        # Ensure index is timezone-aware
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('UTC')
                        all_data.append(df)
                except Exception as e:
                    self.stdout.write(f"  Error loading data: {e}")

            if not all_data:
                continue

            combined_data = pd.concat(all_data).sort_index()

            # Ensure index is timezone-aware
            if combined_data.index.tz is None:
                combined_data.index = combined_data.index.tz_localize('UTC')

            # Split into train and test
            train_data = combined_data[combined_data.index <= train_end_date]
            test_data = combined_data[combined_data.index > train_end_date]
            test_data = test_data.last(f'{test_years}Y')

            self.stdout.write(f"  Train: {len(train_data)} rows, Test: {len(test_data)} rows")

            if len(train_data) < 200:
                self.stdout.write(f"  Insufficient training data, skipping...")
                continue

            if len(test_data) < 50:
                self.stdout.write(f"  Insufficient test data, skipping...")
                continue

            # Build training dataset
            try:
                builder = TrendMLDatasetBuilder(train_data, config, symbol_obj)
                train_dataset, _ = builder.build_dataset(rr_ratio=rr_ratio, lookahead_periods=21)

                if len(train_dataset) == 0:
                    self.stdout.write(f"  No valid training samples, skipping...")
                    continue

                train_dataset['symbol'] = sym
                train_datasets.append(train_dataset)
                self.stdout.write(f"  Training samples: {len(train_dataset)}")

                # Build test dataset
                test_builder = TrendMLDatasetBuilder(test_data, config, symbol_obj)
                test_dataset, _ = test_builder.build_dataset(rr_ratio=rr_ratio, lookahead_periods=21)

                if len(test_dataset) > 0:
                    test_dataset['symbol'] = sym
                    test_datasets.append(test_dataset)
                    self.stdout.write(f"  Test samples: {len(test_dataset)}")

                successful_symbols.append(sym)

            except Exception as e:
                self.stdout.write(f"  Error building dataset: {e}")
                continue

        if not train_datasets:
            self.stdout.write(self.style.ERROR("No training data collected"))
            return

        # Combine training data
        combined_train = pd.concat(train_datasets, axis=0, ignore_index=True)
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"COMBINED TRAINING DATA")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Total training samples: {len(combined_train)}")
        self.stdout.write(f"From {len(successful_symbols)} symbols")

        class_dist = combined_train['label'].value_counts()
        self.stdout.write(f"Class distribution: {class_dist.to_dict()}")

        # Check class balance
        if len(class_dist) < 2:
            self.stdout.write(self.style.ERROR("Only one class in training data. Cannot train model."))
            return

        # Prepare features
        feature_cols = [c for c in combined_train.columns if c not in ['label', 'symbol']]
        X = combined_train[feature_cols].values
        y = combined_train['label'].values

        # Shuffle data
        indices = np.random.permutation(len(combined_train))
        X = X[indices]
        y = y[indices]

        # Train/validation split (80/20)
        split_idx = int(len(combined_train) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        self.stdout.write(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Apply SMOTE if needed
        if sampling_method == 'smote' and len(np.unique(y_train)) > 1:
            self.stdout.write("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_train, y_train = smote.fit_resample(X_train, y_train)
            self.stdout.write(f"After SMOTE: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")

        # Create training dataset
        train_dataset_final = pd.DataFrame(X_train, columns=feature_cols)
        train_dataset_final['label'] = y_train

        # Train model
        trainer = TrendModelTrainer(
            dataset=train_dataset_final,
            feature_names=feature_cols,
            label_name='label'
        )

        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
        }

        self.stdout.write("Training XGBoost model...")
        result = trainer.train_xgboost(
            params=params,
            train_idx=list(range(len(X_train))),
            val_idx=list(range(len(X_train), len(X_train) + len(X_val)))
        )

        self.stdout.write(f"Validation Accuracy: {result['metrics']['accuracy']:.2%}")
        self.stdout.write(f"Validation Precision: {result['metrics']['precision']:.2%}")
        self.stdout.write(f"Validation Recall: {result['metrics']['recall']:.2%}")

        # Calculate expected value
        precision = result['metrics']['precision']
        expected_value = (precision * rr_ratio) - ((1 - precision) * 1)
        self.stdout.write(f"Expected Value per Trade (RR {rr_ratio}:1): {expected_value:.3f}R")

        # Save model with unique names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = f"{output_model_name}_dataset_{timestamp}"
        model_name = f"{output_model_name}_RR{rr_ratio}_{timestamp}"

        # Prepare dataset dictionary
        dataset_dict = {
            'samples': combined_train.to_dict('records'),
            'feature_names': feature_cols,
            'symbols': successful_symbols,
            'train_end_date': train_end_date.isoformat(),
            'test_years': test_years,
            'rr_ratio': rr_ratio,
            'training_date': datetime.now().isoformat(),
            'sampling_method': sampling_method,
            'validation_metrics': result['metrics']
        }

        # Create dataset with unique name
        try:
            mldataset = MLTrendDataset.objects.create(
                name=dataset_name,
                description=f"Training data up to {train_end_date.date()} (RR {rr_ratio}:1, {sampling_method})",
                config=config,
                data=dataset_dict,
                symbol=None,
                frequency=frequency,
                start_date=train_end_date.date(),
                end_date=train_end_date.date(),
                total_samples=len(combined_train),
                feature_names=feature_cols
            )
            self.stdout.write(f"  Created dataset: {dataset_name}")
        except IntegrityError:
            # If name exists, add random suffix
            import random
            dataset_name = f"{output_model_name}_dataset_{timestamp}_{random.randint(1000, 9999)}"
            mldataset = MLTrendDataset.objects.create(
                name=dataset_name,
                description=f"Training data up to {train_end_date.date()} (RR {rr_ratio}:1, {sampling_method})",
                config=config,
                data=dataset_dict,
                symbol=None,
                frequency=frequency,
                start_date=train_end_date.date(),
                end_date=train_end_date.date(),
                total_samples=len(combined_train),
                feature_names=feature_cols
            )
            self.stdout.write(f"  Created dataset: {dataset_name} (unique)")

        # Save model with unique name
        try:
            best_model = trainer.save_model(
                model=result['model'],
                name=model_name,
                version='1.0.0',
                dataset=mldataset,
                config=config,
                metrics=result['metrics'],
                importance=result['feature_importance'],
                params=params,
                rr_ratio=rr_ratio
            )
            self.stdout.write(self.style.SUCCESS(f"✓ Model saved: {best_model.name}"))
            self.stdout.write(f"  Model ID: {best_model.id}")
        except IntegrityError:
            # If model name exists, add random suffix
            import random
            model_name = f"{output_model_name}_RR{rr_ratio}_{timestamp}_{random.randint(1000, 9999)}"
            best_model = trainer.save_model(
                model=result['model'],
                name=model_name,
                version='1.0.0',
                dataset=mldataset,
                config=config,
                metrics=result['metrics'],
                importance=result['feature_importance'],
                params=params,
                rr_ratio=rr_ratio
            )
            self.stdout.write(self.style.SUCCESS(f"✓ Model saved: {best_model.name}"))
            self.stdout.write(f"  Model ID: {best_model.id}")

        # Now validate on test data
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("VALIDATING ON OUT-OF-SAMPLE TEST DATA")
        self.stdout.write(f"{'='*60}")

        if test_datasets:
            combined_test = pd.concat(test_datasets, axis=0, ignore_index=True)
            self.stdout.write(f"Test samples: {len(combined_test)}")
            self.stdout.write(f"Test symbols: {len(test_datasets)}")

            # Make predictions on test data
            X_test = combined_test[feature_cols].values
            y_test = combined_test['label'].values

            y_pred_proba = result['model'].predict_proba(X_test)[:, 1]

            # Test different thresholds
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            self.stdout.write(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'Signals':<10} {'Expected Value':<15}")
            self.stdout.write("-" * 70)

            best_threshold = 0.5
            best_expected = -np.inf
            best_precision = 0
            best_recall = 0
            threshold_results = []

            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                if len(np.unique(y_pred)) > 1:
                    from sklearn.metrics import precision_score, recall_score
                    precision_test = precision_score(y_test, y_pred, zero_division=0)
                    recall_test = recall_score(y_test, y_pred, zero_division=0)
                    signals = y_pred.sum()
                    expected = (precision_test * rr_ratio) - ((1 - precision_test) * 1)

                    status = "✓" if expected > 0 else "✗"
                    self.stdout.write(f"{threshold:.2f}      {precision_test:.2%}      {recall_test:.2%}      {signals:>6}      {expected:>+.3f}R      {status}")

                    threshold_results.append({
                        'threshold': threshold,
                        'precision': precision_test,
                        'recall': recall_test,
                        'signals': int(signals),
                        'expected_value': expected
                    })

                    #if expected > best_expected:
                    if True:
                        best_expected = expected
                        best_threshold = threshold
                        best_precision = precision_test
                        best_recall = recall_test

            # Summary
            self.stdout.write("\n" + "="*60)
            self.stdout.write("OUT-OF-SAMPLE TEST SUMMARY")
            self.stdout.write("="*60)
            self.stdout.write(f"Best Threshold: {best_threshold:.2f}")
            self.stdout.write(f"Expected Value: {best_expected:.3f}R per trade")
            self.stdout.write(f"Test Period: {test_years} years after {train_end_date.date()}")

            # Break-even analysis
            break_even_precision = 1 / (rr_ratio + 1)
            self.stdout.write(f"\nBreak-even Precision Needed: {break_even_precision:.1%}")

            if best_expected > 0:
                self.stdout.write(self.style.SUCCESS(f"\n✓ Model is PROFITABLE on out-of-sample data!"))
                self.stdout.write(f"  For every $100 risked, expect ${best_expected*100:.0f} profit")
                self.stdout.write(f"\nRecommended trading parameters:")
                self.stdout.write(f"  - Probability threshold: {best_threshold:.2f}")
                self.stdout.write(f"  - Risk:Reward ratio: {rr_ratio}:1")
                self.stdout.write(f"  - Expected win rate: {best_precision:.1%}")
            else:
                self.stdout.write(self.style.WARNING(f"\n⚠ Model is NOT PROFITABLE on out-of-sample data"))
                self.stdout.write(f"  Need precision > {break_even_precision:.1%}")
                self.stdout.write(f"  Current best precision: {best_precision:.1%}")

            # Save test results
            test_results_file = f"test_results_{model_name}_{timestamp}.json"
            test_results = {
                'model_id': best_model.id,
                'model_name': model_name,
                'train_end_date': train_end_date.isoformat(),
                'test_years': test_years,
                'rr_ratio': rr_ratio,
                'test_samples': len(combined_test),
                'best_threshold': best_threshold,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'expected_value': best_expected,
                'threshold_analysis': threshold_results,
                'validation_metrics': result['metrics'],
                'feature_importance': {k: float(v) for k, v in result['feature_importance'].items()}
            }

            with open(test_results_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)

            self.stdout.write(f"\nTest results saved to: {test_results_file}")

            # Feature importance analysis
            self.stdout.write("\n" + "="*60)
            self.stdout.write("FEATURE IMPORTANCE ANALYSIS")
            self.stdout.write("="*60)
            importance = result['feature_importance']
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                self.stdout.write("Top 10 Most Important Features:")
                for i, (feature, imp) in enumerate(top_features, 1):
                    self.stdout.write(f"  {i}. {feature}: {imp:.4f}")

        else:
            self.stdout.write(self.style.WARNING("No test data available for validation"))
            self.stdout.write("\nSuggestions:")
            self.stdout.write("  1. Choose an earlier train-end-date (e.g., 2020-12-31)")
            self.stdout.write("  2. Add more symbols with historical data")
            self.stdout.write("  3. Reduce test-years to 1")

