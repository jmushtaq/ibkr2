from django.core.management.base import BaseCommand
from markets.models import Symbol, OHLCVData
from ml_trend.models import TrendConfig, MLTrendDataset, MLTrendModel
from ml_trend.feature_engine import TrendFollowingFeatures
from ml_trend.dataset_builder import TrendMLDatasetBuilder
from ml_trend.model_trainer import TrendModelTrainer
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train balanced trend following model with class imbalance handling'

    def add_arguments(self, parser):
        parser.add_argument('--symbols-file', type=str, required=True, help='File containing list of symbols')
        parser.add_argument('--config-id', type=int, help='TrendConfig ID')
        parser.add_argument('--rr-ratios', type=str, default='1.5,2.0,3.0', help='Risk:Reward ratios to test')
        parser.add_argument('--years', type=int, default=7, help='Years of data to use')
        parser.add_argument('--force', action='store_true', help='Force retraining')
        parser.add_argument('--output-model', type=str, default='balanced_trend_model', help='Output model name')
        parser.add_argument('--sampling-method', type=str, default='smote',
                           choices=['none', 'smote', 'adasyn', 'smote_tomek', 'class_weight'],
                           help='Method to handle class imbalance')
        parser.add_argument('--min-samples', type=int, default=200, help='Minimum samples for training')

    def handle(self, *args, **options):
        symbols_file = options['symbols_file']
        config_id = options.get('config_id')
        rr_ratios = [float(r) for r in options['rr_ratios'].split(',')]
        years = options['years']
        force = options.get('force', False)
        output_model_name = options['output_model']
        sampling_method = options['sampling_method']
        min_samples = options['min_samples']

        # Load symbols
        if not os.path.exists(symbols_file):
            self.stdout.write(self.style.ERROR(f"Symbols file not found: {symbols_file}"))
            return

        with open(symbols_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]

        self.stdout.write(f"Training on {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

        # Get or create default config
        if config_id:
            config = TrendConfig.objects.get(id=config_id)
        else:
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
            if created:
                self.stdout.write(self.style.SUCCESS(f"Created default config: {config.name}"))
            else:
                self.stdout.write(f"Using existing config: {config.name}")

        # Collect data from all symbols
        all_datasets = {}
        successful_symbols = []

        for rr in rr_ratios:
            all_datasets[rr] = []

        for sym in symbols:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Processing symbol: {sym}")
            self.stdout.write(f"{'='*60}")

            try:
                symbol_obj = Symbol.objects.get(ticker=sym)
            except Symbol.DoesNotExist:
                self.stdout.write(self.style.WARNING(f"Symbol {sym} not found, skipping..."))
                continue

            # Get OHLCV data
            ohlcv_records = OHLCVData.objects.filter(
                symbol=symbol_obj,
                frequency='1D'
            ).order_by('year')

            if not ohlcv_records:
                self.stdout.write(self.style.WARNING(f"No OHLCV data found for {sym}, skipping..."))
                continue

            all_data = []
            for record in ohlcv_records:
                try:
                    df = record.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"Error loading data: {e}"))

            if not all_data:
                continue

            combined_data = pd.concat(all_data).sort_index()
            combined_data = combined_data.last(f'{years}Y')
            self.stdout.write(f"Loaded {len(combined_data)} rows for {sym}")

            for rr in rr_ratios:
                self.stdout.write(f"  Building dataset for RR {rr}:1...")
                builder = TrendMLDatasetBuilder(combined_data, config, symbol_obj)
                dataset, scaler = builder.build_dataset(rr_ratio=rr, lookahead_periods=21)

                if len(dataset) == 0:
                    self.stdout.write(f"    No valid samples for RR {rr}:1")
                    continue

                dataset['symbol'] = sym
                all_datasets[rr].append(dataset)
                self.stdout.write(f"    Got {len(dataset)} samples")

            successful_symbols.append(sym)

        if not any(all_datasets.values()):
            self.stdout.write(self.style.ERROR("No valid datasets collected"))
            return

        # Combine datasets
        combined_datasets = {}
        for rr, datasets in all_datasets.items():
            if not datasets:
                continue
            combined = pd.concat(datasets, axis=0, ignore_index=True)
            combined_datasets[rr] = combined
            self.stdout.write(f"\nRR {rr}:1 - Combined {len(combined)} samples")

            # Log imbalance ratio
            class_dist = combined['label'].value_counts()
            if len(class_dist) > 1:
                imbalance_ratio = class_dist[0] / class_dist[1]
                self.stdout.write(f"  Class distribution: 0={class_dist.get(0, 0)}, 1={class_dist.get(1, 0)}")
                self.stdout.write(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

        # Train models with imbalance handling
        best_model = None
        best_metrics = None
        best_rr = None

        for rr, dataset in combined_datasets.items():
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Training for RR {rr}:1 with {sampling_method} sampling")
            self.stdout.write(f"{'='*60}")

            if len(dataset) < min_samples:
                self.stdout.write(self.style.WARNING(f"Only {len(dataset)} samples, need {min_samples}. Skipping..."))
                continue

            # Prepare features
            feature_cols = [c for c in dataset.columns if c not in ['label', 'symbol']]
            X = dataset[feature_cols].values
            y = dataset['label'].values

            # Train/validation split
            split_idx = int(len(dataset) * 0.8)
            X_train_raw = X[:split_idx]
            y_train_raw = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]

            self.stdout.write(f"Train samples: {len(X_train_raw)} (Class 0: {sum(y_train_raw==0)}, Class 1: {sum(y_train_raw==1)})")
            self.stdout.write(f"Validation samples: {len(X_val)}")

            # Apply sampling method
            if sampling_method == 'smote' and len(np.unique(y_train_raw)) > 1:
                self.stdout.write("  Applying SMOTE oversampling...")
                smote = SMOTE(random_state=42, sampling_strategy='auto')
                X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
                self.stdout.write(f"  After SMOTE: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")

            elif sampling_method == 'adasyn' and len(np.unique(y_train_raw)) > 1:
                self.stdout.write("  Applying ADASYN oversampling...")
                adasyn = ADASYN(random_state=42, sampling_strategy='auto')
                X_train, y_train = adasyn.fit_resample(X_train_raw, y_train_raw)
                self.stdout.write(f"  After ADASYN: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")

            elif sampling_method == 'smote_tomek' and len(np.unique(y_train_raw)) > 1:
                self.stdout.write("  Applying SMOTE+TOMEK...")
                smote_tomek = SMOTETomek(random_state=42, sampling_strategy='auto')
                X_train, y_train = smote_tomek.fit_resample(X_train_raw, y_train_raw)
                self.stdout.write(f"  After SMOTE+TOMEK: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")

            elif sampling_method == 'class_weight':
                self.stdout.write("  Using class weights...")
                X_train, y_train = X_train_raw, y_train_raw
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
                self.stdout.write(f"  Class weights: {class_weight_dict}")

            else:
                X_train, y_train = X_train_raw, y_train_raw

            # Create training dataset
            train_dataset = pd.DataFrame(X_train, columns=feature_cols)
            train_dataset['label'] = y_train

            # Train model with optimized parameters for imbalance
            trainer = TrendModelTrainer(
                dataset=train_dataset,
                feature_names=feature_cols,
                label_name='label'
            )

            # XGBoost parameters for imbalanced data
            params = {
                'n_estimators': 300,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'scale_pos_weight': (sum(y_train == 0) / sum(y_train == 1)) if sampling_method != 'class_weight' else class_weight_dict[1],
            }

            if sampling_method == 'class_weight':
                # For class_weight method, use scale_pos_weight
                scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
                params['scale_pos_weight'] = scale_pos_weight
                self.stdout.write(f"  Scale pos weight: {scale_pos_weight:.2f}")

            self.stdout.write("  Training XGBoost model...")
            result = trainer.train_xgboost(
                params=params,
                train_idx=list(range(len(X_train))),
                val_idx=list(range(len(X_train), len(X_train) + len(X_val)))
            )

            self.stdout.write(f"  Accuracy: {result['metrics']['accuracy']:.2%}")
            self.stdout.write(f"  Precision: {result['metrics']['precision']:.2%}")
            self.stdout.write(f"  Recall: {result['metrics']['recall']:.2%}")
            self.stdout.write(f"  F1 Score: {result['metrics']['f1_score']:.2%}")

            # Calculate profit factor for RR ratio
            precision = result['metrics']['precision']
            recall = result['metrics']['recall']

            # Expected value per trade: (win_rate * rr) - (loss_rate * 1)
            # Using precision as win rate when signal is given
            expected_value = (precision * rr) - ((1 - precision) * 1)
            self.stdout.write(f"  Expected Value per trade (RR {rr}:1): {expected_value:.3f}")

            # Score = F1 * expected_value (combined metric)
            combined_score = result['metrics']['f1_score'] * expected_value

            # Track best model
            if combined_score > (best_metrics.get('combined_score', -np.inf) if best_metrics else -np.inf):
                best_rr = rr
                best_metrics = {
                    'accuracy': result['metrics']['accuracy'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1_score': result['metrics']['f1_score'],
                    'expected_value': expected_value,
                    'combined_score': combined_score
                }

                # Save dataset
                dataset_name = f"{output_model_name}_dataset_RR{rr}"
                dataset_dict = {
                    'samples': dataset.to_dict('records'),
                    'feature_names': feature_cols,
                    'symbols': successful_symbols,
                    'sampling_method': sampling_method
                }

                try:
                    mldataset = MLTrendDataset.objects.create(
                        name=dataset_name,
                        description=f"Balanced dataset with {sampling_method} for RR {rr}:1",
                        config=config,
                        data=dataset_dict,
                        symbol=None,
                        frequency='1D',
                        start_date=datetime.now().date(),
                        end_date=datetime.now().date(),
                        total_samples=len(dataset),
                        feature_names=feature_cols
                    )

                    model_name = f"{output_model_name}_RR{rr}_{sampling_method}"
                    best_model = trainer.save_model(
                        model=result['model'],
                        name=model_name,
                        version='1.0.0',
                        dataset=mldataset,
                        config=config,
                        metrics=result['metrics'],
                        importance=result['feature_importance'],
                        params=params,
                        rr_ratio=rr
                    )
                    self.stdout.write(self.style.SUCCESS(f"  Model saved: {model_name}"))

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  Error saving model: {e}"))

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write("TRAINING COMPLETE")
        self.stdout.write("="*60)

        if best_model:
            self.stdout.write(self.style.SUCCESS(f"Best Model: {best_model.name}"))
            self.stdout.write(f"  RR Ratio: {best_rr}:1")
            self.stdout.write(f"  Accuracy: {best_metrics['accuracy']:.2%}")
            self.stdout.write(f"  Precision: {best_metrics['precision']:.2%}")
            self.stdout.write(f"  Recall: {best_metrics['recall']:.2%}")
            self.stdout.write(f"  F1 Score: {best_metrics['f1_score']:.2%}")
            self.stdout.write(f"  Expected Value per trade: {best_metrics['expected_value']:.3f}")

            # Interpretation
            self.stdout.write("\n" + "="*60)
            self.stdout.write("INTERPRETATION")
            self.stdout.write("="*60)

            precision = best_metrics['precision']
            rr = best_rr

            if precision > 0.33:  # Need > 33% win rate for 2:1 RR to break even
                self.stdout.write(self.style.SUCCESS(f"✓ This model is profitable at {rr}:1 risk-reward!"))
                self.stdout.write(f"  Win rate needed: {1/(rr+1):.1%}")
                self.stdout.write(f"  Actual precision: {precision:.1%}")
                self.stdout.write(f"  Expected profit per trade: {(precision * rr) - ((1-precision) * 1):.2f}R")
            else:
                self.stdout.write(self.style.WARNING(f"⚠ Model not yet profitable. Need precision > {1/(rr+1):.1%}"))

            # Feature importance
            importance = best_model.feature_importance
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                self.stdout.write("\nTop 10 Features:")
                for i, (feature, imp) in enumerate(top_features, 1):
                    self.stdout.write(f"  {i}. {feature}: {imp:.4f}")
        else:
            self.stdout.write(self.style.ERROR("No model trained successfully"))
