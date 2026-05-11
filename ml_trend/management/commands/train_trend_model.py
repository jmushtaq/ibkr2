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

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train trend following ML model on single or multiple symbols'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Symbol ticker (single symbol)')
        parser.add_argument('--symbols-file', type=str, help='File containing list of symbols (one per line)')
        parser.add_argument('--config-id', type=int, help='TrendConfig ID')
        parser.add_argument('--rr-ratios', type=str, default='1.5,2.0,3.0', help='Risk:Reward ratios to test')
        parser.add_argument('--years', type=int, default=5, help='Years of data to use')
        parser.add_argument('--force', action='store_true', help='Force retraining')
        parser.add_argument('--output-model', type=str, default='trend_following_model', help='Output model name')

    def handle(self, *args, **options):
        symbol = options.get('symbol')
        symbols_file = options.get('symbols_file')
        config_id = options.get('config_id')
        rr_ratios = [float(r) for r in options['rr_ratios'].split(',')]
        years = options['years']
        force = options.get('force', False)
        output_model_name = options['output_model']

        # Validate input
        if not symbol and not symbols_file:
            self.stdout.write(self.style.ERROR("Either --symbol or --symbols-file must be provided"))
            return

        # Get symbols list
        symbols = []
        if symbol:
            symbols = [symbol]
        elif symbols_file:
            if not os.path.exists(symbols_file):
                self.stdout.write(self.style.ERROR(f"Symbols file not found: {symbols_file}"))
                return
            with open(symbols_file, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]

        self.stdout.write(f"Training trend following model on {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

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

            # Get symbol from database
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

            # Combine data
            all_data = []
            for record in ohlcv_records:
                try:
                    df = record.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f"Error loading data: {e}"))

            if not all_data:
                self.stdout.write(self.style.WARNING(f"No valid data for {sym}, skipping..."))
                continue

            combined_data = pd.concat(all_data).sort_index()
            combined_data = combined_data.last(f'{years}Y')

            self.stdout.write(f"Loaded {len(combined_data)} rows for {sym}")

            # Build datasets for each RR ratio
            for rr in rr_ratios:
                self.stdout.write(f"  Building dataset for RR {rr}:1...")

                builder = TrendMLDatasetBuilder(combined_data, config, symbol_obj)
                dataset, scaler = builder.build_dataset(rr_ratio=rr, lookahead_periods=21)

                if len(dataset) == 0:
                    self.stdout.write(f"    No valid samples for RR {rr}:1")
                    continue

                # Add symbol column to identify which symbol the sample came from
                dataset['symbol'] = sym

                all_datasets[rr].append(dataset)
                self.stdout.write(f"    Got {len(dataset)} samples")

            successful_symbols.append(sym)

        if not any(all_datasets.values()):
            self.stdout.write(self.style.ERROR("No valid datasets collected from any symbol"))
            return

        self.stdout.write(f"\n{'='*60}")
        self.stdout.write("COMBINING DATASETS")
        self.stdout.write(f"{'='*60}")
        self.stdout.write(f"Successfully processed symbols: {', '.join(successful_symbols)}")

        # Combine datasets for each RR ratio
        combined_datasets = {}

        for rr, datasets in all_datasets.items():
            if not datasets:
                self.stdout.write(f"  No datasets for RR {rr}:1")
                continue

            combined = pd.concat(datasets, axis=0, ignore_index=True)
            combined_datasets[rr] = combined
            self.stdout.write(f"  RR {rr}:1 - Combined {len(combined)} samples from {len(datasets)} symbols")

            # Log class distribution
            class_dist = combined['label'].value_counts()
            self.stdout.write(f"    Class distribution: {class_dist.to_dict()}")

        # Train models for each RR ratio
        best_model = None
        best_accuracy = 0
        best_rr = None

        for rr, dataset in combined_datasets.items():
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Training on combined dataset for RR {rr}:1")
            self.stdout.write(f"{'='*60}")
            self.stdout.write(f"Dataset size: {len(dataset)} samples")

            # Check if we have enough samples
            min_samples = 100  # Minimum samples needed for meaningful training
            if len(dataset) < min_samples:
                self.stdout.write(self.style.WARNING(f"Only {len(dataset)} samples, need at least {min_samples}. Skipping..."))
                continue

            # Prepare features
            feature_cols = [c for c in dataset.columns if c not in ['label', 'symbol']]
            X = dataset[feature_cols].values
            y = dataset['label'].values

            # Shuffle data (since it's from multiple symbols)
            indices = np.random.permutation(len(dataset))
            X = X[indices]
            y = y[indices]

            # Train/validation split (80/20)
            split_idx = int(len(dataset) * 0.8)
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]

            self.stdout.write(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

            # Create training dataset
            train_dataset = pd.DataFrame(X_train, columns=feature_cols)
            train_dataset['label'] = y_train

            # Train model
            trainer = TrendModelTrainer(
                dataset=train_dataset,
                feature_names=feature_cols,
                label_name='label'
            )

            self.stdout.write("Training XGBoost model...")

            # Train with optimized parameters
            params = {
                'n_estimators': min(300, len(X_train) // 3),
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

            result = trainer.train_xgboost(
                params=params,
                train_idx=list(range(len(X_train))),
                val_idx=list(range(len(X_train), len(X_train) + len(X_val)))
            )

            self.stdout.write(f"  Accuracy: {result['metrics']['accuracy']:.2%}")
            self.stdout.write(f"  Precision: {result['metrics']['precision']:.2%}")
            self.stdout.write(f"  Recall: {result['metrics']['recall']:.2%}")
            self.stdout.write(f"  F1 Score: {result['metrics']['f1_score']:.2%}")

            # Track best model
            if result['metrics']['accuracy'] > best_accuracy:
                best_accuracy = result['metrics']['accuracy']
                best_rr = rr

                # Save dataset to database
                dataset_name = f"{output_model_name}_dataset_RR{rr}"

                # Create a combined dataset record
                dataset_dict = {
                    'samples': dataset.to_dict('records'),
                    'feature_names': feature_cols,
                    'symbols': successful_symbols,
                    'total_symbols': len(successful_symbols),
                    'samples_per_symbol': {sym: len(dataset[dataset['symbol'] == sym]) for sym in successful_symbols if sym in dataset['symbol'].values}
                }

                try:
                    mldataset = MLTrendDataset.objects.create(
                        name=dataset_name,
                        description=f"Combined trend following dataset from {len(successful_symbols)} symbols with RR {rr}:1",
                        config=config,
                        data=dataset_dict,
                        symbol=None,  # Combined dataset has no single symbol
                        frequency='1D',
                        start_date=dataset['timestamp'].min() if 'timestamp' in dataset.columns else datetime.now().date(),
                        end_date=dataset['timestamp'].max() if 'timestamp' in dataset.columns else datetime.now().date(),
                        total_samples=len(dataset),
                        feature_names=feature_cols
                    )

                    # Save model
                    model_name = f"{output_model_name}_RR{rr}"
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
                    import traceback
                    traceback.print_exc()

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write("TRAINING COMPLETE")
        self.stdout.write("="*60)

        if best_model:
            self.stdout.write(self.style.SUCCESS(f"Best Model: {best_model.name}"))
            self.stdout.write(f"  RR Ratio: {best_rr}:1")
            self.stdout.write(f"  Accuracy: {best_model.performance_metrics['accuracy']:.2%}")
            self.stdout.write(f"  Precision: {best_model.performance_metrics['precision']:.2%}")
            self.stdout.write(f"  Recall: {best_model.performance_metrics['recall']:.2%}")
            self.stdout.write(f"  F1 Score: {best_model.performance_metrics['f1_score']:.2%}")
            self.stdout.write(f"  Training Samples: {best_model.training_samples}")

            # Feature importance
            importance = best_model.feature_importance
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                self.stdout.write("\nTop 10 Features:")
                for i, (feature, imp) in enumerate(top_features, 1):
                    self.stdout.write(f"  {i}. {feature}: {imp:.4f}")
        else:
            self.stdout.write(self.style.ERROR("No model trained successfully"))
            self.stdout.write("\nSuggestions:")
            self.stdout.write("  1. Try increasing years of data: --years 10")
            self.stdout.write("  2. Add more symbols to your symbols file")
            self.stdout.write("  3. Adjust strategy parameters in TrendConfig")
            self.stdout.write("  4. Consider using higher frequency data (1H, 4H)")
            self.stdout.write("  5. Try with different RR ratios: --rr-ratios '1.5,2.0,2.5,3.0'")

