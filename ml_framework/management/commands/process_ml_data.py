from django.core.management.base import BaseCommand
from markets.models import Symbol, OHLCVData
from ml_framework.dataset_manager import MLDatasetManager
from ml_framework.model_trainer import ModelTrainer
from ml_framework.backtest import Backtester
from ml_framework.models import MLDataset, FeatureSet, LabeledDataset, BacktestResult, MLModel
import pandas as pd
import numpy as np
import os
from datetime import datetime

class Command(BaseCommand):
    help = 'Process ML data for a symbol'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, required=True, help='Symbol ticker')
        parser.add_argument('--frequency', type=str, default='1D', help='Data frequency')
        parser.add_argument('--years', type=int, default=5, help='Number of years of data')
        parser.add_argument('--train', action='store_true', help='Train model')
        parser.add_argument('--backtest', action='store_true', help='Run backtest')
        parser.add_argument('--force', action='store_true', help='Force retraining even if model exists')

    def convert_to_native(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self.convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_native(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj

    def handle(self, *args, **options):
        symbol_ticker = options['symbol']
        frequency = options['frequency']
        years = options['years']
        force_retrain = options.get('force', False)

        self.stdout.write(f"Processing ML data for {symbol_ticker}")

        # Get symbol
        try:
            symbol = Symbol.objects.get(ticker=symbol_ticker)
        except Symbol.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Symbol {symbol_ticker} not found"))
            return

        # Get OHLCV data
        ohlcv_records = OHLCVData.objects.filter(
            symbol=symbol,
            frequency=frequency
        ).order_by('year')

        if not ohlcv_records:
            self.stdout.write(self.style.ERROR(f"No OHLCV data found for {symbol_ticker}"))
            return

        # Combine data from multiple years
        all_data = []
        for record in ohlcv_records:
            try:
                df = record.get_data_as_dataframe()
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Error loading data for year {record.year}: {e}"))

        if all_data:
            combined_data = pd.concat(all_data).sort_index()
            # Keep only last X years
            combined_data = combined_data.last(f'{years}Y')

            self.stdout.write(f"Loaded {len(combined_data)} rows of data")
            self.stdout.write(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")

            # Basic data quality check
            self.stdout.write("\nData quality check:")
            self.stdout.write(f"  Missing values:\n{combined_data.isnull().sum()}")
            self.stdout.write(f"  Data types:\n{combined_data.dtypes}")

            # Create dataset manager (pass Symbol object, not string)
            dataset_manager = MLDatasetManager(
                data=combined_data,
                symbol=symbol,  # Pass the Symbol object
                frequency=frequency
            )

            # Feature configuration - comprehensive set
            feature_config = {
                'price': True,
                'technical': True,
                'volatility': True,
                'momentum': True,
                'volume': False,  # Start with False to avoid complexity
                'pattern': False,  # Start with False to avoid complexity
                'cycle': False     # Start with False to avoid complexity
            }

            # Label configuration
            label_config = {
                'type': 'binary',
                'lookahead': '1d',
                'threshold': 0.005,  # 0.5% threshold
                'upside_only': False
            }

            self.stdout.write("\n" + "="*60)
            self.stdout.write("Creating dataset...")
            self.stdout.write("="*60)

            try:
                dataset = dataset_manager.create_dataset(
                    feature_config=feature_config,
                    label_config=label_config,
                    name=f"{symbol_ticker}_{frequency}_dataset",
                    description=f"ML dataset for {symbol_ticker} at {frequency} frequency",
                    scaler_type='standard'
                )

                self.stdout.write(self.style.SUCCESS(f"✓ Created dataset with {len(dataset)} samples"))

                if 'label' in dataset.columns:
                    label_counts = dataset['label'].value_counts()
                    self.stdout.write(f"  Label distribution: {label_counts.to_dict()}")

                    # Calculate class balance
                    total = len(dataset)
                    class_0 = label_counts.get(0, 0)
                    class_1 = label_counts.get(1, 0)
                    self.stdout.write(f"  Class balance: 0={class_0/total:.2%}, 1={class_1/total:.2%}")

                # Split data
                train_df, val_df, test_df = dataset_manager.split_data(
                    dataset,
                    test_size=0.2,
                    validation_size=0.1,
                    time_based=True
                )

                self.stdout.write(f"\nData split:")
                self.stdout.write(f"  Train: {len(train_df)} samples")
                self.stdout.write(f"  Validation: {len(val_df)} samples")
                self.stdout.write(f"  Test: {len(test_df)} samples")

                # Train model if requested
                if options['train']:
                    self.stdout.write("\n" + "="*60)
                    self.stdout.write("Training model...")
                    self.stdout.write("="*60)

                    # Check if model already exists
                    model_name = f"{symbol_ticker}_xgboost"
                    existing_model = MLModel.objects.filter(
                        name=model_name,
                        version='1.0.0'
                    ).first()

                    if existing_model and not force_retrain:
                        self.stdout.write(self.style.WARNING(f"Model {model_name} already exists. Use --force to retrain."))
                        self.stdout.write(f"  Model metrics: {existing_model.performance_metrics}")
                        model = existing_model
                    else:
                        try:
                            trainer = ModelTrainer(
                                dataset=train_df,
                                feature_names=[col for col in train_df.columns if col != 'label'],
                                label_name='label'
                            )

                            # Train XGBoost with optimized parameters
                            self.stdout.write("Training XGBoost model...")

                            params = {
                                'n_estimators': 100,
                                'max_depth': 4,
                                'learning_rate': 0.05,
                                'subsample': 0.8,
                                'colsample_bytree': 0.8,
                                'random_state': 42,
                                'eval_metric': 'logloss' if len(train_df['label'].unique()) <= 2 else 'rmse'
                            }

                            result = trainer.train_xgboost(
                                params=params,
                                train_df=train_df,
                                val_df=val_df,
                                name=model_name,
                                description="XGBoost model for directional prediction"
                            )

                            self.stdout.write(f"\nModel metrics on validation set:")
                            for metric, value in result['metrics'].items():
                                self.stdout.write(f"  {metric}: {value:.4f}")

                            # Save model
                            self.stdout.write("\nSaving model to database...")

                            # Get or create FeatureSet
                            feature_set, fs_created = FeatureSet.objects.get_or_create(
                                name=f"{symbol_ticker}_{frequency}_features",
                                defaults={
                                    'description': f"Feature set for {symbol_ticker}",
                                    'feature_type': 'custom',
                                    'features_json': self.convert_to_native(feature_config)
                                }
                            )
                            self.stdout.write(f"  FeatureSet: {'created' if fs_created else 'existing'}")

                            # Get or create LabeledDataset
                            labeled_dataset, ld_created = LabeledDataset.objects.get_or_create(
                                name=f"{symbol_ticker}_{frequency}_labels",
                                defaults={
                                    'description': "Binary classification labels",
                                    'label_type': 'binary',
                                    'config': self.convert_to_native(label_config)
                                }
                            )
                            self.stdout.write(f"  LabeledDataset: {'created' if ld_created else 'existing'}")

                            # Get or create MLDataset
                            mldataset, md_created = MLDataset.objects.get_or_create(
                                name=f"{symbol_ticker}_{frequency}_dataset",
                                defaults={
                                    'description': f"ML dataset for {symbol_ticker}",
                                    'feature_set': feature_set,
                                    'labeled_dataset': labeled_dataset,
                                    'symbol': symbol,
                                    'frequency': frequency,
                                    'start_date': dataset.index.min().date(),
                                    'end_date': dataset.index.max().date(),
                                    'total_samples': int(len(dataset)),
                                    'feature_names': [col for col in dataset.columns if col != 'label']
                                }
                            )
                            self.stdout.write(f"  MLDataset: {'created' if md_created else 'existing'}")

                            # Save model
                            model = trainer.save_model(
                                model=result['model'],
                                name=model_name,
                                version='1.0.0',
                                dataset=mldataset,
                                feature_set=feature_set,
                                metrics=result['metrics'],
                                importance=result['feature_importance'],
                                params=params
                            )

                            self.stdout.write(self.style.SUCCESS(f"✓ Model saved: {model.name}"))

                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f"Error training model: {e}"))
                            import traceback
                            traceback.print_exc()
                            return

                    # Run backtest if requested
                    if options['backtest'] and model:
                        self.stdout.write("\n" + "="*60)
                        self.stdout.write("Running backtest...")
                        self.stdout.write("="*60)

                        try:
                            backtester = Backtester(
                                data=test_df,
                                model=model,
                                feature_names=[col for col in test_df.columns if col != 'label']
                            )

                            # Test different backtest configurations
                            backtest_configs = [
                                {
                                    'name': 'Simple',
                                    'position_size': 0.1,
                                    'stop_loss': None,
                                    'take_profit': None,
                                    'min_confidence': 0.5
                                },
                                {
                                    'name': 'With Stops',
                                    'position_size': 0.1,
                                    'stop_loss': 0.02,
                                    'take_profit': 0.04,
                                    'min_confidence': 0.5
                                },
                                {
                                    'name': 'High Confidence',
                                    'position_size': 0.1,
                                    'stop_loss': 0.02,
                                    'take_profit': 0.04,
                                    'min_confidence': 0.7
                                }
                            ]

                            best_sharpe = -float('inf')
                            best_config = None
                            best_result = None

                            for config in backtest_configs:
                                self.stdout.write(f"\nBacktest configuration: {config['name']}")

                                backtest_result = backtester.run_backtest(
                                    initial_capital=100000,
                                    position_size=config['position_size'],
                                    stop_loss=config['stop_loss'],
                                    take_profit=config['take_profit'],
                                    commission=0.001,
                                    min_confidence=config['min_confidence']
                                )

                                # Convert numpy types to native Python
                                metrics = self.convert_to_native(backtest_result['metrics'])
                                trades = self.convert_to_native(backtest_result['trades'])
                                equity_curve = self.convert_to_native(backtest_result['equity_curve'])

                                self.stdout.write(f"  Total Return: {metrics['total_return']:.2%}")
                                self.stdout.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                                self.stdout.write(f"  Win Rate: {metrics['win_rate']:.2%}")
                                self.stdout.write(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                                self.stdout.write(f"  Total Trades: {metrics['total_trades']}")

                                # Save backtest results
                                try:
                                    backtest = BacktestResult.objects.create(
                                        model=model,
                                        name=f"{symbol_ticker}_backtest_{config['name']}",
                                        description=f"Backtest with {config['name']} configuration",
                                        start_date=test_df.index.min().date(),
                                        end_date=test_df.index.max().date(),
                                        initial_capital=100000,
                                        position_size=config['position_size'],
                                        stop_loss=config['stop_loss'],
                                        take_profit=config['take_profit'],
                                        total_return=metrics['total_return'],
                                        annualized_return=metrics['annualized_return'],
                                        sharpe_ratio=metrics['sharpe_ratio'],
                                        max_drawdown=metrics['max_drawdown'],
                                        win_rate=metrics['win_rate'],
                                        profit_factor=metrics['profit_factor'],
                                        total_trades=metrics['total_trades'],
                                        winning_trades=metrics['winning_trades'],
                                        losing_trades=metrics['losing_trades'],
                                        trades=trades,
                                        equity_curve=equity_curve
                                    )
                                    self.stdout.write(f"  ✓ Backtest saved: {backtest.name}")
                                except Exception as e:
                                    self.stdout.write(self.style.WARNING(f"  Could not save backtest: {e}"))

                                # Track best configuration
                                if metrics['sharpe_ratio'] > best_sharpe:
                                    best_sharpe = metrics['sharpe_ratio']
                                    best_config = config['name']
                                    best_result = backtest_result

                            self.stdout.write(f"\nBest configuration: {best_config}")
                            self.stdout.write(f"Best Sharpe Ratio: {best_sharpe:.2f}")

                            # Save best configuration results to model
                            if best_result:
                                best_metrics = self.convert_to_native(best_result['metrics'])
                                model.performance_metrics['best_backtest'] = {
                                    'config': best_config,
                                    'metrics': best_metrics
                                }
                                model.save()
                                self.stdout.write(self.style.SUCCESS("✓ Best backtest results saved to model"))

                        except Exception as e:
                            self.stdout.write(self.style.ERROR(f"Error running backtest: {e}"))
                            import traceback
                            traceback.print_exc()

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error in ML pipeline: {e}"))
                import traceback
                traceback.print_exc()
                return

            # Summary
            self.stdout.write("\n" + "="*60)
            self.stdout.write("Summary")
            self.stdout.write("="*60)
            self.stdout.write(f"Symbol: {symbol_ticker}")
            self.stdout.write(f"Frequency: {frequency}")
            self.stdout.write(f"Data period: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
            self.stdout.write(f"Total samples: {len(dataset)}")
            self.stdout.write(f"Features: {len([col for col in dataset.columns if col != 'label'])}")

            if options['train'] and 'model' in locals():
                self.stdout.write(f"\nModel: {model.name}")
                self.stdout.write(f"Algorithm: {model.algorithm}")
                self.stdout.write(f"Training date: {model.training_date}")
                if model.performance_metrics:
                    self.stdout.write("Performance metrics:")
                    for key, value in model.performance_metrics.items():
                        if isinstance(value, (int, float)) and not key.startswith('_'):
                            self.stdout.write(f"  {key}: {value:.4f}")

            self.stdout.write(self.style.SUCCESS("\n✓ Processing complete!"))

        else:
            self.stdout.write(self.style.ERROR("No data found"))

