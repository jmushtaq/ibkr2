from django.core.management.base import BaseCommand
from markets.models import Symbol, OHLCVData
from ml_framework.dataset_manager import MLDatasetManager
from ml_framework.model_trainer import ModelTrainer
from ml_framework.models import MLDataset, FeatureSet, LabeledDataset, BacktestResult, MLModel
import pandas as pd
import numpy as np

class Command(BaseCommand):
    help = 'Train improved models with better strategies'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, required=True, help='Symbol ticker')
        parser.add_argument('--frequency', type=str, default='1D', help='Data frequency')
        parser.add_argument('--years', type=int, default=5, help='Number of years of data')
        parser.add_argument('--strategy', type=str, default='all',
                           choices=['all', 'larger_threshold', 'longer_lookahead', 'trend_following'])
        parser.add_argument('--force', action='store_true', help='Force retraining')

    def handle(self, *args, **options):
        symbol_ticker = options['symbol']
        frequency = options['frequency']
        years = options['years']
        strategy = options['strategy']
        force_retrain = options.get('force', False)

        self.stdout.write(f"Training improved model for {symbol_ticker} with strategy: {strategy}")

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
            self.stdout.write(self.style.ERROR(f"No OHLCV data found"))
            return

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
            return

        combined_data = pd.concat(all_data).sort_index()
        combined_data = combined_data.last(f'{years}Y')

        self.stdout.write(f"Loaded {len(combined_data)} rows of data")

        # Define strategies
        strategies = {
            'larger_threshold': {
                'name': 'Larger Threshold (2%)',
                'label_config': {
                    'type': 'binary',
                    'lookahead': '1w',
                    'threshold': 0.02,
                    'upside_only': False
                },
                'description': 'Predict if price will move >2% in next week'
            },
            'longer_lookahead': {
                'name': 'Longer Lookahead (1 month)',
                'label_config': {
                    'type': 'binary',
                    'lookahead': '1m',
                    'threshold': 0.03,
                    'upside_only': False
                },
                'description': 'Predict if price will move >3% in next month'
            },
            'trend_following': {
                'name': 'Trend Following',
                'label_config': {
                    'type': 'binary',
                    'lookahead': '1m',
                    'threshold': 0.05,
                    'upside_only': True
                },
                'description': 'Predict strong upward moves (>5%) in next month'
            }
        }

        # Select strategy
        if strategy == 'all':
            selected_strategies = strategies
        else:
            selected_strategies = {strategy: strategies[strategy]}

        best_accuracy = 0
        best_strategy = None
        best_model_obj = None
        best_result = None

        for strat_key, strat_info in selected_strategies.items():
            self.stdout.write("\n" + "="*60)
            self.stdout.write(f"Testing strategy: {strat_info['name']}")
            self.stdout.write("="*60)

            # Create dataset manager
            dataset_manager = MLDatasetManager(
                data=combined_data,
                symbol=symbol,
                frequency=frequency
            )

            # Feature configuration
            feature_config = {
                'price': True,
                'technical': True,
                'volatility': True,
                'momentum': True,
                'volume': True,
                'cycle': True
            }

            try:
                # Create dataset
                dataset = dataset_manager.create_dataset(
                    feature_config=feature_config,
                    label_config=strat_info['label_config'],
                    name=f"{symbol_ticker}_{frequency}_{strat_key}_dataset",
                    description=strat_info['description'],
                    scaler_type='standard'
                )

                self.stdout.write(f"Dataset size: {len(dataset)}")

                if 'label' in dataset.columns:
                    label_counts = dataset['label'].value_counts()
                    self.stdout.write(f"Label distribution: {label_counts.to_dict()}")

                    if len(label_counts) < 2:
                        self.stdout.write(self.style.WARNING("Only one class present, skipping..."))
                        continue

                # Split data
                train_df, val_df, test_df = dataset_manager.split_data(
                    dataset,
                    test_size=0.2,
                    validation_size=0.1,
                    time_based=True
                )

                self.stdout.write(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

                # Train model
                trainer = ModelTrainer(
                    dataset=train_df,
                    feature_names=[col for col in train_df.columns if col != 'label'],
                    label_name='label'
                )

                # Best parameters for trend following
                params = {
                    'n_estimators': 150,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }

                self.stdout.write(f"\nTraining model...")

                result = trainer.train_xgboost(
                    params=params,
                    train_df=train_df,
                    val_df=val_df,
                    name=f"{symbol_ticker}_{strat_key}",
                    description=strat_info['description']
                )

                metrics = result['metrics']
                self.stdout.write(f"  Accuracy: {metrics['accuracy']:.4f}")
                self.stdout.write(f"  Precision: {metrics['precision']:.4f}")
                self.stdout.write(f"  Recall: {metrics['recall']:.4f}")
                self.stdout.write(f"  F1 Score: {metrics['f1_score']:.4f}")

                # Track best model
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_strategy = strat_key
                    best_result = result

                    # Save this model to database
                    self.stdout.write("\nSaving best model to database...")

                    # Get or create FeatureSet
                    feature_set, _ = FeatureSet.objects.get_or_create(
                        name=f"{symbol_ticker}_{frequency}_{strat_key}_features",
                        defaults={
                            'description': f"Feature set for {strat_info['name']}",
                            'feature_type': 'custom',
                            'features_json': feature_config
                        }
                    )

                    # Get or create LabeledDataset
                    labeled_dataset, _ = LabeledDataset.objects.get_or_create(
                        name=f"{symbol_ticker}_{frequency}_{strat_key}_labels",
                        defaults={
                            'description': strat_info['description'],
                            'label_type': strat_info['label_config']['type'],
                            'config': strat_info['label_config']
                        }
                    )

                    # Get or create MLDataset
                    mldataset, _ = MLDataset.objects.get_or_create(
                        name=f"{symbol_ticker}_{frequency}_{strat_key}_dataset",
                        defaults={
                            'description': strat_info['description'],
                            'feature_set': feature_set,
                            'labeled_dataset': labeled_dataset,
                            'symbol': symbol,
                            'frequency': frequency,
                            'start_date': dataset.index.min().date(),
                            'end_date': dataset.index.max().date(),
                            'total_samples': len(dataset),
                            'feature_names': [col for col in dataset.columns if col != 'label']
                        }
                    )

                    # Check if model already exists
                    existing_model = MLModel.objects.filter(
                        name=f"{symbol_ticker}_{strat_key}",
                        version='1.0.0'
                    ).first()

                    if existing_model and not force_retrain:
                        self.stdout.write(self.style.WARNING(f"Model already exists. Use --force to overwrite."))
                        best_model_obj = existing_model
                    else:
                        # Delete existing if force
                        if existing_model and force_retrain:
                            existing_model.delete()
                            self.stdout.write(f"Deleted existing model")

                        # Save new model
                        best_model_obj = trainer.save_model(
                            model=result['model'],
                            name=f"{symbol_ticker}_{strat_key}",
                            version='1.0.0',
                            dataset=mldataset,
                            feature_set=feature_set,
                            metrics=result['metrics'],
                            importance=result['feature_importance'],
                            params=params
                        )
                        self.stdout.write(self.style.SUCCESS(f"✓ Model saved: {best_model_obj.name}"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error with strategy {strat_key}: {e}"))
                import traceback
                traceback.print_exc()

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write("Best Model Results")
        self.stdout.write("="*60)

        if best_model_obj:
            self.stdout.write(f"Model ID: {best_model_obj.id}")
            self.stdout.write(f"Strategy: {best_strategy}")
            self.stdout.write(f"Accuracy: {best_accuracy:.4f}")
            self.stdout.write(f"Precision: {best_result['metrics']['precision']:.4f}")
            self.stdout.write(f"Recall: {best_result['metrics']['recall']:.4f}")
            self.stdout.write(f"F1 Score: {best_result['metrics']['f1_score']:.4f}")

            if best_accuracy > 0.55:
                self.stdout.write(self.style.SUCCESS("\n✓ Model saved successfully!"))
                self.stdout.write(f"  Model ID: {best_model_obj.id}")
                self.stdout.write("  Next steps:")
                self.stdout.write(f"  1. Generate signals: python manage.py generate_daily_signals --model-id {best_model_obj.id} --save")
                self.stdout.write(f"  2. Paper trade: python manage.py paper_trade --model-id {best_model_obj.id}")
                self.stdout.write(f"  3. View details: python manage.py analyze_best_model --model-id {best_model_obj.id}")
        else:
            self.stdout.write(self.style.ERROR("No model saved"))

