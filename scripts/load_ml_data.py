import os
import sys
import django
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ibkr_project.settings')
django.setup()

from markets.models import Symbol, OHLCVData, PrecomputedMetrics
from ml_framework.dataset_manager import MLDatasetManager
from ml_framework.model_trainer import ModelTrainer
from ml_framework.backtest import Backtester
from ml_framework.models import MLDataset, FeatureSet, LabeledDataset, MLModel, Prediction, BacktestResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLDataPipeline:
    """Complete ML data pipeline for financial analysis"""

    def __init__(self, symbol_ticker, frequency='1D', years_back=5):
        self.symbol_ticker = symbol_ticker
        self.frequency = frequency
        self.years_back = years_back
        self.symbol = None
        self.raw_data = None
        self.dataset = None
        self.feature_names = None

    def load_data(self):
        """Load OHLCV data from database"""
        logger.info(f"Loading data for {self.symbol_ticker} at {self.frequency} frequency")

        try:
            self.symbol = Symbol.objects.get(ticker=self.symbol_ticker)
        except Symbol.DoesNotExist:
            logger.error(f"Symbol {self.symbol_ticker} not found")
            return False

        # Get OHLCV data
        ohlcv_records = OHLCVData.objects.filter(
            symbol=self.symbol,
            frequency=self.frequency
        ).order_by('year')

        if not ohlcv_records:
            logger.error(f"No OHLCV data found for {self.symbol_ticker}")
            return False

        # Combine data from all years
        dfs = []
        for record in ohlcv_records:
            df = record.get_data_as_dataframe()
            dfs.append(df)

        self.raw_data = pd.concat(dfs).sort_index()

        # Keep only recent years
        cutoff_date = datetime.now() - timedelta(days=self.years_back * 365)
        self.raw_data = self.raw_data[self.raw_data.index >= cutoff_date]

        logger.info(f"Loaded {len(self.raw_data)} rows of data from {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        return True

    def create_features_and_labels(self):
        """Create feature-engineered dataset with labels"""
        logger.info("Creating features and labels")

        # Initialize dataset manager
        manager = MLDatasetManager(
            data=self.raw_data,
            symbol=self.symbol_ticker,
            frequency=self.frequency
        )

        # Configure feature engineering
        feature_config = {
            'all': True  # Use all feature types
        }

        # Configure labeling strategy
        # Let's try different labeling strategies
        labeling_strategies = [
            {
                'name': 'binary_threshold_0.5pct',
                'config': {
                    'type': 'binary',
                    'lookahead': '1d',
                    'threshold': 0.005,  # 0.5% threshold
                    'upside_only': False
                },
                'description': 'Binary classification: up if return > 0.5%, down if return < -0.5%'
            },
            {
                'name': 'binary_threshold_1pct',
                'config': {
                    'type': 'binary',
                    'lookahead': '1d',
                    'threshold': 0.01,  # 1% threshold
                    'upside_only': False
                },
                'description': 'Binary classification: up if return > 1%, down if return < -1%'
            },
            {
                'name': 'regression_1d',
                'config': {
                    'type': 'regression',
                    'lookahead': '1d'
                },
                'description': 'Regression: predict 1-day forward return'
            },
            {
                'name': 'regression_1w',
                'config': {
                    'type': 'regression',
                    'lookahead': '1w'
                },
                'description': 'Regression: predict 1-week forward return'
            },
            {
                'name': 'risk_reward_2pct_4pct',
                'config': {
                    'type': 'risk_reward',
                    'lookahead': '1m',
                    'stop_loss': 0.02,
                    'take_profit': 0.04
                },
                'description': 'Risk-reward: 2% stop loss, 4% take profit over 1 month'
            }
        ]

        # Create datasets for each labeling strategy
        datasets = {}
        for strategy in labeling_strategies:
            logger.info(f"Creating dataset with strategy: {strategy['name']}")

            try:
                dataset = manager.create_dataset(
                    feature_config=feature_config,
                    label_config=strategy['config'],
                    name=f"{self.symbol_ticker}_{self.frequency}_{strategy['name']}",
                    description=strategy['description'],
                    scaler_type='standard'
                )

                datasets[strategy['name']] = {
                    'dataset': dataset,
                    'config': strategy['config']
                }

                logger.info(f"  Created dataset with {len(dataset)} samples")
                logger.info(f"  Class distribution: {dataset['label'].value_counts().to_dict() if 'label' in dataset.columns else 'Regression'}")

            except Exception as e:
                logger.error(f"  Failed to create dataset: {e}")

        self.datasets = datasets
        return datasets

    def train_models(self, test_size=0.2, validation_size=0.1):
        """Train multiple models on different datasets"""
        logger.info("Training models")

        trained_models = {}

        for strategy_name, strategy_data in self.datasets.items():
            dataset = strategy_data['dataset']
            config = strategy_data['config']

            logger.info(f"\nTraining models for strategy: {strategy_name}")

            # Split data
            manager = MLDatasetManager(self.raw_data, self.symbol_ticker, self.frequency)
            train_df, val_df, test_df = manager.split_data(
                dataset,
                test_size=test_size,
                validation_size=validation_size,
                time_based=True
            )

            logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            # Get feature names (all columns except label)
            feature_names = [col for col in train_df.columns if col != 'label']

            # Initialize trainer
            trainer = ModelTrainer(
                dataset=train_df,
                feature_names=feature_names,
                label_name='label'
            )

            # Define different hyperparameter configurations
            hyperparameter_configs = {
                'xgboost_default': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'xgboost_deep': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'xgboost_shallow': {
                    'n_estimators': 50,
                    'max_depth': 4,
                    'learning_rate': 0.15,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42
                },
                'lightgbm_default': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbose': -1
                },
                'random_forest_default': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }

            # Train multiple models
            models_for_strategy = {}

            for model_name, params in hyperparameter_configs.items():
                try:
                    logger.info(f"    Training {model_name}...")

                    if 'xgboost' in model_name:
                        result = trainer.train_xgboost(
                            params=params,
                            train_df=train_df,
                            val_df=val_df,
                            name=f"{self.symbol_ticker}_{strategy_name}_{model_name}",
                            description=f"XGBoost model for {strategy_name}"
                        )
                    elif 'lightgbm' in model_name:
                        result = trainer.train_lightgbm(
                            params=params,
                            train_df=train_df,
                            val_df=val_df,
                            name=f"{self.symbol_ticker}_{strategy_name}_{model_name}",
                            description=f"LightGBM model for {strategy_name}"
                        )
                    elif 'random_forest' in model_name:
                        result = trainer.train_random_forest(
                            params=params,
                            train_df=train_df,
                            val_df=val_df,
                            name=f"{self.symbol_ticker}_{strategy_name}_{model_name}",
                            description=f"Random Forest model for {strategy_name}"
                        )
                    else:
                        continue

                    # Save model metrics
                    models_for_strategy[model_name] = {
                        'model': result['model'],
                        'metrics': result['metrics'],
                        'feature_importance': result['feature_importance'],
                        'params': params,
                        'trainer': trainer
                    }

                    logger.info(f"      Validation metrics: {result['metrics']}")

                except Exception as e:
                    logger.error(f"      Failed to train {model_name}: {e}")

            trained_models[strategy_name] = models_for_strategy

        self.trained_models = trained_models
        return trained_models

    def backtest_models(self, initial_capital=100000, position_size=0.1):
        """Backtest all trained models"""
        logger.info("\nRunning backtests")

        backtest_results = {}

        for strategy_name, models in self.trained_models.items():
            logger.info(f"\nBacktesting strategy: {strategy_name}")

            backtest_results[strategy_name] = {}

            for model_name, model_data in models.items():
                logger.info(f"  Backtesting {model_name}...")

                # Use validation+test data for backtesting (no lookahead bias)
                # We need to recreate the full dataset with features
                manager = MLDatasetManager(self.raw_data, self.symbol_ticker, self.frequency)

                # Recreate the dataset to get features for backtesting period
                # This is a simplified approach - in production, you'd want to maintain feature history

                # For now, use the test data split
                dataset = self.datasets[strategy_name]['dataset']
                train_df, val_df, test_df = manager.split_data(
                    dataset,
                    test_size=0.2,
                    validation_size=0.1,
                    time_based=True
                )

                # Use test data for backtesting
                backtest_data = test_df.copy()

                # We need the original OHLCV data for the test period
                test_start = backtest_data.index[0]
                test_end = backtest_data.index[-1]

                ohlcv_test = self.raw_data.loc[test_start:test_end]

                # Initialize backtester
                feature_names = [col for col in backtest_data.columns if col != 'label']
                X_test = backtest_data[feature_names].values

                # Make predictions
                model = model_data['model']
                predictions = model.predict(X_test)

                # For classification models, use probability for confidence
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_test)[:, 1]
                else:
                    probabilities = np.ones(len(predictions)) * 0.5

                # Add predictions to backtest data
                backtest_data['prediction'] = predictions
                backtest_data['probability'] = probabilities

                # Create backtester
                backtester = Backtester(
                    data=ohlcv_test,
                    model=model,
                    feature_names=feature_names
                )

                # Run backtest with different configurations
                backtest_configs = [
                    {
                        'name': 'simple',
                        'position_size': position_size,
                        'stop_loss': None,
                        'take_profit': None,
                        'min_confidence': 0.5
                    },
                    {
                        'name': 'with_stops',
                        'position_size': position_size,
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'min_confidence': 0.5
                    },
                    {
                        'name': 'high_confidence',
                        'position_size': position_size,
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'min_confidence': 0.7
                    }
                ]

                config_results = {}
                for config in backtest_configs:
                    try:
                        result = backtester.run_backtest(
                            initial_capital=initial_capital,
                            position_size=config['position_size'],
                            stop_loss=config['stop_loss'],
                            take_profit=config['take_profit'],
                            min_confidence=config['min_confidence']
                        )

                        config_results[config['name']] = result

                        logger.info(f"    {config['name']} backtest results:")
                        logger.info(f"      Total Return: {result['metrics']['total_return']:.2%}")
                        logger.info(f"      Sharpe Ratio: {result['metrics']['sharpe_ratio']:.2f}")
                        logger.info(f"      Win Rate: {result['metrics']['win_rate']:.2%}")
                        logger.info(f"      Max Drawdown: {result['metrics']['max_drawdown']:.2%}")
                        logger.info(f"      Total Trades: {result['metrics']['total_trades']}")

                    except Exception as e:
                        logger.error(f"      Failed backtest for {config['name']}: {e}")

                backtest_results[strategy_name][model_name] = config_results

        self.backtest_results = backtest_results
        return backtest_results

    def save_to_database(self):
        """Save models and backtest results to database"""
        logger.info("\nSaving results to database")

        saved_models = []

        for strategy_name, models in self.trained_models.items():
            # Get or create FeatureSet
            feature_set, _ = FeatureSet.objects.get_or_create(
                name=f"{self.symbol_ticker}_{self.frequency}_features",
                defaults={
                    'description': f"Feature set for {self.symbol_ticker}",
                    'feature_type': 'custom',
                    'features_json': {'all': True}
                }
            )

            # Get or create LabeledDataset
            label_config = self.datasets[strategy_name]['config']
            labeled_dataset, _ = LabeledDataset.objects.get_or_create(
                name=f"{self.symbol_ticker}_{self.frequency}_{strategy_name}_labels",
                defaults={
                    'description': f"Labeling strategy: {strategy_name}",
                    'label_type': label_config['type'],
                    'config': label_config
                }
            )

            # Get or create MLDataset
            mldataset, _ = MLDataset.objects.get_or_create(
                name=f"{self.symbol_ticker}_{self.frequency}_{strategy_name}_dataset",
                defaults={
                    'description': f"Dataset for {strategy_name}",
                    'feature_set': feature_set,
                    'labeled_dataset': labeled_dataset,
                    'symbol': self.symbol,
                    'frequency': self.frequency,
                    'version': '1.0.0'
                }
            )

            for model_name, model_data in models.items():
                # Check if model already exists
                model_obj, created = MLModel.objects.get_or_create(
                    name=f"{self.symbol_ticker}_{strategy_name}_{model_name}",
                    version='1.0.0',
                    defaults={
                        'description': f"Model trained on {strategy_name}",
                        'model_type': 'classification' if label_config['type'] in ['binary', 'multi_class'] else 'regression',
                        'algorithm': model_name.split('_')[0],
                        'dataset': mldataset,
                        'feature_set': feature_set,
                        'hyperparameters': model_data['params'],
                        'training_samples': len(self.datasets[strategy_name]['dataset']),
                        'validation_samples': 0,
                        'test_samples': 0,
                        'performance_metrics': model_data['metrics'],
                        'feature_importance': model_data['feature_importance']
                    }
                )

                saved_models.append(model_obj)

                # Save backtest results if available
                if hasattr(self, 'backtest_results') and strategy_name in self.backtest_results:
                    if model_name in self.backtest_results[strategy_name]:
                        for config_name, result in self.backtest_results[strategy_name][model_name].items():
                            BacktestResult.objects.create(
                                model=model_obj,
                                name=f"{model_name}_{config_name}",
                                description=f"Backtest with {config_name} configuration",
                                start_date=self.raw_data.index[-int(len(self.raw_data)*0.2)],
                                end_date=self.raw_data.index[-1],
                                initial_capital=100000,
                                position_size=0.1,
                                stop_loss=0.02 if 'stops' in config_name else None,
                                take_profit=0.04 if 'stops' in config_name else None,
                                total_return=result['metrics']['total_return'],
                                annualized_return=result['metrics']['annualized_return'],
                                sharpe_ratio=result['metrics']['sharpe_ratio'],
                                max_drawdown=result['metrics']['max_drawdown'],
                                win_rate=result['metrics']['win_rate'],
                                profit_factor=result['metrics']['profit_factor'],
                                total_trades=result['metrics']['total_trades'],
                                winning_trades=result['metrics']['winning_trades'],
                                losing_trades=result['metrics']['losing_trades'],
                                trades=result['trades'],
                                equity_curve=result['equity_curve']
                            )

        logger.info(f"Saved {len(saved_models)} models to database")
        return saved_models

def main():
    """Main execution function"""

    # Example 1: Process AAPL data
    logger.info("=" * 60)
    logger.info("ML Data Pipeline for AAPL")
    logger.info("=" * 60)

    pipeline = MLDataPipeline(
        symbol_ticker='AAPL',
        frequency='1D',
        years_back=5
    )

    # Load data
    if not pipeline.load_data():
        return

    # Create datasets with different labeling strategies
    pipeline.create_features_and_labels()

    # Train models
    pipeline.train_models()

    # Backtest models
    pipeline.backtest_models()

    # Save results to database
    pipeline.save_to_database()

    # Example 2: Compare different symbols
    logger.info("\n" + "=" * 60)
    logger.info("Comparing Multiple Symbols")
    logger.info("=" * 60)

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    all_results = {}

    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")
        pipeline_symbol = MLDataPipeline(
            symbol_ticker=symbol,
            frequency='1D',
            years_back=5
        )

        if pipeline_symbol.load_data():
            pipeline_symbol.create_features_and_labels()
            pipeline_symbol.train_models()
            pipeline_symbol.backtest_models()

            # Store best model performance
            best_performance = 0
            best_config = None

            if hasattr(pipeline_symbol, 'backtest_results'):
                for strategy, models in pipeline_symbol.backtest_results.items():
                    for model, configs in models.items():
                        for config_name, result in configs.items():
                            if result['metrics']['sharpe_ratio'] > best_performance:
                                best_performance = result['metrics']['sharpe_ratio']
                                best_config = {
                                    'symbol': symbol,
                                    'strategy': strategy,
                                    'model': model,
                                    'config': config_name,
                                    'metrics': result['metrics']
                                }

            all_results[symbol] = best_config

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("Best Model for Each Symbol")
    logger.info("=" * 60)

    for symbol, best in all_results.items():
        if best:
            logger.info(f"\n{symbol}:")
            logger.info(f"  Strategy: {best['strategy']}")
            logger.info(f"  Model: {best['model']}")
            logger.info(f"  Config: {best['config']}")
            logger.info(f"  Sharpe Ratio: {best['metrics']['sharpe_ratio']:.2f}")
            logger.info(f"  Total Return: {best['metrics']['total_return']:.2%}")
            logger.info(f"  Win Rate: {best['metrics']['win_rate']:.2%}")
        else:
            logger.info(f"\n{symbol}: No successful models")

    # Example 3: Generate trading signals for next month
    logger.info("\n" + "=" * 60)
    logger.info("Generating Trading Signals for Next Month")
    logger.info("=" * 60)

    generate_trading_signals()

def generate_trading_signals():
    """Generate trading signals using the best models"""

    # Get the best model from database (based on Sharpe ratio)
    best_model = MLModel.objects.filter(is_active=True).order_by('-performance_metrics__sharpe_ratio').first()

    if not best_model:
        logger.error("No trained models found")
        return

    logger.info(f"Using best model: {best_model.name}")
    logger.info(f"Performance: {best_model.performance_metrics}")

    # Load the model
    import joblib
    model = joblib.load(best_model.model_file_path)

    # Get recent data for the symbol
    symbol = best_model.dataset.symbol
    frequency = best_model.dataset.frequency

    ohlcv_records = OHLCVData.objects.filter(
        symbol=symbol,
        frequency=frequency
    ).order_by('-year')[:2]  # Get last 2 years

    dfs = []
    for record in ohlcv_records:
        df = record.get_data_as_dataframe()
        dfs.append(df)

    recent_data = pd.concat(dfs).sort_index()
    recent_data = recent_data.last('3M')  # Last 3 months

    # Recreate features
    manager = MLDatasetManager(
        data=recent_data,
        symbol=symbol.ticker,
        frequency=frequency
    )

    # Get feature configuration from FeatureSet
    feature_config = best_model.feature_set.features_json

    # Generate features for recent data
    dataset = manager.create_dataset(
        feature_config=feature_config,
        label_config={'type': 'binary', 'lookahead': '1d'},  # Placeholder
        name="temp",
        description="Temporary dataset for prediction"
    )

    # Make predictions
    feature_names = best_model.dataset.feature_names
    X = dataset[feature_names].values
    predictions = model.predict(X)

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = np.ones(len(predictions)) * 0.5

    # Add predictions to data
    recent_data = recent_data.loc[dataset.index]
    recent_data['prediction'] = predictions
    recent_data['probability'] = probabilities

    # Generate signals
    signals = recent_data[recent_data['probability'] >= 0.7]

    logger.info(f"\nTrading Signals for {symbol.ticker} (Confidence >= 70%):")
    for date, row in signals.iterrows():
        signal = "BUY" if row['prediction'] > 0.5 else "SELL"
        logger.info(f"  {date.date()}: {signal} at ${row['close']:.2f} (Confidence: {row['probability']:.1%})")

    # Save predictions to database
    for date, row in signals.iterrows():
        Prediction.objects.create(
            model=best_model,
            symbol=symbol,
            frequency=frequency,
            timestamp=date,
            prediction_type='signal',
            prediction_value=1 if row['prediction'] > 0.5 else 0,
            confidence=row['probability'],
            features_used=row[feature_names].to_dict()
        )

    logger.info(f"\nSaved {len(signals)} predictions to database")

if __name__ == "__main__":
    main()
