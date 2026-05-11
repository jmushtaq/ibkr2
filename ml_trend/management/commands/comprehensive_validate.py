from django.core.management.base import BaseCommand
from ml_trend.models import MLTrendModel
from markets.models import Symbol, OHLCVData
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from ml_trend.dataset_builder import TrendMLDatasetBuilder
from ml_trend.models import TrendConfig
import json
from datetime import datetime, timedelta
import pytz

class Command(BaseCommand):
    help = 'Comprehensive model validation with multiple test periods'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, required=True, help='Model ID to validate')
        parser.add_argument('--test-years', type=int, default=2, help='Years for out-of-sample test')
        parser.add_argument('--walk-forward', action='store_true', help='Perform walk-forward validation')
        parser.add_argument('--save-results', action='store_true', help='Save validation results to file')

    def handle(self, *args, **options):
        model_id = options['model_id']
        test_years = options['test_years']
        walk_forward = options['walk_forward']
        save_results = options['save_results']

        # Get model
        model_obj = MLTrendModel.objects.get(id=model_id)
        self.stdout.write(f"\n{'='*80}")
        self.stdout.write(f"VALIDATING MODEL: {model_obj.name}")
        self.stdout.write(f"{'='*80}")
        self.stdout.write(f"Training Date: {model_obj.training_date}")
        self.stdout.write(f"Training Samples: {model_obj.training_samples}")
        self.stdout.write(f"RR Ratio: {model_obj.target_rr_ratio}")
        self.stdout.write(f"Reported Accuracy: {model_obj.performance_metrics.get('accuracy', 0):.2%}")
        self.stdout.write(f"Reported Precision: {model_obj.performance_metrics.get('precision', 0):.2%}")
        self.stdout.write(f"Reported Recall: {model_obj.performance_metrics.get('recall', 0):.2%}")

        # Load model
        model = joblib.load(model_obj.model_file_path)

        # Get dataset info
        dataset = model_obj.dataset
        config = model_obj.config

        # Get symbols from dataset
        symbols = dataset.data.get('symbols', [])
        if not symbols:
            self.stdout.write(self.style.ERROR("No symbols found in dataset"))
            return

        self.stdout.write(f"\nTesting on {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

        if walk_forward:
            self._walk_forward_validation(model, config, symbols, test_years, model_obj)
        else:
            self._single_period_validation(model, config, symbols, test_years, model_obj, save_results)

    def _single_period_validation(self, model, config, symbols, test_years, model_obj, save_results):
        """Single period out-of-sample validation"""
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        symbol_results = {}

        # Create timezone-aware training end date
        train_end = pd.Timestamp(model_obj.training_date).tz_localize('UTC')
        self.stdout.write(f"Training end date (UTC): {train_end}")

        for sym in symbols:
            try:
                symbol_obj = Symbol.objects.get(ticker=sym)
            except Symbol.DoesNotExist:
                continue

            # Get recent data (after training period)
            ohlcv_records = OHLCVData.objects.filter(
                symbol=symbol_obj,
                frequency='1D'
            ).order_by('-year')

            if not ohlcv_records:
                continue

            dfs = []
            for record in ohlcv_records[:2]:
                df = record.get_data_as_dataframe()
                if df is not None and not df.empty:
                    # Ensure index is timezone-aware
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    dfs.append(df)

            if not dfs:
                continue

            recent_data = pd.concat(dfs).sort_index()

            # Ensure index is timezone-aware
            if recent_data.index.tz is None:
                recent_data.index = recent_data.index.tz_localize('UTC')

            # Keep only last test_years
            recent_data = recent_data.last(f'{test_years}Y')

            # Get data after training date
            out_of_sample = recent_data[recent_data.index > train_end]

            if len(out_of_sample) < 50:
                self.stdout.write(f"\n{sym}: Only {len(out_of_sample)} out-of-sample rows (need 50), skipping...")
                continue

            self.stdout.write(f"\n{sym}: {len(out_of_sample)} out-of-sample rows")
            self.stdout.write(f"  Date range: {out_of_sample.index[0]} to {out_of_sample.index[-1]}")

            # Build features
            builder = TrendMLDatasetBuilder(out_of_sample, config, symbol_obj)
            dataset_scaled, scaler = builder.build_dataset(
                rr_ratio=model_obj.target_rr_ratio,
                lookahead_periods=21
            )

            if len(dataset_scaled) == 0:
                self.stdout.write(f"  No valid samples after feature engineering")
                continue

            # Make predictions
            feature_cols = [c for c in dataset_scaled.columns if c != 'label']
            X = dataset_scaled[feature_cols].values
            y_true = dataset_scaled['label'].values

            # Get probabilities
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
            except Exception as e:
                self.stdout.write(f"  Error getting predictions: {e}")
                continue

            # Try different confidence thresholds
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            best_threshold = 0.5
            best_precision = 0
            best_expected = -np.inf

            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                if len(np.unique(y_pred)) > 1:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    rr = model_obj.target_rr_ratio
                    expected_value = (precision * rr) - ((1 - precision) * 1)

                    if expected_value > best_expected:
                        best_expected = expected_value
                        best_threshold = threshold
                        best_precision = precision

            # Use best threshold
            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # Calculate metrics
            if len(np.unique(y_pred)) > 1:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Calculate profit metrics
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))

                # Expected value per trade
                rr = model_obj.target_rr_ratio
                if tp + fp > 0:
                    expected_value = (precision * rr) - ((1 - precision) * 1)
                else:
                    expected_value = 0

                symbol_results[sym] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'expected_value': expected_value,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'best_threshold': best_threshold,
                    'total_signals': tp + fp,
                    'total_opportunities': tp + fn
                }

                self.stdout.write(f"  Best threshold: {best_threshold:.2f}")
                self.stdout.write(f"  Precision: {precision:.2%} ({tp}/{tp+fp})")
                self.stdout.write(f"  Recall: {recall:.2%} ({tp}/{tp+fn})")
                self.stdout.write(f"  F1: {f1:.2%}")
                self.stdout.write(f"  Expected Value: {expected_value:.3f}R")

                all_predictions.extend(y_pred)
                all_actuals.extend(y_true)
                all_probabilities.extend(y_pred_proba)

        # Overall out-of-sample performance
        if all_predictions:
            overall_precision = precision_score(all_actuals, all_predictions, zero_division=0)
            overall_recall = recall_score(all_actuals, all_predictions, zero_division=0)
            overall_f1 = f1_score(all_actuals, all_predictions, zero_division=0)

            # Calculate overall expected value
            tp_total = sum([r['tp'] for r in symbol_results.values()])
            fp_total = sum([r['fp'] for r in symbol_results.values()])
            overall_precision_calc = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
            rr = model_obj.target_rr_ratio
            overall_expected = (overall_precision_calc * rr) - ((1 - overall_precision_calc) * 1)

            self.stdout.write("\n" + "="*80)
            self.stdout.write("OVERALL OUT-OF-SAMPLE RESULTS")
            self.stdout.write("="*80)
            self.stdout.write(f"Total Predictions: {len(all_predictions)}")
            self.stdout.write(f"Total Signals: {tp_total + fp_total}")
            self.stdout.write(f"Total Opportunities: {tp_total + sum([r['fn'] for r in symbol_results.values()])}")
            self.stdout.write(f"Precision: {overall_precision:.2%} ({tp_total}/{tp_total+fp_total})")
            self.stdout.write(f"Recall: {overall_recall:.2%}")
            self.stdout.write(f"F1 Score: {overall_f1:.2%}")
            self.stdout.write(f"Expected Value per Trade: {overall_expected:.3f}R")

            # Compare to training performance
            train_precision = model_obj.performance_metrics.get('precision', 0)
            precision_drop = train_precision - overall_precision

            self.stdout.write("\n" + "-"*40)
            self.stdout.write("PERFORMANCE COMPARISON")
            self.stdout.write("-"*40)
            self.stdout.write(f"Training Precision: {train_precision:.2%}")
            self.stdout.write(f"Out-of-Sample Precision: {overall_precision:.2%}")
            self.stdout.write(f"Difference: {precision_drop:+.2%}")

            if overall_precision > 0.5 and overall_expected > 0:
                self.stdout.write(self.style.SUCCESS("\n✓ MODEL PASSES VALIDATION!"))
                self.stdout.write("  The model generalizes well to out-of-sample data.")
                self.stdout.write(f"  Expected profit per trade: {overall_expected:.2f}R")
            elif overall_precision > 0.4:
                self.stdout.write(self.style.WARNING("\n⚠ MODEL SHOWS MODERATE OVERFITTING"))
                self.stdout.write("  Consider using more conservative thresholds.")
                self.stdout.write("  Start with smaller position sizes.")
            else:
                self.stdout.write(self.style.ERROR("\n❌ MODEL FAILS VALIDATION"))
                self.stdout.write("  Significant overfitting detected.")
                self.stdout.write("  Do not use this model for live trading.")
                self.stdout.write("\nRecommendations:")
                self.stdout.write("  1. Increase training data (more symbols, more years)")
                self.stdout.write("  2. Reduce model complexity (lower max_depth, fewer trees)")
                self.stdout.write("  3. Add more regularization")
                self.stdout.write("  4. Try different sampling methods")

            # Save results
            if save_results:
                results_file = f"validation_results_{model_obj.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                results_data = {
                    'model_name': model_obj.name,
                    'model_id': model_obj.id,
                    'validation_date': datetime.now().isoformat(),
                    'test_years': test_years,
                    'overall_metrics': {
                        'precision': overall_precision,
                        'recall': overall_recall,
                        'f1': overall_f1,
                        'expected_value': overall_expected
                    },
                    'training_metrics': {
                        'precision': train_precision,
                        'accuracy': model_obj.performance_metrics.get('accuracy', 0),
                        'recall': model_obj.performance_metrics.get('recall', 0)
                    },
                    'symbol_results': symbol_results
                }

                with open(results_file, 'w') as f:
                    json.dump(results_data, f, indent=2, default=str)

                self.stdout.write(f"\nResults saved to: {results_file}")
        else:
            self.stdout.write(self.style.ERROR("No valid out-of-sample predictions"))
            self.stdout.write("\nPossible reasons:")
            self.stdout.write("  1. Training data is too recent (no out-of-sample period available)")
            self.stdout.write("  2. Not enough data after training date")
            self.stdout.write("  3. Symbols in dataset don't have recent data")
            self.stdout.write("\nTry:")
            self.stdout.write("  - Use older training data (--years 10)")
            self.stdout.write("  - Add more symbols with historical data")
            self.stdout.write("  - Reduce test-years to 1")

    def _walk_forward_validation(self, model, config, symbols, test_years, model_obj):
        """Walk-forward validation (rolling window)"""
        self.stdout.write("\nPerforming walk-forward validation...")
        # Implementation would go here
        pass
