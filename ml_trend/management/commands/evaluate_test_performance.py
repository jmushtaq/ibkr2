from django.core.management.base import BaseCommand
from ml_trend.models import MLTrendModel
from markets.models import Symbol, OHLCVData
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from ml_trend.dataset_builder import TrendMLDatasetBuilder
import json
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = 'Evaluate model performance on test data'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, required=True, help='Model ID to evaluate')
        parser.add_argument('--threshold', type=float, default=0.6, help='Probability threshold')
        parser.add_argument('--detailed', action='store_true', help='Show detailed per-symbol results')

    def handle(self, *args, **options):
        model_id = options['model_id']
        threshold = options['threshold']
        detailed = options['detailed']

        # Load model
        model_obj = MLTrendModel.objects.get(id=model_id)
        model = joblib.load(model_obj.model_file_path)
        config = model_obj.config
        rr_ratio = model_obj.target_rr_ratio

        self.stdout.write(f"\n{'='*80}")
        self.stdout.write(f"EVALUATING MODEL ON TEST DATA")
        self.stdout.write(f"{'='*80}")
        self.stdout.write(f"Model: {model_obj.name}")
        self.stdout.write(f"RR Ratio: {rr_ratio}:1")
        self.stdout.write(f"Probability Threshold: {threshold}")
        self.stdout.write(f"Training Date: {model_obj.training_date}")
        self.stdout.write(f"Training Samples: {model_obj.training_samples}")

        # Get symbols from dataset
        dataset = model_obj.dataset
        symbols = dataset.data.get('symbols', [])

        # Handle timezone properly
        train_end_date_str = dataset.data.get('train_end_date', '2019-12-31')
        try:
            # Try to parse as timestamp first (if it's already in ISO format)
            train_end_date = pd.Timestamp(train_end_date_str)
            # If it's timezone-aware, convert to UTC for comparison
            if train_end_date.tz is not None:
                train_end_date = train_end_date.tz_convert('UTC')
            else:
                train_end_date = train_end_date.tz_localize('UTC')
        except:
            # If parsing fails, use the date string directly
            train_end_date = pd.Timestamp(train_end_date_str).tz_localize('UTC')

        self.stdout.write(f"\nTest Period: {train_end_date.date()} to present")
        self.stdout.write(f"Total symbols in dataset: {len(symbols)}")

        # Collect test data results
        symbol_results = {}
        all_y_true = []
        all_y_pred = []
        all_probas = []

        processed_count = 0
        skipped_count = 0

        for sym in symbols[:100]:  # Limit to 100 for speed, remove limit for full evaluation
            try:
                symbol_obj = Symbol.objects.get(ticker=sym)
            except Symbol.DoesNotExist:
                skipped_count += 1
                continue

            # Get test data (after training date)
            ohlcv_records = OHLCVData.objects.filter(
                symbol=symbol_obj,
                frequency='1D'
            ).order_by('year')

            if not ohlcv_records:
                skipped_count += 1
                continue

            dfs = []
            for record in ohlcv_records:
                try:
                    df = record.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        # Ensure index is timezone-aware
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('UTC')
                        dfs.append(df)
                except Exception as e:
                    continue

            if not dfs:
                skipped_count += 1
                continue

            combined_data = pd.concat(dfs).sort_index()

            # Ensure index is timezone-aware
            if combined_data.index.tz is None:
                combined_data.index = combined_data.index.tz_localize('UTC')

            test_data = combined_data[combined_data.index > train_end_date]

            if len(test_data) < 50:
                skipped_count += 1
                continue

            # Build features
            try:
                builder = TrendMLDatasetBuilder(test_data, config, symbol_obj)
                dataset_scaled, _ = builder.build_dataset(rr_ratio=rr_ratio, lookahead_periods=21)

                if len(dataset_scaled) == 0:
                    skipped_count += 1
                    continue

                # Make predictions
                feature_cols = [c for c in dataset_scaled.columns if c != 'label']
                X = dataset_scaled[feature_cols].values
                y_true = dataset_scaled['label'].values
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = (y_pred_proba >= threshold).astype(int)

                # Calculate metrics
                if len(np.unique(y_pred)) > 1:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    try:
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
                    except:
                        roc_auc = 0.5

                    tp = np.sum((y_pred == 1) & (y_true == 1))
                    fp = np.sum((y_pred == 1) & (y_true == 0))
                    fn = np.sum((y_pred == 0) & (y_true == 1))
                    tn = np.sum((y_pred == 0) & (y_true == 0))

                    expected_value = (precision * rr_ratio) - ((1 - precision) * 1)

                    symbol_results[sym] = {
                        'samples': len(y_true),
                        'signals': tp + fp,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn,
                        'tn': tn,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'roc_auc': roc_auc,
                        'expected_value': expected_value
                    }

                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
                    all_probas.extend(y_pred_proba)
                    processed_count += 1

                    if detailed and processed_count <= 20:  # Show first 20 symbols
                        self.stdout.write(f"\n{sym}:")
                        self.stdout.write(f"  Samples: {len(y_true)}, Signals: {tp+fp}")
                        self.stdout.write(f"  Precision: {precision:.2%} ({tp}/{tp+fp})")
                        self.stdout.write(f"  Recall: {recall:.2%} ({tp}/{tp+fn})")
                        self.stdout.write(f"  Expected Value: {expected_value:.3f}R")
                else:
                    skipped_count += 1

            except Exception as e:
                skipped_count += 1
                if detailed:
                    self.stdout.write(f"\n{sym}: Error - {str(e)[:50]}")
                continue

        # Overall test performance
        self.stdout.write(f"\n\nProcessed: {processed_count} symbols, Skipped: {skipped_count} symbols")

        if all_y_true:
            overall_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
            overall_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
            overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
            overall_roc_auc = roc_auc_score(all_y_true, all_probas) if all_probas else 0.5

            tp_total = sum([r['tp'] for r in symbol_results.values()])
            fp_total = sum([r['fp'] for r in symbol_results.values()])
            fn_total = sum([r['fn'] for r in symbol_results.values()])
            tn_total = sum([r['tn'] for r in symbol_results.values()])

            overall_expected = (overall_precision * rr_ratio) - ((1 - overall_precision) * 1)

            self.stdout.write("\n" + "="*80)
            self.stdout.write("OVERALL TEST SET PERFORMANCE")
            self.stdout.write("="*80)
            self.stdout.write(f"Total Test Samples: {len(all_y_true)}")
            self.stdout.write(f"Total Symbols Tested: {len(symbol_results)}")
            self.stdout.write(f"\nConfusion Matrix:")
            self.stdout.write(f"  True Positives:  {tp_total}")
            self.stdout.write(f"  False Positives: {fp_total}")
            self.stdout.write(f"  True Negatives:  {tn_total}")
            self.stdout.write(f"  False Negatives: {fn_total}")
            self.stdout.write(f"\nMetrics:")
            self.stdout.write(f"  Precision: {overall_precision:.2%} ({tp_total}/{tp_total+fp_total})")
            self.stdout.write(f"  Recall:    {overall_recall:.2%} ({tp_total}/{tp_total+fn_total})")
            self.stdout.write(f"  F1 Score:  {overall_f1:.2%}")
            self.stdout.write(f"  ROC-AUC:   {overall_roc_auc:.2%}")
            self.stdout.write(f"\nTrading Metrics (RR {rr_ratio}:1):")
            self.stdout.write(f"  Expected Value per Trade: {overall_expected:.3f}R")
            self.stdout.write(f"  For every $100 risked: ${overall_expected*100:.0f} profit")
            self.stdout.write(f"  Total Signals: {tp_total + fp_total}")

            # Break-even analysis
            break_even = 1 / (rr_ratio + 1)
            self.stdout.write(f"\nBreak-even Precision Needed: {break_even:.1%}")

            if overall_precision > break_even:
                self.stdout.write(self.style.SUCCESS(f"\n✓ Model is PROFITABLE on test data!"))
                self.stdout.write(f"  Actual precision: {overall_precision:.1%} > {break_even:.1%} needed")
            else:
                self.stdout.write(self.style.WARNING(f"\n⚠ Model is NOT PROFITABLE on test data"))
                self.stdout.write(f"  Need precision > {break_even:.1%}, got {overall_precision:.1%}")

            # Best and worst performing symbols (only if detailed)
            if detailed and len(symbol_results) > 0:
                self.stdout.write("\n" + "="*80)
                self.stdout.write("BEST PERFORMING SYMBOLS")
                self.stdout.write("="*80)
                best_symbols = sorted(symbol_results.items(), key=lambda x: x[1]['expected_value'], reverse=True)[:10]
                for sym, res in best_symbols:
                    if res['expected_value'] > 0:
                        self.stdout.write(f"  {sym}: {res['expected_value']:.3f}R (Precision: {res['precision']:.1%}, Signals: {res['signals']})")

                self.stdout.write("\n" + "="*80)
                self.stdout.write("WORST PERFORMING SYMBOLS")
                self.stdout.write("="*80)
                worst_symbols = sorted(symbol_results.items(), key=lambda x: x[1]['expected_value'])[:10]
                for sym, res in worst_symbols:
                    self.stdout.write(f"  {sym}: {res['expected_value']:.3f}R (Precision: {res['precision']:.1%}, Signals: {res['signals']})")

        else:
            self.stdout.write(self.style.ERROR("No test data found for evaluation"))
            self.stdout.write("\nPossible reasons:")
            self.stdout.write("  1. The model was trained on data up to the present (no out-of-sample period)")
            self.stdout.write("  2. The symbols in the dataset don't have data after the training date")
            self.stdout.write("  3. Not enough data after the training date (need at least 50 samples per symbol)")

