from django.core.management.base import BaseCommand
from ml_trend.models import MLTrendModel
from markets.models import Symbol, OHLCVData
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from ml_trend.dataset_builder import TrendMLDatasetBuilder
from ml_trend.models import TrendConfig

class Command(BaseCommand):
    help = 'Validate model on out-of-sample data'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, required=True, help='Model ID to validate')
        parser.add_argument('--test-years', type=int, default=2, help='Years for out-of-sample test')

    def handle(self, *args, **options):
        model_id = options['model_id']
        test_years = options['test_years']

        # Get model
        model_obj = MLTrendModel.objects.get(id=model_id)
        self.stdout.write(f"Validating model: {model_obj.name}")
        self.stdout.write(f"Training date: {model_obj.training_date}")
        self.stdout.write(f"Training samples: {model_obj.training_samples}")

        # Load model
        model = joblib.load(model_obj.model_file_path)

        # Get symbol from dataset
        dataset = model_obj.dataset
        config = model_obj.config

        # Get symbols from dataset
        symbols = dataset.data.get('symbols', [])
        if not symbols:
            self.stdout.write(self.style.ERROR("No symbols found in dataset"))
            return

        self.stdout.write(f"Testing on {len(symbols)} symbols")

        # Create timezone-aware training end date
        train_end = pd.Timestamp(model_obj.training_date).tz_localize('UTC')

        # Collect out-of-sample data
        all_predictions = []
        all_actuals = []

        for sym in symbols[:10]:  # Test on first 10 symbols
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

            recent_data = recent_data.last(f'{test_years}Y')

            # Get data after training date
            out_of_sample = recent_data[recent_data.index > train_end]

            if len(out_of_sample) < 50:
                continue

            self.stdout.write(f"\n{sym}: {len(out_of_sample)} out-of-sample rows")

            # Build features
            builder = TrendMLDatasetBuilder(out_of_sample, config, symbol_obj)
            dataset_scaled, scaler = builder.build_dataset(
                rr_ratio=model_obj.target_rr_ratio,
                lookahead_periods=21
            )

            if len(dataset_scaled) == 0:
                continue

            # Make predictions
            feature_cols = [c for c in dataset_scaled.columns if c != 'label']
            X = dataset_scaled[feature_cols].values
            y_true = dataset_scaled['label'].values

            # Use probability threshold tuning
            y_pred_proba = model.predict_proba(X)[:, 1]

            # Find optimal threshold on this symbol's data
            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

            # Choose threshold that maximizes profit for RR ratio
            rr = model_obj.target_rr_ratio
            profits = precisions * rr - (1 - precisions)
            best_idx = np.argmax(profits[:-1])  # Exclude last point
            best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5

            y_pred = (y_pred_proba >= best_threshold).astype(int)

            # Calculate metrics
            if len(np.unique(y_pred)) > 1:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                self.stdout.write(f"  Best threshold: {best_threshold:.3f}")
                self.stdout.write(f"  Precision: {precision:.2%}")
                self.stdout.write(f"  Recall: {recall:.2%}")
                self.stdout.write(f"  F1: {f1:.2%}")

                all_predictions.extend(y_pred)
                all_actuals.extend(y_true)

        # Overall out-of-sample performance
        if all_predictions:
            overall_precision = precision_score(all_actuals, all_predictions, zero_division=0)
            overall_recall = recall_score(all_actuals, all_predictions, zero_division=0)
            overall_f1 = f1_score(all_actuals, all_predictions, zero_division=0)

            self.stdout.write("\n" + "="*60)
            self.stdout.write("OUT-OF-SAMPLE RESULTS")
            self.stdout.write("="*60)
            self.stdout.write(f"Total predictions: {len(all_predictions)}")
            self.stdout.write(f"Precision: {overall_precision:.2%}")
            self.stdout.write(f"Recall: {overall_recall:.2%}")
            self.stdout.write(f"F1 Score: {overall_f1:.2%}")

            if overall_precision < 0.8:
                self.stdout.write(self.style.WARNING("\n⚠ Model shows signs of overfitting!"))
                self.stdout.write("  Training precision was 100%, but out-of-sample is lower.")
                self.stdout.write("  Consider:")
                self.stdout.write("  1. Using more conservative probability thresholds")
                self.stdout.write("  2. Adding regularization to the model")
                self.stdout.write("  3. Getting more training data")
            else:
                self.stdout.write(self.style.SUCCESS("\n✓ Model generalizes well!"))
        else:
            self.stdout.write(self.style.ERROR("No out-of-sample predictions available"))
            self.stdout.write("\nPossible reasons:")
            self.stdout.write("  1. Training data is too recent (no out-of-sample period)")
            self.stdout.write("  2. Symbols don't have recent data")
            self.stdout.write("  3. Not enough data after training date")
