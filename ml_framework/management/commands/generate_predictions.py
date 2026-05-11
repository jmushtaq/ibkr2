from django.core.management.base import BaseCommand
from ml_framework.models import MLModel, Prediction
from markets.models import Symbol, OHLCVData
from ml_framework.dataset_manager import MLDatasetManager
import pandas as pd
import joblib
from datetime import datetime

class Command(BaseCommand):
    help = 'Generate predictions using trained models'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, required=True, help='Symbol ticker')
        parser.add_argument('--model-id', type=int, help='Specific model to use')
        parser.add_argument('--save', action='store_true', help='Save predictions to database')

    def handle(self, *args, **options):
        symbol_ticker = options['symbol']
        model_id = options.get('model_id')
        save_to_db = options.get('save', False)

        self.stdout.write(f"Generating predictions for {symbol_ticker}")

        # Get symbol
        try:
            symbol = Symbol.objects.get(ticker=symbol_ticker)
        except Symbol.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Symbol {symbol_ticker} not found"))
            return

        # Get models
        if model_id:
            models = MLModel.objects.filter(id=model_id, is_active=True)
        else:
            models = MLModel.objects.filter(dataset__symbol=symbol, is_active=True)

        if not models:
            self.stdout.write(self.style.ERROR(f"No models found for {symbol_ticker}"))
            return

        # Get recent OHLCV data
        ohlcv_records = OHLCVData.objects.filter(
            symbol=symbol,
            frequency='1D'
        ).order_by('-year')[:2]

        if not ohlcv_records:
            self.stdout.write(self.style.ERROR(f"No OHLCV data found for {symbol_ticker}"))
            return

        # Combine data
        dfs = []
        for record in ohlcv_records:
            df = record.get_data_as_dataframe()
            if df is not None and not df.empty:
                dfs.append(df)

        recent_data = pd.concat(dfs).sort_index()
        self.stdout.write(f"Loaded {len(recent_data)} rows of recent data")

        for model_obj in models:
            self.stdout.write(f"\nUsing model: {model_obj.name}")

            # Get feature config
            feature_config = model_obj.feature_set.features_json
            frequency = model_obj.dataset.frequency

            # Create dataset manager
            manager = MLDatasetManager(
                data=recent_data,
                symbol=symbol_ticker,
                frequency=frequency
            )

            try:
                # Create dataset with features
                dataset = manager.create_dataset(
                    feature_config=feature_config,
                    label_config={'type': 'binary', 'lookahead': '1d'},
                    name=f"prediction_{symbol_ticker}",
                    description="Temporary dataset for prediction",
                    scaler_type='standard'
                )

                if len(dataset) == 0:
                    self.stdout.write(self.style.WARNING("No valid samples for prediction"))
                    continue

                # Load model
                model = joblib.load(model_obj.model_file_path)

                # Make predictions
                feature_names = model_obj.dataset.feature_names
                X = dataset[feature_names].values
                predictions = model.predict(X)

                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)[:, 1]
                else:
                    probabilities = [0.5] * len(predictions)

                # Get the latest prediction
                latest_prediction = predictions[-1]
                latest_confidence = probabilities[-1]
                latest_date = dataset.index[-1]

                self.stdout.write(f"Latest date: {latest_date.date()}")
                self.stdout.write(f"Prediction: {'BUY' if latest_prediction == 1 else 'SELL'}")
                self.stdout.write(f"Confidence: {latest_confidence:.2%}")

                # Save to database if requested
                if save_to_db:
                    pred = Prediction.objects.create(
                        model=model_obj,
                        symbol=symbol,
                        frequency=frequency,
                        timestamp=datetime.now(),
                        prediction_type='signal',
                        prediction_value=float(latest_prediction),
                        confidence=float(latest_confidence),
                        features_used=dict(zip(feature_names, X[-1].tolist()))
                    )
                    self.stdout.write(self.style.SUCCESS(f"Prediction saved: {pred.id}"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error generating prediction: {e}"))
                import traceback
                traceback.print_exc()
