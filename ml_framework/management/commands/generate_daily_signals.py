from django.core.management.base import BaseCommand
from ml_framework.models import MLModel, Prediction
from markets.models import Symbol, OHLCVData
from ml_framework.dataset_manager import MLDatasetManager
import pandas as pd
import joblib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Generate daily trading signals using the best models'

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, help='Specific symbol to process')
        parser.add_argument('--model-id', type=int, help='Specific model to use')
        parser.add_argument('--min-confidence', type=float, default=0.6, help='Minimum confidence for signals')
        parser.add_argument('--save', action='store_true', help='Save predictions to database')

    def handle(self, *args, **options):
        self.stdout.write("Generating trading signals...")

        min_confidence = options['min_confidence']
        save_to_db = options['save']

        # Get models to use
        if options['model_id']:
            models = MLModel.objects.filter(id=options['model_id'], is_active=True)
        else:
            # Get best model per symbol based on accuracy
            models = MLModel.objects.filter(is_active=True).order_by('-performance_metrics__accuracy')

            # Group by symbol and take best model
            best_models = {}
            for model in models:
                if model.dataset.symbol:
                    symbol = model.dataset.symbol
                    if symbol not in best_models:
                        best_models[symbol] = model
            models = list(best_models.values())

        if options['symbol']:
            models = [m for m in models if m.dataset.symbol and m.dataset.symbol.ticker == options['symbol']]

        if not models:
            self.stdout.write(self.style.ERROR("No models found"))
            return

        signals = []

        for model in models:
            symbol = model.dataset.symbol
            if not symbol:
                continue

            self.stdout.write(f"\nProcessing {symbol.ticker} with model: {model.name}")

            try:
                # Get recent OHLCV data
                ohlcv_records = OHLCVData.objects.filter(
                    symbol=symbol,
                    frequency=model.dataset.frequency
                ).order_by('-year')[:2]

                dfs = []
                for record in ohlcv_records:
                    df = record.get_data_as_dataframe()
                    if df is not None and not df.empty:
                        dfs.append(df)

                if not dfs:
                    self.stdout.write(self.style.WARNING("No recent data available"))
                    continue

                recent_data = pd.concat(dfs).sort_index()
                recent_data = recent_data.last('3M')  # Last 3 months

                # Create dataset manager
                manager = MLDatasetManager(
                    data=recent_data,
                    symbol=symbol.ticker,
                    frequency=model.dataset.frequency
                )

                # Generate features
                dataset = manager.create_dataset(
                    feature_config=model.feature_set.features_json,
                    label_config={'type': 'binary', 'lookahead': '1d'},
                    name=f"temp_{symbol.ticker}",
                    description="Temporary dataset for prediction",
                    scaler_type='standard'
                )

                if len(dataset) == 0:
                    self.stdout.write(self.style.WARNING("No valid samples for prediction"))
                    continue

                # Load model
                loaded_model = joblib.load(model.model_file_path)

                # Make predictions
                feature_names = model.dataset.feature_names
                X = dataset[feature_names].values
                predictions = loaded_model.predict(X)

                if hasattr(loaded_model, 'predict_proba'):
                    probabilities = loaded_model.predict_proba(X)[:, 1]
                else:
                    probabilities = [0.5] * len(predictions)

                # Get latest prediction
                latest_prediction = predictions[-1]
                latest_confidence = probabilities[-1]
                latest_date = dataset.index[-1]
                latest_price = recent_data['close'].iloc[-1]

                # Determine signal
                if latest_prediction == 1 and latest_confidence >= min_confidence:
                    signal = "BUY"
                    strength = "STRONG" if latest_confidence >= 0.8 else "MODERATE"
                elif latest_prediction == 0 and latest_confidence >= min_confidence:
                    signal = "SELL"
                    strength = "STRONG" if latest_confidence >= 0.8 else "MODERATE"
                else:
                    signal = "NEUTRAL"
                    strength = "WEAK"

                signal_info = {
                    'symbol': symbol.ticker,
                    'model': model.name,
                    'date': latest_date,
                    'price': latest_price,
                    'signal': signal,
                    'confidence': latest_confidence,
                    'strength': strength
                }

                signals.append(signal_info)

                self.stdout.write(f"  Date: {latest_date.date()}")
                self.stdout.write(f"  Price: ${latest_price:.2f}")
                self.stdout.write(f"  Signal: {signal} ({strength})")
                self.stdout.write(f"  Confidence: {latest_confidence:.1%}")

                # Save to database
                if save_to_db:
                    pred = Prediction.objects.create(
                        model=model,
                        symbol=symbol,
                        frequency=model.dataset.frequency,
                        timestamp=datetime.now(),
                        prediction_type='signal',
                        prediction_value=float(latest_prediction),
                        confidence=float(latest_confidence),
                        features_used=dict(zip(feature_names, X[-1].tolist()))
                    )
                    self.stdout.write(f"  ✓ Prediction saved (ID: {pred.id})")

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error: {e}"))
                import traceback
                traceback.print_exc()

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write("DAILY SIGNALS SUMMARY")
        self.stdout.write("="*60)

        for signal in signals:
            color = self.style.SUCCESS if signal['signal'] == 'BUY' else self.style.ERROR if signal['signal'] == 'SELL' else self.style.WARNING
            self.stdout.write(color(f"{signal['symbol']}: {signal['signal']} @ ${signal['price']:.2f} (Confidence: {signal['confidence']:.1%})"))

        self.stdout.write(f"\nTotal signals: {len(signals)}")
        self.stdout.write(self.style.SUCCESS("\n✓ Signal generation complete"))
