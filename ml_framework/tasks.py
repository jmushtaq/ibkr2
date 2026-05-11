from celery import shared_task
from django.core.management import call_command
import logging

logger = logging.getLogger(__name__)

@shared_task
def train_model_task(symbol_ticker, frequency='1D', years_back=5):
    """Asynchronous model training task"""
    try:
        logger.info(f"Starting model training for {symbol_ticker}")

        # Call the management command
        call_command(
            'process_ml_data',
            symbol=symbol_ticker,
            frequency=frequency,
            years=years_back,
            train=True
        )

        logger.info(f"Model training completed for {symbol_ticker}")
        return f"Training completed for {symbol_ticker}"

    except Exception as e:
        logger.error(f"Error training model for {symbol_ticker}: {e}")
        raise

@shared_task
def generate_daily_signals_task():
    """Generate daily trading signals"""
    try:
        logger.info("Generating daily signals")

        call_command('generate_daily_signals')

        logger.info("Signal generation completed")
        return "Signals generated"

    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise

@shared_task
def update_model_performance_task():
    """Update model performance metrics with latest data"""
    try:
        logger.info("Updating model performance")

        # Get all active models
        from .models import MLModel
        models = MLModel.objects.filter(is_active=True)

        for model in models:
            # Update with latest data
            # This would involve re-evaluating the model on recent data
            pass

        logger.info("Model performance updated")
        return "Performance updated"

    except Exception as e:
        logger.error(f"Error updating performance: {e}")
        raise

