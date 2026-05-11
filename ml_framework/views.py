from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.shortcuts import render

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .models import MLModel, Prediction, BacktestResult, MLDataset
from markets.models import Symbol, OHLCVData
from .dataset_manager import MLDatasetManager
import joblib

@require_http_methods(["GET"])
def get_predictions(request, symbol_ticker):
    """Get latest predictions for a symbol"""
    try:
        symbol = Symbol.objects.get(ticker=symbol_ticker)

        # Get latest predictions
        predictions = Prediction.objects.filter(
            symbol=symbol,
            prediction_type='signal'
        ).order_by('-timestamp')[:100]

        data = []
        for pred in predictions:
            data.append({
                'timestamp': pred.timestamp.isoformat(),
                'prediction': 'BUY' if pred.prediction_value == 1 else 'SELL',
                'confidence': pred.confidence,
                'model': pred.model.name,
                'actual_outcome': pred.actual_outcome,
                'was_correct': pred.was_correct
            })

        return JsonResponse({
            'symbol': symbol_ticker,
            'predictions': data
        })

    except Symbol.DoesNotExist:
        return JsonResponse({'error': 'Symbol not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_model_performance(request, model_id):
    """Get performance metrics for a specific model"""
    try:
        model = MLModel.objects.get(id=model_id)

        # Get backtest results
        backtests = model.backtests.order_by('-created_at')[:10]

        performance = {
            'model_name': model.name,
            'algorithm': model.algorithm,
            'model_type': model.model_type,
            'training_date': model.training_date,
            'metrics': model.performance_metrics,
            'feature_importance': model.feature_importance,
            'backtests': []
        }

        for backtest in backtests:
            performance['backtests'].append({
                'name': backtest.name,
                'sharpe_ratio': backtest.sharpe_ratio,
                'total_return': backtest.total_return,
                'win_rate': backtest.win_rate,
                'max_drawdown': backtest.max_drawdown,
                'total_trades': backtest.total_trades
            })

        return JsonResponse(performance)

    except MLModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def list_models(request):
    """List all trained models with their performance"""
    models = MLModel.objects.filter(is_active=True).order_by('-training_date')

    data = []
    for model in models:
        # Get best backtest
        best_backtest = model.backtests.order_by('-sharpe_ratio').first()

        model_data = {
            'id': model.id,
            'name': model.name,
            'algorithm': model.algorithm,
            'symbol': model.dataset.symbol.ticker if model.dataset.symbol else None,
            'frequency': model.dataset.frequency,
            'training_date': model.training_date,
            'accuracy': model.performance_metrics.get('accuracy'),
            'sharpe_ratio': best_backtest.sharpe_ratio if best_backtest else None,
            'total_return': best_backtest.total_return if best_backtest else None
        }
        data.append(model_data)

    return JsonResponse({'models': data})

@csrf_exempt
@require_http_methods(["POST"])
def generate_prediction(request):
    """Generate real-time prediction for a symbol"""
    try:
        data = json.loads(request.body)
        symbol_ticker = data.get('symbol')
        model_id = data.get('model_id')

        if not symbol_ticker or not model_id:
            return JsonResponse({'error': 'Missing symbol or model_id'}, status=400)

        # Get symbol and model
        symbol = Symbol.objects.get(ticker=symbol_ticker)
        model_obj = MLModel.objects.get(id=model_id)

        # Get the feature configuration from the model
        feature_config = model_obj.feature_set.features_json
        frequency = model_obj.dataset.frequency

        # Determine how much historical data we need based on frequency
        # We need enough data to compute all features (especially moving averages)
        if frequency == '1D':
            lookback_days = 252  # 1 year of daily data
        elif frequency == '1H':
            lookback_days = 30  # 1 month of hourly data
        else:
            lookback_days = 60  # Default

        # Get recent OHLCV data (more data than we need for feature calculation)
        ohlcv_records = OHLCVData.objects.filter(
            symbol=symbol,
            frequency=frequency
        ).order_by('-year', '-created_at')

        if not ohlcv_records:
            return JsonResponse({'error': f'No OHLCV data found for {symbol_ticker}'}, status=404)

        # Get the last 2 years of data to ensure we have enough for indicators
        dfs = []
        for record in ohlcv_records[:2]:  # Get last 2 years
            try:
                df = record.get_data_as_dataframe()
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading data: {e}")

        if not dfs:
            return JsonResponse({'error': 'No data available'}, status=404)

        recent_data = pd.concat(dfs).sort_index()

        # Ensure we have enough data points
        if len(recent_data) < 100:
            return JsonResponse({'error': f'Insufficient data: only {len(recent_data)} points available'}, status=400)

        # Keep only the last lookback_days
        recent_data = recent_data.last(f'{lookback_days}D')

        print(f"Recent data shape: {recent_data.shape}")
        print(f"Recent data date range: {recent_data.index[0]} to {recent_data.index[-1]}")

        # Load model
        model = joblib.load(model_obj.model_file_path)

        # Create a temporary dataset for prediction
        # We need to create a dataset with the same features as training
        manager = MLDatasetManager(
            data=recent_data,
            symbol=symbol_ticker,
            frequency=frequency
        )

        # Create dataset with features only (no labels needed for prediction)
        # Use a try-except to handle potential issues
        try:
            # Create a temporary dataset name
            temp_name = f"temp_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create dataset with the same feature config
            # We don't need labels for prediction
            temp_dataset = manager.create_dataset(
                feature_config=feature_config,
                label_config={'type': 'binary', 'lookahead': '1d'},  # Dummy label config
                name=temp_name,
                description="Temporary dataset for prediction",
                scaler_type='standard'
            )

            if temp_dataset.empty:
                return JsonResponse({'error': 'Failed to create dataset for prediction'}, status=500)

            # Get the latest row for prediction
            latest_row = temp_dataset.iloc[-1:]

            if latest_row.empty:
                return JsonResponse({'error': 'No valid data for prediction'}, status=500)

            # Make prediction
            feature_names = model_obj.dataset.feature_names
            X = latest_row[feature_names].values

            if X.shape[0] == 0:
                return JsonResponse({'error': 'No features available for prediction'}, status=500)

            prediction = model.predict(X)[0]

            if hasattr(model, 'predict_proba'):
                confidence = model.predict_proba(X)[0][1] if prediction == 1 else model.predict_proba(X)[0][0]
            else:
                confidence = 0.5

            # Save prediction
            pred = Prediction.objects.create(
                model=model_obj,
                symbol=symbol,
                frequency=frequency,
                timestamp=datetime.now(),
                prediction_type='signal',
                prediction_value=float(prediction),
                confidence=float(confidence),
                features_used=dict(zip(feature_names, X[0].tolist()))
            )

            # Clean up temporary dataset (optional - could be left for debugging)
            # MLDataset.objects.filter(name=temp_name).delete()

            return JsonResponse({
                'symbol': symbol_ticker,
                'prediction': 'BUY' if prediction == 1 else 'SELL',
                'confidence': float(confidence),
                'timestamp': pred.timestamp.isoformat(),
                'model': model_obj.name,
                'price': float(recent_data['close'].iloc[-1]),
                'date': recent_data.index[-1].isoformat()
            })

        except Exception as e:
            print(f"Error creating prediction dataset: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': f'Failed to prepare features: {str(e)}'}, status=500)

    except Symbol.DoesNotExist:
        return JsonResponse({'error': 'Symbol not found'}, status=404)
    except MLModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'}, status=404)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def generate_simple_prediction(request):
    """Generate simple prediction without full dataset creation"""
    try:
        data = json.loads(request.body)
        symbol_ticker = data.get('symbol')
        model_id = data.get('model_id')

        if not symbol_ticker or not model_id:
            return JsonResponse({'error': 'Missing symbol or model_id'}, status=400)

        # Get symbol and model
        symbol = Symbol.objects.get(ticker=symbol_ticker)
        model_obj = MLModel.objects.get(id=model_id)

        # Get the feature configuration
        feature_config = model_obj.feature_set.features_json
        frequency = model_obj.dataset.frequency

        # Get recent OHLCV data
        ohlcv_records = OHLCVData.objects.filter(
            symbol=symbol,
            frequency=frequency
        ).order_by('-year')[:2]

        dfs = []
        for record in ohlcv_records:
            df = record.get_data_as_dataframe()
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return JsonResponse({'error': 'No data available'}, status=404)

        recent_data = pd.concat(dfs).sort_index()

        # Get the latest complete data point
        latest_data = recent_data.iloc[-1:]

        # For a simple prediction, we could use the latest price and volume
        # This is a simplified version - you might want to precompute some features

        # Load model
        model = joblib.load(model_obj.model_file_path)

        # For now, return a message that we need to precompute features
        return JsonResponse({
            'error': 'Simple prediction not implemented yet. Please ensure the model has been trained with recent data.',
            'last_price': float(latest_data['close'].iloc[-1]),
            'last_date': latest_data.index[-1].isoformat(),
            'suggestion': 'Run the ML pipeline with recent data first'
        }, status=501)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_backtest_results(request, backtest_id):
    """Get detailed backtest results"""
    try:
        backtest = BacktestResult.objects.get(id=backtest_id)

        result = {
            'id': backtest.id,
            'name': backtest.name,
            'model': backtest.model.name,
            'start_date': backtest.start_date,
            'end_date': backtest.end_date,
            'metrics': {
                'total_return': backtest.total_return,
                'annualized_return': backtest.annualized_return,
                'sharpe_ratio': backtest.sharpe_ratio,
                'max_drawdown': backtest.max_drawdown,
                'win_rate': backtest.win_rate,
                'profit_factor': backtest.profit_factor,
                'total_trades': backtest.total_trades,
                'winning_trades': backtest.winning_trades,
                'losing_trades': backtest.losing_trades
            },
            'trades': backtest.trades[:100],  # Limit to 100 trades for performance
            'equity_curve': backtest.equity_curve
        }

        return JsonResponse(result)

    except BacktestResult.DoesNotExist:
        return JsonResponse({'error': 'Backtest not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def dashboard(request):
    """ML Trading Dashboard"""
    return render(request, 'ml_framework/dashboard.html')
