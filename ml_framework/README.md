# 1. Run initial data processing and model training
python manage.py process_ml_data --symbol AAPL --frequency 1D --years 5 --train

# 2. Generate daily signals
python manage.py generate_daily_signals

# 3. Query predictions via API
curl http://localhost:8000/ml/api/predictions/AAPL/

# 4. List available models
curl http://localhost:8000/ml/api/models/

# 5. Get detailed model performance
curl http://localhost:8000/ml/api/models/1/

# 6. Generate real-time prediction
curl -X POST http://localhost:8000/ml/api/predict/ -H "Content-Type: application/json" -d '{"symbol": "AAPL", "model_id": 1}'
OR
	# Generate prediction without saving
	python manage.py generate_predictions --symbol AAPL --model-id 1

	# Generate and save prediction
	python manage.py generate_predictions --symbol AAPL --model-id 1 --save

# JM

# --force (save)
python manage.py train_improved_model --symbol AAPL --years 7 --strategy trend_following --force
python manage.py analyze_best_model --model-id 6
python manage.py generate_daily_signals --model-id 6


# 7. Run Jupyter notebook for analysis
jupyter notebook notebooks/ml_analysis.ipynb

# 8. Schedule daily tasks with Celery
celery -A ibkr_project beat --loglevel=info



This comprehensive framework provides:

Complete end-to-end ML pipeline from data loading to model deployment

Multiple labeling strategies for different trading objectives

Feature engineering with 100+ financial features

Multiple model algorithms with hyperparameter tuning

Backtesting framework with realistic trade execution

Database storage for all ML artifacts

REST API for real-time predictions

Jupyter notebooks for interactive analysis

Celery tasks for automated processing

Docker support for containerized deployment

The framework is modular and extensible, allowing you to add new features, models, or trading strategies as needed.
