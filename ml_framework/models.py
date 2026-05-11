from django.db import models
from django.contrib.postgres.fields import JSONField
from django.core.validators import MinValueValidator, MaxValueValidator
from markets.models import Symbol, OHLCVData, PrecomputedMetrics, TechnicalIndicators
import json
from datetime import date

class FeatureSet(models.Model):
    """
    Stores computed feature sets for ML analysis
    """
    FEATURE_TYPE_CHOICES = [
        ('price', 'Price-based Features'),
        ('technical', 'Technical Indicators'),
        ('volatility', 'Volatility Features'),
        ('volume', 'Volume Features'),
        ('momentum', 'Momentum Features'),
        ('pattern', 'Pattern Recognition'),
        ('fundamental', 'Fundamental Features'),
        ('custom', 'Custom Features'),
    ]

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    feature_type = models.CharField(max_length=20, choices=FEATURE_TYPE_CHOICES)
    features_json = models.JSONField(default=dict, help_text="Dictionary of feature names and their configurations")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.feature_type})"

class LabeledDataset(models.Model):
    """
    Stores labeled datasets for supervised learning
    """
    LABEL_TYPE_CHOICES = [
        ('binary', 'Binary Classification (Up/Down)'),
        ('multi_class', 'Multi-class Classification'),
        ('regression', 'Regression'),
        ('risk_reward', 'Risk-Reward Ratio'),
    ]

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    label_type = models.CharField(max_length=20, choices=LABEL_TYPE_CHOICES)

    # Configuration for labeling strategy
    config = models.JSONField(default=dict, help_text="Labeling configuration (lookahead periods, thresholds, etc.)")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.label_type})"

class MLDataset(models.Model):
    """
    Stores the actual ML dataset with features and labels
    """
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    # References to feature set and labeling strategy
    feature_set = models.ForeignKey(FeatureSet, on_delete=models.PROTECT, related_name='datasets')
    labeled_dataset = models.ForeignKey(LabeledDataset, on_delete=models.PROTECT, related_name='datasets')

    # Data storage
    data = models.JSONField(default=dict, help_text="Structured data with features and labels")

    # Metadata
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='ml_datasets', null=True, blank=True)
    frequency = models.CharField(max_length=10, choices=OHLCVData.FREQUENCY_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    total_samples = models.IntegerField(default=0)
    feature_names = models.JSONField(default=list, help_text="List of feature names")

    # Versioning
    version = models.CharField(max_length=20, default='1.0.0')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'version']
        indexes = [
            models.Index(fields=['symbol', 'frequency']),
            models.Index(fields=['start_date', 'end_date']),
        ]

    def __str__(self):
        return f"{self.name} v{self.version}"

    def to_dataframe(self):
        """Convert stored data to pandas DataFrame"""
        import pandas as pd
        if self.data and 'samples' in self.data:
            df = pd.DataFrame(self.data['samples'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()

    def get_feature_matrix(self):
        """Return X (features) and y (labels) as numpy arrays"""
        import numpy as np
        df = self.to_dataframe()
        if df.empty:
            return np.array([]), np.array([])

        X = df[self.feature_names].values
        y = df['label'].values if 'label' in df.columns else np.array([])
        return X, y

class MLModel(models.Model):
    """
    Stores trained ML models and their metadata
    """
    MODEL_TYPE_CHOICES = [
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('time_series', 'Time Series'),
    ]

    ALGORITHM_CHOICES = [
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('catboost', 'CatBoost'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('logistic_regression', 'Logistic Regression'),
        ('neural_network', 'Neural Network'),
        ('lstm', 'LSTM'),
    ]

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    algorithm = models.CharField(max_length=30, choices=ALGORITHM_CHOICES)

    # References
    dataset = models.ForeignKey(MLDataset, on_delete=models.PROTECT, related_name='models')
    feature_set = models.ForeignKey(FeatureSet, on_delete=models.PROTECT, related_name='models')

    # Model configuration
    hyperparameters = models.JSONField(default=dict, help_text="Model hyperparameters")

    # Training metadata
    training_date = models.DateField(auto_now_add=True)
    training_samples = models.IntegerField()
    validation_samples = models.IntegerField()
    test_samples = models.IntegerField()

    # Performance metrics
    performance_metrics = models.JSONField(default=dict, help_text="Accuracy, precision, recall, etc.")

    # Feature importance
    feature_importance = models.JSONField(default=dict, help_text="Feature importance scores")

    # Model file path (if saved to disk)
    model_file_path = models.CharField(max_length=500, blank=True, help_text="Path to saved model file")

    # Versioning
    version = models.CharField(max_length=20)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'version']

    def __str__(self):
        return f"{self.name} v{self.version} ({self.algorithm})"

class Prediction(models.Model):
    """
    Stores model predictions for backtesting and live trading
    """
    PREDICTION_TYPE_CHOICES = [
        ('signal', 'Trading Signal'),
        ('probability', 'Probability'),
        ('price_target', 'Price Target'),
    ]

    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='predictions')
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name='predictions')
    frequency = models.CharField(max_length=10, choices=OHLCVData.FREQUENCY_CHOICES)
    timestamp = models.DateTimeField()

    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPE_CHOICES)
    prediction_value = models.FloatField()
    confidence = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(1)])

    # Actual outcome (for tracking accuracy)
    actual_outcome = models.FloatField(null=True, blank=True)
    was_correct = models.BooleanField(null=True, blank=True)

    # Features used for this prediction (optional, for debugging)
    features_used = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['model', 'symbol', '-timestamp']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"{self.symbol.ticker} - {self.prediction_value:.2f} ({self.timestamp})"

class BacktestResult(models.Model):
    """
    Stores backtest results for model evaluation
    """
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='backtests')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    # Backtest configuration
    start_date = models.DateField()
    end_date = models.DateField()
    initial_capital = models.FloatField(default=100000)
    position_size = models.FloatField(default=0.1, help_text="Position size as % of capital")
    stop_loss = models.FloatField(null=True, blank=True, help_text="Stop loss percentage")
    take_profit = models.FloatField(null=True, blank=True, help_text="Take profit percentage")

    # Performance metrics
    total_return = models.FloatField()
    annualized_return = models.FloatField()
    sharpe_ratio = models.FloatField()
    max_drawdown = models.FloatField()
    win_rate = models.FloatField()
    profit_factor = models.FloatField()
    total_trades = models.IntegerField()
    winning_trades = models.IntegerField()
    losing_trades = models.IntegerField()

    # Detailed results
    trades = models.JSONField(default=list, help_text="List of all trades")
    equity_curve = models.JSONField(default=list, help_text="Equity curve over time")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.total_return:.2%} return"

