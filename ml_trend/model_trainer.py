import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix)
import joblib
import os
from datetime import datetime
import logging

from .models import MLTrendModel

logger = logging.getLogger(__name__)


class TrendModelTrainer:
    """
    Train ML models for trend following
    """

    def __init__(self, dataset: pd.DataFrame, feature_names: List[str], label_name: str = 'label'):
        """
        Initialize trainer

        Args:
            dataset: DataFrame with features and labels
            feature_names: List of feature column names
            label_name: Name of label column
        """
        self.dataset = dataset
        self.feature_names = feature_names
        self.label_name = label_name

        # Prepare data
        self.X = dataset[feature_names].values
        self.y = dataset[label_name].values

    def train_xgboost(self, params: Dict = None, train_idx: List = None, val_idx: List = None) -> Dict:
        """
        Train XGBoost model
        """
        if params is None:
            params = {
                'n_estimators': min(200, len(self.X) // 2),  # Reduce trees for small datasets
                'max_depth': 3,  # Smaller depth for small datasets
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }

        if train_idx is not None and val_idx is not None:
            # Ensure indices are within bounds
            train_idx = [i for i in train_idx if i < len(self.X)]
            val_idx = [i for i in val_idx if i < len(self.X)]

            if len(train_idx) == 0 or len(val_idx) == 0:
                # Fall back to simple split
                split = int(len(self.X) * 0.8)
                train_idx = list(range(split))
                val_idx = list(range(split, len(self.X)))

            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X[val_idx]
            y_val = self.y[val_idx]

            # Check if we have at least one sample in each class
            unique_train = np.unique(y_train)
            unique_val = np.unique(y_val)

            if len(unique_train) < 2:
                logger.warning(f"Training data has only one class: {unique_train}")
                # Return dummy result
                return {
                    'model': None,
                    'metrics': {'accuracy': 0.5, 'precision': 0, 'recall': 0, 'f1_score': 0},
                    'feature_importance': {},
                    'params': params
                }

            model = xgb.XGBClassifier(**params)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(self.X, self.y)
            X_val, y_val = self.X, self.y

        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = self._evaluate(y_val, y_pred, y_pred_proba)
        importance = dict(zip(self.feature_names, model.feature_importances_))

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'params': params
        }

    def train_lightgbm(self, params: Dict = None, train_idx: List = None, val_idx: List = None) -> Dict:
        """
        Train LightGBM model
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }

        if train_idx is not None and val_idx is not None:
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X[val_idx]
            y_val = self.y[val_idx]

            model = lgb.LGBMClassifier(**params)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model = lgb.LGBMClassifier(**params)
            model.fit(self.X, self.y)
            X_val, y_val = self.X, self.y

        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = self._evaluate(y_val, y_pred, y_pred_proba)
        importance = dict(zip(self.feature_names, model.feature_importances_))

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'params': params
        }

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict:
        """
        Evaluate model performance
        """
        y_true = y_true.astype(int)
        y_pred = np.round(y_pred).astype(int)

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                metrics['roc_auc'] = float(roc_auc) if not np.isnan(roc_auc) else 0.5
            except:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_positives'] = int(cm[1, 1]) if len(cm) > 1 else 0
        metrics['false_positives'] = int(cm[0, 1]) if len(cm) > 1 else 0
        metrics['true_negatives'] = int(cm[0, 0]) if len(cm) > 1 else 0
        metrics['false_negatives'] = int(cm[1, 0]) if len(cm) > 1 else 0

        return metrics

    def save_model(self, model: object, name: str, version: str, dataset, config,
                metrics: Dict, importance: Dict, params: Dict, rr_ratio: float) -> MLTrendModel:
        """
        Save model to database and disk
        """
        import math

        # Determine algorithm
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            algorithm = 'xgboost'
        elif isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            algorithm = 'lightgbm'
        elif isinstance(model, RandomForestClassifier):
            algorithm = 'random_forest'
        else:
            algorithm = 'unknown'

        # Convert numpy types to Python native types and handle NaN
        def clean_nan(obj):
            """Recursively replace NaN with None for JSON serialization"""
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            elif isinstance(obj, (np.float32, np.float64)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return clean_nan(obj.tolist())
            else:
                return obj

        # Clean metrics, importance, and params
        metrics_clean = clean_nan(metrics)
        importance_clean = clean_nan(importance)
        params_clean = clean_nan(params)

        # Ensure all required metrics exist
        if metrics_clean.get('roc_auc') is None:
            metrics_clean['roc_auc'] = 0.5

        # Create model record
        try:
            ml_model = MLTrendModel.objects.create(
                name=name,
                description=f"{algorithm} model for trend following with RR {rr_ratio}:1",
                dataset=dataset,
                config=config,
                algorithm=algorithm,
                hyperparameters=params_clean,
                target_rr_ratio=rr_ratio,
                target_lookahead=21,  # 1 month lookahead
                training_date=datetime.now().date(),
                training_samples=len(self.X),
                performance_metrics=metrics_clean,
                feature_importance=importance_clean,
                version=version,
                is_active=True
            )
        except Exception as e:
            logger.error(f"Error creating model record: {e}")
            # Try with even more aggressive cleaning
            def deep_clean(obj):
                if isinstance(obj, dict):
                    return {k: deep_clean(v) for k, v in obj.items() if v is not None}
                elif isinstance(obj, list):
                    return [deep_clean(v) for v in obj if v is not None]
                elif isinstance(obj, float):
                    if np.isnan(obj) or np.isinf(obj):
                        return 0.0
                    return obj
                else:
                    return obj

            metrics_cleaner = deep_clean(metrics_clean)
            importance_cleaner = deep_clean(importance_clean)

            ml_model = MLTrendModel.objects.create(
                name=name,
                description=f"{algorithm} model for trend following with RR {rr_ratio}:1",
                dataset=dataset,
                config=config,
                algorithm=algorithm,
                hyperparameters=params_clean,
                target_rr_ratio=rr_ratio,
                target_lookahead=21,
                training_date=datetime.now().date(),
                training_samples=len(self.X),
                performance_metrics=metrics_cleaner,
                feature_importance=importance_cleaner,
                version=version,
                is_active=True
            )

        # Save model to disk
        model_path = f"ml_models/trend_models/{name}_v{version}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # Update model record with file path
        ml_model.model_file_path = model_path
        ml_model.save()

        logger.info(f"Model saved: {name} v{version}")
        logger.info(f"  Accuracy: {metrics_clean.get('accuracy', 0):.2%}")
        logger.info(f"  Precision: {metrics_clean.get('precision', 0):.2%}")
        logger.info(f"  F1 Score: {metrics_clean.get('f1_score', 0):.2%}")

        return ml_model

    def _convert_to_native(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

