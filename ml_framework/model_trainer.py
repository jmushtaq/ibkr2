import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, mean_squared_error, mean_absolute_error)
import joblib
import os
from datetime import datetime

from .models import MLModel, MLDataset, FeatureSet

class ModelTrainer:
    """
    Train and manage ML models
    """

    def __init__(self, dataset: pd.DataFrame, feature_names: List[str], label_name: str = 'label'):
        """
        Initialize model trainer

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

    def train_xgboost(self,
                     params: Dict = None,
                     train_df: pd.DataFrame = None,
                     val_df: pd.DataFrame = None,
                     name: str = "xgboost_model",
                     description: str = "") -> Dict:
        """
        Train XGBoost model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        # Check if classification or regression
        if len(np.unique(self.y)) <= 10:
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)

        print(f"Training XGBoost model: {name}")

        if train_df is not None and val_df is not None:
            X_train = train_df[self.feature_names].values
            y_train = train_df[self.label_name].values
            X_val = val_df[self.feature_names].values
            y_val = val_df[self.label_name].values

            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(self.X, self.y)
            X_val, y_val = self.X, self.y

        # Evaluate
        y_pred = model.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        # Feature importance
        importance = dict(zip(self.feature_names, model.feature_importances_))

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'params': params
        }

    def train_lightgbm(self,
                      params: Dict = None,
                      train_df: pd.DataFrame = None,
                      val_df: pd.DataFrame = None,
                      name: str = "lightgbm_model",
                      description: str = "") -> Dict:
        """
        Train LightGBM model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }

        # Check if classification or regression
        if len(np.unique(self.y)) <= 10:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        print(f"Training LightGBM model: {name}")

        if train_df is not None and val_df is not None:
            X_train = train_df[self.feature_names].values
            y_train = train_df[self.label_name].values
            X_val = val_df[self.feature_names].values
            y_val = val_df[self.label_name].values

            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(self.X, self.y)
            X_val, y_val = self.X, self.y

        # Evaluate
        y_pred = model.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        # Feature importance
        importance = dict(zip(self.feature_names, model.feature_importances_))

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'params': params
        }

    def train_random_forest(self,
                           params: Dict = None,
                           train_df: pd.DataFrame = None,
                           val_df: pd.DataFrame = None,
                           name: str = "random_forest_model",
                           description: str = "") -> Dict:
        """
        Train Random Forest model
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }

        # Check if classification or regression
        if len(np.unique(self.y)) <= 10:
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)

        print(f"Training Random Forest model: {name}")

        if train_df is not None:
            X_train = train_df[self.feature_names].values
            y_train = train_df[self.label_name].values
            model.fit(X_train, y_train)

            if val_df is not None:
                X_val = val_df[self.feature_names].values
                y_val = val_df[self.label_name].values
            else:
                X_val, y_val = X_train, y_train
        else:
            model.fit(self.X, self.y)
            X_val, y_val = self.X, self.y

        # Evaluate
        y_pred = model.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        # Feature importance
        importance = dict(zip(self.feature_names, model.feature_importances_))

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': importance,
            'params': params
        }

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        # Check if classification or regression
        is_classification = len(np.unique(y_true)) <= 10

        if is_classification:
            # Convert to integers for classification
            y_true = y_true.astype(int)
            y_pred = np.round(y_pred).astype(int)

            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),  # Convert to float
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            }

            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred))
                except:
                    metrics['roc_auc'] = 0.5
        else:
            # Regression metrics
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
            }

        return metrics

    def save_model(self, model: object, name: str, version: str,
                dataset: MLDataset, feature_set: FeatureSet,
                metrics: Dict, importance: Dict, params: Dict) -> MLModel:
        """
        Save model to database and disk
        """
        import numpy as np

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
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

        # Determine model type
        if len(np.unique(self.y)) <= 10:
            model_type = 'classification'
        else:
            model_type = 'regression'

        # Determine algorithm
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            algorithm = 'xgboost'
        elif isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            algorithm = 'lightgbm'
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            algorithm = 'random_forest'
        else:
            algorithm = 'gradient_boosting'

        # Convert metrics, importance, and params to native Python types
        metrics_native = convert_to_native(metrics)
        importance_native = convert_to_native(importance)
        params_native = convert_to_native(params)

        # Ensure all values are JSON serializable
        self.stdout = getattr(self, 'stdout', None)
        if self.stdout:
            self.stdout.write(f"  Converting metrics for JSON: {list(metrics_native.keys())}")

        # Create model record
        try:
            ml_model = MLModel.objects.create(
                name=name,
                description=f"{algorithm} model trained on {dataset.name}",
                model_type=model_type,
                algorithm=algorithm,
                dataset=dataset,
                feature_set=feature_set,
                hyperparameters=params_native,
                training_date=datetime.now().date(),
                training_samples=int(len(self.X)),  # Convert to int
                validation_samples=0,
                test_samples=0,
                performance_metrics=metrics_native,
                feature_importance=importance_native,
                version=version,
                is_active=True
            )
        except Exception as e:
            print(f"Error creating model record: {e}")
            # Try to save with simplified metrics if there's still an issue
            simplified_metrics = {}
            for k, v in metrics_native.items():
                try:
                    # Test JSON serialization
                    import json
                    json.dumps({k: v})
                    simplified_metrics[k] = v
                except:
                    simplified_metrics[k] = str(v)

            ml_model = MLModel.objects.create(
                name=name,
                description=f"{algorithm} model trained on {dataset.name}",
                model_type=model_type,
                algorithm=algorithm,
                dataset=dataset,
                feature_set=feature_set,
                hyperparameters=params_native,
                training_date=datetime.now().date(),
                training_samples=int(len(self.X)),
                validation_samples=0,
                test_samples=0,
                performance_metrics=simplified_metrics,
                feature_importance=importance_native,
                version=version,
                is_active=True
            )

        # Save model to disk
        model_path = f"ml_models/models/{name}_v{version}.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # Update model record with file path
        ml_model.model_file_path = model_path
        ml_model.save()

        print(f"Model saved: {name} v{version}")
        print(f"  Path: {model_path}")
        print(f"  Metrics: {metrics_native}")

        return ml_model

