"""
Model training module for the data science collaboration project.

This module contains functions for training, evaluating, and saving
machine learning models.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from typing import Tuple, Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class to handle model training, evaluation, and saving.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        
    def prepare_data(self, data_path: str, target_column: str, test_size: float = 0.2) -> Tuple:
        """
        Load and prepare data for training.
        
        Args:
            data_path (str): Path to the processed data file
            target_column (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if self._is_classification(y) else None
            )
            
            logger.info(f"Training set shape: {X_train.shape}")
            logger.info(f"Test set shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Check if the problem is classification or regression."""
        return y.dtype == 'object' or y.nunique() <= 10
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            problem_type: str = 'auto') -> Dict:
        """
        Train multiple models and compare their performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            problem_type (str): 'classification', 'regression', or 'auto'
            
        Returns:
            Dict: Trained models and their scores
        """
        if problem_type == 'auto':
            problem_type = 'classification' if self._is_classification(y_train) else 'regression'
        
        logger.info(f"Training models for {problem_type} problem...")
        
        if problem_type == 'classification':
            models_to_train = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42)
            }
            scoring_metric = 'accuracy'
        else:
            models_to_train = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'SVR': SVR()
            }
            scoring_metric = 'neg_mean_squared_error'
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring_metric)
            mean_score = cv_scores.mean()
            
            self.models[name] = model
            self.model_scores[name] = mean_score
            
            logger.info(f"{name} CV Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with score: {self.model_scores[best_model_name]:.4f}")
        
        return self.models
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str = 'RandomForest') -> Any:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Name of the model to tune
            
        Returns:
            Best model after hyperparameter tuning
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        if self._is_classification(y_train):
            if model_name == 'RandomForest':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                scoring = 'accuracy'
        else:
            if model_name == 'RandomForest':
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                scoring = 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        predictions = model.predict(X_test)
        
        if self._is_classification(y_test):
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, predictions)
            }
            
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            
        else:
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            logger.info(f"Test RMSE: {rmse:.4f}")
            logger.info(f"Test R² Score: {r2:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, model: Any, feature_names: list, top_n: int = 10) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            top_n (int): Number of top features to display
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            logger.info(f"Plotted top {top_n} feature importances")
        else:
            logger.warning("Model does not have feature_importances_ attribute")
    
    def save_model(self, model: Any, model_path: str, method: str = 'joblib') -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_path (str): Path to save the model
            method (str): Serialization method ('joblib' or 'pickle')
        """
        try:
            if method == 'joblib':
                joblib.dump(model, model_path)
            elif method == 'pickle':
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            logger.info(f"Model saved successfully to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str, method: str = 'joblib') -> Any:
        """
        Load trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            method (str): Serialization method ('joblib' or 'pickle')
            
        Returns:
            Loaded model
        """
        try:
            if method == 'joblib':
                model = joblib.load(model_path)
            elif method == 'pickle':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Prepare data (example paths)
        data_path = "../data/processed/processed_data.csv"
        target_column = "target"  # Replace with actual target column name
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(data_path, target_column)
        
        # Train multiple models
        models = trainer.train_multiple_models(X_train, y_train)
        
        # Hyperparameter tuning for best model
        best_model = trainer.hyperparameter_tuning(X_train, y_train, 'RandomForest')
        
        # Evaluate model
        metrics = trainer.evaluate_model(best_model, X_test, y_test)
        print("Evaluation Metrics:", metrics)
        
        # Plot feature importance
        trainer.plot_feature_importance(best_model, X_train.columns.tolist())
        
        # Save model
        trainer.save_model(best_model, "../models/best_model.joblib")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

"""Model training utilities."""

from sklearn.ensemble import GradientBoostingClassifier

def train_model(X_train, y_train):
    """Train a machine learning model."""
    # Use Gradient Boosting for better performance
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    main()
