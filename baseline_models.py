# baseline_models.py
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
import numpy as np
import pandas as pd

class BaselineModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Baseline models for fraud detection with enhanced evaluation
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=None):
        """
        Train a Random Forest model with advanced configuration
        
        Args:
            n_estimators (int): Number of trees in forest
            max_depth (int, optional): Maximum tree depth
        
        Returns:
            RandomForestClassifier: Trained model with detailed analysis
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced',  # Handle imbalanced datasets
            n_jobs=-1  # Use all cores
        )
        model.fit(self.X_train, self.y_train)
        
        # Detailed model analysis
        results = self.comprehensive_model_evaluation(model)
        results['feature_importances'] = self._get_feature_importances(model)
        results['model_description'] = "Random Forest: Wang et al. (2024) Approach"
        
        return results
    
    def train_xgboost(self, n_estimators=100, learning_rate=0.1):
        """
        Train an XGBoost model with advanced configuration
        
        Args:
            n_estimators (int): Number of boosting rounds
            learning_rate (float): Boosting learning rate
        
        Returns:
            XGBClassifier: Trained model with detailed analysis
        """
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            scale_pos_weight=1,  # Handle class imbalance
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        
        # Detailed model analysis
        results = self.comprehensive_model_evaluation(model)
        results['feature_importances'] = self._get_feature_importances(model)
        results['model_description'] = "XGBoost: Liu et al. (2023) Approach"
        
        return results
    
    def train_logistic_regression(self, C=1.0):
        """
        Train a Logistic Regression model
        
        Args:
            C (float): Inverse of regularization strength
        
        Returns:
            LogisticRegression: Trained model with detailed analysis
        """
        model = LogisticRegression(
            C=C,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        model.fit(self.X_train, self.y_train)
        
        # Detailed model analysis
        results = self.comprehensive_model_evaluation(model)
        results['model_description'] = "Logistic Regression: Baseline Approach"
        
        return results
    
    def train_svm(self, C=1.0, kernel='rbf'):
        """
        Train a Support Vector Machine (SVM) model
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('linear', 'rbf', etc.)
        
        Returns:
            SVC: Trained model with detailed analysis
        """
        model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        
        # Detailed model analysis
        results = self.comprehensive_model_evaluation(model)
        results['model_description'] = f"SVM ({kernel} kernel): Baseline Approach"
        
        return results
    
    def comprehensive_model_evaluation(self, model):
        """
        Comprehensive model evaluation with multiple metrics
        
        Args:
            model: Trained classifier
        
        Returns:
            dict: Detailed model performance metrics
        """
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_recall_fscore_support(self.y_test, y_pred, average='binary')[0],
            'recall': precision_recall_fscore_support(self.y_test, y_pred, average='binary')[1],
            'f1_score': precision_recall_fscore_support(self.y_test, y_pred, average='binary')[2],
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        
        return metrics
    
    def _get_feature_importances(self, model):
        """
        Extract feature importances with ranking
        
        Args:
            model: Trained model
        
        Returns:
            pd.DataFrame: Ranked feature importances
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        return None