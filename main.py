# main.py
import os
import sys
import logging
import torch
import mlflow
import json

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

from data_generator import AdvancedTransactionSimulator
from preprocessing import TransactionNetworkPreprocessor
from baseline_models import BaselineModels
from trainer import FraudDetectionTrainer
from model import AdvancedFraudDetectionModel

def setup_logging():
    """Configure logging for the entire pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/fraud_detection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def save_model_results(results, filename='model_results.json'):
    """
    Save model evaluation results to a JSON file
    
    Args:
        results (dict): Model evaluation results
        filename (str): Output filename
    """
    with open(os.path.join('logs', filename), 'w') as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=4)

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Set MLflow tracking
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("Fraud Detection Research")

        # 1. Generate Synthetic Dataset
        logger.info("🚀 Generating Synthetic Transaction Dataset")
        simulator = AdvancedTransactionSimulator(
            num_accounts=5000, 
            num_transactions=100000, 
            fraud_ratio=0.07
        )
        simulator.save_dataset('data/synthetic_fraud_dataset.csv')

        # 2. Preprocess Data
        logger.info("🔍 Preprocessing Transaction Data")
        preprocessor = TransactionNetworkPreprocessor('data/synthetic_fraud_dataset.csv')
        processed_data = preprocessor.prepare_data()

        # 3. Train Baseline Models
        logger.info("📊 Training Baseline Models")
        baseline_models = BaselineModels(
            processed_data['X_train'], 
            processed_data['y_train'], 
            processed_data['X_test'], 
            processed_data['y_test']
        )
        
        # Train and evaluate different models
        models_to_train = {
            'Random Forest': baseline_models.train_random_forest(),
            'XGBoost': baseline_models.train_xgboost(),
            'Logistic Regression': baseline_models.train_logistic_regression(),
            'SVM': baseline_models.train_svm()
        }
        
        # Save results for each model
        for name, results in models_to_train.items():
            logger.info(f"{name} Metrics: {results}")
            save_model_results(results, f'{name.replace(" ", "_")}_results.json')

        # 4. Prepare Advanced Model
        logger.info("🧠 Preparing Advanced Fraud Detection Model")
        input_dim = processed_data['X_train'].shape[1]
        advanced_model = AdvancedFraudDetectionModel(input_dim)

        # 5. Train Advanced Model
        logger.info("🚀 Training Advanced Fraud Detection Model")
        trainer = FraudDetectionTrainer(
            model=advanced_model,
            train_data=processed_data,
            test_data=processed_data
        )
        trainer.train()

        logger.info("✅ Fraud Detection Experiment Completed Successfully")

    except Exception as e:
        logger.error(f"❌ Error in fraud detection pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()