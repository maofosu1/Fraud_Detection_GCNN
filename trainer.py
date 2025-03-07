# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import logging
from torch_geometric.loader import DataLoader as GeometricDataLoader
import os
from datetime import datetime

class FraudDetectionTrainer:
    def __init__(self, model, train_data, test_data, learning_rate=1e-3, batch_size=32, epochs=50, 
                 lambda1=1.0, lambda2=0.5, use_graph_data=True, save_dir='models'):
        """
        Enhanced trainer for the fraud detection model with multi-task learning and triplet loss
        
        Args:
            model (nn.Module): Fraud detection model
            train_data (dict): Training data including graph data
            test_data (dict): Test data including graph data
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            epochs (int): Number of epochs
            lambda1 (float): Weight for fraud detection loss
            lambda2 (float): Weight for anomaly detection loss
            use_graph_data (bool): Whether to use graph data
            save_dir (str): Directory to save models
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_graph_data = use_graph_data
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss(weight=self._get_class_weights())
        self.regression_criterion = nn.MSELoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    def _get_class_weights(self):
        """
        Calculate class weights to handle imbalanced data
        
        Returns:
            torch.Tensor: Class weights
        """
        y_train = self.train_data['y_train']
        class_counts = np.bincount(y_train.astype(int))
        total = len(y_train)
        
        # Calculate weights inversely proportional to class frequencies
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def train(self):
        """
        Train the model with graph data, LSTM sequences, and multi-task learning
        """
        # Prepare data loaders based on graph or tabular data
        if self.use_graph_data and 'graph_data' in self.train_data:
            self.logger.info("Using graph-based data for training")
            train_loader = GeometricDataLoader([self.train_data['graph_data']], batch_size=self.batch_size)
            test_loader = GeometricDataLoader([self.test_data['graph_data']], batch_size=self.batch_size)
        else:
            self.logger.info("Using tabular data for training")
            train_loader = self._prepare_tabular_data_loader(self.train_data)
            test_loader = self._prepare_tabular_data_loader(self.test_data)
        
        # Start MLflow run
        mlflow.set_experiment("Advanced Fraud Detection")
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "model_type": type(self.model).__name__,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "use_graph_data": self.use_graph_data
            })
            
            # Initialize best metrics
            best_f1 = 0
            best_auc = 0
            best_epoch = 0
            
            # Training loop with early stopping
            patience = 10
            patience_counter = 0
            min_delta = 0.001  # Minimum improvement to consider
            
            # Track metrics for plotting
            train_losses = []
            val_metrics = {
                'f1': [], 'auc': [], 'accuracy': [], 'precision': [], 'recall': []
            }
            
            for epoch in range(self.epochs):
                # Train epoch
                epoch_metrics = self._train_epoch(train_loader, epoch)
                train_losses.append(epoch_metrics['train_loss'])
                
                # Evaluate
                test_metrics = self.evaluate(test_loader)
                
                # Update learning rate based on F1 score
                self.scheduler.step(test_metrics['f1'])
                
                # Log metrics
                self._log_metrics(epoch_metrics, test_metrics, epoch)
                
                # Save metrics for plotting
                for k, v in test_metrics.items():
                    if k in val_metrics:
                        val_metrics[k].append(v)
                
                # Check for best model
                if test_metrics['f1'] > best_f1 + min_delta:
                    best_f1 = test_metrics['f1']
                    best_auc = test_metrics['auc']
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    self._save_model(f"best_model_f1_{best_f1:.4f}.pt")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log best results
            mlflow.log_metrics({
                "best_f1": best_f1,
                "best_auc": best_auc,
                "best_epoch": best_epoch
            })
            
            # Create and log performance plots
            self._create_and_log_plots(train_losses, val_metrics)
            
            # Log best model to MLflow
            mlflow.pytorch.log_model(self.model, "best_model")
            
            return {
                "best_f1": best_f1,
                "best_auc": best_auc,
                "best_epoch": best_epoch
            }
    
    def _train_epoch(self, data_loader, epoch):
        """
        Train for one epoch
        
        Args:
            data_loader: Data loader for training
            epoch: Current epoch number
        
        Returns:
            dict: Training metrics
        """
        self.model.train()
        epoch_loss = 0
        fraud_losses = 0
        anomaly_losses = 0
        triplet_losses = 0
        
        for i, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            # Process batch based on data type
            if self.use_graph_data and hasattr(batch, 'x'):
                # Graph data
                batch = batch.to(self.device)
                pred, anomaly_pred, embeddings = self.model(
                    batch.x, batch.edge_index, batch.batch
                )
                y = batch.y
                
                # Calculate anomaly targets (using node degrees as a proxy)
                if hasattr(batch, 'edge_attr'):
                    anomaly_target = batch.edge_attr
                else:
                    # If no edge attributes, use node degrees normalized
                    anomaly_target = torch.ones_like(y, dtype=torch.float32)
            else:
                # Tabular data
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Create dummy edge_index for tabular data
                edge_index = self._create_dummy_edge_index(x.size(0)).to(self.device)
                
                # Forward pass
                pred, anomaly_pred, embeddings = self.model(x, edge_index)
                
                # Use original values as anomaly targets
                anomaly_target = x[:, 0].unsqueeze(1)  # Using first feature as proxy
            
            # Classification loss
            fraud_loss = self.classification_criterion(pred, y)
            
            # Anomaly detection loss
            anomaly_loss = self.regression_criterion(anomaly_pred, anomaly_target)
            
            # Triplet loss for enhanced class separation
            triplet_loss = self.model.triplet_loss(embeddings, y)
            
            # Combined loss with weighting
            loss = triplet_loss + self.lambda1 * fraud_loss + self.lambda2 * anomaly_loss
            
            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss components
            epoch_loss += loss.item()
            fraud_losses += fraud_loss.item()
            anomaly_losses += anomaly_loss.item()
            triplet_losses += triplet_loss.item()
            
            # Log progress
            if (i + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch+1} | Batch {i+1}/{len(data_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average losses
        avg_loss = epoch_loss / len(data_loader)
        avg_fraud_loss = fraud_losses / len(data_loader)
        avg_anomaly_loss = anomaly_losses / len(data_loader)
        avg_triplet_loss = triplet_losses / len(data_loader)
        
        return {
            'train_loss': avg_loss,
            'fraud_loss': avg_fraud_loss,
            'anomaly_loss': avg_anomaly_loss,
            'triplet_loss': avg_triplet_loss
        }
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset with comprehensive metrics
        
        Args:
            data_loader: Data loader for evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        all_anomaly_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Process batch based on data type
                if self.use_graph_data and hasattr(batch, 'x'):
                    # Graph data
                    batch = batch.to(self.device)
                    pred, anomaly_pred, _ = self.model(
                        batch.x, batch.edge_index, batch.batch
                    )
                    y = batch.y
                else:
                    # Tabular data
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Create dummy edge_index for tabular data
                    edge_index = self._create_dummy_edge_index(x.size(0)).to(self.device)
                    
                    # Forward pass
                    pred, anomaly_pred, _ = self.model(x, edge_index)
                
                # Extract predictions and probabilities
                probs = F.softmax(pred, dim=1)[:, 1].cpu().numpy()
                preds = pred.argmax(dim=1).cpu().numpy()
                labels = y.cpu().numpy()
                anomaly = anomaly_pred.cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_anomaly_preds.extend(anomaly)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute classification metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Compute AUC and AUPRC
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auprc': auprc
        }
    
    def _log_metrics(self, train_metrics, test_metrics, epoch):
        """
        Log metrics to MLflow and console
        
        Args:
            train_metrics (dict): Training metrics
            test_metrics (dict): Test metrics
            epoch (int): Current epoch
        """
        # Combine metrics for logging
        metrics = {}
        for k, v in train_metrics.items():
            metrics[k] = v
        
        for k, v in test_metrics.items():
            metrics[f'test_{k}'] = v
        
        # Log to MLflow
        mlflow.log_metrics(metrics, step=epoch)
        
        # Log to console
        self.logger.info(f"Epoch {epoch+1}/{self.epochs} | "
                         f"Loss: {train_metrics['train_loss']:.4f} | "
                         f"Test F1: {test_metrics['f1']:.4f} | "
                         f"Test AUC: {test_metrics['auc']:.4f}")
    
    def _create_and_log_plots(self, train_losses, val_metrics):
        """
        Create and log performance plots
        
        Args:
            train_losses (list): Training losses per epoch
            val_metrics (dict): Validation metrics per epoch
        """
        # Create directory for plots
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        
        # Metrics plot
        plt.figure(figsize=(12, 8))
        for metric, values in val_metrics.items():
            if values:  # Check if not empty
                plt.plot(values, label=f'Test {metric.upper()}')
        
        plt.title('Test Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        metrics_plot_path = os.path.join(plots_dir, 'metrics_plot.png')
        plt.savefig(metrics_plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(loss_plot_path)
        mlflow.log_artifact(metrics_plot_path)
    
    def _save_model(self, filename):
        """
        Save model checkpoint
        
        Args:
            filename (str): Filename for the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epochs
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        self.logger.info(f"Model saved to {os.path.join(self.save_dir, filename)}")
    
    def _prepare_tabular_data_loader(self, data):
        """
        Prepare a DataLoader from tabular dataset
        
        Args:
            data (dict): Dataset containing features and labels
        
        Returns:
            DataLoader: Data loader for training/evaluation
        """
        x = torch.tensor(data['X_train'].values, dtype=torch.float32)
        y = torch.tensor(data['y_train'].values, dtype=torch.long)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _create_dummy_edge_index(self, num_nodes):
        """
        Create a dummy edge index for tabular data (fully connected graph)
        
        Args:
            num_nodes (int): Number of nodes
        
        Returns:
            torch.Tensor: Edge index tensor
        """
        # Create edges for a simple chain
        source = torch.arange(0, num_nodes - 1)
        target = torch.arange(1, num_nodes)
        
        # Combine into edge_index
        edge_index = torch.stack([
            torch.cat([source, target]),
            torch.cat([target, source])
        ])
        
        return edge_index