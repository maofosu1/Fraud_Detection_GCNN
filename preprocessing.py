# preprocessing.py
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict

class TransactionNetworkPreprocessor:
    def __init__(self, dataset_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Advanced preprocessor for transaction network data
        
        Args:
            dataset_path (str): Path to the CSV dataset
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.dataset = pd.read_csv(dataset_path)
        self.test_size = test_size
        
    def clean_data(self):
        """
        Clean and prepare raw transaction data
        
        Returns:
            Cleaned DataFrame
        """
        # Drop duplicate and null values
        df = self.dataset.dropna().drop_duplicates()
        
        # Convert timestamp and extract temporal features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['sender_id', 'recipient_id', 'sender_income_level']  # Include 'sender_income_level'
        
        for col in categorical_cols:
            if col in df.columns:  # Check if the column exists in the dataset
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        return df, label_encoders
    
    def feature_engineering(self, df: pd.DataFrame):
        """
        Create advanced features for fraud detection
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        # Ensure the required columns are present
        required_columns = ['sender_id_encoded', 'amount', 'is_fraud']  # Removed 'sender_age'
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing in the dataset.")
        
        # Aggregate account-level features
        account_features = df.groupby('sender_id_encoded').agg({
            'amount': ['mean', 'std', 'max', 'min'],
            'is_fraud': 'sum'
        }).reset_index()
        
        # Flatten the multi-level column names
        account_features.columns = [
            'sender_id_encoded', 
            'avg_transaction_amount', 
            'std_transaction_amount', 
            'max_transaction_amount', 
            'min_transaction_amount',
            'fraud_count'
        ]
        
        # Merge account features
        df = df.merge(account_features, on='sender_id_encoded', how='left')
        
        # Compute account's transaction count as a proxy for transaction history length
        df['transaction_history_length'] = df.groupby('sender_id_encoded')['sender_id_encoded'].transform('count')
        
        # Derived features
        df['amount_to_avg_ratio'] = df['amount'] / df['avg_transaction_amount']
        df['fraud_likelihood_score'] = (df['fraud_count'] / df['transaction_history_length']).fillna(0)
        
        return df
    
    def scale_features(self, df: pd.DataFrame):
        """
        Apply advanced scaling techniques
        
        Args:
            df (pd.DataFrame): DataFrame with engineered features
        
        Returns:
            Scaled DataFrame
        """
        # Select numeric features for scaling
        numeric_features = [
            'amount', 'transaction_history_length', 
            'avg_transaction_amount', 'std_transaction_amount', 
            'max_transaction_amount', 'min_transaction_amount',
            'amount_to_avg_ratio', 'fraud_likelihood_score',
            'hour_of_day', 'day_of_week', 'month',
            'sender_id_encoded', 'recipient_id_encoded'
        ]
        
        # Ensure all numeric features are converted to float
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Scale numeric features
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        # One-hot encode categorical features
        categorical_features = ['sender_income_level']
        for col in categorical_features:
            if col in df.columns:  # Check if the column exists
                df = pd.get_dummies(df, columns=[col])
        
        return df
    
    def create_graph_data(self, df: pd.DataFrame):
        """
        Create PyTorch Geometric Data object for graph neural network
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame
        
        Returns:
            PyTorch Geometric Data object
        """
        # Get unique nodes and their features
        unique_nodes = pd.concat([
            df[['sender_id_encoded', 'transaction_history_length', 'is_fraud']],
            df[['recipient_id_encoded', 'transaction_history_length', 'is_fraud']]
        ]).drop_duplicates(subset='sender_id_encoded')
        
        # Create node features
        node_features = unique_nodes[['transaction_history_length']].values
        node_labels = unique_nodes['is_fraud'].values
        
        # Create edge index
        edges = df[['sender_id_encoded', 'recipient_id_encoded']].values
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)
        
        # Create PyG Data object
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        
        return graph_data
    
    def prepare_data(self):
        """
        Complete preprocessing pipeline
        
        Returns:
            Dictionary with processed data components
        """
        # Clean and encode data
        cleaned_df, label_encoders = self.clean_data()
        
        # Feature engineering
        engineered_df = self.feature_engineering(cleaned_df)
        
        # Scale features
        scaled_df = self.scale_features(engineered_df)
        
        # Create graph data
        graph_data = self.create_graph_data(scaled_df)
        
        # Prepare features and target
        features_columns = [col for col in scaled_df.columns if col not in ['is_fraud', 'timestamp']]
        X = scaled_df[features_columns]
        y = scaled_df['is_fraud']
        
        # Verify all columns are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'graph_data': graph_data,
            'full_preprocessed_data': scaled_df,
            'label_encoders': label_encoders
        }

def main():
    preprocessor = TransactionNetworkPreprocessor('data/synthetic_fraud_dataset.csv')
    processed_data = preprocessor.prepare_data()
    
    print("Preprocessing Complete")
    print(f"Training Samples: {len(processed_data['X_train'])}")
    print(f"Testing Samples: {len(processed_data['X_test'])}")
    print(f"Fraud Ratio in Training: {processed_data['y_train'].mean():.2%}")
    print(f"Fraud Ratio in Testing: {processed_data['y_test'].mean():.2%}")

if __name__ == "__main__":
    main()