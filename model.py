# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

class AdvancedFraudDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lstm_layers=2, embedding_dim=32, dropout=0.3, margin=0.5):
        """
        Hybrid Graph-based Deep Learning Framework for Fraud Detection
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Dimension of hidden layers
            lstm_layers (int): Number of LSTM layers
            embedding_dim (int): Dimension of the final embedding
            dropout (float): Dropout rate
            margin (float): Margin for Triplet Loss
        """
        super(AdvancedFraudDetectionModel, self).__init__()
        
        self.margin = margin
        
        # Graph Convolutional Layers
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Graph Attention Layer for suspicious transactions
        self.gat = GATConv(hidden_dim, hidden_dim, heads=2, dropout=dropout)
        
        # LSTM for temporal dependency modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Embedding projection
        self.embedding = nn.Linear(hidden_dim * 2, embedding_dim)
        
        # Classification header
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification: fraud or legitimate
        )
        
        # Anomaly detection header (multi-task learning)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Regression task: predict transaction value anomaly
        )
    
    def forward(self, x, edge_index, batch=None, edge_attr=None, seq_len=10):
        """
        Forward pass with graph convolution, attention, LSTM, and multi-task learning
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Graph edge indices
            batch (torch.Tensor): Batch indices for multiple graphs
            edge_attr (torch.Tensor): Edge attributes/features
            seq_len (int): Length of sequences for LSTM
        
        Returns:
            tuple: 
                - Classification logits
                - Anomaly scores
                - Node embeddings (for triplet loss)
        """
        # Graph Convolution
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        
        # Graph Attention
        x = self.gat(x, edge_index)
        
        # If batch is provided, use it for global pooling
        if batch is not None:
            # Reshape for LSTM (batch_size, seq_len, features)
            batch_size = batch.max().item() + 1
            x_seq = x.view(batch_size, seq_len, -1)
            
            # LSTM for temporal dependencies
            lstm_out, _ = self.lstm(x_seq)
            
            # Temporal attention
            attn_weights = F.softmax(self.temporal_attention(lstm_out), dim=1)
            x = torch.sum(lstm_out * attn_weights, dim=1)
        else:
            # If no batch, treat the entire graph as one sequence
            x_seq = x.unsqueeze(0)  # Add batch dimension
            lstm_out, _ = self.lstm(x_seq)
            x = lstm_out.mean(dim=1).squeeze(0)  # Average over sequence
        
        # Embedding space (for triplet loss)
        embedding = self.embedding(x)
        
        # Classification and anomaly detection (multi-task)
        logits = self.classifier(embedding)
        anomaly_score = self.anomaly_detector(embedding)
        
        return logits, anomaly_score, embedding
    
    def triplet_loss(self, embeddings, labels):
        """
        Compute triplet loss to enhance class separation
        
        Args:
            embeddings (torch.Tensor): Embeddings from the model
            labels (torch.Tensor): Class labels
        
        Returns:
            torch.Tensor: Triplet loss value
        """
        # Get mask for positive and negative pairs
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings)
        
        # Find hardest positive and negative samples
        pos_dist = torch.max(dist_mat * pos_mask, dim=1)[0]
        neg_dist = torch.min(dist_mat * neg_mask + 1e6 * (1 - neg_mask), dim=1)[0]
        
        # Compute triplet loss with margin
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return loss.mean()