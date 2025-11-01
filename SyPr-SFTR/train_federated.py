import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.transformer import SymbolicTransformer
from src.federated import FederatedClient, FederatedServer

def create_synthetic_dataset(num_samples: int = 1000, seq_length: int = 32, vocab_size: int = 1000):
    """Create a synthetic dataset for testing federated learning."""
    # Create random input and label tensors
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Create dataset
    dataset = TensorDataset(input_ids, labels, attention_mask)
    return dataset

def create_client_loaders(dataset, num_clients: int = 5, batch_size: int = 8):
    """Split dataset into client datasets and create data loaders."""
    # Split dataset into non-overlapping datasets for each client
    client_sizes = [len(dataset) // num_clients] * num_clients
    client_sizes[-1] += len(dataset) % num_clients  # Add remainder to last client
    
    client_datasets = random_split(dataset, client_sizes)
    
    # Create data loaders for each client
    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_datasets
    ]
    
    return client_loaders

def setup_federated_learning(
    num_clients: int = 5,
    vocab_size: int = 1000,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    use_dp: bool = False,
    use_compression: bool = True,
    personalization_layers: Optional[List[str]] = None,
    batch_size: int = 8,
    num_rounds: int = 10,
    local_epochs: int = 1,
    lr: float = 1e-4,
):
    """Set up and run federated learning."""
    # Create model
    model = SymbolicTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_symbolic=True,
        use_probabilistic=True,
        use_self_supervised=True
    )
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(
        num_samples=1000,
        seq_length=32,
        vocab_size=vocab_size
    )
    
    # Split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create client data loaders
    train_loaders = create_client_loaders(
        train_dataset,
        num_clients=num_clients,
        batch_size=batch_size
    )
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize server
    server = FederatedServer(
        model=model,
        num_clients=num_clients,
        test_loader=test_loader,
        fraction_fit=1.0,  # Use all clients for training
        fraction_evaluate=1.0,  # Use all clients for evaluation
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        personalization_layers=personalization_layers or [],
        use_secure_aggregation=False,
        aggregation_method="fedavg"
    )
    
    # Start the server
    server.start(num_rounds=num_rounds, server_address="[::]:8080")

def main():
    # Configuration
    config = {
        "num_clients": 5,
        "vocab_size": 1000,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "use_dp": False,
        "use_compression": True,
        "personalization_layers": ["lm_head"],  # Personalize the language modeling head
        "batch_size": 8,
        "num_rounds": 10,
        "local_epochs": 1,
        "lr": 1e-4,
    }
    
    # Start federated learning
    setup_federated_learning(**config)

if __name__ == "__main__":
    main()
