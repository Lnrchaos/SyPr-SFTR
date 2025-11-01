import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional

from src.transformer import SymbolicTransformer
from src.federated import FederatedClient

def start_client(
    client_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int = 1000,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    use_dp: bool = False,
    use_compression: bool = True,
    personalization_layers: Optional[List[str]] = None,
    local_epochs: int = 1,
    lr: float = 1e-4,
    server_address: str = "[::]:8080"
):
    """Start a federated learning client."""
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
    
    # Create client
    client = FederatedClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_dp=use_dp,
        use_compression=use_compression,
        personalization_layers=personalization_layers or [],
        local_epochs=local_epochs,
        lr=lr
    )
    
    # Start client
    client.start(server_address=server_address, client_id=client_id)

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start a federated learning client.")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of training samples per client")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--server-address", type=str, default="[::]:8080", help="Server address")
    args = parser.parse_args()
    
    # Create synthetic dataset for this client
    input_ids = torch.randint(0, 1000, (args.num_samples, 32))  # 1000 vocab size, 32 sequence length
    labels = torch.randint(0, 1000, (args.num_samples, 32))
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Create dataset and data loaders
    dataset = TensorDataset(input_ids, labels, attention_mask)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Start client
    start_client(
        client_id=args.client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_dp=False,
        use_compression=True,
        personalization_layers=["lm_head"],
        local_epochs=1,
        lr=1e-4,
        server_address=args.server_address
    )

if __name__ == "__main__":
    main()
