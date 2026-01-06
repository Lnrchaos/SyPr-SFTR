# SyPr-SFTR: Symbolic-Probabilistic-SelfSupervised-Federated Learning Transformer

A cutting-edge framework that combines symbolic reasoning, probabilistic modeling, and self-supervised learning in a federated learning setup.

## 🌟 Features

- **Symbolic Reasoning**: Integrates neural and symbolic AI for better interpretability
- **Probabilistic Modeling**: Quantifies prediction uncertainty
- **Self-Supervised Learning**: Leverages unlabeled data with contrastive and predictive tasks
- **Federated Learning**: Privacy-preserving distributed training
- **Advanced Optimization**: Supports multiple aggregation strategies (FedAvg, FedProx, FedAdam)
- **Privacy Protection**: Differential privacy and secure aggregation

## 🚀 Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Lnrchaos/SyPr-SFTR.git
   cd SyPr-SFTR
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```python
from src.transformer import SymbolicTransformer
import torch

# Initialize model
model = SymbolicTransformer(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=6,
    num_heads=12,
    use_symbolic=True,
    use_probabilistic=True,
    use_self_supervised=True
)

# Example input
input_ids = torch.randint(0, 30000, (1, 32))  # [batch_size, seq_len]

# Forward pass
outputs = model(input_ids)

# Generate text
generated = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
```

## 🧪 Testing

Run the test suite to verify the architecture:

```bash
python -m pytest tests/
```

## 🏗️ Project Structure

```
SyPr-SFTR/
├── src/
│   ├── __init__.py
│   ├── transformer.py      # Main transformer architecture
│   ├── symbolic.py         # Symbolic reasoning layer
│   ├── probabilistic.py    # Probabilistic modeling
│   ├── self_supervised.py  # Self-supervised learning
│   └── federated.py        # Federated learning components
├── tests/
│   └── test_architecture.py
├── requirements.txt
└── README.md
```

## 🤖 Federated Learning Example

### Server

```python
from src.federated import FederatedServer
import torch.nn as nn

# Initialize model
model = nn.Sequential(
    nn.Embedding(1000, 64),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=64, nhead=8),
        num_layers=2
    ),
    nn.Linear(64, 10)
)

# Create and start server
server = FederatedServer(
    model=model,
    num_clients=10,
    fraction_fit=0.5,
    use_secure_aggregation=True,
    aggregation_method="fedadam"
)
server.start(num_rounds=100)
```

### Client

```python
from src.federated import FederatedClient
from torch.utils.data import DataLoader, TensorDataset

# Create dummy data
train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 10, (100,))
train_loader = DataLoader(
    TensorDataset(train_data, train_labels),
    batch_size=32,
    shuffle=True
)

# Create client
client = FederatedClient(
    model=model,
    train_loader=train_loader,
    val_loader=train_loader,
    use_dp=True,
    use_compression=True,
    personalization_layers=["2"]  # Don't aggregate the final layer
)

# Start client (in practice, this would run on a different machine)
import flwr as fl
fl.client.start_numpy_client(server_address="[::]:8080", client=client)
```

## 📊 Features in Detail

### Symbolic Reasoning
- Neural-symbolic integration for interpretable AI
- Extracts human-readable symbolic expressions
- Can be used for rule extraction and verification

### Probabilistic Modeling
- Quantifies prediction uncertainty
- Supports Bayesian neural networks
- Useful for active learning and risk assessment

### Self-Supervised Learning
- Contrastive learning for better representations
- Masked prediction tasks
- Reduces need for labeled data

### Federated Learning
- Privacy-preserving distributed training
- Multiple aggregation strategies
- Secure aggregation with differential privacy

## 📚 Documentation

For detailed documentation, please see the [docs](docs/) directory.

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).
