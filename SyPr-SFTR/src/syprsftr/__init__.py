"""
SyPr-SFTR: Symbolic, Probabilistic, and Self-Supervised Transformer with Federated Learning

A PyTorch-based library for building interpretable and privacy-preserving AI models.
"""

__version__ = "0.1.0"

# Core imports
from .models.transformer import SymbolicTransformer
from .federated.client import FederatedClient
from .federated.server import FederatedServer

__all__ = [
    'SymbolicTransformer',
    'FederatedClient',
    'FederatedServer',
]
