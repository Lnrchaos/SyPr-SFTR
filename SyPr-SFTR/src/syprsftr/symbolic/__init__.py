import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import symbols, lambdify
import numpy as np

class SymbolicLayer(nn.Module):
    """
    Symbolic reasoning layer that combines neural and symbolic operations.
    Can learn to apply symbolic operations to the input features.
    """
    def __init__(self, input_dim, output_dim, num_operations=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_operations = num_operations
        
        # Learnable parameters for symbolic operations
        self.op_weights = nn.Parameter(torch.randn(num_operations))
        self.op_selector = nn.Linear(input_dim, num_operations)
        
        # Symbolic operations (can be expanded)
        self.operations = [
            lambda x: x,  # Identity
            lambda x: x ** 2,
            lambda x: torch.sin(x),
            lambda x: torch.exp(-x),
            lambda x: 1 / (1 + torch.exp(-x))  # Sigmoid
        ]
        
        # Output projection
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Get operation weights
        op_scores = F.softmax(self.op_selector(x), dim=-1)  # [batch_size, num_ops]
        
        # Apply each operation and combine
        results = []
        for op in self.operations:
            results.append(op(x).unsqueeze(-1))  # [batch_size, seq_len, hidden_dim, 1]
        
        # Stack and weight the results
        results = torch.cat(results, dim=-1)  # [batch_size, seq_len, hidden_dim, num_ops]
        weighted_results = results * op_scores.unsqueeze(-2)  # Weight each operation
        
        # Combine and project
        combined = weighted_results.sum(dim=-1)  # [batch_size, seq_len, hidden_dim]
        return self.proj(combined)
    
    def get_symbolic_expression(self, input_symbols):
        """Convert the learned operations to symbolic expressions"""
        x = symbols('x')
        op_expressions = [
            x,                    # Identity
            x**2,                 # Square
            symbols('sin(x)'),    # Sine
            symbols('exp(-x)'),   # Exponential decay
            1/(1 + symbols('exp(-x)'))  # Sigmoid
        ]
        
        # Get the dominant operation for each position
        op_probs = F.softmax(self.op_selector.weight.detach(), dim=0)
        dominant_ops = op_probs.argmax(dim=1)
        
        # Convert to symbolic expression
        symbolic_expr = 0
        for i, op_idx in enumerate(dominant_ops):
            symbolic_expr += op_expressions[op_idx] * self.proj.weight[0, i].item()
            
        return symbolic_expr.simplify()
