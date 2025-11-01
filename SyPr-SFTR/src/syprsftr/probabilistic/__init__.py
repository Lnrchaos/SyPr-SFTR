import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

class ProbabilisticLayer(nn.Module):
    """
    Probabilistic layer that models uncertainty in predictions.
    Outputs both mean and variance for each feature.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Heads for mean and variance
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
        # Initialize small variance
        nn.init.constant_(self.logvar_head.weight, 0.1)
        nn.init.constant_(self.logvar_head.bias, -1.0)
        
    def forward(self, x, return_distribution=False):
        """
        Forward pass through the probabilistic layer.
        
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            return_distribution: If True, returns the distribution object
            
        Returns:
            If return_distribution is False, returns a tuple of (mean, var)
            If return_distribution is True, returns a Normal distribution object
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        
        if return_distribution:
            return dist.Normal(mean, std)
        return mean, std**2  # Return mean and variance
    
    def kl_divergence(self, q_dist, p_dist=None):
        """
        Compute KL divergence between q_dist and p_dist (or standard normal if None)
        """
        if p_dist is None:
            # Standard normal prior
            p_dist = dist.Normal(torch.zeros_like(q_dist.mean), 
                               torch.ones_like(q_dist.stddev))
        return dist.kl_divergence(q_dist, p_dist).sum(-1)
    
    def sample(self, x, num_samples=1):
        """Sample from the learned distribution"""
        dist = self.forward(x, return_distribution=True)
        return dist.rsample((num_samples,))  # [num_samples, batch_size, ..., output_dim]
