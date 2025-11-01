import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SelfSupervisedHead(nn.Module):
    """
    Self-supervised learning head with contrastive and predictive tasks.
    Can be used for pre-training or as a regularizer during supervised training.
    """
    def __init__(self, input_dim: int, proj_dim: int = 256, 
                 num_contrastive_views: int = 2, temperature: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_views = num_contrastive_views
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 2, proj_dim)
        )
        
        # Predictive task head (predicts masked tokens or next token)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised learning.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask_ratio: Ratio of tokens to mask for masked prediction task
            
        Returns:
            contrastive_loss: Contrastive learning loss
            predictive_loss: Masked prediction loss
        """
        # Contrastive learning
        contrastive_loss = self.contrastive_learning(x)
        
        # Masked prediction task
        predictive_loss = self.masked_prediction(x, mask_ratio)
        
        return contrastive_loss, predictive_loss
    
    def contrastive_learning(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using multiple views of the input"""
        batch_size = x.size(0)
        
        # Create multiple views (in practice, apply different augmentations)
        views = []
        for _ in range(self.num_views):
            # Apply different augmentations here (e.g., random masking, noise, etc.)
            augmented_view = self.augment(x)
            projected = self.projection(augmented_view)
            views.append(F.normalize(projected, dim=-1))
        
        # Compute similarity matrix
        sim_matrix = torch.mm(views[0].view(-1, self.proj_dim), 
                             views[1].view(-1, self.proj_dim).t()) / self.temperature
        
        # Contrastive loss (NT-Xent)
        labels = torch.arange(batch_size, device=x.device)
        loss = (F.cross_entropy(sim_matrix, labels) + 
                F.cross_entropy(sim_matrix.t(), labels)) / 2
        
        return loss
    
    def masked_prediction(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Predict masked tokens in the input sequence"""
        batch_size, seq_len, _ = x.size()
        
        # Create random mask
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(batch_size, seq_len, device=x.device)
        for i in range(batch_size):
            mask[i, torch.randperm(seq_len)[:num_masked]] = 1
        mask = mask.bool()
        
        # Mask the input
        masked_x = x.clone()
        masked_x[mask.unsqueeze(-1).expand_as(x)] = 0
        
        # Predict the original values
        predictions = self.predictor(masked_x)
        
        # Compute MSE loss only for masked positions
        loss = F.mse_loss(
            predictions[mask.unsqueeze(-1).expand_as(x)],
            x[mask.unsqueeze(-1).expand_as(x)],
            reduction='mean'
        )
        
        return loss
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to create different views"""
        # Add Gaussian noise
        noise = torch.randn_like(x) * 0.1
        return x + noise
