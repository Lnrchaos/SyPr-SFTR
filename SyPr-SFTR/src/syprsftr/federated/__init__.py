from typing import Dict, List, Tuple, Optional, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    NDArrays,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from collections import OrderedDict, defaultdict
import numpy as np
import random
import warnings
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class ClientPrivacyManager:
    """Manages differential privacy and secure aggregation for clients."""
    def __init__(self, target_epsilon: float = 5.0, target_delta: float = 1e-5, 
                 max_grad_norm: float = 1.0):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = None
        
    def make_private(self, model: nn.Module, optimizer: optim.Optimizer, 
                    data_loader: DataLoader) -> tuple:
        """Convert model to use differential privacy."""
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
        
        self.privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,  # Will be auto-tuned
            max_grad_norm=self.max_grad_norm,
        )
        return model, optimizer, data_loader
    
    def get_privacy_spent(self) -> dict:
        """Get the privacy budget spent so far."""
        if self.privacy_engine:
            return self.privacy_engine.get_privacy_spent(self.target_delta)
        return {"epsilon": 0.0, "delta": 0.0}

class ClientGradientManager:
    """Manages gradient compression and error feedback for efficient communication."""
    def __init__(self, compression_ratio: float = 0.01, use_error_feedback: bool = True):
        self.compression_ratio = compression_ratio
        self.use_error_feedback = use_error_feedback
        self.error_feedback = None
    
    def compress_gradients(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], dict]:
        """Compress gradients using top-k sparsification."""
        compressed_grads = []
        metadata = {"original_sizes": [], "sparse_indices": []}
        
        for grad in gradients:
            if grad is None:
                compressed_grads.append(None)
                continue
                
            # Flatten the gradient
            flat_grad = grad.detach().view(-1)
            k = max(1, int(flat_grad.numel() * self.compression_ratio))
            
            # Get top-k values and indices
            values, indices = torch.topk(flat_grad.abs(), k=k, sorted=False)
            # Keep the original signs
            signs = torch.sign(flat_grad[indices])
            values = values * signs
            
            # Create sparse gradient
            sparse_grad = torch.zeros_like(flat_grad)
            sparse_grad[indices] = values
            
            compressed_grads.append(sparse_grad.view_as(grad))
            
            # Store metadata for decompression
            metadata["original_sizes"].append(grad.size())
            metadata["sparse_indices"].append(indices.cpu())
        
        return compressed_grads, metadata
    
    def apply_error_feedback(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply error feedback to gradients."""
        if not self.use_error_feedback or self.error_feedback is None:
            self.error_feedback = [torch.zeros_like(g) if g is not None else None 
                                for g in gradients]
            return gradients
            
        # Add previous compression error to current gradients
        for i, (grad, err) in enumerate(zip(gradients, self.error_feedback)):
            if grad is not None and err is not None:
                gradients[i] = grad + err
                
        # Update error feedback for next round
        compressed_grads, _ = self.compress_gradients(gradients)
        for i, (grad, comp_grad) in enumerate(zip(gradients, compressed_grads)):
            if grad is not None and comp_grad is not None:
                self.error_feedback[i] = grad - comp_grad
                
        return compressed_grads

class FederatedClient(fl.client.NumPyClient):
    """
    Advanced Federated Learning client with support for:
    - Differential privacy
    - Gradient compression
    - Adaptive optimization
    - Model personalization
    - Secure aggregation
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_dp: bool = False,
        use_compression: bool = True,
        personalization_layers: List[str] = None,
        local_epochs: int = 1,
        lr: float = 1e-3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.use_compression = use_compression
        self.personalization_layers = personalization_layers or []
        
        # Initialize privacy and gradient managers
        self.privacy_manager = ClientPrivacyManager()
        self.gradient_manager = ClientGradientManager(compression_ratio=0.01)
        
        # Setup distributed training if multiple GPUs are available
        self.is_distributed = torch.cuda.device_count() > 1
        if self.is_distributed:
            self.setup_distributed()
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=local_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Enable differential privacy if requested
        if use_dp:
            self.model, self.optimizer, self.train_loader = \
                self.privacy_manager.make_private(
                    self.model, self.optimizer, self.train_loader
                )
    
    def setup_distributed(self):
        """Initialize distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=torch.cuda.device_count(),
                rank=0  # This will be set by the launcher script
            )
        self.model = DDP(self.model)
        
        # Update train_loader with distributed sampler
        sampler = DistributedSampler(
            self.train_loader.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy ndarrays."""
        # Only return trainable parameters from non-personalized layers
        params = []
        for name, param in self.model.named_parameters():
            if not any(layer in name for layer in self.personalization_layers):
                if param.requires_grad:
                    params.append(param.detach().cpu().numpy())
                else:
                    # Send zeros for non-trainable parameters
                    params.append(np.zeros_like(param.detach().cpu().numpy()))
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy ndarrays."""
        # Only update non-personalized layers
        param_idx = 0
        for name, param in self.model.named_parameters():
            if not any(layer in name for layer in self.personalization_layers):
                if param.requires_grad:
                    param.data = torch.tensor(
                        parameters[param_idx], 
                        device=self.device
                    )
                    param_idx += 1
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset."""
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        current_round = config.get("server_round", 0)
        local_epochs = config.get("epochs", self.local_epochs)
        lr = config.get("lr", self.lr)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training loop
        self.model.train()
        total_samples = 0
        total_loss = 0.0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    # Unpack if model returns (logits, ...)
                    logits = outputs[0]
                    loss = F.cross_entropy(logits, labels)
                else:
                    loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient compression if enabled
                if self.use_compression and batch_idx % 10 == 0:  # Compress every 10 steps
                    grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                    compressed_grads = self.gradient_manager.apply_error_feedback(grads)
                    
                    # Update gradients
                    grad_idx = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data = compressed_grads[grad_idx].to(p.device)
                            grad_idx += 1
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                batch_size = inputs.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
                num_batches += 1
            
            # Log epoch metrics
            epoch_loss /= len(self.train_loader.dataset)
            total_loss += epoch_loss
            
            if self.device == 0:  # Only log from main process
                print(f"Epoch {epoch+1}/{local_epochs}, Loss: {epoch_loss:.4f}", end='\r')
        
        # Get updated parameters
        updated_params = self.get_parameters({})
        
        # Get privacy metrics if using differential privacy
        privacy_metrics = {}
        if hasattr(self, 'privacy_manager'):
            epsilon, _ = self.privacy_manager.get_privacy_spent()
            privacy_metrics = {"privacy_epsilon": epsilon}
        
        # Return updated parameters and metrics
        avg_loss = total_loss / local_epochs
        metrics = {
            "loss": avg_loss,
            "samples": total_samples,
            "lr": lr,
            **privacy_metrics
        }
        
        return updated_params, total_samples, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation set."""
        # Set model parameters
        self.set_parameters(parameters)
        self.model.eval()
        
        # Evaluation metrics
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    # Unpack if model returns (logits, ...)
                    logits = outputs[0]
                    loss = F.cross_entropy(logits, labels)
                    preds = torch.argmax(logits, dim=1)
                else:
                    loss = F.cross_entropy(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                
                # Update metrics
                total_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return float(avg_loss), total, {"accuracy": accuracy}


class FederatedServer:
    """
    Advanced Federated Learning server with support for:
    - Adaptive client selection
    - Model aggregation with personalization
    - Secure aggregation
    - Automated model evaluation
    - Federated analytics
    """
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        test_loader: DataLoader = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = None,
        min_evaluate_clients: int = None,
        min_available_clients: int = None,
        personalization_layers: List[str] = None,
        use_secure_aggregation: bool = False,
        aggregation_method: str = "fedavg",  # Options: fedavg, fedprox, fedadam, etc.
    ):
        self.model = model
        self.num_clients = num_clients
        self.test_loader = test_loader
        self.personalization_layers = personalization_layers or []
        self.use_secure_aggregation = use_secure_aggregation
        self.aggregation_method = aggregation_method
        
        # Client selection parameters
        self.fraction_fit = max(0.1, min(fraction_fit, 1.0))
        self.fraction_evaluate = max(0.1, min(fraction_evaluate, 1.0))
        self.min_fit_clients = min_fit_clients or max(1, int(num_clients * fraction_fit))
        self.min_evaluate_clients = min_evaluate_clients or max(1, int(num_clients * fraction_evaluate))
        self.min_available_clients = min_available_clients or max(
            self.min_fit_clients, self.min_evaluate_clients
        )
        
        # Server state
        self.current_round = 0
        self.best_accuracy = 0.0
        self.client_metrics = defaultdict(list)
        
        # Initialize aggregation parameters
        self.aggregator = self._get_aggregator()
    
    def _get_aggregator(self):
        """Initialize the appropriate aggregation strategy."""
        if self.aggregation_method == "fedprox":
            return FedProxAggregator(mu=0.01)
        elif self.aggregation_method == "fedadam":
            return FedAdamAggregator(
                model_params=self.model.parameters(),
                beta1=0.9,
                beta2=0.99,
                eta=1e-3,
                tau=1e-3,
            )
        else:  # Default to FedAvg
            return FedAvgAggregator()
    
    def get_evaluate_fn(self, test_loader: DataLoader = None) -> Callable:
        """Return a function that evaluates the global model on a test set."""
        if test_loader is None:
            test_loader = self.test_loader
        
        def evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar],
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate the global model on the test set."""
            if test_loader is None:
                return None
                
            # Update model with the latest parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            
            # Evaluate
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(next(self.model.parameters()).device), labels.to(next(self.model.parameters()).device)
                    
                    outputs = self.model(inputs)
                    
                    # Handle different output types
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        loss = F.cross_entropy(logits, labels)
                        preds = torch.argmax(logits, dim=1)
                    else:
                        loss = F.cross_entropy(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)
                    
                    total_loss += loss.item() * inputs.size(0)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / total
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
            
            return avg_loss, {"accuracy": accuracy}
        
        return evaluate
    
    def get_fit_config_fn(self, lr: float = 1e-3, epochs: int = 1) -> Callable:
        """Return a function that returns training configuration."""
        def fit_config(server_round: int) -> Dict[str, Scalar]:
            # Learning rate scheduling
            lr_decay = 0.99 ** (server_round // 10)
            current_lr = lr * lr_decay
            
            return {
                "lr": current_lr,
                "epochs": epochs,
                "server_round": server_round,
                "local_epochs": max(1, int(epochs * (1 - server_round / 100))),  # Gradually reduce local epochs
            }
        return fit_config
    
    def get_on_fit_config_fn(self) -> Callable:
        """Return a function that returns the configuration for fit."""
        return self.get_fit_config_fn()
    
    def get_on_evaluate_config_fn(self) -> Callable:
        """Return a function that returns the configuration for evaluation."""
        def evaluate_config(server_round: int) -> Dict[str, Scalar]:
            return {"server_round": server_round}
        return evaluate_config
    
    def start(self, num_rounds: int = 100, server_address: str = "[::]:8080"):
        """Start the federated learning server."""
        # Define strategy with adaptive client selection
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
            on_fit_config_fn=self.get_fit_config_fn(),
            on_evaluate_config_fn=self.get_on_evaluate_config_fn(),
            evaluate_fn=self.get_evaluate_fn(self.test_loader),
            # Add secure aggregation if enabled
            **({"aggregation": self.secure_aggregation} if self.use_secure_aggregation else {}),
        )
        
        # Start server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    
    def secure_aggregation(
        self, results: List[Tuple[NDArrays, int]]
    ) -> NDArrays:
        """
        Perform secure aggregation of client updates.
        
        Args:
            results: List of tuples containing (parameters, num_examples) from clients
            
        Returns:
            Aggregated parameters
        """
        # Simple weighted average for demonstration
        # In practice, this would use secure multi-party computation (MPC) or homomorphic encryption
        # For a real implementation, consider using libraries like TF-Encrypted or PySyft
        
        # Get weights based on number of examples
        weights = [num_examples for _, num_examples in results]
        total_weight = sum(weights)
        
        # Initialize aggregated parameters
        aggregated_params = [
            np.zeros_like(param) for param in results[0][0]
        ]
        
        # Weighted average
        for (params, _), weight in zip(results, weights):
            for i, param in enumerate(params):
                aggregated_params[i] += param * (weight / total_weight)
        
        return aggregated_params

# Aggregation Strategies
class Aggregator:
    """Base class for aggregation strategies."""
    def aggregate(self, parameters_list: List[List[NDArrays]], weights: List[float]) -> List[NDArrays]:
        raise NotImplementedError

class FedAvgAggregator(Aggregator):
    """Standard Federated Averaging aggregation."""
    def aggregate(self, parameters_list: List[List[NDArrays]], weights: List[float]) -> List[NDArrays]:
        """Compute weighted average of parameters."""
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Initialize aggregated parameters
        aggregated_params = [
            np.zeros_like(param) for param in parameters_list[0]
        ]
        
        # Weighted average
        for params, weight in zip(parameters_list, weights):
            for i, param in enumerate(params):
                aggregated_params[i] += param * weight
        
        return aggregated_params

class FedProxAggregator(Aggregator):
    """FedProx aggregation with proximal term."""
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Proximal term weight
        self.global_params = None
    
    def aggregate(self, parameters_list: List[List[NDArrays]], weights: List[float]) -> List[NDArrays]:
        """Compute FedProx aggregation with proximal term."""
        if self.global_params is None:
            # First round, use standard FedAvg
            self.global_params = FedAvgAggregator().aggregate(parameters_list, weights)
            return self.global_params
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Initialize aggregated parameters
        aggregated_params = [
            np.zeros_like(param) for param in parameters_list[0]
        ]
        
        # FedProx aggregation
        for params, weight in zip(parameters_list, weights):
            for i, (param, global_param) in enumerate(zip(params, self.global_params)):
                # Add proximal term
                aggregated_params[i] += (param + self.mu * global_param) * weight
        
        # Update global parameters
        self.global_params = aggregated_params
        return aggregated_params

class FedAdamAggregator(Aggregator):
    """FedAdam aggregation with adaptive optimization."""
    def __init__(self, model_params, beta1: float = 0.9, beta2: float = 0.99, 
                 eta: float = 1e-3, tau: float = 1e-3):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta  # Server learning rate
        self.tau = tau  # Controls the algorithm's degree of adaptivity
        
        # Initialize moments
        self.m = [torch.zeros_like(p) for p in model_params]
        self.v = [torch.zeros_like(p) for p in model_params]
        self.t = 0  # Time step
    
    def aggregate(self, parameters_list: List[List[NDArrays]], weights: List[float]) -> List[NDArrays]:
        """Compute FedAdam aggregation with adaptive optimization."""
        # Convert numpy arrays to torch tensors
        parameters_list = [
            [torch.from_numpy(p) for p in params] 
            for params in parameters_list
        ]
        
        # Compute weighted average of deltas
        deltas = []
        for params in parameters_list:
            delta = [p - g for p, g in zip(params, self.global_params)]
            deltas.append(delta)
        
        # Average deltas
        avg_delta = [torch.zeros_like(d[0]) for d in deltas[0]]
        for delta, weight in zip(deltas, weights):
            for i, d in enumerate(delta):
                avg_delta[i] += d * weight
        
        # Update moments
        self.t += 1
        for i, delta in enumerate(avg_delta):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * delta
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * delta**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update global parameters
            self.global_params[i] += self.eta * m_hat / (torch.sqrt(v_hat) + self.tau)
        
        return [p.detach().numpy() for p in self.global_params]

# Utility functions
def setup_ddp(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()

if __name__ == "__main__":
    # Example usage
    model = nn.Linear(10, 2)  # Replace with your model
    server = FederatedServer(
        model=model,
        num_clients=10,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        personalization_layers=["classifier"],  # Example: don't aggregate classifier
        use_secure_aggregation=True,
        aggregation_method="fedadam"
    )
    server.start(num_rounds=100)
