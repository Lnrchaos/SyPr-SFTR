import torch
from src import SymbolicTransformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
config = {
    'vocab_size': 10000,  # Size of vocabulary
    'max_seq_length': 128,  # Maximum sequence length
    'hidden_dim': 256,  # Hidden dimension of the model
    'num_layers': 6,  # Number of transformer layers
    'num_heads': 8,  # Number of attention heads
    'batch_size': 32,  # Batch size
    'num_epochs': 5,  # Number of training epochs
}

def create_sample_data(vocab_size, num_samples=1000, seq_length=128):
    """Create sample data for demonstration"""
    # Generate random token IDs (0 is usually reserved for padding)
    input_ids = np.random.randint(1, vocab_size, size=(num_samples, seq_length))
    
    # Create labels (shifted input for language modeling)
    labels = np.roll(input_ids, -1, axis=1)
    labels[:, -1] = 0  # Set last token to padding
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = np.ones_like(input_ids)
    
    return torch.tensor(input_ids, dtype=torch.long), \
           torch.tensor(labels, dtype=torch.long), \
           torch.tensor(attention_mask, dtype=torch.long)

def train_model():
    # Create sample data
    input_ids, labels, attention_mask = create_sample_data(
        config['vocab_size'], 
        num_samples=1000,
        seq_length=config['max_seq_length']
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, labels, attention_mask)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    model = SymbolicTransformer(
        vocab_size=config['vocab_size'],
        max_seq_length=config['max_seq_length'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_symbolic=True,
        use_probabilistic=True,
        use_self_supervised=True
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        total_loss = 0
        
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids_batch, labels_batch, attention_mask_batch = batch
            
            # Forward pass
            outputs = model(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                labels=labels_batch,
                return_self_supervised_loss=True
            )
            
            if isinstance(outputs, tuple):
                logits, contrastive_loss, predictive_loss = outputs
                lm_loss = criterion(
                    logits.view(-1, config['vocab_size']), 
                    labels_batch.view(-1)
                )
                # Combine losses
                loss = lm_loss + 0.1 * contrastive_loss + 0.1 * predictive_loss
            else:
                logits = outputs
                loss = criterion(
                    logits.view(-1, config['vocab_size']), 
                    labels_batch.view(-1)
                )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'symbolic_transformer_model.pt')
    print("Model training complete!")
    
    # Demonstrate model capabilities
    demo_model_capabilities(model, device, config['vocab_size'])

def demo_model_capabilities(model, device, vocab_size):
    """Demonstrate various capabilities of the trained model"""
    print("\n--- Model Capabilities Demo ---")
    
    # Generate sample input
    input_ids = torch.randint(1, vocab_size, (1, 10)).to(device)  # 10 tokens
    
    # 1. Basic forward pass
    with torch.no_grad():
        logits = model(input_ids)
    print(f"\n1. Model output shape: {logits.shape} (batch_size, seq_len, vocab_size)")
    
    # 2. Get symbolic expressions
    if model.use_symbolic:
        expressions = model.get_symbolic_expressions(input_ids[0:1, :5])  # First 5 tokens
        print("\n2. Symbolic expressions for first 5 tokens:")
        for idx, expr in expressions.items():
            print(f"  Token {idx}: {expr}")
    
    # 3. Get uncertainty estimates
    if model.use_probabilistic:
        uncertainty = model.get_uncertainty_estimates(input_ids[0:1, :3])  # First 3 tokens
        print("\n3. Uncertainty estimates (mean, variance, uncertainty):")
        for i in range(3):
            print(f"  Token {i}:")
            print(f"    Mean: {uncertainty['mean'][0, i, :5].round(4)}...")
            print(f"    Variance: {uncertainty['variance'][0, i, :5].round(6)}...")
            print(f"    Uncertainty: {uncertainty['uncertainty'][0, i, :5].round(6)}...")
    
    # 4. Text generation
    if hasattr(model, 'generate'):
        print("\n4. Text generation (sampling from the model):")
        generated = model.generate(
            input_ids[:, :5],  # Start with first 5 tokens
            max_length=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_return_sequences=2
        )
        print(f"  Generated sequences (shape: {generated.shape}):")
        for i, seq in enumerate(generated):
            print(f"  Sample {i+1}: {seq.tolist()}")

if __name__ == "__main__":
    train_model()
