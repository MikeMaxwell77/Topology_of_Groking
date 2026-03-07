"""
Grokking + TDA Analysis
Train a tiny transformer on modular arithmetic and track topological changes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from ripser import ripser
import os

# ============================================================================
# DATA
# ============================================================================

class ModularArithmeticDataset(Dataset):
    """Dataset for (a ^ c + b ^ c) mod p =d """
    def __init__(self, p=1, c=1, train=True, train_fraction=0.3, seed=42):
        self.p = p
        np.random.seed(seed)
        R=113 #range of dataset
        # Generate ALL possible pairs
        all_pairs = []
        for a in range(R):
            for b in range(R):
                d = (a ** c + b ** c) % p
                all_pairs.append((a, b, c, p, d))

        
        all_pairs = np.array(all_pairs)
        np.random.shuffle(all_pairs)
        
        # Split train/val
        split_idx = int(len(all_pairs) * train_fraction)
        if train:
            self.data = all_pairs[:split_idx]
        else:
            self.data = all_pairs[split_idx:]
    
    def add_dataset(self, dataset):
        self.datasets.append(dataset)
        self.concat = ConcatDataset(self.datasets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, b, c, p, d = self.data[idx]
        return torch.tensor([a, b, c, p], dtype=torch.long), torch.tensor(d, dtype=torch.long)

# Seperate out the datset into smaller sections
       

#
# ============================================================================
# MODEL
# ============================================================================

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=114, d_model=128, n_heads=4, n_layers=2, d_ff=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        self.seq_len = 4
        self.pos_embed = nn.Parameter(torch.randn(self.seq_len, d_model) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embed(x) + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last position
        logits = self.output(x)
        return logits
    
    def get_hidden_states(self, x, layer_idx):
        """Extract hidden states from specific layer"""
        x = self.embed(x) + self.pos_embed.unsqueeze(0)
        
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if i == layer_idx:
                return x[:, -1, :].detach()
        
        return x[:, -1, :].detach()


# ============================================================================
# TDA FUNCTIONS
# ============================================================================

def compute_topology(hidden_states, max_samples=1000, maxdim=1):
    """
    Compute persistent homology on hidden states
    
    Args:
        hidden_states: numpy array [n_samples, hidden_dim]
        max_samples: subsample for computational efficiency
        maxdim: maximum homology dimension to compute
    
    Returns:
        dict with topological features
    """
    # Subsample if needed
    if len(hidden_states) > max_samples:
        idx = np.random.choice(len(hidden_states), max_samples, replace=False)
        hidden_states = hidden_states[idx]
    
    # Compute persistent homology
    try:
        result = ripser(hidden_states, maxdim=maxdim)
        diagrams = result['dgms']
        
        # Extract Betti numbers (counting topological features)
        betti_0 = len(diagrams[0]) - 1 if len(diagrams[0]) > 0 else 0  # Connected components
        betti_1 = len(diagrams[1]) if len(diagrams) > 1 else 0  # Loops/holes
        
        # Total persistence (sum of feature lifetimes)
        total_persistence_0 = np.sum(diagrams[0][:, 1] - diagrams[0][:, 0]) if len(diagrams[0]) > 0 else 0
        total_persistence_1 = np.sum(diagrams[1][:, 1] - diagrams[1][:, 0]) if len(diagrams) > 1 and len(diagrams[1]) > 0 else 0
        
        # Average persistence (measure of topological complexity)
        avg_persistence_1 = total_persistence_1 / max(betti_1, 1)
        
        return {
            'betti_0': betti_0,
            'betti_1': betti_1,
            'total_persistence_0': float(total_persistence_0),
            'total_persistence_1': float(total_persistence_1),
            'avg_persistence_1': float(avg_persistence_1),
            'diagrams': diagrams
        }
    except Exception as e:
        print(f"TDA computation failed: {e}")
        return {
            'betti_0': 0,
            'betti_1': 0,
            'total_persistence_0': 0.0,
            'total_persistence_1': 0.0,
            'avg_persistence_1': 0.0,
            'diagrams': None
        }


def extract_all_hidden_states(model, loader, device, layer_idx):
    """Extract hidden states for entire dataset at specific layer"""
    model.eval()
    all_states = []
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            states = model.get_hidden_states(inputs, layer_idx)
            all_states.append(states.cpu().numpy())
    
    return np.vstack(all_states)


def analyze_topology_all_layers(model, loader, device, n_layers):
    """Compute topology for all layers"""
    topology_per_layer = {}
    
    for layer_idx in range(n_layers):
        print(f"  Analyzing layer {layer_idx}...")
        hidden_states = extract_all_hidden_states(model, loader, device, layer_idx)
        topology = compute_topology(hidden_states)
        topology_per_layer[layer_idx] = topology
    
    return topology_per_layer


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    return correct / total


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(history):
    """Plot training curves and topology evolution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy curves
    ax = axes[0, 0]
    epochs = history['epoch']
    ax.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Grokking: Train vs Val Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Betti-1 (holes) over time for each layer
    ax = axes[0, 1]
    for layer_idx in range(len(history['topology'][0])):
        betti_1_vals = [history['topology'][i][layer_idx]['betti_1'] for i in range(len(epochs))]
        ax.plot(epochs, betti_1_vals, label=f'Layer {layer_idx}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti-1 (Number of Holes)')
    ax.set_title('Topological Complexity: Betti-1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total persistence over time
    ax = axes[1, 0]
    for layer_idx in range(len(history['topology'][0])):
        persistence_vals = [history['topology'][i][layer_idx]['total_persistence_1'] 
                          for i in range(len(epochs))]
        ax.plot(epochs, persistence_vals, label=f'Layer {layer_idx}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Persistence (Betti-1)')
    ax.set_title('Topological Complexity: Total Persistence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Val accuracy vs Betti-1 (phase diagram)
    ax = axes[1, 1]
    for layer_idx in range(len(history['topology'][0])):
        betti_1_vals = [history['topology'][i][layer_idx]['betti_1'] for i in range(len(epochs))]
        val_acc = history['val_acc']
        scatter = ax.scatter(betti_1_vals, val_acc, c=epochs, cmap='viridis', 
                           alpha=0.6, label=f'Layer {layer_idx}')
    ax.set_xlabel('Betti-1 (Topological Complexity)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Phase Diagram: Topology vs Performance')
    plt.colorbar(scatter, ax=ax, label='Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grokking_topology_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'grokking_topology_analysis.png'")
    plt.show()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    # Hyperparameters
    P = 113  # Modulus (prime number)
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 512
    TRAIN_FRACTION = 0.3
    BATCH_SIZE = 512
    LR = 1e-3
    WEIGHT_DECAY = 1.0  # CRUCIAL for grokking
    NUM_EPOCHS = 10000
    LOG_INTERVAL = 100  # Log and compute TDA every N epochs
    TDA_INTERVAL = 500  # Compute expensive TDA every N epochs (within LOG_INTERVAL)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ModularArithmeticDataset(p=P, train=True, train_fraction=TRAIN_FRACTION)
    val_dataset = ModularArithmeticDataset(p=P, train=False, train_fraction=TRAIN_FRACTION)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = TinyTransformer(vocab_size=P+2, d_model=D_MODEL, n_heads=N_HEADS, 
                           n_layers=N_LAYERS, d_ff=D_FF).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # History tracking
    history = {
        'epoch': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'topology': []  # Will store list of dicts (one per layer)
    }
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    stage = 0 
    val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Check for logging AND stage switching
        if epoch % LOG_INTERVAL == 0:
            val_acc = evaluate(model, val_loader, device)
            
            print(f"\nEpoch {epoch:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # STAGE SWITCHING LOGIC (Now inside the log interval)
            if val_acc > .95:
                if stage == 0:
                    print(">>> Switching to Stage 1: Modular Addition (c=1)")
                    stage = 1
                    new_train = ModularArithmeticDataset(p=P, c=1, train=True, train_fraction=TRAIN_FRACTION)
                    new_val = ModularArithmeticDataset(p=P, c=1, train=False, train_fraction=TRAIN_FRACTION)
                    train_dataset = ConcatDataset([train_dataset, new_train])
                    val_dataset = ConcatDataset([val_dataset, new_val])
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    val_acc = 0.0 # Reset so we don't trigger Stage 2 immediately

                elif stage == 1:
                    print(">>> Switching to Stage 2: Quadratic Modular (c=2)")
                    stage = 2
                    new_train_2 = ModularArithmeticDataset(p=P, c=2, train=True, train_fraction=TRAIN_FRACTION)
                    new_val_2 = ModularArithmeticDataset(p=P, c=2, train=False, train_fraction=TRAIN_FRACTION)
                    train_dataset = ConcatDataset([train_dataset, new_train_2])
                    val_dataset = ConcatDataset([val_dataset, new_val_2])
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    val_acc = 0.0

            # Record basic metrics... (rest of your history code)
            
            # Record basic metrics
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            
            # Compute TDA (expensive, so do less frequently)
            if epoch % TDA_INTERVAL == 0:
                print("  Computing topology...")
                topology = analyze_topology_all_layers(model, val_loader, device, N_LAYERS)
                history['topology'].append(topology)
                
                # Print topology summary
                for layer_idx in range(N_LAYERS):
                    topo = topology[layer_idx]
                    print(f"    Layer {layer_idx}: Betti-0={topo['betti_0']}, "
                          f"Betti-1={topo['betti_1']}, "
                          f"Persistence={topo['total_persistence_1']:.3f}")
            else:
                # Placeholder for non-TDA epochs (for consistent indexing)
                if len(history['topology']) > 0:
                    history['topology'].append(history['topology'][-1])
                else:
                    history['topology'].append({i: {'betti_0': 0, 'betti_1': 0, 
                                                    'total_persistence_0': 0.0,
                                                    'total_persistence_1': 0.0,
                                                    'avg_persistence_1': 0.0} 
                                               for i in range(N_LAYERS)})
            
            # Check if grokking happened
            if val_acc > 0.9 and epoch > 0:
                print("\n" + "=" * 80)
                print("GROKKING DETECTED! Validation accuracy > 90%")
                print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    # Final evaluation
    final_val_acc = evaluate(model, val_loader, device)
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    
    # Plot results
    #wprint("\nGenerating plots...")
    #plot_results(history)
    
    # Save history
    import pickle
    with open('grokking_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("History saved to 'grokking_history.pkl'")
    
    return model, history


if __name__ == "__main__":
    main()