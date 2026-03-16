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
import warnings

# ===========================================================================
# Warnings
# =========================================================================

warnings.filterwarnings("ignore", category=UserWarning, module="persim")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

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

from scipy.linalg import eigh
from persim import wasserstein
from sklearn.metrics.pairwise import cosine_similarity


def intrinsic_dimension(hidden_states):
    """
    Participation ratio intrinsic dimension estimate.
    """
    X = hidden_states - hidden_states.mean(0)
    cov = np.cov(X, rowvar=False)
    eigvals = np.maximum(eigh(cov, eigvals_only=True), 1e-12)
    return float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))

"""
ripser has a problem with "non finite death times" so we need to remove them before
computing wasserstein shift.
"""
# Remove infinite death points
def remove_infinite(dgm):
    return dgm[np.isfinite(dgm[:, 1])]


def compute_simplex_score(means_matrix):
    """
    Measure how close class means are to forming an equiangular simplex
    Perfect simplex = all pairwise cosine similarities equal to -1/(C-1)
    where C is number of classes
    """
    if len(means_matrix) < 2:
        return 0.0
    
    # Normalize to unit sphere
    means_normalized = means_matrix / (np.linalg.norm(means_matrix, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine similarities
    cos_sim = cosine_similarity(means_normalized)
    
    # For equiangular simplex, off-diagonal should be -1/(C-1)
    C = len(means_matrix)
    target = -1.0 / (C - 1)
    
    # Extract off-diagonal elements
    off_diag = cos_sim[~np.eye(C, dtype=bool)]
    
    # Measure deviation from target (negative so higher = better)
    simplex_score = -np.std(off_diag - target)
    
    return float(simplex_score)


def compute_neural_collapse_metrics(hidden_states, labels, max_samples=1000):
    """
    Compute Neural Collapse metrics:
    - NC1: Within-class variability collapse
    - NC2: Simplex ETF formation (class means equiangular)
    Also computes Betti-2 for 3D void detection
    """
    # Subsample if needed
    if len(hidden_states) > max_samples:
        idx = np.random.choice(len(hidden_states), max_samples, replace=False)
        hidden_states = hidden_states[idx]
        labels = labels[idx]
    
    try:
        # NC1: Within-class variance
        unique_classes = np.unique(labels)
        class_means = {}
        for c in unique_classes:
            class_mask = labels == c
            if np.sum(class_mask) > 0:
                class_means[c] = hidden_states[class_mask].mean(axis=0)
        
        within_class_var = 0.0
        total_samples = 0
        for c in unique_classes:
            class_mask = labels == c
            n_samples = np.sum(class_mask)
            if n_samples > 0:
                var = np.var(hidden_states[class_mask], axis=0).mean()
                within_class_var += var * n_samples
                total_samples += n_samples
        
        if total_samples > 0:
            within_class_var /= total_samples
        
        # NC2: Simplex ETF score
        if len(class_means) >= 2:
            means_matrix = np.array([class_means[c] for c in sorted(class_means.keys())])
            simplex_score = compute_simplex_score(means_matrix)
        else:
            simplex_score = 0.0
        
        # Compute Betti-2 (3D voids) on hidden states
        result = ripser(hidden_states, maxdim=2)
        diagrams = result['dgms']
        
        betti_2 = len(diagrams[2]) if len(diagrams) > 2 else 0
        total_persistence_2 = (np.sum(diagrams[2][:, 1] - diagrams[2][:, 0]) 
                              if len(diagrams) > 2 and len(diagrams[2]) > 0 else 0.0)
        
        return {
            'nc_within_class_var': float(within_class_var),
            'nc_simplex_score': simplex_score,
            'nc_betti_2': betti_2,
            'nc_total_persistence_2': float(total_persistence_2),
            'nc_num_classes': len(class_means)
        }
        
    except Exception as e:
        print(f"Neural collapse computation failed: {e}")
        return {
            'nc_within_class_var': 0.0,
            'nc_simplex_score': 0.0,
            'nc_betti_2': 0,
            'nc_total_persistence_2': 0.0,
            'nc_num_classes': 0
        }


def compute_topology(hidden_states, labels=None, prev_diagrams=None, max_samples=800, maxdim=2):
    """
    Compute topology metrics including neural collapse if labels provided
    """
    if len(hidden_states) > max_samples:
        idx = np.random.choice(len(hidden_states), max_samples, replace=False)
        hidden_states = hidden_states[idx]
        if labels is not None:
            labels = labels[idx]

    hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        result = ripser(hidden_states, maxdim=maxdim)
        diagrams = result['dgms']

        topology_stats = {}

        for dim in range(maxdim + 1):
            if dim < len(diagrams) and len(diagrams[dim]) > 0:
                dgm = diagrams[dim]
                lifetimes = dgm[:, 1] - dgm[:, 0]
                topology_stats[f'betti_{dim}']             = len(lifetimes)
                topology_stats[f'total_persistence_{dim}'] = float(np.sum(lifetimes))
                topology_stats[f'avg_persistence_{dim}']   = float(np.mean(lifetimes))
                topology_stats[f'max_persistence_{dim}']   = float(np.max(lifetimes))
                topology_stats[f'var_persistence_{dim}']   = float(np.var(lifetimes))
                topology_stats[f'long_lived_{dim}']        = int(np.sum(lifetimes > np.percentile(lifetimes, 75)))
            else:
                topology_stats[f'betti_{dim}']             = 0
                topology_stats[f'total_persistence_{dim}'] = 0.0
                topology_stats[f'avg_persistence_{dim}']   = 0.0
                topology_stats[f'max_persistence_{dim}']   = 0.0
                topology_stats[f'var_persistence_{dim}']   = 0.0
                topology_stats[f'long_lived_{dim}']        = 0

            shift = 0.0
            if prev_diagrams is not None and dim < len(prev_diagrams) and dim < len(diagrams):
                d1 = remove_infinite(prev_diagrams[dim])
                d2 = remove_infinite(diagrams[dim])
                if len(d1) > 0 and len(d2) > 0:
                    shift = float(wasserstein(d1, d2))
            topology_stats[f'wasserstein_shift_{dim}'] = shift

        topology_stats['intrinsic_dim'] = intrinsic_dimension(hidden_states)
        topology_stats['diagrams'] = diagrams
        
        # Add neural collapse metrics if labels provided
        if labels is not None:
            nc_metrics = compute_neural_collapse_metrics(hidden_states, labels)
            topology_stats.update(nc_metrics)
        
        return topology_stats

    except Exception as e:
        print(f"TDA computation failed: {e}")
        fallback = {'intrinsic_dim': 0.0, 'diagrams': []}
        for d in range(maxdim + 1):
            fallback[f'betti_{d}']             = 0
            fallback[f'total_persistence_{d}'] = 0.0
            fallback[f'avg_persistence_{d}']   = 0.0
            fallback[f'max_persistence_{d}']   = 0.0
            fallback[f'var_persistence_{d}']   = 0.0
            fallback[f'long_lived_{d}']        = 0
            fallback[f'wasserstein_shift_{d}'] = 0.0
        
        # Add empty NC metrics
        if labels is not None:
            fallback.update({
                'nc_within_class_var': 0.0,
                'nc_simplex_score': 0.0,
                'nc_betti_2': 0,
                'nc_total_persistence_2': 0.0,
                'nc_num_classes': 0
            })
        
        return fallback


def analyze_topology_all_layers(model,
                                loader,
                                device,
                                n_layers,
                                prev_topology=None,
                                maxdim=2):
    """
    Compute extended topology for all transformer layers.
    Now includes neural collapse metrics by extracting labels.
    """
    topology_per_layer = {}

    for layer_idx in range(n_layers):
        # Extract hidden states AND labels
        hidden_states, labels = extract_all_hidden_states_with_labels(
            model, loader, device, layer_idx
        )

        prev_diagrams = None
        if prev_topology is not None:
            prev_diagrams = prev_topology[layer_idx]['diagrams']

        topology = compute_topology(
            hidden_states,
            labels=labels,
            prev_diagrams=prev_diagrams,
            maxdim=maxdim
        )

        topology_per_layer[layer_idx] = topology

    return topology_per_layer

# ======================================================================
# HIDDEN STATE EXTRACTION
# ======================================================================

def extract_all_hidden_states(model, loader, device, layer_idx):
    """Original function - just hidden states"""
    model.eval()
    all_states = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            states = model.get_hidden_states(inputs, layer_idx)
            states_np = states.cpu().numpy()
            if np.any(np.isnan(states_np)) or np.any(np.isinf(states_np)):
                print(f"Warning: NaN/Inf in layer {layer_idx} hidden states, skipping batch")
                continue
            all_states.append(states_np)

    if len(all_states) == 0:
        print(f"Warning: all batches had NaN/Inf for layer {layer_idx}, returning zeros")
        return np.zeros((10, model.d_model))

    return np.nan_to_num(np.vstack(all_states), nan=0.0, posinf=0.0, neginf=0.0)


def extract_all_hidden_states_with_labels(model, loader, device, layer_idx):
    """New function - returns both hidden states and labels for NC metrics"""
    model.eval()
    all_states = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            states = model.get_hidden_states(inputs, layer_idx)
            states_np = states.cpu().numpy()
            labels_np = targets.cpu().numpy()
            
            if np.any(np.isnan(states_np)) or np.any(np.isinf(states_np)):
                print(f"Warning: NaN/Inf in layer {layer_idx} hidden states, skipping batch")
                continue
                
            all_states.append(states_np)
            all_labels.append(labels_np)

    if len(all_states) == 0:
        print(f"Warning: all batches had NaN/Inf for layer {layer_idx}, returning zeros")
        return np.zeros((10, model.d_model)), np.zeros(10, dtype=int)

    hidden_states = np.nan_to_num(np.vstack(all_states), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.concatenate(all_labels)
    
    return hidden_states, labels


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
    prev_topology = None

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
            
            # Record basic metrics
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)

            # Compute TDA (expensive, so do less frequently)
            if epoch % TDA_INTERVAL == 0:
                
                print(f"\n" + "="*40)
                print(f"TOPOLOGY REPORT: EPOCH {epoch}")
                print("="*40)
                
                # We pass prev_topology so Wasserstein Shift can be calculated
                current_topology = analyze_topology_all_layers(
                    model, val_loader, device, N_LAYERS, 
                    prev_topology=prev_topology, maxdim=2
                )
                
                for layer_idx in range(N_LAYERS):
                    t = current_topology[layer_idx]
                    print(f"\n[LAYER {layer_idx}]")
                    
                    for d in range(3): # Dimensions 0, 1, 2
                        print(f"  H{d} | Betti: {t[f'betti_{d}']:<3} | "
                              f"Max_P: {t[f'max_persistence_{d}']:7.3f} | "
                              f"Shift: {t[f'wasserstein_shift_{d}']:7.3f}")
                    
                    print(f"  ID (Intrinsic Dim): {t['intrinsic_dim']:.4f}")
                    
                    # Print Neural Collapse metrics if available
                    if 'nc_within_class_var' in t:
                        print(f"  NC | Within-Class Var: {t['nc_within_class_var']:.6f} | "
                              f"Simplex Score: {t['nc_simplex_score']:.6f}")
                        print(f"     | Betti-2: {t['nc_betti_2']:<3} | "
                              f"Persistence-2: {t['nc_total_persistence_2']:7.3f} | "
                              f"Classes: {t['nc_num_classes']}")

                # Update memory for the next shift calculation
                prev_topology = current_topology
                history['topology'].append(current_topology)
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
    #print("\nGenerating plots...")
    #plot_results(history)
    
    # Save history
    import pickle
    with open('grokking_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("History saved to 'grokking_history.pkl'")
    
    return model, history


if __name__ == "__main__":
    main()