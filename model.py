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
    def __init__(self, p=1, c=1, r=1, train=True, train_fraction=0.3, seed=42):
        self.p = p

        np.random.seed(seed)
        R=r #range of dataset
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
            activation="relu",
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

# ============================================================================
# DATASET TOPOLOGY ANALYSIS
# ============================================================================

def compute_dataset_topology(dataset, p, c, max_samples=2000):
    """
    Compute the ground truth topology of the dataset.
    For modular arithmetic (a^c + b^c) mod p, we expect:
    - c=1: circular structure  
    - c=2: More complex structure
    """
    print(f"\n" + "="*60)
    print(f"ANALYZING DATASET TOPOLOGY: (a^{c} + b^{c}) mod {p}")
    print("="*60)
    
    # Sample from dataset
    n_samples = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    # Extract the computational space: [a, b, result]
    data_points = []
    for idx in indices:
        inputs, target = dataset[idx]
        a, b, exp, mod = inputs.numpy()
        result = target.item()
        # Embed in 3D space: (a, b, result)
        data_points.append([a, b, result])
    
    data_points = np.array(data_points)
    
    # Normalize to [0, 1] for better TDA
    data_normalized = data_points / np.array([p, p, p])
    
    print(f"Dataset shape: {data_normalized.shape}")
    print(f"Data range: [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")
    
    # Compute persistent homology
    print("\nComputing persistent homology of dataset...")
    result = ripser(data_normalized, maxdim=2)
    diagrams = result['dgms']
    
    # Analyze results
    print("\nDataset Topological Structure:")
    for dim in range(3):
        if dim < len(diagrams) and len(diagrams[dim]) > 0:
            dgm = diagrams[dim]
            lifetimes = dgm[:, 1] - dgm[:, 0]
            
            # Remove infinite persistence (connected components)
            finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
            
            print(f"\n  H{dim} (Dimension {dim}):")
            print(f"    Betti number: {len(lifetimes)}")
            if len(finite_lifetimes) > 0:
                print(f"    Max persistence: {np.max(finite_lifetimes):.4f}")
                print(f"    Mean persistence: {np.mean(finite_lifetimes):.4f}")
                print(f"    Long-lived features (>75th percentile): {np.sum(finite_lifetimes > np.percentile(finite_lifetimes, 75))}")
    
    # For c=0 or c=1, check if we see circular structure (Betti-1 > 0)
    if c <= 1:
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            prominent_loops = np.sum(diagrams[1][:, 1] - diagrams[1][:, 0] > 0.1)
            print(f"\n  ? Circular structure detected: {prominent_loops} prominent loops (expected for mod arithmetic)")
        else:
            print(f"\n  ? Warning: No loops detected (expected circular structure for mod arithmetic)")
    
    return {
        'diagrams': diagrams,
        'data_points': data_normalized,
        'stats': {
            'betti_0': len(diagrams[0]) if len(diagrams) > 0 else 0,
            'betti_1': len(diagrams[1]) if len(diagrams) > 1 else 0,
            'betti_2': len(diagrams[2]) if len(diagrams) > 2 else 0,
        }
    }


def compute_wasserstein_distance_to_ideal(model_diagrams, ideal_diagrams, maxdim=2):
    """
    Compute Wasserstein distance between model's learned topology 
    and the ideal dataset topology.
    
    Lower distance = model has learned structure closer to ground truth
    """
    distances = {}
    
    for dim in range(maxdim + 1):
        if (dim < len(model_diagrams) and dim < len(ideal_diagrams) and
            len(model_diagrams[dim]) > 0 and len(ideal_diagrams[dim]) > 0):
            
            d1 = remove_infinite(ideal_diagrams[dim])
            d2 = remove_infinite(model_diagrams[dim])
            
            if len(d1) > 0 and len(d2) > 0:
                dist = wasserstein(d1, d2)
                distances[f'wasserstein_to_ideal_{dim}'] = float(dist)
            else:
                distances[f'wasserstein_to_ideal_{dim}'] = 0.0
        else:
            distances[f'wasserstein_to_ideal_{dim}'] = 0.0
    
    return distances


def visualize_dataset_topology(dataset_topology, save_path='dataset_topology.png'):
    """
    Visualize the persistence diagrams of the dataset
    """
    from persim import plot_diagrams
    
    diagrams = dataset_topology['diagrams']
    
    plt.figure(figsize=(12, 4))
    
    # Plot persistence diagrams
    plt.subplot(1, 2, 1)
    plot_diagrams(diagrams, show=False)
    plt.title('Dataset Topology: Persistence Diagrams')
    
    # Plot data in 3D
    plt.subplot(1, 2, 2)
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.gcf().add_subplot(1, 2, 2, projection='3d')
    
    data = dataset_topology['data_points']
    # Sample for visualization
    sample_idx = np.random.choice(len(data), min(1000, len(data)), replace=False)
    sample_data = data[sample_idx]
    
    ax.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], 
               alpha=0.3, s=1, c=sample_data[:, 2], cmap='viridis')
    ax.set_xlabel('Input a (normalized)')
    ax.set_ylabel('Input b (normalized)')
    ax.set_zlabel('Output (normalized)')
    ax.set_title('Dataset Embedding in 3D')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nDataset topology visualization saved to '{save_path}'")
    plt.close()


# ============================================================================
# UPDATED COMPUTE_TOPOLOGY - Now with distance to ideal
# ============================================================================

def compute_topology(hidden_states, labels=None, prev_diagrams=None, 
                    ideal_diagrams=None, max_samples=800, maxdim=2):
    """
    Compute topology metrics including neural collapse if labels provided
    AND distance to ideal topology if ideal_diagrams provided
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
            print(f"\n  Analyzing H{dim} features...")
            print(f"\n    Total features: {len(diagrams[dim]) if dim < len(diagrams) else 0}")
            print(f"\n    Shape: {diagrams[dim].shape if dim < len(diagrams) else 'N/A'}")
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

            # Wasserstein shift from previous checkpoint
            shift = 0.0
            if prev_diagrams is not None and dim < len(prev_diagrams) and dim < len(diagrams):
                d1 = remove_infinite(prev_diagrams[dim])
                d2 = remove_infinite(diagrams[dim])
                if len(d1) > 0 and len(d2) > 0:
                    shift = float(wasserstein(d1, d2))
            topology_stats[f'wasserstein_shift_{dim}'] = shift

        # NEW: Wasserstein distance to ideal dataset topology
        if ideal_diagrams is not None:
            ideal_distances = compute_wasserstein_distance_to_ideal(
                diagrams, ideal_diagrams, maxdim=maxdim
            )
            topology_stats.update(ideal_distances)

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
            fallback[f'wasserstein_to_ideal_{d}'] = 0.0
        
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


# ============================================================================
# UPDATED ANALYZE_TOPOLOGY_ALL_LAYERS
# ============================================================================

def analyze_topology_all_layers(model,
                                loader,
                                device,
                                n_layers,
                                prev_topology=None,
                                ideal_topology=None,
                                maxdim=2):
    """
    Compute extended topology for all transformer layers.
    Now includes neural collapse metrics AND distance to ideal topology.
    """
    topology_per_layer = {}
    
    # Extract ideal diagrams if provided
    ideal_diagrams = ideal_topology['diagrams'] if ideal_topology is not None else None

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
            ideal_diagrams=ideal_diagrams,
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
def print_residuals(outputs, targets):
    # Print the residuals at each point
    targets_vector = np.zeros(len(outputs),len(outputs[0]), dtype=int)
    # assign values in target
    for target, i in targets.cpu().numpy():
        targets_vector[i][target]=target
    residuals = np.subtract(outputs,targets_vector)
    print(f"Residuals: {residuals}")


    
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # Print residuals for debugging
        print_residuals(outputs, targets)
        
        # Loss
        loss = nn.CrossEntropyLoss()(outputs, targets)
        #loss = torch.clamp(loss, min=.3)  
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
    C = 1    # Exponent (1 for pure addition - circular structure)
    R = 300  # Range of a and b (0 to R-1)
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 512
    TRAIN_FRACTION = 0.3
    BATCH_SIZE = 512
    LR = 1e-3
    WEIGHT_DECAY = 1.0
    NUM_EPOCHS = 10000
    LOG_INTERVAL = 1
    TDA_INTERVAL = 10
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ModularArithmeticDataset(p=P, c=C, r=R, train=True, train_fraction=TRAIN_FRACTION)
    val_dataset = ModularArithmeticDataset(p=P, c=C, train=False, train_fraction=TRAIN_FRACTION)
    
    # ========================================================================
    # NEW: ANALYZE DATASET TOPOLOGY FIRST
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: ANALYZING GROUND TRUTH DATASET TOPOLOGY")
    print("="*80)
    
    # Combine train and val for complete dataset topology
    full_dataset = ConcatDataset([train_dataset, val_dataset])
    ideal_topology = compute_dataset_topology(full_dataset, p=P, c=C, max_samples=2000)
    
    # Visualize it
    # visualize_dataset_topology(ideal_topology, save_path='dataset_topology.png')
    
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODEL AND TRACKING CONVERGENCE TO IDEAL")
    print("="*80)
    # ========================================================================
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nTrain size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = TinyTransformer(vocab_size=R+2, d_model=D_MODEL, n_heads=N_HEADS, 
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
        'topology': [],
        'ideal_topology': ideal_topology  # Store for later analysis
    }
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    stage = 0 
    val_acc = 0.0
    prev_topology = None

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        if epoch % LOG_INTERVAL == 0:
            # get L1 and L2 norms of weights for monitoring
            #l1_norm = sum(p.abs().sum() for p in model.parameters())
            #l2_norm = sum(p.pow(2).sum() for p in model.parameters()).sqrt()
            l1_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=1).item()
            l2_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2).item()


            # Rest of Code
            val_acc = evaluate(model, val_loader, device)
            
            print(f"\nEpoch {epoch:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | L1: {l1_norm:.4f} | L2: {l2_norm:.4f}")

            # STAGE SWITCHING LOGIC
            if val_acc > .95:
                if stage == 0:
                    print(">>> Switching to Stage 1: Quadratic Addition (c=2)")
                    stage = 1
                    new_train = ModularArithmeticDataset(p=P, c=2, r=R, train=True, train_fraction=TRAIN_FRACTION)
                    new_val = ModularArithmeticDataset(p=P, c=2, r=R, train=False, train_fraction=TRAIN_FRACTION)
                    train_dataset = ConcatDataset([train_dataset, new_train])
                    val_dataset = ConcatDataset([val_dataset, new_val])
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    
                    # Recompute ideal topology for new task
                    full_dataset = ConcatDataset([train_dataset, val_dataset])
                    ideal_topology = compute_dataset_topology(full_dataset, p=P, c=2, max_samples=2000)
                    val_acc = 0.0

                elif stage == 1:
                    print(">>> Switching to Stage 2: Cubic Modular (c=3)")
                    stage = 2
                    new_train_2 = ModularArithmeticDataset(p=P, c=3, r=R, train=True, train_fraction=TRAIN_FRACTION)
                    new_val_2 = ModularArithmeticDataset(p=P, c=3, r=R,train=False, train_fraction=TRAIN_FRACTION)
                    train_dataset = ConcatDataset([train_dataset, new_train_2])
                    val_dataset = ConcatDataset([val_dataset, new_val_2])
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
                    
                    # Recompute ideal topology for new task
                    full_dataset = ConcatDataset([train_dataset, val_dataset])
                    ideal_topology = compute_dataset_topology(full_dataset, p=P, c=3, max_samples=2000)
                    val_acc = 0.0
            
            # Record basic metrics
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)

            # Compute TDA
            if epoch % TDA_INTERVAL == 0:
                
                print(f"\n" + "="*40)
                print(f"TOPOLOGY REPORT: EPOCH {epoch}")
                print("="*40)
                
                # Pass ideal_topology to compute distances
                current_topology = analyze_topology_all_layers(
                    model, val_loader, device, N_LAYERS, 
                    prev_topology=prev_topology,
                    ideal_topology=ideal_topology,
                    maxdim=2
                )
                
                for layer_idx in range(N_LAYERS):
                    t = current_topology[layer_idx]
                    print(f"\n[LAYER {layer_idx}]")
                    
                    for d in range(3):
                        print(f"  H{d} | Betti: {t[f'betti_{d}']:<3} | "
                              f"Max_P: {t[f'max_persistence_{d}']:7.3f} | "
                              f"Shift: {t[f'wasserstein_shift_{d}']:7.3f} | "
                              f"?Ideal: {t.get(f'wasserstein_to_ideal_{d}', 0.0):7.3f}")
                    
                    print(f"  ID (Intrinsic Dim): {t['intrinsic_dim']:.4f}")
                    
                    # Print Neural Collapse metrics if available
                    if 'nc_within_class_var' in t:
                        print(f"  NC | Within-Class Var: {t['nc_within_class_var']:.6f} | "
                              f"Simplex Score: {t['nc_simplex_score']:.6f}")
                        print(f"     | Betti-2: {t['nc_betti_2']:<3} | "
                              f"Persistence-2: {t['nc_total_persistence_2']:7.3f} | "
                              f"Classes: {t['nc_num_classes']}")

                prev_topology = current_topology
                history['topology'].append(current_topology)
            else:
                if len(history['topology']) > 0:
                    history['topology'].append(history['topology'][-1])
                else:
                    history['topology'].append({i: {'betti_0': 0, 'betti_1': 0, 
                                                    'total_persistence_0': 0.0,
                                                    'total_persistence_1': 0.0,
                                                    'avg_persistence_1': 0.0} 
                                               for i in range(N_LAYERS)})
            
            if val_acc > 0.9 and epoch > 0:
                print("\n" + "=" * 80)
                print("GROKKING DETECTED! Validation accuracy > 90%")
                print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    final_val_acc = evaluate(model, val_loader, device)
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    
    import pickle
    with open('grokking_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("History saved to 'grokking_history.pkl'")
    
    return model, history
"""

**What this does:**

1. **Analyzes dataset topology FIRST** before training
2. For `c=0` (pure addition mod p), expects to find **circular structure** (Betti-1 > 0)
3. **Tracks Wasserstein distance to ideal** (`?Ideal` column) - shows how close the model's representations are to the ground truth topology
4. **Hypothesis**: Distance to ideal should **decrease** during grokking as the model learns the true structure
5. **Visualizes dataset** in 3D and shows persistence diagrams

The output will now show:
```
H1 | Betti: 45  | Max_P: 0.234 | Shift: 0.012 | ?Ideal: 0.456
"""

if __name__ == "__main__":
    main()