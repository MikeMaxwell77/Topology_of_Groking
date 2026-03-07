#quick look

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

#file_path = 'C:/Users/mikey/OneDrive/Documents/USCB/Research/grokking_history.pkl'
file_path = 'grokking_history.pkl'

with open(file_path, 'rb') as f:
    first_byte = f.read(1)
    if first_byte != b'\x80':
        # If the first byte isn't the pickle start, keep reading until we find it
        # Or just skip the first byte if you're sure it's just one newline
        print(f"Skipping unexpected leading byte: {first_byte}")
    else:
        f.seek(0) # Go back to start if it was fine
        
    history = pickle.load(f)#, weights_only=False)
# ==============================================================
# EXTENDED TOPOLOGY ANALYZER
# ==============================================================

def plot_interactive(history):
    """
    Full topology + training analysis dashboard
    """

    epochs = history['epoch']
    n_checkpoints = len(epochs)

    if not history['topology']:
        print("No topology data found.")
        return

    n_layers = len(history['topology'][0])

    # Auto-detect available Betti dimensions
    sample_layer = history['topology'][0][0]
    betti_dims = sorted([
        int(k.split('_')[1])
        for k in sample_layer.keys()
        if k.startswith("betti_")
    ])

    print(f"Detected Betti dimensions: {betti_dims}")

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))

    # ==========================================================
    # ROW 1 Training Dynamics
    # ==========================================================
    
    axes[0,0].plot(epochs, history['train_acc'], label='Train', linewidth=2)
    axes[0,0].plot(epochs, history['val_acc'], label='Val', linewidth=2)
    axes[0,0].set_title("Accuracy")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(epochs, history['train_loss'], linewidth=2)
    axes[0,1].set_title("Training Loss")
    axes[0,1].grid(True, alpha=0.3)

    # Intrinsic dimension
    for layer_idx in range(n_layers):
        intrinsic = [
            history['topology'][i][layer_idx]['intrinsic_dim']
            for i in range(n_checkpoints)
        ]
        axes[0,2].plot(epochs, intrinsic, label=f'Layer {layer_idx}', linewidth=2)

    axes[0,2].set_title("Intrinsic Dimension")
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)

    # ==========================================================
    # ROW 2 Betti Numbers
    # ==========================================================

    for dim_idx, dim in enumerate(betti_dims):
        ax = axes[1, dim_idx]
        for layer_idx in range(n_layers):
            vals = [
                history['topology'][i][layer_idx][f'betti_{dim}']
                for i in range(n_checkpoints)
            ]
            ax.plot(epochs, vals, label=f'L{layer_idx}', linewidth=2)
        ax.set_title(f"Betti-{dim}")
        ax.grid(True, alpha=0.3)
        if dim_idx == 0:
            ax.legend()

    # ==========================================================
    # ROW 3 Total Persistence
    # ==========================================================

    for dim_idx, dim in enumerate(betti_dims):
        ax = axes[2, dim_idx]
        for layer_idx in range(n_layers):
            vals = [
                history['topology'][i][layer_idx][f'total_persistence_{dim}']
                for i in range(n_checkpoints)
            ]
            ax.plot(epochs, vals, label=f'L{layer_idx}', linewidth=2)
        ax.set_title(f"Total Persistence H{dim}")
        ax.grid(True, alpha=0.3)

    # ==========================================================
    # ROW 4 Wasserstein Shifts (Phase Transitions)
    # ==========================================================

    for dim_idx, dim in enumerate(betti_dims):
        ax = axes[3, dim_idx]
        for layer_idx in range(n_layers):
            vals = [
                history['topology'][i][layer_idx][f'wasserstein_shift_{dim}']
                for i in range(n_checkpoints)
            ]
            ax.plot(epochs, vals, label=f'L{layer_idx}', linewidth=2)
        ax.set_title(f"Wasserstein Shift H{dim}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ExtendedTopologyAnalysis.png", dpi=300)

# Print what's in there
print("Keys in history:", history.keys())
print("\nNumber of checkpoints:", len(history['epoch']))
print("Epochs recorded:", history['epoch'])

# Print some basic stats
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")

# Check if grokking happened
val_accs = np.array(history['val_acc'])
if np.max(val_accs) > 0.9:
    grok_idx = np.where(val_accs > 0.9)[0][0]
    grok_epoch = history['epoch'][grok_idx]
    print(f"\n Grokking detected at epoch {grok_epoch}")
else:
    print("\n  No grokking detected yet (val acc < 90%)")

# Print topology info
print("\n" + "="*60)
print("TOPOLOGY SUMMARY")
print("="*60)

if len(history['topology']) > 0:
    # Get number of layers
    n_layers = len(history['topology'][0])
    print(f"Number of layers tracked: {n_layers}")
    
    # Show topology at first and last checkpoint
    print("\nFirst checkpoint:")
    for layer_idx in range(n_layers):
        topo = history['topology'][0][layer_idx]
        print(f"  Layer {layer_idx}: Betti-0={topo['betti_0']}, "
              f"Betti-1={topo['betti_1']}, "
              f"Persistence={topo['total_persistence_1']:.3f}")
    
    print("\nLast checkpoint:")
    for layer_idx in range(n_layers):
        topo = history['topology'][-1][layer_idx]
        print(f"  Layer {layer_idx}: Betti-0={topo['betti_0']}, "
              f"Betti-1={topo['betti_1']}, "
              f"Persistence={topo['total_persistence_1']:.3f}")
# topology summary
print("\n" + "="*60)
print("TOPOLOGY SUMMARY")
print("="*60)

if history['topology']:
    n_layers = len(history['topology'][0])

    final_topo = history['topology'][-1]

    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}")
        layer_data = final_topo[layer_idx]

        for key in sorted(layer_data.keys()):
            if key != "diagrams":
                print(f"  {key}: {layer_data[key]}")
