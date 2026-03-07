import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ModularArithmeticDataset(Dataset):
    """
    Dataset for a + b = c (mod p)
    """
    def __init__(self, p=113, train=True, train_fraction=0.3, seed=42):
        self.p = p
        np.random.seed(seed)
        
        # Generate ALL possible pairs
        all_pairs = []
        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                all_pairs.append((a, b, c))
        
        all_pairs = np.array(all_pairs)
        np.random.shuffle(all_pairs)
        
        # Split train/val
        split_idx = int(len(all_pairs) * train_fraction)
        if train:
            self.data = all_pairs[:split_idx]
        else:
            self.data = all_pairs[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, b, c = self.data[idx]
        # Return as tokens: [a, b] -> c
        return torch.tensor([a, b], dtype=torch.long), torch.tensor(c, dtype=torch.long)

# Create datasets
train_dataset = ModularArithmeticDataset(p=113, train=True, train_fraction=0.3)
val_dataset = ModularArithmeticDataset(p=113, train=False, train_fraction=0.3)

print(f"Train size: {len(train_dataset)}")  # ~3,800 examples
print(f"Val size: {len(val_dataset)}")      # ~8,900 examples

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)