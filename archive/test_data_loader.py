
import torch
from polypgate.data.dataset import get_dataloaders

# Test the data loader
data_root = "hyper-kvasir-labeled-images (1)/labeled-images"
train_loader, val_loader, test_loader = get_dataloaders(data_root, batch_size=4)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Check a batch
batch = next(iter(train_loader))
print(f"Batch keys: {batch.keys()}")
print(f"Image shape: {batch['image'].shape}")
print(f"Labels: {batch['label']}")
