
import torch
import numpy as np
from vit_classifier import PolypViT
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolypViT(num_classes=2, pretrained=False).to(device)

# Let's create a dummy input
dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
image_tensor = transform(image=dummy_image)["image"].unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    logits, attn_weights = model(image_tensor)

print(f"Logits shape: {logits.shape}")
print(f"Number of attention weight tensors: {len(attn_weights)}")
if len(attn_weights) > 0:
    print(f"First attention weight shape: {attn_weights[0].shape}")
else:
    print("No attention weights captured!")
