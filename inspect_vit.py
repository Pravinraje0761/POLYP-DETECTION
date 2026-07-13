
import torch
from vit_classifier import PolypViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolypViT(num_classes=2, pretrained=False).to(device)

print("ViT Model Structure:")
print(model)

print("\nEncoder Layers:")
for i, layer in enumerate(model.vit.encoder.layers):
    print(f"\nLayer {i}:")
    print(layer)
