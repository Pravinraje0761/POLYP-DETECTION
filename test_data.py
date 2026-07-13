
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample from archive
data_root = "archive (1)/PNG"
original_dir = os.path.join(data_root, "Original")
gt_dir = os.path.join(data_root, "Ground Truth")
sample_name = os.listdir(original_dir)[0]
print("Sample name:", sample_name)

img = np.array(Image.open(os.path.join(original_dir, sample_name)).convert("RGB"))
mask = np.array(Image.open(os.path.join(gt_dir, sample_name)).convert("L"))

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Mask min/max:", mask.min(), mask.max())

# Plot them
fig, axes = plt.subplots(1, 2, figsize=(12,6))
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Ground Truth Mask")
plt.savefig("test_data.png")
print("Saved test image to test_data.png")
plt.close()
