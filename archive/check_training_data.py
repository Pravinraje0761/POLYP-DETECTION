
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    data_root = "archive (1)/PNG"
    original_dir = os.path.join(data_root, "Original")
    gt_dir = os.path.join(data_root, "Ground Truth")
    
    image_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".png")])[:10]
    
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(original_dir, img_file)
        mask_path = os.path.join(gt_dir, img_file)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(img_file)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(img)
        axes[2, i].imshow(mask, cmap='jet', alpha=0.3)
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("GT Mask")
    axes[2, 0].set_ylabel("Overlay")
    
    plt.tight_layout()
    plt.savefig("training_data_check.png", dpi=200, bbox_inches="tight")
    print("Saved training_data_check.png!")

if __name__ == "__main__":
    main()
