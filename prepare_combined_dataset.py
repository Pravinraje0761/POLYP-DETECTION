
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def create_output_dirs(output_root):
    images_dir = os.path.join(output_root, "images")
    masks_dir = os.path.join(output_root, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    return images_dir, masks_dir


def process_archive_dataset(input_dir, images_out, masks_out, prefix="archive_"):
    original_dir = os.path.join(input_dir, "Original")
    gt_dir = os.path.join(input_dir, "Ground Truth")
    
    files = [f for f in sorted(os.listdir(original_dir)) if f.endswith(".png")]
    
    for idx, filename in enumerate(tqdm(files, desc="Processing Archive Dataset")):
        img_path = os.path.join(original_dir, filename)
        mask_path = os.path.join(gt_dir, filename)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask_np)
        
        out_filename = f"{prefix}{idx:04d}.png"
        img.save(os.path.join(images_out, out_filename))
        mask.save(os.path.join(masks_out, out_filename))


def process_kvasir_seg(input_dir, images_out, masks_out, prefix="kvasir_"):
    images_dir = os.path.join(input_dir, "images")
    masks_dir = os.path.join(input_dir, "masks")
    
    files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith((".jpg", ".png"))]
    
    for idx, filename in enumerate(tqdm(files, desc="Processing Kvasir-SEG")):
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)
        
        if not os.path.exists(mask_path):
            base = os.path.splitext(filename)[0]
            for ext in [".jpg", ".png"]:
                test_mask = os.path.join(masks_dir, base + ext)
                if os.path.exists(test_mask):
                    mask_path = test_mask
                    break
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            mask_np = np.array(mask)
            mask_np = (mask_np > 127).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np)
            
            out_filename = f"{prefix}{idx:04d}.png"
            img.save(os.path.join(images_out, out_filename))
            mask.save(os.path.join(masks_out, out_filename))


def process_test_dataset(input_dir, images_out, masks_out):
    for dataset_name in ["CVC-300", "CVC-ColonDB"]:
        ds_dir = os.path.join(input_dir, dataset_name)
        img_dir = os.path.join(ds_dir, "images")
        mask_dir = os.path.join(ds_dir, "masks")
        
        files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith((".jpg", ".png"))]
        
        prefix = dataset_name.lower().replace("-", "_") + "_"
        for idx, filename in enumerate(tqdm(files, desc=f"Processing {dataset_name}")):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                
                mask_np = np.array(mask)
                mask_np = (mask_np > 127).astype(np.uint8) * 255
                mask = Image.fromarray(mask_np)
                
                out_filename = f"{prefix}{idx:04d}.png"
                img.save(os.path.join(images_out, out_filename))
                mask.save(os.path.join(masks_out, out_filename))


def main():
    print("=" * 80)
    print("Polyp Segmentation Dataset Combiner")
    print("=" * 80)
    
    archive_dir = "archive (1)/PNG"
    kvasir_dir = "C:/Users/ADMIN/Downloads/kvasir-seg/Kvasir-SEG"
    test_dir = "C:/Users/ADMIN/Downloads/TestDataset/TestDataset"
    output_dir = "combined_dataset"
    
    print("\nCreating output directories...")
    images_out, masks_out = create_output_dirs(output_dir)
    
    print("\nProcessing datasets...")
    if os.path.exists(archive_dir):
        process_archive_dataset(archive_dir, images_out, masks_out)
    else:
        print(f"Warning: Archive dataset not found at {archive_dir}")
    
    if os.path.exists(kvasir_dir):
        process_kvasir_seg(kvasir_dir, images_out, masks_out)
    else:
        print(f"Warning: Kvasir-SEG dataset not found at {kvasir_dir}")
    
    if os.path.exists(test_dir):
        process_test_dataset(test_dir, images_out, masks_out)
    else:
        print(f"Warning: TestDataset not found at {test_dir}")
    
    num_images = len(os.listdir(images_out))
    num_masks = len(os.listdir(masks_out))
    print("\n" + "=" * 80)
    print(f"Dataset preparation complete!")
    print(f"Total images: {num_images}")
    print(f"Total masks: {num_masks}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
