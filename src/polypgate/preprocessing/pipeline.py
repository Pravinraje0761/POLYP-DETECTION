"""
Preprocessing module for polyp detection.
Includes ROI extraction and enhancement techniques.
"""
import numpy as np
import cv2
import torch


def soft_roi_extraction(image, prob_mask):
    """
    Stage 2: Soft ROI Extraction using probability mask.
    ROI = ProbabilityMask × OriginalImage
    
    Args:
        image (np.array): Original RGB image (H, W, 3)
        prob_mask (np.array): Probability mask (H, W)
        
    Returns:
        tuple: (original_image, prob_mask, roi_image)
    """
    # Ensure same dimensions
    if image.shape[:2] != prob_mask.shape:
        prob_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    
    # Normalize prob_mask to [0, 1]
    prob_mask = prob_mask.astype(np.float32)
    if prob_mask.max() > 1.0:
        prob_mask /= 255.0
    
    # Soft ROI: element-wise multiplication
    roi_image = image * prob_mask[..., np.newaxis]
    roi_image = np.clip(roi_image, 0, 255).astype(np.uint8)
    
    return image, prob_mask, roi_image


def contrast_stretching(image):
    """
    Contrast stretching to expand intensity range.
    
    Args:
        image (np.array): Input image
        
    Returns:
        np.array: Contrast-stretched image
    """
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    
    stretched = np.clip((image - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    
    return stretched


def gamma_correction(image, gamma=1.0):
    """
    Gamma correction for illumination correction.
    
    Args:
        image (np.array): Input image
        gamma (float): Gamma value (1.0 = no change)
        
    Returns:
        np.array: Gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)


def white_balance_correction(image):
    """
    White balance correction using gray world assumption.
    
    Args:
        image (np.array): Input RGB image
        
    Returns:
        np.array: White-balanced image
    """
    # Calculate mean of each channel
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    
    # Calculate gray world mean
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    
    # Calculate correction factors
    corr_r = mean_gray / mean_r
    corr_g = mean_gray / mean_g
    corr_b = mean_gray / mean_b
    
    # Apply correction
    balanced = image.copy().astype(np.float32)
    balanced[:, :, 0] *= corr_r
    balanced[:, :, 1] *= corr_g
    balanced[:, :, 2] *= corr_b
    
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    
    return balanced


def rgb_channel_normalization(image):
    """
    RGB channel normalization to standardize color distribution.
    
    Args:
        image (np.array): Input RGB image
        
    Returns:
        np.array: Normalized RGB image
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    
    for channel in range(3):
        channel_data = image[:, :, channel].astype(np.float32)
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        
        normalized[:, :, channel] = (channel_data - mean) / (std + 1e-8)
    
    # Scale back to [0, 255]
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8) * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def selective_rgb_enhancement(image, prob_mask):
    """
    Stage 3: Selective RGB Enhancement.
    Applies enhancement only inside ROI.
    
    Args:
        image (np.array): Original RGB image
        prob_mask (np.array): Probability mask
        
    Returns:
        tuple: (original_image, enhanced_roi, blended_output)
    """
    # Step 1: Apply all enhancements to the entire image
    enhanced = image.copy()
    enhanced = contrast_stretching(enhanced)
    enhanced = gamma_correction(enhanced, gamma=1.2)
    enhanced = white_balance_correction(enhanced)
    enhanced = rgb_channel_normalization(enhanced)
    
    # Step 2: Blend enhanced and original using prob_mask
    if image.shape[:2] != prob_mask.shape:
        prob_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    
    prob_mask = prob_mask[..., np.newaxis]
    blended = (enhanced * prob_mask + image * (1 - prob_mask)).astype(np.uint8)
    
    return image, enhanced, blended


def lab_clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Stage 4: LAB + CLAHE Enhancement.
    Converts RGB to LAB, applies CLAHE to L channel, converts back.
    
    Args:
        image (np.array): Input RGB image
        clip_limit (float): CLAHE clip limit
        tile_grid_size (tuple): CLAHE tile grid size
        
    Returns:
        tuple: (before_clahe, after_clahe)
    """
    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    # Merge channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert back to RGB
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return image, rgb_clahe


def complete_preprocessing_pipeline(image, seg_model, device):
    """
    Complete preprocessing pipeline:
    Segmentation → Soft ROI → RGB Enhancement → LAB-CLAHE
    
    Args:
        image (np.array): Input RGB image
        seg_model (nn.Module): A-DenseUNet segmentation model
        device (torch.device): Device
        
    Returns:
        dict: All intermediate results
    """
    # Preprocess for segmentation
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    
    transform = Compose([
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    input_tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    
    # Get segmentation mask
    seg_model.eval()
    with torch.no_grad():
        prob_mask = seg_model(input_tensor)
        prob_mask = torch.sigmoid(prob_mask).squeeze().cpu().numpy()
    
    # Resize mask back to original image size
    prob_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    
    # Stage 2: Soft ROI Extraction
    orig_img, soft_mask, roi_img = soft_roi_extraction(image, prob_mask)
    
    # Stage 3: Selective RGB Enhancement
    orig_rgb, enhanced_rgb, blended_rgb = selective_rgb_enhancement(image, prob_mask)
    
    # Stage 4: LAB + CLAHE
    before_clahe, after_clahe = lab_clahe_enhancement(blended_rgb)
    
    return {
        "original": orig_img,
        "prob_mask": soft_mask,
        "roi": roi_img,
        "enhanced_rgb": enhanced_rgb,
        "blended_rgb": blended_rgb,
        "before_clahe": before_clahe,
        "after_clahe": after_clahe
    }
