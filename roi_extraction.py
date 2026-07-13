
"""
Stage 2: Soft ROI Extraction Module (UPGRADED).
UPGRADES:
- Soft ROI with context preservation
- Expands mask slightly and adds Gaussian blur for smooth boundary
- Preserves 10-20% background context
"""
import numpy as np
import cv2


def expand_mask(mask, expansion_ratio=0.1):
    """
    Expand mask slightly to preserve context.
    
    Args:
        mask (np.array): Binary or probability mask [0-1]
        expansion_ratio (float): Ratio to expand mask (0.1 = 10%)
    
    Returns:
        np.array: Expanded mask
    """
    h, w = mask.shape
    expansion_pixels = int(min(h, w) * expansion_ratio)
    
    # Find contours to get bounding box
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    # Get bounding box of all contours
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + bw)
        y_max = max(y_max, y + bh)
    
    # Expand bounding box
    x_min = max(0, x_min - expansion_pixels)
    y_min = max(0, y_min - expansion_pixels)
    x_max = min(w, x_max + expansion_pixels)
    y_max = min(h, y_max + expansion_pixels)
    
    # Create expanded mask
    expanded_mask = np.zeros_like(mask)
    expanded_mask[y_min:y_max, x_min:x_max] = 1
    
    return expanded_mask


def soft_roi_extraction(image, prob_mask, expansion_ratio=0.1, blur_kernel_size=5):
    """
    Stage 2: Soft ROI Extraction using probability mask (UPGRADED).
    ROI = ProbabilityMask × OriginalImage
    
    Args:
        image (np.array): Original RGB image (H, W, 3)
        prob_mask (np.array): Probability mask (H, W)
        expansion_ratio (float): Ratio to expand mask for context
        blur_kernel_size (int): Kernel size for Gaussian blur (odd)
    
    Returns:
        tuple: (original_image, prob_mask, roi_image)
    """
    # Ensure same dimensions
    if image.shape[:2] != prob_mask.shape:
        prob_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    
    # Normalize prob_mask to [0-1]
    prob_mask = prob_mask.astype(np.float32)
    if prob_mask.max() > 1.0:
        prob_mask /= 255.0
    
    # Expand mask slightly for context
    expanded_mask = expand_mask(prob_mask, expansion_ratio)
    
    # Combine with original mask and smooth with Gaussian blur
    combined_mask = np.maximum(prob_mask, expanded_mask * 0.3)  # 30% weight for context
    if blur_kernel_size > 0 and blur_kernel_size % 2 == 1:
        combined_mask = cv2.GaussianBlur(combined_mask, (blur_kernel_size, blur_kernel_size), 0)
    
    # Soft ROI: element-wise multiplication
    roi_image = image * combined_mask[..., np.newaxis]
    roi_image = np.clip(roi_image, 0, 255).astype(np.uint8)
    
    return image, combined_mask, roi_image

