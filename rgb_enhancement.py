
"""
Stage 3: Selective RGB Enhancement Module (UPGRADED).
UPGRADES:
- Smart adaptive enhancement based on image statistics
- Only applies contrast stretching/gamma correction when needed
- Medical-image-safe preprocessing
"""
import numpy as np
import cv2


def contrast_stretching(image, low_percentile=2, high_percentile=98):
    """
    Contrast stretching to expand intensity range.
    
    Args:
        image (np.array): Input image
        low_percentile (int): Lower percentile for clipping
        high_percentile (int): Upper percentile for clipping
    
    Returns:
        np.array: Contrast-stretched image
    """
    p_low = np.percentile(image, low_percentile)
    p_high = np.percentile(image, high_percentile)
    
    stretched = np.clip((image - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
    
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
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    
    corr_r = mean_gray / mean_r
    corr_g = mean_gray / mean_g
    corr_b = mean_gray / mean_b
    
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
    
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8) * 255
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return normalized


def selective_rgb_enhancement(image, prob_mask):
    """
    Stage 3: Selective RGB Enhancement (UPGRADED: SMART ADAPTIVE).
    Applies enhancement only inside ROI and only when needed.
    
    Args:
        image (np.array): Original RGB image
        prob_mask (np.array): Probability mask
    
    Returns:
        tuple: (original_image, enhanced_roi, blended_output)
    """
    # Ensure same dimensions
    if image.shape[:2] != prob_mask.shape:
        prob_mask = cv2.resize(prob_mask, (image.shape[1], image.shape[0]))
    
    # Compute image statistics to decide if enhancement is needed
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Apply adaptive enhancement
    enhanced = image.copy()
    
    # Only apply contrast stretching if std is low (low contrast)
    if std_brightness < 50:
        enhanced = contrast_stretching(enhanced)
    
    # Only apply gamma correction if mean brightness is very low/high
    if mean_brightness < 80:
        enhanced = gamma_correction(enhanced, gamma=1.3)  # Brighten
    elif mean_brightness > 180:
        enhanced = gamma_correction(enhanced, gamma=0.8)  # Darken
    
    # Always apply white balance and normalization for consistency
    enhanced = white_balance_correction(enhanced)
    enhanced = rgb_channel_normalization(enhanced)
    
    # Blend enhanced and original using soft mask
    prob_mask = prob_mask.astype(np.float32)
    if prob_mask.max() > 1.0:
        prob_mask /= 255.0
    prob_mask = prob_mask[..., np.newaxis]
    blended = (enhanced * prob_mask + image * (1 - prob_mask)).astype(np.uint8)
    
    return image, enhanced, blended

