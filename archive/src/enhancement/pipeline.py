import cv2
import numpy as np
import torch
from PIL import Image


def extract_roi(image, mask, threshold=0.5, padding=10):
    """
    Extract Region of Interest from image using segmentation mask
    """
    # Convert mask to binary
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Crop ROI
    roi = image[y:y+h, x:x+w]
    
    return roi


def selective_rgb_enhancement(image):
    """
    Selective RGB Enhancement
    """
    img = image.astype(np.float32) / 255.0
    
    # Split channels
    r, g, b = cv2.split(img)
    
    # Enhance based on channel properties
    r_enhanced = np.clip(r * 1.1 + 0.05, 0, 1)
    g_enhanced = np.clip(g * 1.05 + 0.02, 0, 1)
    b_enhanced = np.clip(b * 0.95, 0, 1)
    
    # Merge back
    enhanced = cv2.merge([r_enhanced, g_enhanced, b_enhanced])
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced


def lab_clahe_enhancement(image):
    """
    LAB + CLAHE Enhancement
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_enhanced = cv2.merge([l_clahe, a, b])
    
    # Convert back to RGB
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return rgb_enhanced


def enhance_pipeline(image, mask):
    """
    Complete enhancement pipeline: ROI extraction + Selective RGB + LAB-CLAHE
    """
    # Extract ROI
    roi = extract_roi(image, mask)
    
    # Apply selective RGB enhancement
    rgb_enhanced = selective_rgb_enhancement(roi)
    
    # Apply LAB + CLAHE enhancement
    final_enhanced = lab_clahe_enhancement(rgb_enhanced)
    
    return final_enhanced
