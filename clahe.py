"""
Stage 4: LAB + CLAHE Enhancement Module.
"""
import cv2
import numpy as np


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
