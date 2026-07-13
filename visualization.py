"""
Visualization module for all stages.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def visualize_stage2(original, mask, roi, save_path=None):
    """
    Visualize Stage 2: ROI Extraction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Probability Mask")
    axes[1].axis('off')
    
    axes[2].imshow(roi)
    axes[2].set_title("Soft ROI Image")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_stage3(original, enhanced, blended, save_path=None):
    """
    Visualize Stage 3: Selective RGB Enhancement.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced ROI")
    axes[1].axis('off')
    
    axes[2].imshow(blended)
    axes[2].set_title("Blended Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_stage4(before, after, save_path=None):
    """
    Visualize Stage 4: LAB + CLAHE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(before)
    axes[0].set_title("Before CLAHE")
    axes[0].axis('off')
    
    axes[1].imshow(after)
    axes[1].set_title("After CLAHE")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_stage8(original, mask, roi, enhanced, gradcam_overlay, gradcam_heatmap, save_path=None):
    """
    Visualize Stage 8: Complete Grad-CAM visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title("2. Probability Mask")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(roi)
    axes[0, 2].set_title("3. ROI Image")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(enhanced)
    axes[1, 0].set_title("4. Enhanced Image")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Grad-CAM Overlay")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(gradcam_heatmap, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Raw Heatmap")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_stage9(gradcam_overlay, attn_overlay, attn_heatmap, save_path=None):
    """
    Visualize Stage 9: Attention Rollout comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Grad-CAM Overlay")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(attn_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Attention Rollout Overlay")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Attention Heatmap")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_table(metrics_dict, save_path=None):
    """
    Create a comparison table for ablation study (Stage 11).
    """
    import pandas as pd
    
    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df
