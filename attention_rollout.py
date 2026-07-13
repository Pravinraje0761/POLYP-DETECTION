"""
Stage 9: Attention Rollout Implementation.
"""
import torch
import numpy as np
import cv2


def attention_rollout(attn_weights, start_layer=0, head_fusion="mean", discard_ratio=0.0):
    """
    Attention Rollout for Vision Transformer.
    
    Args:
        attn_weights: List of attention weights from each layer
        start_layer: Starting layer for rollout
        head_fusion: How to fuse attention heads ('mean' or 'max')
        discard_ratio: Ratio of low attention values to discard
        
    Returns:
        Rollout attention map
    """
    # Get batch_size = attn_weights[0].size(0)
    batch_size = attn_weights[0].size(0)
    num_tokens = attn_weights[0].size(-1)
    device = attn_weights[0].device
    
    result = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    for attn in attn_weights[start_layer:]:
        # Average or max attention heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion: {head_fusion}")
            
        # Discard low attention values
        if discard_ratio > 0.0:
            flat = attn_fused.view(attn_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            for b in range(batch_size):
                flat[b, indices[b]] = 0
            attn_fused = flat.view(attn_fused.size())
            
        # Add residual connection
        I = torch.eye(num_tokens, device=device).unsqueeze(0)
        attn_fused = (attn_fused + 1.0 * I) / 2
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)
        
        # Matrix multiplication for rollout
        result = torch.matmul(attn_fused, result)
    
    return result


def visualize_attention_rollout(image, rollout, image_size=224):
    """
    Visualize Attention Rollout.
    
    Args:
        image: Input image
        rollout: Rollout attention map
        image_size: Target image size
        
    Returns:
        Overlay and raw heatmap
    """
    # Ensure rollout has batch dimension
    if rollout.dim() == 2:
        rollout = rollout.unsqueeze(0)  # Add batch dim
    
    # Get rollout for class token to all patches
    mask = rollout[0, 0, 1:]  # (num_patches)
    num_patches = mask.shape[0]
    hw = int(np.sqrt(num_patches))
    mask = mask.reshape(hw, hw)
    
    # Normalize
    mask = mask.detach().cpu().numpy()
    mask = mask - np.min(mask)
    mask = mask / (np.max(mask) + 1e-8)
    
    # Resize to original image size
    img_h, img_w = image.shape[:2]
    mask = cv2.resize(mask, (img_w, img_h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    # Convert image to BGR
    if isinstance(image, np.ndarray):
        if image.shape[-1] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
    else:
        image_bgr = np.array(image)
        if image_bgr.shape[-1] == 3:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    
    # Overlay
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    
    return overlay, heatmap
