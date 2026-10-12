import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
        
    def save_activations(self, module, input, output):
        self.activations = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pool gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam


def attention_rollout(attn_weights, start_layer=0, head_fusion="max", discard_ratio=0.0):
    """
    Attention Rollout from Transformers
    """
    result = torch.eye(attn_weights[0].size(-1))
    device = attn_weights[0].device
    result = result.to(device)
    
    for attn in attn_weights[start_layer:]:
        # Average attention heads
        if head_fusion == "mean":
            attn = attn.mean(dim=1)
        elif head_fusion == "max":
            attn = attn.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion: {head_fusion}")
            
        # Discard low attention values
        if discard_ratio > 0.0:
            flat = attn.view(attn.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0
            attn = flat.view(attn.size())
            
        # Add residual
        I = torch.eye(attn.size(-1)).to(device)
        attn = (attn + 1.0 * I) / 2
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        result = torch.matmul(attn, result)
        
    return result


def visualize_cam(image, cam, image_size=224):
    """
    Visualize GradCAM
    """
    # Resize CAM
    cam = cv2.resize(cam, (image_size, image_size))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Convert image to BGR
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return overlay


def visualize_attention_rollout(image, rollout, image_size=224):
    """
    Visualize Attention Rollout
    """
    # Get rollout for class token
    mask = rollout[0, 1:].reshape(1, -1)
    mask = mask.reshape(14, 14)  # Assuming patch size 16
    
    # Normalize
    mask = mask.detach().cpu().numpy()
    mask = mask - np.min(mask)
    mask = mask / (np.max(mask) + 1e-8)
    
    # Resize
    mask = cv2.resize(mask, (image_size, image_size))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    # Convert image to BGR
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Overlay
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    return overlay
