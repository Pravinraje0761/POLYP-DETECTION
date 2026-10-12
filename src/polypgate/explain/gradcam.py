"""
Stage 8: Grad-CAM Implementation.
"""
import torch
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM implementation for Vision Transformer.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: Vision Transformer model
            target_layer: Target layer to hook
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)
        
    def _save_activations(self, module, input, output):
        """Save forward activations"""
        self.activations = output
        
    def _save_gradients(self, module, grad_input, grad_output):
        """Save backward gradients"""
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        NOTE: No torch.no_grad()!
        """
        self.model.eval()
        
        # Forward pass
        logits, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Verify shapes
        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")
        assert activations is not None, "Activations are None!"
        assert activations.sum() != 0, "Activations are all zero!"
        
        # For ViT, we need to process the patch embeddings
        # ViT activations shape: (num_tokens, embed_dim) or similar
        # Let's take the spatial tokens (exclude CLS token)
        if len(activations.shape) == 2:  # (num_tokens, dim)
            # Assume first token is CLS, rest are spatial
            spatial_activations = activations[1:]
            spatial_gradients = gradients[1:]
            
            # Reshape to 2D (assume 14x14 for patch size 16)
            num_patches = spatial_activations.shape[0]
            hw = int(np.sqrt(num_patches))
            spatial_activations = spatial_activations.reshape(hw, hw, -1)
            spatial_gradients = spatial_gradients.reshape(hw, hw, -1)
            
            # Global average pool gradients
            weights = np.mean(spatial_gradients, axis=(0, 1))
            
            # Weighted sum
            cam = np.zeros(spatial_activations.shape[0:2], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * spatial_activations[:, :, i]
        elif len(activations.shape) == 3:  # (seq_len, batch, dim) - adjust as needed
            # Handle different activation shapes if needed
            spatial_activations = activations[:, 1:].reshape(-1)  # Simplified
            # For simplicity, let's assume we have 2D activations
            cam = np.mean(activations, axis=-1)
        else:
            raise ValueError(f"Unexpected activations shape: {activations.shape}")
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam


def visualize_gradcam(image, cam, image_size=224):
    """
    Visualize Grad-CAM: original + mask + ROI + enhanced + overlay + raw heatmap.
    """
    # Resize heatmap to match original image size
    img_h, img_w = image.shape[:2]
    cam_resized = cv2.resize(cam, (img_w, img_h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    
    # Convert image to BGR for OpenCV
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
