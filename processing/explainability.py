import torch
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# EXPLAINABILITY MODULE
# ============================================================================

class GradCamYOLOCompatibleWrapper:
    """
    A specialized wrapper for GradCAM that intercepts .eval() and provides
    access to the underlying model's parameters and .zero_grad() method.
    """
    def __init__(self, model):
        self.model = model
        self.training = False

    def __call__(self, x):
        """
        This accepts a tensor input from GradCAM and returns a 2D tensor [batch_size, num_classes].
        """
        # GradCAM passes a tensor, but YOLO expects PIL Image or numpy array
        # Convert tensor back to numpy/PIL for YOLO processing
        
        # x shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Process each image in the batch
        all_probs = []
        for i in range(batch_size):
            # Extract single image: [channels, height, width]
            img_tensor = x[i]
            
            # Denormalize (reverse the normalization applied in preprocessing)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)
            img_tensor = img_tensor * std + mean
            
            # Convert to numpy and then PIL
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Run YOLO inference
            results_list = self.model(pil_img, verbose=False)
            first_result = results_list[0]
            probs_object = first_result.probs
            
            # Get probabilities as 1D tensor
            probs_data = probs_object.data
            all_probs.append(probs_data)
        
        # Stack all probabilities into 2D tensor [batch_size, num_classes]
        output = torch.stack(all_probs)
        return output

    def eval(self):
        return self

    def parameters(self):
        return self.model.parameters()
        
    def zero_grad(self):
        # YOLO models don't have zero_grad, so we'll pass
        pass

    def children(self):
        return iter([])


class ExplainabilityModule:
    """
    Provides visual explanations using alternative methods since YOLO 
    doesn't support GradCAM out of the box due to its architecture.
    """

    def __init__(self, model=None, target_layer=None):
        """
        Initialize explainability module.
        Note: GradCAM doesn't work with YOLO models, so we use alternative methods.
        """
        self.model = model
        self.target_layer = target_layer

    def generate_heatmap(self, image_tensor, target_category=None):
        """
        Generate a saliency-style heatmap using edge detection and intensity analysis.
        This is a fallback since GradCAM doesn't work with YOLO models.
        """
        # Denormalize the image tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = image_tensor * std + mean
        
        # Convert to numpy
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Gaussian blur to create smooth heatmap
        edges_blurred = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
        
        # Normalize to 0-1 range
        if edges_blurred.max() > 0:
            heatmap = edges_blurred / edges_blurred.max()
        else:
            heatmap = edges_blurred
        
        # Apply additional processing to highlight damage areas
        # Use intensity variations
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Detect areas with high intensity variation (potential damage)
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize magnitude
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        
        # Combine edge and gradient information
        combined_heatmap = (heatmap * 0.5 + magnitude * 0.5)
        
        # Apply threshold to focus on significant areas
        combined_heatmap = np.where(combined_heatmap > 0.2, combined_heatmap, 0)
        
        # Renormalize
        if combined_heatmap.max() > 0:
            combined_heatmap = combined_heatmap / combined_heatmap.max()
        
        return combined_heatmap

    def visualize(self, original_image, heatmap):
        """Overlay heatmap on original image"""
        img_array = np.array(original_image.resize((224, 224)))
        img_array = img_array.astype(np.float32) / 255.0

        # Create colormap visualization
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0
        
        # Blend with original image
        alpha = 0.5
        visualization = (alpha * heatmap_colored + (1 - alpha) * img_array)
        visualization = (visualization * 255).astype(np.uint8)
        
        return visualization
