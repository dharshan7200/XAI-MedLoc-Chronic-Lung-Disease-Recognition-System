import torch
import torch.nn.functional as F
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook for gradients
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        # Note: We need to register the forward hook or get the activation in another way
        # For simplicity in this script, we'll assume we can pass x through the model 
        # but we need to capture the feature map of the target layer.
        
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output)
            
        handle = self.target_layer.register_forward_hook(hook_fn)
        
        # Forward
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        handle.remove()
        
        # Grad-CAM generation
        gradients = self.gradients
        activations = feature_maps[0]
        
        # Global Average Pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the activations
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()

def generate_lime_explanation(model, image_path, preprocess_transform):
    """
    Generates LIME explanation for a single image.
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Read image using PIL to keep consistency with loading
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    # Resize for the LIME explainer (it likes numpy arrays)
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized)

    def batch_predict(images):
        # images is a numpy array (N, H, W, C)
        # Convert to torch tensor
        batch = torch.stack([preprocess_transform(Image.fromarray(i)) for i in images])
        
        # Move to model device
        device = next(model.parameters()).device
        batch = batch.to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
        return probs.cpu().numpy()

    # Generate explanation
    # predicting top 1 label for simplicity in demo
    explanation = explainer.explain_instance(
        img_np, 
        batch_predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=50 # Low for speed in demo
    )
    
    return explanation, img_np
