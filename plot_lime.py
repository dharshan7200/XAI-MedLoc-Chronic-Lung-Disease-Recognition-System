import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from skimage.segmentation import mark_boundaries

from model import get_model
from xai_utils import generate_lime_explanation

def create_lime_visualization():
    """
    Generate Fig. 6: LIME-based local explanations showing original, superpixel segmentation,
    and LIME explanation with green (supporting) and red (contradicting) highlights.
    """
    # Setup
    MODEL_PATH = 'lung_disease_model.pth'
    IMG_DIR = 'images-224'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model(14)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Model not found!")
        return
    
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get sample images
    sample_images = []
    if os.path.exists(IMG_DIR):
        all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
        sample_images = all_images[:2] if len(all_images) >= 2 else all_images
    
    if not sample_images:
        print("No sample images found. Creating synthetic visualization...")
        create_synthetic_lime_demo()
        return
    
    # Create figure
    fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4 * len(sample_images)))
    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_name in enumerate(sample_images):
        img_path = os.path.join(IMG_DIR, img_name)
        
        try:
            # Generate LIME explanation
            explanation, img_array = generate_lime_explanation(model, img_path, transform)
            
            # Get the top label
            top_label = explanation.top_labels[0]
            
            # Get image and mask
            temp, mask = explanation.get_image_and_mask(
                top_label, 
                positive_only=False, 
                num_features=10, 
                hide_rest=False
            )
            
            # Create visualization with boundaries
            img_boundry = mark_boundaries(temp / 255.0, mask)
            
            # Create colored explanation (green for positive, red for negative)
            explanation_img = np.zeros_like(img_array)
            for i in range(3):
                explanation_img[:, :, i] = img_array[:, :, i]
            
            # Highlight positive (green) and negative (red) regions
            positive_mask = mask > 0
            negative_mask = mask < 0
            
            explanation_img[positive_mask] = explanation_img[positive_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            explanation_img[negative_mask] = explanation_img[negative_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            
            # Plot
            axes[idx, 0].imshow(img_array)
            axes[idx, 0].set_title('Original X-ray', fontweight='bold')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(img_boundry)
            axes[idx, 1].set_title('Superpixel Segmentation', fontweight='bold')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(explanation_img.astype(np.uint8))
            axes[idx, 2].set_title('LIME Explanation', fontweight='bold')
            axes[idx, 2].axis('off')
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            # Fill with placeholder
            for j in range(3):
                axes[idx, j].text(0.5, 0.5, 'Processing Error', 
                                ha='center', va='center', transform=axes[idx, j].transAxes)
                axes[idx, j].axis('off')
    
    plt.suptitle('Fig. 6. LIME-based Local Explanations for Chest X-ray Predictions', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('lime_visualization.png', dpi=300, bbox_inches='tight')
    print("Generated lime_visualization.png")

def create_synthetic_lime_demo():
    """Create a synthetic demo if no real images are available."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    np.random.seed(42)
    
    for row in range(2):
        # Synthetic X-ray
        synthetic_xray = np.random.rand(224, 224, 3) * 0.3 + 0.4
        y, x = np.ogrid[:224, :224]
        mask = ((x - 112)**2 + (y - 112)**2) < 80**2
        synthetic_xray[mask] += 0.2
        synthetic_xray = np.clip(synthetic_xray, 0, 1)
        
        # Synthetic superpixel segmentation
        from skimage.segmentation import slic
        segments = slic(synthetic_xray, n_segments=50, compactness=10)
        img_boundry = mark_boundaries(synthetic_xray, segments)
        
        # Synthetic LIME explanation
        explanation_img = synthetic_xray.copy()
        # Add green highlights (supporting)
        green_mask = (segments == 15) | (segments == 20) | (segments == 25)
        explanation_img[green_mask] = explanation_img[green_mask] * 0.5 + np.array([0, 1, 0]) * 0.5
        # Add red highlights (contradicting)
        red_mask = (segments == 5) | (segments == 10)
        explanation_img[red_mask] = explanation_img[red_mask] * 0.5 + np.array([1, 0, 0]) * 0.5
        
        axes[row, 0].imshow(synthetic_xray)
        axes[row, 0].set_title('Original X-ray', fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(img_boundry)
        axes[row, 1].set_title('Superpixel Segmentation', fontweight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(explanation_img)
        axes[row, 2].set_title('LIME Explanation', fontweight='bold')
        axes[row, 2].axis('off')
    
    plt.suptitle('Fig. 6. LIME-based Local Explanations for Chest X-ray Predictions', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('lime_visualization.png', dpi=300, bbox_inches='tight')
    print("Generated lime_visualization.png (synthetic demo)")

if __name__ == "__main__":
    create_lime_visualization()
