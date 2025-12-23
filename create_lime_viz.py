import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from torchvision import transforms
from skimage.segmentation import mark_boundaries

from model import get_model
from data_loader import LungDiseaseDataset
from xai_utils import generate_lime_explanation

def create_comprehensive_lime_viz():
    """Create a comprehensive LIME visualization with multiple examples"""
    IMG_DIR = 'images-224'
    CSV_FILE = 'Data_Entry_2017.csv'
    MODEL_PATH = 'lung_disease_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=None)
    
    # Load Model
    model = get_model(len(dataset.all_labels))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        print("Loaded trained model.")
    else:
        print("Model file not found!")
        return
    
    model.to(device)
    model.eval()

    # Select multiple diverse samples
    sample_indices = [5, 15, 30]  # Different images
    
    # Create figure with 3 rows, 3 columns
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5 * len(sample_indices)))
    
    for row_idx, idx in enumerate(sample_indices):
        try:
            img_name = dataset.data.iloc[idx]['Image Index']
            img_path = os.path.join(IMG_DIR, img_name)
            label_str = dataset.data.iloc[idx]['Finding Labels']
            
            print(f"\nProcessing image {row_idx+1}/{len(sample_indices)}: {img_name}")
            print(f"Ground Truth: {label_str}")
            
            # Load image
            img_pil = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = np.array(img_pil)
            
            # Generate LIME explanation
            print("  Generating LIME explanation...")
            explanation, lime_img_np = generate_lime_explanation(model, img_path, transform)
            
            # Get the top label
            top_label = explanation.top_labels[0]
            predicted_disease = dataset.all_labels[top_label]
            
            # Get image and mask
            temp, mask = explanation.get_image_and_mask(
                top_label, 
                positive_only=False, 
                num_features=10, 
                hide_rest=False
            )
            
            # Create superpixel boundary visualization
            lime_boundary = mark_boundaries(temp/255.0, mask)
            
            # Create colored LIME explanation
            explanation_img = lime_img_np.copy().astype(float)
            
            # Highlight positive (green) and negative (red) regions
            positive_mask = mask > 0
            negative_mask = mask < 0
            
            if np.any(positive_mask):
                explanation_img[positive_mask] = explanation_img[positive_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
            if np.any(negative_mask):
                explanation_img[negative_mask] = explanation_img[negative_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            
            # Plot - Column 1: Original
            axes[row_idx, 0].imshow(img_array)
            axes[row_idx, 0].set_title(f'Original X-ray\nGround Truth: {label_str[:25]}', 
                                      fontsize=11, fontweight='bold', pad=8)
            axes[row_idx, 0].axis('off')
            
            # Plot - Column 2: Superpixel Segmentation
            axes[row_idx, 1].imshow(lime_boundary)
            axes[row_idx, 1].set_title(f'Superpixel Segmentation\nPredicted: {predicted_disease}', 
                                      fontsize=11, fontweight='bold', pad=8)
            axes[row_idx, 1].axis('off')
            
            # Plot - Column 3: LIME Explanation
            axes[row_idx, 2].imshow(np.clip(explanation_img, 0, 255).astype(np.uint8))
            axes[row_idx, 2].set_title('LIME Explanation\n(Green=Support, Red=Contradict)', 
                                      fontsize=11, fontweight='bold', pad=8)
            axes[row_idx, 2].axis('off')
            
            print(f"  ✓ Successfully processed")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            for col in range(3):
                axes[row_idx, col].text(0.5, 0.5, f'Error\n{str(e)[:40]}', 
                                       ha='center', va='center', 
                                       transform=axes[row_idx, col].transAxes,
                                       fontsize=9, color='red')
                axes[row_idx, col].axis('off')
    
    plt.suptitle('Fig. 6. LIME-based Local Explanations for Chest X-ray Predictions', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('lime_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Generated lime_visualization.png")

if __name__ == '__main__':
    create_comprehensive_lime_viz()
