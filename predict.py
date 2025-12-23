import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from PIL import Image
from torchvision import transforms
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Import our custom modules
from model import get_model
from xai_utils import GradCAM, generate_lime_explanation

def predict_and_explain(image_path, model_path, output_path):
    print(f"Processing {image_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Define Labels (Hardcoded from data_loader logic for standalone usage)
    all_labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    # 2. Load Model
    model = get_model(len(all_labels))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return

    model.to(device)
    model.eval()

    # 3. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    
    print("\nTop Prdeictions:")
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f"  {all_labels[idx]}: {prob.item():.4f}")
        
    top_class_idx = top_indices[0][0].item()
    top_class_name = all_labels[top_class_idx]

    # 5. Explanations
    
    # A. Grad-CAM
    print("Generating Grad-CAM...")
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(img_tensor, class_idx=top_class_idx)
    
    # Process Heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + img_bgr 
    superimposed_img = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)

    # B. LIME
    print("Generating LIME explanation...")
    explanation, lime_img_np = generate_lime_explanation(model, image_path, transform)
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=5, 
        hide_rest=False
    )
    lime_boundary = mark_boundaries(temp/255.0 + 0.5, mask)

    # 6. Save Composite Output
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img_pil)
    ax[0].set_title(f"Input: {os.path.basename(image_path)}")
    ax[0].axis('off')
    
    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"Grad-CAM: {top_class_name}")
    ax[1].axis('off')
    
    ax[2].imshow(lime_boundary)
    ax[2].set_title(f"LIME: {top_class_name}")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    # Hardcoded defaults as per user request
    IMG_PATH = 'sample_input.png'
    MODEL_FILE = 'lung_disease_model.pth'
    OUT_PATH = 'sample_output.png'
    
    if not os.path.exists(IMG_PATH):
        # Fallback for demo purposes if file doesn't exist
        print(f"'{IMG_PATH}' not found. Looking for a fallback...")
        if os.path.exists('images-224'):
            files = os.listdir('images-224')
            if files:
                fallback = os.path.join('images-224', files[0])
                print(f"Using '{fallback}' as sample input.")
                # Copy it to sample_input.png for consistency
                import shutil
                shutil.copy(fallback, IMG_PATH)
                print(f"Created {IMG_PATH} from fallback.")
    
    predict_and_explain(IMG_PATH, MODEL_FILE, OUT_PATH)
