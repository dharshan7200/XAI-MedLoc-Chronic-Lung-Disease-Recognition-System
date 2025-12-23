import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
from torchvision import transforms
from skimage.segmentation import mark_boundaries

from model import get_model
from data_loader import LungDiseaseDataset
from xai_utils import GradCAM, generate_lime_explanation

def visualize_results():
    IMG_DIR = 'images-224'
    CSV_FILE = 'Data_Entry_2017.csv'
    MODEL_PATH = 'lung_disease_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Dataset just to get a random image (or pick specific)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=None) # Raw dataset for accessing filenames
    
    # Load Model
    model = get_model(len(dataset.all_labels))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        print("Loaded trained model.")
    else:
        print("Model file not found. Using random weights (untrained) for demo.")
    
    model.to(device)
    model.eval()

    # Pick a sample
    idx = 25 # Changed to analyze a different image
    img_name = dataset.data.iloc[idx]['Image Index']
    img_path = os.path.join(IMG_DIR, img_name)
    label_str = dataset.data.iloc[idx]['Finding Labels']
    
    print(f"Analyzing {img_name}")
    print(f"Ground Truth: {label_str}")

    # Preprocess
    img_pil = Image.open(img_path).convert('RGB').resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Predict
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    top_prob, top_class_idx = torch.max(probs, dim=1)
    top_class_name = dataset.all_labels[top_class_idx.item()]
    
    print(f"Prediction: {top_class_name} ({top_prob.item():.4f})")

    # Grad-CAM
    # Target the last bottleneck block of ResNet50
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    heatmap = grad_cam(img_tensor, class_idx=top_class_idx.item())
    
    # Overlay Heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img_np = np.array(img_pil)
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    superimposed_img = heatmap * 0.4 + img_bgr 
    superimposed_img = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)

    # LIME
    print("Generating LIME explanation (this may take a moment)...")
    explanation, lime_img_np = generate_lime_explanation(model, img_path, transform)
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=10, 
        hide_rest=False
    )
    lime_boundary = mark_boundaries(temp/255.0 + 0.5, mask)

    # Plot with better spacing and visible titles
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    
    ax[0].imshow(img_pil)
    ax[0].set_title(f"Original X-ray\n{label_str[:30]}", fontsize=12, fontweight='bold', pad=10)
    ax[0].axis('off')
    
    ax[1].imshow(superimposed_img)
    ax[1].set_title(f"Grad-CAM\nPrediction: {top_class_name}", fontsize=12, fontweight='bold', pad=10)
    ax[1].axis('off')
    
    ax[2].imshow(lime_boundary)
    ax[2].set_title("LIME Explanation\n(Superpixel Boundaries)", fontsize=12, fontweight='bold', pad=10)
    ax[2].axis('off')
    
    plt.suptitle('Fig. 6. XAI Explanations for Chest X-ray Disease Prediction', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('lime_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved lime_visualization.png")

if __name__ == '__main__':
    visualize_results()
