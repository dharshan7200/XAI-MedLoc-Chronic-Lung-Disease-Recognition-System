import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd

from model import get_model
from data_loader import LungDiseaseDataset

def evaluate_model():
    BATCH_SIZE = 16
    IMG_DIR = 'images-224'
    CSV_FILE = 'Data_Entry.csv'
    MODEL_PATH = 'lung_disease_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = get_model(len(dataset.all_labels))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    else:
        print("Model not found!")
        return
        
    model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate Metrics
    print("\n--- Model Performance ---")
    try:
        # AUROC (Average across classes)
        # Note: roc_auc_score might error if a class has only 0 or only 1 in targets
        # We try to calculate it safely
        aucs = []
        for i, class_name in enumerate(dataset.all_labels):
            try:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aucs.append(score)
                print(f"{class_name:<20}: AUC = {score:.4f}")
            except ValueError:
                # This happens if a class is not present in the batch
                print(f"{class_name:<20}: AUC = N/A (No positive samples)")
        
        if aucs:
            print(f"\nMean AUC: {np.mean(aucs):.4f}")
            
        # Accuracy (Subset Accuracy - exact match)
        # This is very harsh for multi-label, so we also do a threshold-based accuracy
        threshold = 0.5
        binary_preds = (all_preds > threshold).astype(int)
        
        # Calculate accuracy per class (Binary accuracy)
        acc_per_class = np.mean(binary_preds == all_targets, axis=0)
        print(f"Mean Per-Class Accuracy: {np.mean(acc_per_class):.4f}")
        
    except Exception as e:
        print(f"Error evaluating metrics: {e}")

if __name__ == '__main__':
    evaluate_model()
