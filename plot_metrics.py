import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

from model import get_model
from data_loader import LungDiseaseDataset

def plot_learning_curve():
    log_file = 'training_log.csv'
    if not os.path.exists(log_file):
        print("No training log found. Skipping Learning Curve.")
        return
        
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', color='b', label='Train Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], marker='x', linestyle='--', color='r', label='Val Loss')
    
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")

def plot_roc_curves():
    # Setup model and data
    BATCH_SIZE = 16
    IMG_DIR = 'images-224'
    CSV_FILE = 'Data_Entry.csv'
    MODEL_PATH = 'lung_disease_model.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(len(dataset.all_labels))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Model not found. Skipping ROC metrics.")
        return
        
    model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    
    print("Generating predictions for ROC...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Plot ROC for all classes
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(dataset.all_labels):
        try:
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_preds[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        except Exception:
            pass # Skip classes with no positive samples in this subset
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) by Class')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.savefig('performance_metrics.png')
    print("Saved performance_metrics.png")

if __name__ == '__main__':
    plot_learning_curve()
    plot_roc_curves()
