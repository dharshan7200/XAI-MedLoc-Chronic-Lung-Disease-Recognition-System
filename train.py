import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np

from data_loader import LungDiseaseDataset
from model import get_model

def train():
    # Parameters
    BATCH_SIZE = 16
    EPOCHS = 10  # Increased epochs
    LR = 1e-4
    IMG_DIR = 'images-224'
    CSV_FILE = 'Data_Entry_2017.csv'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Components
    # Enhanced transforms with Data Augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR)
    
    # Quick split
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(full_dataset))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate dataset instances to apply different transforms
    train_dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=train_transform)
    val_dataset = LungDiseaseDataset(CSV_FILE, IMG_DIR, transform=val_transform)
    
    # Use Subset with the same indices but different underlying transforms
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(len(full_dataset.all_labels)).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training Loop
    log_file = 'training_log.csv'
    # Initialize log file with headers
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
        epoch_train_loss = train_running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
        epoch_val_loss = val_running_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Step the scheduler
        scheduler.step()
        
        # Log to CSV
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{epoch_train_loss:.4f},{epoch_val_loss:.4f}\n")
            
        # Optional: Save intermediate best model
        # if epoch_val_loss < best_val_loss: ...
    
    # Save final model
    torch.save(model.state_dict(), 'lung_disease_model.pth')
    print("Model saved to lung_disease_model.pth")

if __name__ == '__main__':
    if os.path.exists('images-224') and os.path.exists('Data_Entry_2017.csv'):
        train()
    else:
        print("Data not found. Please ensure CSV and 'images-224' are present.")
