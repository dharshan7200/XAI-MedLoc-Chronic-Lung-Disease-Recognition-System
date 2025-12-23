import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LungDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Define the exact list of classes as per NIH dataset standard
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # Filter data to only include images that actually exist in the directory
        # This prevents runtime errors during training
        self.data['exists'] = self.data['Image Index'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
        self.data = self.data[self.data['exists']].reset_index(drop=True)
        print(f"Dataset loaded: {len(self.data)} images found out of original CSV.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert('RGB')
        
        # Labels are pipe-separated, e.g., "Cardiomegaly|Emphysema"
        label_str = self.data.iloc[idx]['Finding Labels']
        
        # Create multi-hot vector
        label = torch.zeros(len(self.all_labels), dtype=torch.float32)
        for i, disease in enumerate(self.all_labels):
            if disease in label_str:
                label[i] = 1.0
                
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == '__main__':
    # Test the loader (Smoke Test)
    from torchvision import transforms
    
    # Simple transform
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Adjust paths as per user environment
    csv_path = 'Data_Entry.csv'
    img_path = 'images-224'
    
    if os.path.exists(csv_path) and os.path.exists(img_path):
        ds = LungDiseaseDataset(csv_path, img_path, transform=t)
        if len(ds) > 0:
            img, lbl = ds[0]
            print(f"Sample Image Shape: {img.shape}")
            print(f"Sample Label: {lbl}")
            print("DataLoader test passed.")
        else:
            print("Dataset empty!")
    else:
        print("Files not found for test.")
