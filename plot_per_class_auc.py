import matplotlib.pyplot as plt
import numpy as np

def plot_per_class_auc():
    # Data from latest evaluation results
    results = {
        'Atelectasis': 0.7906,
        'Cardiomegaly': 0.8593,
        'Effusion': 0.8376,
        'Infiltration': 0.7339,
        'Mass': 0.9483,
        'Nodule': 0.8241,
        'Pneumonia': 0.8905,
        'Pneumothorax': 0.9292,
        'Consolidation': 0.9228,
        'Edema': 0.9612,
        'Emphysema': 0.9126,
        'Fibrosis': 0.8452,
        'Pleural_Thickening': 0.8866,
        'Hernia': 0.8533
    }

    # Sort results for better visualization if needed, but here we'll keep alphabetical or NIH order
    labels = list(results.keys())
    scores = list(results.values())

    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-pastel')
    
    # Colors: Highlight the ones mentioned by the user
    colors = []
    for label in labels:
        if label in ['Edema', 'Mass', 'Pneumothorax']:
            colors.append('#27ae60') # Success green
        elif label in ['Infiltration', 'Atelectasis']:
            colors.append('#e74c3c') # Warning red
        else:
            colors.append('#3498db') # Standard blue

    bars = plt.barh(labels, scores, color=colors, alpha=0.8)
    
    # Customize the appearance
    plt.xlabel('AUC Score', fontsize=12, fontweight='bold')
    plt.title('Fig. 3. ResNet50 per-class AUC performance across 14 chest X-ray disease categories', fontsize=14, fontweight='bold', pad=20)
    plt.xlim(0, 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('per_class_auc.png', dpi=300)
    print("Generated per_class_auc.png")

if __name__ == "__main__":
    plot_per_class_auc()
