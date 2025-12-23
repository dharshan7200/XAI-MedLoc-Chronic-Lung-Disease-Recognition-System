import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_shap_summary_plot():
    """
    Generate Fig. 7: SHAP summary plot showing global feature importance
    for the ResNet50 model across all 14 disease categories.
    
    Note: This creates a representative visualization. For real SHAP values,
    you would need to run shap.DeepExplainer on the actual model.
    """
    
    # 14 disease categories
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Representative feature names (image regions/patterns)
    features = [
        'Upper Lung Field Opacity',
        'Lower Lung Field Opacity',
        'Cardiac Silhouette Size',
        'Pleural Space Density',
        'Mediastinal Width',
        'Lung Texture Pattern',
        'Costophrenic Angle',
        'Hilar Region Density',
        'Peripheral Lung Opacity',
        'Central Lung Opacity',
        'Diaphragm Position',
        'Rib Cage Alignment',
        'Lung Volume',
        'Vascular Pattern'
    ]
    
    # Generate synthetic SHAP values for demonstration
    # In reality, these would come from shap.DeepExplainer
    np.random.seed(42)
    n_samples = 100
    n_features = len(features)
    
    # Create SHAP values and feature values
    shap_values = []
    feature_values = []
    
    for i in range(n_features):
        # Generate SHAP values with varying importance
        importance = (n_features - i) / n_features  # Higher features are more important
        shap_vals = np.random.randn(n_samples) * importance * 0.5
        
        # Generate corresponding feature values (normalized 0-1)
        feat_vals = np.random.rand(n_samples)
        
        shap_values.append(shap_vals)
        feature_values.append(feat_vals)
    
    # Create the summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort features by mean absolute SHAP value
    mean_abs_shap = [np.mean(np.abs(sv)) for sv in shap_values]
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    
    # Plot each feature
    for idx, feature_idx in enumerate(sorted_indices):
        y_pos = idx
        shap_vals = shap_values[feature_idx]
        feat_vals = feature_values[feature_idx]
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(shap_vals, [y_pos] * len(shap_vals), 
                           c=feat_vals, cmap='coolwarm', 
                           alpha=0.6, s=20, vmin=0, vmax=1)
    
    # Customize plot
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in sorted_indices])
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=12, fontweight='bold')
    ax.set_title('Fig. 7. SHAP Summary Plot - Global Feature Importance for ResNet50', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Feature Value\n(Blue = Low, Red = High)', 
                   rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print("Generated shap_summary_plot.png")
    print("\nNote: This is a representative visualization.")
    print("For actual SHAP values, run: shap.DeepExplainer(model, background_data)")

if __name__ == "__main__":
    create_shap_summary_plot()
