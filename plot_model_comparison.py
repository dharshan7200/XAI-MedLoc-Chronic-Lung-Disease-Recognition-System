import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison():
    # Models and Metrics
    models = ["Logistic Regression", "Decision Tree" , "Random Forest", "SVM", "CNN", "Ensemble"]
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    # Representative performance values (scaled 0-1)
    # Ordered as: [LR, DT, RF, SVM, CNN, Ensemble] for each metric
    data = {
        "Accuracy": [0.72, 0.70, 0.81, 0.78, 0.89, 0.93],
        "Precision": [0.70, 0.68, 0.80, 0.77, 0.88, 0.92],
        "Recall": [0.68, 0.65, 0.78, 0.75, 0.87, 0.91],
        "F1-Score": [0.69, 0.66, 0.79, 0.76, 0.87, 0.91]
    }

    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    # Professional color palette
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    for i, (metric_name, values) in enumerate(data.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric_name, color=colors[i], alpha=0.85)
        # ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8) # Optional: too crowded?
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Comparative Performance Analysis Across Architectures', fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks(x + (width * (len(metrics) - 1) / 2))
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.legend(loc='upper left', ncols=4, fontsize=10)
    ax.set_ylim(0, 1.15)
    
    # Grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    print("Generated model_performance_comparison.png")

if __name__ == "__main__":
    plot_model_comparison()
