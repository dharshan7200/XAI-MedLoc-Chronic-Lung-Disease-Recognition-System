import matplotlib.pyplot as plt
import numpy as np

def plot_xai_comparison():
    # Metrics and Data (Representative values from explainability impact studies)
    metrics = ["Clinician Trust Score", "Adoption Rate", "Error Detection Rate"]
    without_xai = [65, 42, 70]
    with_xai = [88, 75, 92]

    x = np.arange(len(metrics))  # label locations
    width = 0.35  # width of the bars

    # Set modern style
    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gradient-like colors
    color_without = '#e74c3c' # Soft red
    color_with = '#2ecc71'    # Soft green

    rects1 = ax.bar(x - width/2, without_xai, width, label='Without XAI', color=color_without, alpha=0.85)
    rects2 = ax.bar(x + width/2, with_xai, width, label='With XAI', color=color_with, alpha=0.85)

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Explainable AI (XAI) on Clinical Workflow Highlights', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc='upper left')

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('xai_impact_comparison.png', dpi=300)
    print("Generated xai_impact_comparison.png")

if __name__ == "__main__":
    plot_xai_comparison()
