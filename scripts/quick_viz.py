"""Quick visualization of encoding results."""
import matplotlib.pyplot as plt
import numpy as np

# Your data
layer_data = {
    0: [85.6, 2.6, 2.2, 2.0, 7.7],
    1: [20.3, 20.2, 20.2, 19.5, 19.7],
    2: [19.7, 21.5, 21.0, 19.5, 18.4],
    3: [19.8, 20.6, 20.2, 20.0, 19.4],
}

encoding_names = ['Rate', 'Temporal', 'Population', 'Burst', 'Adaptive']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for layer_idx, weights in layer_data.items():
    weights_norm = np.array(weights) / 100
    bars = axes[layer_idx].bar(encoding_names, weights_norm, color=colors)
    
    # Highlight dominant
    max_idx = np.argmax(weights_norm)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    axes[layer_idx].set_title(f'Layer {layer_idx}', fontsize=14, fontweight='bold')
    axes[layer_idx].set_ylabel('Usage Frequency')
    axes[layer_idx].set_ylim([0, 1])
    axes[layer_idx].tick_params(axis='x', rotation=45)
    axes[layer_idx].grid(axis='y', alpha=0.3)
    
    # Add entropy measure
    entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))
    axes[layer_idx].text(0.95, 0.95, f'Entropy: {entropy:.2f}',
                        transform=axes[layer_idx].transAxes,
                        ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Learned Encoding Preferences by Layer', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/encoding_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/plots/encoding_analysis.png")
plt.show()