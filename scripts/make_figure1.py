"""
Figure 1: Learned encoding preferences show hierarchical specialization.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
layer_data = {
    0: [85.6, 2.6, 2.2, 2.0, 7.7],
    1: [20.3, 20.2, 20.2, 19.5, 19.7],
    2: [19.7, 21.5, 21.0, 19.5, 18.4],
    3: [19.8, 20.6, 20.2, 20.0, 19.4],
}

encoding_names = ['Rate', 'Temporal', 'Population', 'Burst', 'Adaptive']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure
fig = plt.figure(figsize=(16, 10))

# Main plots - encoding distribution
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                      left=0.08, right=0.95, top=0.88, bottom=0.35)

for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    weights = np.array(layer_data[i]) / 100
    
    bars = ax.bar(encoding_names, weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Highlight dominant
    max_idx = np.argmax(weights)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    ax.set_title(f'Layer {i}', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Usage Frequency', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add entropy
    entropy = -np.sum(weights * np.log(weights + 1e-10))
    max_entropy = np.log(5)
    pct = entropy / max_entropy * 100
    
    ax.text(0.95, 0.95, f'H={entropy:.2f}\n({pct:.0f}% max)',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=11, fontweight='bold')
    
    # Add interpretation
    if i == 0:
        ax.text(0.5, 0.5, 'SPECIALIZED', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='red', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'ADAPTIVE', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='green', alpha=0.3)

# Bottom: Entropy progression
ax_entropy = fig.add_subplot(gs[1, :2])
layers = list(range(4))
entropies = [0.588, 1.609, 1.609, 1.609]
max_entropy = np.log(5)

ax_entropy.plot(layers, entropies, 'o-', linewidth=3, markersize=12, 
                color='#2ca02c', label='Observed Entropy')
ax_entropy.axhline(max_entropy, linestyle='--', color='red', linewidth=2, 
                   label='Maximum Entropy', alpha=0.7)
ax_entropy.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.5)

ax_entropy.set_xlabel('Layer', fontsize=14, fontweight='bold')
ax_entropy.set_ylabel('Entropy (nats)', fontsize=14, fontweight='bold')
ax_entropy.set_title('Encoding Diversity Across Layers', fontsize=16, fontweight='bold')
ax_entropy.set_xticks(layers)
ax_entropy.set_ylim([-0.1, 1.8])
ax_entropy.grid(True, alpha=0.3)
ax_entropy.legend(fontsize=12, loc='lower right')

# Add annotations
ax_entropy.annotate('Specialized\n(Low Diversity)', 
                   xy=(0, 0.588), xytext=(0.5, 0.3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))

ax_entropy.annotate('Adaptive\n(Maximum Diversity)', 
                   xy=(1, 1.609), xytext=(2, 1.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Bottom right: Dominance comparison
ax_dom = fig.add_subplot(gs[1, 2:])
dominance = [85.6, 20.3, 21.5, 20.6]
colors_dom = ['red', 'green', 'green', 'green']

bars = ax_dom.bar(layers, dominance, color=colors_dom, alpha=0.6, edgecolor='black', linewidth=2)
ax_dom.axhline(20, linestyle='--', color='blue', linewidth=2, label='Uniform (20%)', alpha=0.7)

ax_dom.set_xlabel('Layer', fontsize=14, fontweight='bold')
ax_dom.set_ylabel('Dominant Encoding Usage (%)', fontsize=14, fontweight='bold')
ax_dom.set_title('Specialization vs. Flexibility', fontsize=16, fontweight='bold')
ax_dom.set_xticks(layers)
ax_dom.set_ylim([0, 100])
ax_dom.grid(axis='y', alpha=0.3)
ax_dom.legend(fontsize=12)

# Overall title
fig.suptitle('Context-Adaptive Spike Encoding: Emergent Hierarchical Specialization', 
             fontsize=20, fontweight='bold', y=0.98)

# Add caption
caption = (
    "Figure 1: Our context-adaptive spike encoding discovers a hierarchical strategy without explicit supervision. "
    "Layer 0 specializes in rate encoding (85.6% usage, entropy=0.59), while Layers 1-3 maintain maximum encoding "
    "diversity (entropy=1.609, 100% of theoretical maximum). This emergent pattern suggests that early layers benefit "
    "from consistent encoding while deeper layers require adaptive flexibility for semantic processing."
)

fig.text(0.5, 0.02, caption, ha='center', fontsize=11, style='italic', wrap=True)

plt.savefig('results/plots/figure1_encoding_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('results/plots/figure1_encoding_analysis.pdf', bbox_inches='tight')
print("✓ Saved Figure 1 (PNG and PDF)")
plt.show()