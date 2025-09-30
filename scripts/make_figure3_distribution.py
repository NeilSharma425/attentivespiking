"""
Figure 3: Layer-wise spike distribution showing efficiency source.
"""

import matplotlib.pyplot as plt
import numpy as np

# Your data
layers = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3']
spike_counts = [28252, 7375, 7163, 7159]
sparsity_attn = [53.7, 86.8, 86.3, 86.3]
sparsity_ffn = [85.0, 97.2, 98.1, 98.2]

fig = plt.figure(figsize=(16, 5))

# Subplot 1: Spike count by layer
ax1 = plt.subplot(131)
bars = ax1.bar(layers, spike_counts, color=['red', 'green', 'green', 'green'], 
               alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Spikes', fontsize=12, fontweight='bold')
ax1.set_title('Spike Distribution by Layer', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add percentages
for i, (bar, count) in enumerate(zip(bars, spike_counts)):
    pct = count / sum(spike_counts) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, count + 1000, 
             f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Subplot 2: Sparsity comparison
ax2 = plt.subplot(132)
x = np.arange(len(layers))
width = 0.35

bars1 = ax2.bar(x - width/2, sparsity_attn, width, label='Attention', 
                color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, sparsity_ffn, width, label='FFN',
                color='lightcoral', edgecolor='black', linewidth=1.5)

ax2.set_ylabel('Sparsity (%)', fontsize=12, fontweight='bold')
ax2.set_title('Sparsity by Component', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(layers)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 100])

# Subplot 3: Encoding distribution
ax3 = plt.subplot(133)
encoding_data = {
    'Layer 0': [79.0, 0.4, 0.6, 0.7, 19.3],
    'Layer 1': [20.1, 20.5, 19.9, 19.9, 19.7],
    'Layer 2': [20.0, 20.6, 20.3, 19.8, 19.3],
    'Layer 3': [19.8, 20.5, 20.2, 20.0, 19.5],
}

encoding_names = ['Rate', 'Temporal', 'Pop', 'Burst', 'Adaptive']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

x = np.arange(len(layers))
bottom = np.zeros(len(layers))

for i, enc_name in enumerate(encoding_names):
    values = [encoding_data[layer][i] for layer in layers]
    ax3.bar(x, values, bottom=bottom, label=enc_name, color=colors[i], 
            edgecolor='white', linewidth=0.5)
    bottom += values

ax3.set_ylabel('Encoding Distribution (%)', fontsize=12, fontweight='bold')
ax3.set_title('Encoding Selection by Layer', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(layers)
ax3.legend(fontsize=9, loc='upper right')
ax3.set_ylim([0, 100])

plt.suptitle('Adaptive Encoding: Hierarchical Specialization Drives Efficiency', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/figure3_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 3 saved")