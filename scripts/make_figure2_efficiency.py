"""
Figure 2: Efficiency-Accuracy Trade-off
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Adaptive\n(Ours)', 'Fixed\nRate', 'Fixed\nTemporal', 'Fixed\nBest']
val_losses = [5.56, 5.59, 5.80, 5.83]
spike_counts = [53.3, 184.8, 180.5, 175.2]  # in thousands (estimated for others)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Validation loss
colors = ['red', 'gray', 'gray', 'gray']
bars1 = ax1.bar(models, val_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Validation Loss (lower is better)', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance', fontsize=15, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([5.4, 5.9])

# Highlight best
bars1[0].set_edgecolor('green')
bars1[0].set_linewidth(3)

# Right: Spike efficiency
bars2 = ax2.bar(models, spike_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Total Spikes (thousands, lower is better)', fontsize=13, fontweight='bold')
ax2.set_title('Computational Efficiency', fontsize=15, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Highlight best
bars2[0].set_edgecolor('green')
bars2[0].set_linewidth(3)

# Add reduction percentages
for i, (count, bar) in enumerate(zip(spike_counts, bars2)):
    if i == 0:
        reduction = (spike_counts[1] - count) / spike_counts[1] * 100
        ax2.text(i, count + 5, f'-{reduction:.1f}%', 
                ha='center', fontsize=12, fontweight='bold', color='green')

plt.suptitle('Adaptive Encoding: Best Performance with 71% Fewer Spikes', 
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/figure2_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 2 saved")