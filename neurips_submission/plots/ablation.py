import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# Real data
means = np.array([
    [40.9691206, 43.63046277, 43.8595757, 44.1897189], # nlp
    [51.41051925, 54.37509214, 55.31913423, 57.8510169], # bbh
    [40.1940164, 41.5289663, 42.66715723, 43.71338979], # mmlu
    [21.00, 23.75, 21.00, 22.5], # arc
])
stds = np.array([
    [0.75, 0.45, 0.57, 0.55],
    [0.76, 0.88, 0.72, 0.78],
    [0.73, 0.65, 0.62, 0.54],
])
ylims = [48.6897189, 60.8510169, 46.71338979, 25.75]

labels = ['neither', 'no leave-one-out', 'no token-dropout', 'both']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
titles = ['NLP-LR', 'BBH', 'MMLU', 'ARC']

# Fontsize settings
title_fs = 16
ylabel_fs = 14
ytick_fs = 12
label_fs = 12
legend_fs = 16

# Create subplots without shared y-axis scales
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
x = np.arange(len(labels))
width = 0.6

for i, ax in enumerate(axes):
    if i < 3:
        bars = ax.bar(
            x, means[i], yerr=stds[i], capsize=5,
            color=colors, alpha=0.6, width=width
        )
    else:
        bars = ax.bar(
            x, means[i],
            color=colors, alpha=0.6, width=width
        )
    # Add data labels on top of each bar
    for j, bar in enumerate(bars):
        height = bar.get_height()
        error = stds[i][j] if i < 3 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + error + 0.5,
            f'{height:.1f}',
            ha='center', va='bottom',
            fontsize=label_fs
        )
    # Titles and labels
    ax.set_title(titles[i], fontsize=title_fs)
    if i == 0 or i == 2:
        ax.set_ylabel('Accuracy (%)', fontsize=ylabel_fs)
    ax.tick_params(axis='y', labelsize=ytick_fs)
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0, ylims[i])

# Single legend for all subplots
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in colors]
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=legend_fs)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# Save and close
plt.savefig('plots/ablation.pdf', dpi=200)
plt.savefig('plots/ablation.png', dpi=200)
plt.close()
