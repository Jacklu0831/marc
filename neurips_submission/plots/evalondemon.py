import matplotlib.pyplot as plt

# Data (0–100 scale)
datasets = ['ARC', 'MetaICL-LR', 'BBH', 'MMLU']
accuracies = [22.61, 81.92, 84.13, 89.34]  # replace these with your accuracy values

fontsize = 15

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(datasets, accuracies, color='skyblue')

# Y-axis label with larger font
ax.set_ylabel('Accuracy (%)', fontsize=fontsize)

# Increase tick label fontsize
ax.tick_params(axis='both', which='major', labelsize=fontsize)

# Annotate bars with large annotations
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=fontsize)

# Set limits and layout
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('plots/evalondemon.pdf', dpi=200)
plt.savefig('plots/evalondemon.png', dpi=200)
plt.close()