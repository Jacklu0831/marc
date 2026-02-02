import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data from your tables
# -------------------------
# Table 3: Corruption probabilities (p)
p = np.array([0, 25, 50, 75, 100])
icl_p_nlplr    = np.array([35.6, 36.1, 35.4, 34.7, 31.7])
prefix_p_nlplr = np.array([42.0, 41.2, 38.7, 34.9, 31.4])
ctkv_p_nlplr   = np.array([44.2, 42.2, 39.9, 35.6, 31.4])

icl_p_mmlu     = np.array([41.2, 41.3, 40.7, 40.4, 40.2])
prefix_p_mmlu  = np.array([39.9, 39.4, 38.7, 38.3, 37.4])
ctkv_p_mmlu    = np.array([43.7, 42.5, 41.6, 40.5, 39.1])

# -------------------------
# Plot aesthetics
# -------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 140,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
plt.tight_layout(pad=1.75)

markers = dict(icl='o', prefix='o', ctkv='o')
alpha_pt = 0.7
ms = 6

def style_axes(ax, show_ylabel=False):
    ax.grid(True, linewidth=0.5, alpha=0.4)
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=18)

# Small x-offsets to reduce point overlap for identical x locations
def offsets(n, width=0.35):
    base = np.linspace(-width, width, n)
    return base

xofs3 = offsets(3, width=0.25)

# -------------------------
# Panel 1: NLP-LR vs corruption p (Table 3, left)
# -------------------------
ax = axes[0]
ax.plot(p + xofs3[0], icl_p_nlplr,     marker=markers['icl'],    markersize=6, alpha=alpha_pt, label="ICL")
ax.plot(p + xofs3[1], prefix_p_nlplr,  marker=markers['prefix'], markersize=6, alpha=alpha_pt, label="Prefix Tuning (m=32)")
ax.plot(p + xofs3[2], ctkv_p_nlplr,    marker=markers['ctkv'],   markersize=6, alpha=alpha_pt, label="CT-KV")
ax.set_title("NLP-LR", fontsize=18)
ax.set_xlabel("Corruption Probability $p$ (%)", fontsize=18)
style_axes(ax, show_ylabel=True)
legend = ax.legend(loc="upper right", frameon=False, ncols=1)
for text in legend.get_texts():
    text.set_fontweight('bold')

# -------------------------
# Panel 2: MMLU vs corruption p (Table 3, right)
# -------------------------
ax = axes[1]
ax.plot(p + xofs3[0], icl_p_mmlu,     marker=markers['icl'],    markersize=6, alpha=alpha_pt)
ax.plot(p + xofs3[1], prefix_p_mmlu,  marker=markers['prefix'], markersize=6, alpha=alpha_pt)
ax.plot(p + xofs3[2], ctkv_p_mmlu,    marker=markers['ctkv'],   markersize=6, alpha=alpha_pt)
ax.set_title("MMLU", fontsize=18)
ax.set_xlabel("Corruption Probability $p$ (%)", fontsize=18)
style_axes(ax, show_ylabel=False)

# -------------------------
# Floating panel tag (b)
# -------------------------
fig.text(0.005, 0.98, "(b)", ha="left", va="top", fontsize=22, fontweight='bold')

plt.savefig('scatterplot_part2.png', dpi=200)