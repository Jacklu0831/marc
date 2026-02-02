import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data from your tables
# -------------------------
# Table 2: Few-shot examples (k)
k_nlplr = np.array([8, 16, 24, 32])
k_mmlu  = np.array([16, 32, 48, 64])
icl_k_nlplr   = np.array([37.8, 37.1, 35.6, 36.1])
prefix_k_nlplr= np.array([40.5, 41.7, 42.0, 44.0])
ctkv_k_nlplr  = np.array([43.1, 45.5, 46.6, 48.9])

icl_k_mmlu    = np.array([43.3, 43.4, 43.8, 44.0])
prefix_k_mmlu = np.array([42.4, 43.0, 44.1, 44.2])
ctkv_k_mmlu   = np.array([45.5, 45.8, 46.3, 47.6])

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
# Panel 1: NLP-LR vs k (Table 2, left)
# -------------------------
ax = axes[0]
ax.plot(k_nlplr + xofs3[0], icl_k_nlplr,   marker=markers['icl'],    markersize=6, alpha=alpha_pt, label="ICL")
ax.plot(k_nlplr + xofs3[1], prefix_k_nlplr,marker=markers['prefix'], markersize=6, alpha=alpha_pt, label="Prefix Tuning (m=32)")
ax.plot(k_nlplr + xofs3[2], ctkv_k_nlplr,  marker=markers['ctkv'],   markersize=6, alpha=alpha_pt, label="CT-KV")
ax.set_title("NLP-LR", fontsize=18)
ax.set_xlabel("# Demonstration Examples", fontsize=18)
style_axes(ax, show_ylabel=True)
legend = ax.legend(loc="upper left", frameon=False, ncols=1)
for text in legend.get_texts():
    text.set_fontweight('bold')

# -------------------------
# Panel 2: MMLU vs k (Table 2, second)
# -------------------------
ax = axes[1]
ax.plot(k_mmlu + xofs3[0], icl_k_mmlu,    marker=markers['icl'],    markersize=6, alpha=alpha_pt)
ax.plot(k_mmlu + xofs3[1], prefix_k_mmlu, marker=markers['prefix'], markersize=6, alpha=alpha_pt)
ax.plot(k_mmlu + xofs3[2], ctkv_k_mmlu,   marker=markers['ctkv'],   markersize=6, alpha=alpha_pt)
ax.set_title("MMLU", fontsize=18)
ax.set_xlabel("# Demonstration Examples", fontsize=18)
style_axes(ax, show_ylabel=False)

# -------------------------
# Floating panel tag (a)
# -------------------------
fig.text(0.005, 0.98, "(a)", ha="left", va="top", fontsize=22, fontweight='bold')

plt.savefig('scatterplot_part1.png', dpi=200)