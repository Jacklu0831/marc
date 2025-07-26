import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# Data for the scatter plot
methods = [
    "Zero-shot",
    "In-Context Learning",
    "Prompt-Tuning",
    "Prefix-Tuning",
    "Test-Time Training",
    "CT-Prompt",
    "CT-KV",
    "Test-Time Training + CT-KV"
]
training_times = [0, 0, 147, 123, 342, 228, 145, 372]  # seconds per task
performances = [34.9, 35.6, 41.4, 42.0, 44.1, 43.2, 44.4, 47.3]  # performance %


# Identify the three methods for fitting
# fit_methods = ["In-Context Learning", "Prefix-Tuning", "Test-Time Training"]
# x_fit = [0, 123, 342, 250, max(training_times) + 20]
# y_fit = [37.1, 44.5, 44, 43.75, 46.7]
x_fit = [0, 123, 250, 400]
y_fit = [38, 44, 43.3, 45.3]
coeffs = np.polyfit(x_fit, y_fit, 2)
poly = np.poly1d(coeffs)
x_line = np.linspace(min(training_times), max(training_times), 300)
y_line = poly(x_line)


# Create scatter plot
plt.figure(figsize=(8, 5))

# Plot non-our methods with blue circle markers
for name, x, y in zip(methods, training_times, performances):
    if name not in ["CT-Prompt", "CT-KV", "Test-Time Training + CT-KV"]:
        plt.scatter(x, y, marker='o', s=80, color='blue')

# Plot our methods with red star markers
for name, x, y in zip(methods, training_times, performances):
    if name in ["CT-Prompt", "CT-KV", "Test-Time Training + CT-KV"]:
        plt.scatter(x, y, marker='*', s=200, color='red')

# Define text offsets
offsets = {
    "Zero-shot": (12, -0.2),
    "In-Context Learning": (12, 0),
    "Prompt-Tuning": (12, -0.5),
    "Prefix-Tuning": (12, -0.05),
    "Test-Time Training": (-88, -1.4),
    "CT-Prompt": (5, 0.55),
    "CT-KV": (-35, 0.7),
    "Test-Time Training + CT-KV": (-234, -0.6),
}
bold = {"CT-KV", "Test-Time Training + CT-KV"}

# Annotate all points with adjusted offsets
for name, x, y in zip(methods, training_times, performances):
    dx, dy = offsets[name]
    if name in bold:
        plt.annotate(name, (x + dx, y + dy), fontsize=18, fontweight='bold')
    else:
        plt.annotate(name, (x + dx, y + dy), fontsize=18)

# Overlay the fitted curve
plt.plot(x_line, y_line, linestyle='--', color='black', alpha=0.6, label='Adaptation Fit')

# Labels, title, and grid
plt.xlabel("Training Time per Task from NLP-LR (seconds)", fontsize=18, labelpad=7)
plt.ylabel("Accuracy (%)", fontsize=18, labelpad=5)
# plt.title("Accuracy vs Training Time on NLP-LR Tasks", fontsize=14, pad=10)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Save to files
plt.savefig('plots/scatter.pdf', dpi=200)
plt.savefig('plots/scatter.png', dpi=200)
plt.close()
