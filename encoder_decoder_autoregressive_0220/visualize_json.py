import numpy as np
import json
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from typing import List, Dict

def visualize_task(
        task: List[Dict[str, List]],
        name: str = "",
        out_path: str = "temp.jpg",
    ) -> None:
    # do some parsing
    grids_row1 = []
    for t in task[:-1]:
        grids_row1.append(t["input"])
        grids_row1.append(t["output"])
    grids_row2 = [task[-1]["input"], task[-1]["output"]]

    color_map_list = [
        "#000000",  # 0: black
        "#FF5733",  # 1: orange-red
        "#33FF57",  # 2: bright green
        "#3357FF",  # 3: bright blue
        "#FFFF33",  # 4: yellow
        "#FF33FF",  # 5: magenta
        "#33FFFF",  # 6: cyan
        "#FF8C00",  # 7: dark orange
        "#8A2BE2",  # 8: blue-violet
        "#FF1493",  # 9: deep pink
    ]
    cmap = ListedColormap(color_map_list)
    n_cols = max(len(grids_row1), len(grids_row2))
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 8))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    # Plot each grid in the first row
    for i, grid in enumerate(grids_row1):
        ax = axes[0, i]
        ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=9)
        ax.axis("off")
    if len(grids_row1) < n_cols:
        for j in range(len(grids_row1), n_cols):
            axes[0, j].axis("off")
    # Plot each grid in the second row
    for i, grid in enumerate(grids_row2):
        ax = axes[1, i]
        ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=9)
        ax.axis("off")
    if len(grids_row2) < n_cols:
        for j in range(len(grids_row2), n_cols):
            axes[1, j].axis("off")
    # format
    fig.suptitle(name, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


<<<<<<< HEAD:encoder_decoder_autoregressive_0212/visualize_json.py
# x = json.load(open("/scratch/zy3101/re-arc/train_data/tasks/99fa7670.json", 'r'))
# visualize_task(x[:10], out_path="temp1.jpg")

# x = json.load(open("/scratch/zy3101/re-arc/arc_original/training/99fa7670.json", 'r'))
# visualize_task(x["train"] + x["test"], out_path="temp2.jpg")

for i in range(1, 5):
    x = json.load(open(f"/scratch/zy3101/ConceptARC/corpus/Count/Count{i}.json", 'r'))
=======
# x = json.load(open("./data/re-arc/train_data/tasks/99fa7670.json", 'r'))
# visualize_task(x[:10], out_path="temp1.jpg")

# x = json.load(open("./data/re-arc/arc_original/training/99fa7670.json", 'r'))
# visualize_task(x["train"] + x["test"], out_path="temp2.jpg")

for i in range(1, 5):
    x = json.load(open(f"./data/ConceptARC/corpus/Count/Count{i}.json", 'r'))
>>>>>>> origin/main:encoder_decoder_autoregressive_0220/visualize_json.py
    visualize_task(x["train"] + x["test"], out_path=f"temp{i}.jpg")
