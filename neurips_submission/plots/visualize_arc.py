import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

color_map_list = [
    "#000000", # 0: black
    "#0074D9", # 1: blue
    "#FF4136", # 2: red
    "#2ECC40", # 3: green
    "#FFDC00", # 4: yellow
    "#AAAAAA", # 5: grey
    "#F012BE", # 6: fuchsia
    "#FF851B", # 7: orange
    "#7FDBFF", # 8: teal
    "#870C25", # 9: brown
    "#ffffff", # 10: white (background, unused if values 0–9)
]

def visualize_grid(data, path):
    """
    Render a pure color grid with no axes, legend, or margins.

    Args:
        data (list of list of int): 2D grid values 0–9.
        path (str): File path to save the image.
    """
    # Convert to integer NumPy array
    grid = np.array(data, dtype=int)

    # Create colormap for values 0–9
    cmap = ListedColormap(color_map_list)

    # Set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')

    # Remove axes, ticks, and margins
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save and close
    fig.savefig(path, dpi=200)
    plt.close(fig)


json_file = 'data/re-arc/arc_original/evaluation/0becf7df.json'
grids = json.load(open(json_file, 'r'))

for train_i, grid in enumerate(grids['train']):
    visualize_grid(grid['input'], f'plots/arc_example/train{train_i}_input')
    visualize_grid(grid['output'], f'plots/arc_example/train{train_i}_output')
visualize_grid(grids['test'][0]['input'], f'plots/arc_example/test_input')
visualize_grid(grids['test'][0]['output'], f'plots/arc_example/test_output')