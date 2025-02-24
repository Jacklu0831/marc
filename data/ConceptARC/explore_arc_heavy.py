from multiprocessing import Pool
from datasets import load_dataset

dataset = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")["train"]
print(len(dataset), 'tasks')

colors = set(range(10))
def probe_dataset(data_i):
    grid_dims = []
    data = dataset[data_i]["examples"]
    for pair in data:
        assert len(pair) == 2
        for grid in pair:
            grid_dims.append(len(grid))
            grid_dims.append(len(grid[0]))
            assert set([item for sublist in grid for item in sublist]).issubset(colors), (data_i, grid)
    return len(data), min(grid_dims), max(grid_dims)


with Pool(16) as pool:
    results = pool.map(probe_dataset, list(range(len(dataset))))
num_pairs = [x[0] for x in results]
min_grid_dims = [x[1] for x in results]
max_grid_dims = [x[2] for x in results]
print('num pairs min/max', min(num_pairs), max(num_pairs))
print('grid dims min/max', min(min_grid_dims), max(max_grid_dims))

# num pairs min/max 4 59
# grid dims min/max 1 3672
