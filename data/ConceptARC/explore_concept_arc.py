import json
import glob

files = glob.glob(f'corpus/*/*.json')
files += glob.glob(f'MinimalTasks/*.json')
print(len(files), 'json')

train_num_io = []
test_num_io = []
total_num_io = []
grid_dims = []
for file in files:
    task = json.load(open(file, 'r'))
    assert set(task.keys()) == set(["train", "test"])
    train_num_io.append(len(task["train"]))
    test_num_io.append(len(task["test"]))
    total_num_io.append(len(task["train"]) + len(task["test"]))
    for pair in task["train"] + task["test"]:
        inp, out = pair["input"], pair["output"]
        grid_dims += [len(inp), len(inp[0]), len(out), len(out[0])]

print('num train pair min/max', min(train_num_io), max(train_num_io))
print('num test pair min/max', min(test_num_io), max(test_num_io))
print('num total pair min/max', min(total_num_io), max(total_num_io))
print('grid dim min/max', min(grid_dims), max(grid_dims))

# 176 json
# num train pair min/max 1 5
# num test pair min/max 3 3
# num total pair min/max 4 8
# grid dim min/max 1 25