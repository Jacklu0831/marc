import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os


color_map_list = [
    "#000000", # 0: black
    "#0074D9", # 1: blue # correct
    "#FF4136", # 2: red
    "#2ECC40", # 3: green # correct
    "#FFDC00", # 4: yellow
    "#AAAAAA", # 5: grey
    "#F012BE", # 6: fuchsia
    "#FF851B", # 7: orange
    "#7FDBFF", # 8: teal
    "#870C25", # 9: brown
    "#ffffff", # 10: white
]


icl_paths = [
    'encoder_decoder/outputs_eval/eval_arc_part1_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_part2_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_part3_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_part4_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_part5_0317_noprogram_base/eval_pred_gt.json',
]
ct_paths = [
    'encoder_decoder/outputs_eval/eval_arc_gs100_lr3e-3_dropnone_tokendrop0.1_part1_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_gs100_lr3e-3_dropnone_tokendrop0.1_part2_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_gs100_lr3e-3_dropnone_tokendrop0.1_part3_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_gs100_lr3e-3_dropnone_tokendrop0.1_part4_0317_noprogram_base/eval_pred_gt.json',
    'encoder_decoder/outputs_eval/eval_arc_gs100_lr3e-3_dropnone_tokendrop0.1_part5_0317_noprogram_base/eval_pred_gt.json',
]

def solved_tasks(paths):
    solved_tasks = set()
    for p in paths:
        obj = json.load(open(p, 'r'))
        task_names = set(t.split('-')[0] for t in obj)
        total_solved = 0
        for t in task_names:
            results = [v for o, v in obj.items() if o.startswith(t)]
            assert len(results) > 0
            if all(x[0][0] == x[0][1] for x in results):
                solved_tasks.add(t)
    return solved_tasks

icl_solved_tasks = solved_tasks(icl_paths)
ct_solved_tasks = solved_tasks(ct_paths)
print(len(icl_solved_tasks))
print(len(ct_solved_tasks))

all_tasks = set(p.split('.')[0] for p in os.listdir('data/re-arc/arc_original/evaluation'))
all_tasks.difference(icl_solved_tasks.union(ct_solved_tasks))

print('solved->solved', len(icl_solved_tasks.intersection(ct_solved_tasks)))
print('solved->unsolved', len(icl_solved_tasks.difference(ct_solved_tasks)))
print('unsolved->solved', len(ct_solved_tasks.difference(icl_solved_tasks)))
print('unsolved->unsolved', len(all_tasks.difference(icl_solved_tasks.union(ct_solved_tasks))))

# total acc: 13.25 -> 23.75 (53, 95)
# solved->solved 44
# solved->unsolved 9
# unsolved->solved 51
# unsolved->unsolved 296


def visualize_grid(data, path, cell_size=1.0, dpi=200):
    # 1. Numeric array & dimensions
    grid = np.array(data, dtype=int)
    rows, cols = grid.shape

    # 2. Colormap
    cmap = ListedColormap(color_map_list)

    # 3. Figure sized so that
    #    width_in = cols * cell_size
    #    height_in = rows * cell_size
    fig, ax = plt.subplots(
        figsize=(cols * cell_size, rows * cell_size),
        dpi=dpi
    )

    ax.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=9,
        interpolation='nearest'
    )

    # 4. Strip away axes & padding
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 5. Save & clean up
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def str_to_2dlist(s):
    grid = [[int(y) for y in list(row)] for row in s.split('\n')]
    grid = grid[1:] # no dim
    assert not (len(grid) == 0 or any(len(r) == 0 for r in grid))
    assert len(set(len(r) for r in grid)) == 1
    return grid



# visualize_grid([
#     [3, 8, 3, 3],
#     [3, 4, 3, 0],
#     [0, 4, 0, 0],
# ], path='temp.jpg')

# exit()


# visualize!
icl_only_tasks = icl_solved_tasks.difference(ct_solved_tasks)
ct_only_tasks = ct_solved_tasks.difference(icl_solved_tasks)

all_icl_pred_gt = {}
for path in icl_paths:
    all_icl_pred_gt.update(json.load(open(path, 'r')))
all_ct_pred_gt = {}
for path in ct_paths:
    all_ct_pred_gt.update(json.load(open(path, 'r')))

# filter out tasks with more than one

os.makedirs('plots/icl_only_tasks/', exist_ok=True)
os.makedirs('plots/ct_only_tasks/', exist_ok=True)

# for task in icl_only_tasks:
#     task_dir = f'plots/icl_only_tasks/{task}'

#     icl_pred_gt = [pred_gt for task_name, pred_gt in all_icl_pred_gt.items() if task_name.startswith(task)]
#     ct_pred_gt = [pred_gt for task_name, pred_gt in all_ct_pred_gt.items() if task_name.startswith(task)]
#     if len(icl_pred_gt) > 1 or len(ct_pred_gt) > 1:
#         continue
#     assert len(icl_pred_gt[0]) == 1 and len(ct_pred_gt[0]) == 1
#     icl_pred, icl_gt = icl_pred_gt[0][0]
#     ct_pred, ct_gt = ct_pred_gt[0][0]
#     try:
#         icl_pred, icl_gt = str_to_2dlist(icl_pred), str_to_2dlist(icl_gt)
#         ct_pred, ct_gt = str_to_2dlist(ct_pred), str_to_2dlist(ct_gt)
#     except:
#         continue
#     assert icl_gt == ct_gt

#     assert icl_pred == icl_gt
#     assert ct_pred != ct_gt

#     original_data = json.load(open(f'data/re-arc/arc_original/evaluation/{task}.json', 'r'))
#     assert len(original_data['test']) == 1
#     assert original_data['test'][0]['output'] == icl_gt

#     # demon
#     os.makedirs(task_dir, exist_ok=True)
#     for train_i, grid in enumerate(original_data['train']):
#         visualize_grid(grid['input'], f'{task_dir}/train{train_i}_input')
#         visualize_grid(grid['output'], f'{task_dir}/train{train_i}_output')

#     # test
#     visualize_grid(original_data['test'][0]['input'], f'{task_dir}/test_input')
#     visualize_grid(icl_pred, f'{task_dir}/icl_pred')
#     visualize_grid(ct_pred, f'{task_dir}/ct_pred')

for task in ct_only_tasks:
    task_dir = f'plots/ct_only_tasks/{task}'

    ct_pred_gt = [pred_gt for task_name, pred_gt in all_ct_pred_gt.items() if task_name.startswith(task)]
    icl_pred_gt = [pred_gt for task_name, pred_gt in all_icl_pred_gt.items() if task_name.startswith(task)]
    if len(ct_pred_gt) > 1 or len(icl_pred_gt) > 1:
        continue
    assert len(ct_pred_gt[0]) == 1 and len(icl_pred_gt[0]) == 1
    ct_pred, ct_gt = ct_pred_gt[0][0]
    icl_pred, icl_gt = icl_pred_gt[0][0]
    try:
        ct_pred, ct_gt = str_to_2dlist(ct_pred), str_to_2dlist(ct_gt)
        icl_pred, icl_gt = str_to_2dlist(icl_pred), str_to_2dlist(icl_gt)
    except:
        continue
    assert ct_gt == icl_gt

    assert ct_pred == ct_gt
    assert icl_pred != icl_gt

    original_data = json.load(open(f'data/re-arc/arc_original/evaluation/{task}.json', 'r'))
    assert len(original_data['test']) == 1
    assert original_data['test'][0]['output'] == ct_gt

    # demon
    os.makedirs(task_dir, exist_ok=True)
    for train_i, grid in enumerate(original_data['train']):
        visualize_grid(grid['input'], f'{task_dir}/train{train_i}_input')
        visualize_grid(grid['output'], f'{task_dir}/train{train_i}_output')

    # test
    visualize_grid(original_data['test'][0]['input'], f'{task_dir}/test_input')
    visualize_grid(ct_pred, f'{task_dir}/ct_pred')
    visualize_grid(icl_pred, f'{task_dir}/ct_pred')
