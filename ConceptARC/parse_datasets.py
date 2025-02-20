import copy
from multiprocessing import Pool
from datasets import load_dataset
import string
from pathlib import Path
import os
import json
import glob
import random

random.seed(0)

conceptarc_output_dir = "/scratch/yl11330/ConceptARC/concept_arc_parsed"
archeavy_task_ids_path = "/scratch/yl11330/ConceptARC/arc_heavy_task_ids.txt"
os.system(f"rm -rf {conceptarc_output_dir} {archeavy_task_ids_path}")
os.makedirs(conceptarc_output_dir)

# get 800 existing task ids from arc train and re-arc and arc eval
arc_original_training_tasks = "/scratch/yl11330/re-arc/arc_original/training"
arc_original_evaluation_tasks = "/scratch/yl11330/re-arc/arc_original/evaluation"
re_arc_tasks = "/scratch/yl11330/re-arc/train_data/tasks"

arc_original_training_ids = [Path(p).stem for p in glob.glob(f"{arc_original_training_tasks}/*.json")]
arc_original_evaluation_ids = [Path(p).stem for p in glob.glob(f"{arc_original_evaluation_tasks}/*.json")]
re_arc_task_ids = [Path(p).stem for p in glob.glob(f"{re_arc_tasks}/*.json")]
assert len(arc_original_training_ids) == 400 and len(set(arc_original_training_ids)) == len(arc_original_training_ids)
assert len(arc_original_evaluation_ids) == 400 and len(set(arc_original_evaluation_ids)) == len(arc_original_evaluation_ids)
assert set(re_arc_task_ids) == set(arc_original_training_ids)
existing_ids = set(arc_original_training_ids).union(set(arc_original_evaluation_ids))
assert len(existing_ids) == 800


class TaskIDGenerator:
    def __init__(self, task_ids):
        assert isinstance(task_ids, set)
        self.task_ids = task_ids
        self.characters = string.digits + "".join(['a', 'b', 'c', 'd', 'e', 'f'])

    def get_id(self):
        conflict = True
        new_id = None
        while conflict:
            new_id = ''.join(random.choice(self.characters) for _ in range(8))
            if new_id not in self.task_ids:
                self.task_ids.add(new_id)
                conflict = False
        return new_id

generator = TaskIDGenerator(copy.deepcopy(existing_ids))

# parse concept arc
concept_arc_files = glob.glob(f'corpus/*/*.json')
concept_arc_files += glob.glob(f'MinimalTasks/*.json')
print(len(concept_arc_files), 'concept arc json tasks')
concept_arc_task_ids = [generator.get_id() for _ in range(len(concept_arc_files))]
for task_id, file in zip(concept_arc_task_ids, concept_arc_files):
    task = json.load(open(file, 'r'))
    assert set(task.keys()) == set(["train", "test"])
    pairs = task["train"] + task["test"]
    with open(f"{conceptarc_output_dir}/{task_id}.json", 'w') as f:
        json.dump(pairs, f)
print('done saving concept arc')
assert len(generator.task_ids) == len(existing_ids) + len(concept_arc_files), len(generator.task_ids)


# parse heavy arc (nevermind, toooooooo many files)
heavy_arc = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")["train"]
print(len(heavy_arc), 'tasks')

# def save_heavy_arc(idx):
#     def float2dgrid_to_int(grid):
#         int_grid = []
#         for row in grid:
#             assert all(x in set(range(10)) for x in row)
#             int_grid.append([int(x) for x in row])
#         return int_grid

#     data = heavy_arc[idx]['examples']
#     task_id = arc_heavy_task_ids[idx]
#     parsed = []
#     for d in data:
#         assert len(d) == 2
#         parsed.append({
#             "input": float2dgrid_to_int(d[0]),
#             "output":float2dgrid_to_int(d[1]),
#         })
#     with open(f"{archeavy_output_dir}/{task_id}.json", 'w') as f:
#         json.dump(pairs, f)

arc_heavy_task_ids = [generator.get_id() for _ in range(len(heavy_arc))]
with open(archeavy_task_ids_path, 'a') as f:
    for task_id in arc_heavy_task_ids:
        f.write(f"{task_id}\n")

assert len(generator.task_ids) == len(existing_ids) + len(concept_arc_files) + len(arc_heavy_task_ids), len(generator.task_ids)