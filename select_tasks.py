import json
import csv

csv_path = 'task_info_selected.csv'
inp_path = 'kaggle_dataset/arc-agi_evaluation_challenges.json'
out_path = 'kaggle_dataset/arc-agi_evaluation_challenges_selected.json'
inp_path = 'kaggle_dataset/arc-agi_evaluation_solutions.json'
out_path = 'kaggle_dataset/arc-agi_evaluation_solutions_selected.json'

with open(csv_path, mode='r') as file:
    csv_reader = csv.reader(file)
    data_as_tuples = [tuple(row) for row in csv_reader]
    data_as_tuples = data_as_tuples[1:] # first row contains col names
    select_task_ids = [d[0] for d in data_as_tuples]
    assert len(select_task_ids) == len(set(select_task_ids))

with open(inp_path, 'r') as file:
    all_data = json.load(file)

all_data = {task_id: data for task_id, data in all_data.items() if task_id in set(select_task_ids)}
with open(out_path, 'w') as file:
    json.dump(all_data, file)
