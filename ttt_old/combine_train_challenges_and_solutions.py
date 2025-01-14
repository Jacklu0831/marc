import json


challenges_path = "kaggle_dataset/arc-agi_training_challenges.json"
solutions_path = "kaggle_dataset/arc-agi_training_solutions.json"
combined_path = "kaggle_dataset/arc-agi_training_combined.json"

challenges = json.load(open(challenges_path, 'r'))
solutions = json.load(open(solutions_path, 'r'))
assert challenges.keys() == solutions.keys()

for i in challenges:
    c = challenges[i]
    s = solutions[i]
    assert len(c['test']) == len(s) # a solution for each test
    for query, answer in zip(c['test'], s):
        challenges[i]['train'].append({'input': query['input'], 'output': answer})

json.dump(challenges, open(combined_path, 'w'))
