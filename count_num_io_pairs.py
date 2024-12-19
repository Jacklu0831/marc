import os
import json
import matplotlib.pyplot as plt

json_path = 'kaggle_dataset/arc-agi_evaluation_challenges.json'
json_path = 'kaggle_dataset/arc-agi_training_challenges.json'

tasks = json.load(open(json_path, 'r'))
min_count, max_count = 1000, 0
counts = []
for t in tasks.values():
    count = len(t['train'])
    min_count = min(min_count, count)
    max_count = max(max_count, count)
    counts.append(count)
print('min', min_count)
print('max', max_count)

plt.hist(counts)
plt.savefig('temp.jpg')
plt.close()