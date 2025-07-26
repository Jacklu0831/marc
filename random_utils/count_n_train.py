from collections import Counter
import json
from glob import glob

n_train = []
json_files = glob("data/re-arc/arc_original/evaluation/*.json")
for f in json_files:
    j = json.load(open(f, 'r'))
    n_train.append(len(j['train']))
print(dict(Counter(n_train)))