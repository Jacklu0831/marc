import torch
import pickle

# with open('train_old.pkl', 'rb') as f:
#     old = pickle.load(f).parsed_data
# with open('train_new.pkl', 'rb') as f:
#     new = pickle.load(f).parsed_data
with open('eval_old.pkl', 'rb') as f:
    old = pickle.load(f).parsed_data
with open('eval_new.pkl', 'rb') as f:
    new = pickle.load(f).parsed_data

for x in old:
    x['task_id'] = x['task_id'].replace('_', '-')

assert len(old) == len(new)
for x1, x2 in zip(old, new):
    assert set(x1.keys()) == set(x2.keys())
    for k in x1:
        v1 = x1[k]
        v2 = x2[k]
        if isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2)
        else:
            assert v1 == v2, (v1, v2)
print('same')
