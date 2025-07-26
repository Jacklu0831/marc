# test how fast cpu can generate data

import time
import torch
from oracle_fit import create_ground_truth_net

N = 10000

start = time.time()
for _ in range(N):
    ground_truth_net = create_ground_truth_net(20, 100)
    X = torch.randn(101, 20)
    Y = ground_truth_net(X)
print(time.time() - start)
breakpoint()