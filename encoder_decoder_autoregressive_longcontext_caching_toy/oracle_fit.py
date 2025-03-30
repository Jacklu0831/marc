import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Two-layer network architecture (no biases)
class TwoLayerNet(nn.Module):
    def __init__(self, d, r):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(d, r, bias=False)
        self.fc2 = nn.Linear(r, 1, bias=False)

    def forward(self, X):
        hidden = torch.relu(self.fc1(X))
        return self.fc2(hidden)


def create_ground_truth_net(d, r, generator) -> TwoLayerNet:
    # Create a TwoLayerNet instance for ground truth and initialize it accordingly.
    net = TwoLayerNet(d, r)
    with torch.no_grad():
        # Initialize fc1 weights from N(0,I)
        net.fc1.weight.copy_(torch.randn(r, d, generator=generator))
        # Initialize fc2 weights from N(0, 2/r)
        net.fc2.weight.copy_(torch.randn(1, r, generator=generator) * (2 / r) ** 0.5)
    # Freeze the parameters
    for param in net.parameters():
        param.requires_grad = False
    return net


def run_trial(num_train=100, d=20, r=100, num_train_steps=5000, lr=5e-3, batch_size=10):
    device = torch.device('cuda')

    # Create the ground truth network instance and generate data from it.
    ground_truth_net = create_ground_truth_net(d, r, None).to(device)

    # Total examples: num_train for training + 1 for testing
    k_total = num_train + 1
    X = torch.randn(k_total, d, device=device)
    with torch.no_grad():
        Y = ground_truth_net(X)  # [k_total, 1]

    # Split data: first num_train for training, last one for testing.
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_test = X[num_train].unsqueeze(0)  # [1, d]
    Y_test = Y[num_train]

    # Initialize a fresh TwoLayerNet for training.
    model = TwoLayerNet(d, r).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # type: ignore
    loss_fn = nn.MSELoss()

    # Training loop using mini-batches of size 'batch_size'.
    for _ in range(num_train_steps):
        indices = random.sample(range(num_train), batch_size)
        optimizer.zero_grad()
        loss = loss_fn(model(X_train[indices]), Y_train[indices])
        loss.backward()
        optimizer.step()

    # Evaluate on the held-out example; compute squared error normalized by input dimension.
    with torch.no_grad():
        error = (model(X_test) - Y_test) ** 2 / d
    return error.item()


def evaluate_trials(num_trials=1280):
    errors = []
    for _ in range(num_trials):
        err = run_trial()
        errors.append(err)
        print(np.mean(errors))
        # 0.16 - 0.18 range
    return np.mean(errors), np.std(errors)


if __name__ == "__main__":
    mean_error, std_error = evaluate_trials(num_trials=1280)
    print(f"Mean normalized squared error: {mean_error:.4f} +/- {std_error:.4f}")
