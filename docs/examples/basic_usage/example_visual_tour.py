r"""
Visual tour of curvature matrices
=================================

This tutorial visualizes different curvature matrices for a model with
sufficiently small parameter space.

First, the imports.
"""

import numpy
import torch
from torch import nn

from curvlinops import GGNLinearOperator, HessianLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Setup
# -----
#
# We will create a synthetic classification task, a small CNN, and use
# cross-entropy error as loss function.

num_data = 100
batch_size = 16
in_channels = 3
in_features_shape = (in_channels, 10, 10)
num_classes = 5

# dataset
dataset = torch.utils.data.TensorDataset(
    torch.rand(num_data, *in_features_shape),  # X
    torch.randint(size=(num_data,), low=0, high=num_classes),  # y
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# model
model = nn.Sequential(
    nn.Conv2d(in_channels, 5, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(5, 5, 5, padding=2, stride=2),
    nn.Sigmoid(),
    nn.Conv2d(5, 1, 3, padding=1),
    nn.Flatten(),
    nn.Linear(25, num_classes),
).to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]

loss_function = nn.CrossEntropyLoss(reduction="mean").to(DEVICE)

print("Total parameters:", sum(p.numel() for p in params))
