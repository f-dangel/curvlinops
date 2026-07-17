"""
An attempt
==========

"""
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HEREDIR = Path.cwd()
print(HEREDIR)
TEST_PATH = HEREDIR / "assets" / "test_ggn_cifar10_vgg11bn.png"
TRAIN_PATH = HEREDIR / "assets" / "train_ggn_cifar10_vgg11bn.png"

plt.axis("off")
plt.imshow(mpimg.imread(TEST_PATH))


# %%
#
# Another image

# plt.axis("off")
# plt.imshow(mpimg.imread(TRAIN_PATH))

# %%
#
# TODO Below

from typing import List

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from curvlinops import GGNLinearOperator, LanczosApproximateLogSpectrumCached

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO Consider staring this
# https://github.com/chenyaofo/pytorch-cifar-models
model = (
    torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True
    )
    .eval()
    .to(device)
)
print(model)

loss_function = torch.nn.CrossEntropyLoss().to(device)

# # TODO Consider staring https://github.com/chenyaofo/image-classification-codebase
# # from https://github.com/chenyaofo/image-classification-codebase/blob/bb71353b25d32288286e76bb0f758fc4a22c62f7/conf/cifar10.conf#L11-L26


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=1
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=1
)

samples_per_class = 136


def subsample_classification_dataset(dataset, per_class, batch_size=64):
    counts = {}

    subsampled_inputs = []
    subsampled_labels = []

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    for X, y in dataloader:
        N = X.shape[0]

        for n in range(N):
            X_n, y_n = X[n].unsqueeze(0), y[n].unsqueeze(0)
            y_n_item = y_n.item()

            if y_n_item not in counts.keys():
                counts[y_n_item] = 0

            if counts[y_n_item] < per_class:
                subsampled_inputs.append(X_n)
                subsampled_labels.append(y_n)
                counts[y_n_item] = counts[y_n_item] + 1

    for c in counts.values():
        assert c == per_class

    subsampled_inputs = torch.cat(subsampled_inputs)
    subsampled_labels = torch.cat(subsampled_labels)

    print(subsampled_inputs.shape)
    print(subsampled_labels.shape)

    return torch.utils.data.TensorDataset(subsampled_inputs, subsampled_labels)


trainset_subsampled = subsample_classification_dataset(trainset, samples_per_class)
trainloader_subsampled = torch.utils.data.DataLoader(
    trainset_subsampled, batch_size=64, shuffle=False, pin_memory="cuda" in str(device)
)


# params = [p for p in model.parameters() if p.requires_grad]
# GGN = GGNLinearOperator(
#     model, loss_function, params, trainloader_subsampled, progressbar=True
# )

# # spectral density hyperparameters
# num_points = 1024
# margin = 0.05
# ncv = 256

# boundaries = (0.0, None)

# cache = LanczosApproximateLogSpectrumCached(GGN, ncv, boundaries=boundaries)

# num_repeats = 10
# kappas = [1.01, 1.04, 2]  # not specified in the paper → hand-tuned
# epsilon = 1e-5


# def plot_log_spectrum(num_repeats: int, kappas: List[float]):
#     _, ax = plt.subplots(ncols=len(kappas), figsize=(12, 3), sharex=True, sharey=True)
#     plt.suptitle(f"num_repeats = {num_repeats}")

#     for idx, kappa in enumerate(kappas):
#         grid, density = cache.approximate_log_spectrum(
#             num_repeats=num_repeats,
#             num_points=num_points,
#             kappa=kappa,
#             margin=margin,
#             epsilon=epsilon,
#         )

#         ax[idx].loglog(grid, density, label=rf"$\kappa = {kappa}$")
#         ax[idx].fill_between(grid, density, 0)
#         ax[idx].legend()

#         ax[idx].set_xlabel("Eigenvalue")
#         ax[idx].set_ylabel("Spectral density")
#         ax[idx].set_ylim(bottom=1e-14, top=1e-2)


# plot_log_spectrum(1, kappas)
