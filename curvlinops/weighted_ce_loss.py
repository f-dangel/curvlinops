"""Cross-entropy loss with data-dependent weights."""

from torch import Tensor, ones
from torch.nn import CrossEntropyLoss, Parameter


class CrossEntropyLossWeighted(CrossEntropyLoss):

    def __init__(self, num_data: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_data = num_data
        self.data_weights = Parameter(0.5 * ones(num_data))

    def forward(self, input: Tensor, target_and_data_idx: Tensor) -> Tensor:
        """

        Args:
            target_and_data_idx: Has shape (N, 2) where N is the batch size.
                First entry is the label used for cross-entropy, second entry
                is the data point's index, which we will use to look up the
                weight.
        """
        # split targets and data point indices
        target, data_idx = target_and_data_idx.split([1, 1], dim=1)
        target = target.squeeze(1)
        data_idx = data_idx.squeeze(1)

        # compute the unreduced loss
        actual_reduction = self.reduction
        self.reduction = "none"
        loss = super().forward(input, target)
        self.reduction = actual_reduction

        # apply the weights
        weights = self.data_weights[data_idx].clamp(min=0.0, max=1.0)
        loss = loss * weights

        # do the reduction
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError
