r"""Usage with Huggingface LLMs
===============================

This example demonstrates how to work with Huggingface (HF) language models.

As always, let's first import the required functionality.
Remember to run :code:`pip install -U transformers datasets`
"""

from collections import UserDict
from collections.abc import MutableMapping

import numpy as np
import torch
import torch.utils.data as data_utils
from datasets import Dataset
from torch import nn
from transformers import (
    DataCollatorWithPadding,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

from curvlinops import GGNLinearOperator

# make deterministic
torch.manual_seed(0)
np.random.seed(0)

# %%
#
# Data
# ----
#
# We will use synthetic data for simplicity. But obviously this can
# be replaced with any HF dataloader.

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

data = [
    {"text": "Today is hot, but I will manage!!!!", "label": 1},
    {"text": "Tomorrow is cold", "label": 0},
    {"text": "Carpe diem", "label": 1},
    {"text": "Tempus fugit", "label": 1},
]
dataset = Dataset.from_list(data)


def tokenize(row):
    return tokenizer(row["text"])


dataset = dataset.map(tokenize, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataloader = data_utils.DataLoader(
    dataset, batch_size=100, collate_fn=DataCollatorWithPadding(tokenizer)
)

# %%
#
# Let's check the batch emitted by HF. We will see that it is a :code:`UserDict`,
# containing the input and label tensors. Note that :code:`UserDict` is
# :code:`MutableMapping`, so it is compatible with :code:`curvlinops`.

data = next(iter(dataloader))
print(f"Is the data a UserDict? {isinstance(data, UserDict)}")
for k, v in data.items():
    print(k, v.shape)


# %%
#
# Model
# -----
#
# Curvlinops supports general :code:`UserDict` inputs. However, everything must
# be handled inside the :code:`forward` function of the model. This gives
# the users the most flexibility, without much overhead.
#
# Let's wrap the HF model to conform this requirement then.


class MyGPT2(nn.Module):
    """
    Huggingface LLM wrapper.

    Args:
        tokenizer: The tokenizer used for preprocessing the text data. Needed
            since the model needs to know the padding token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained("gpt2")
        config.pad_token_id = tokenizer.pad_token_id
        config.num_labels = 2
        self.hf_model = GPT2ForSequenceClassification.from_pretrained(
            "gpt2", config=config
        )

        # For simplicity, only enable grad for the last layer
        for p in self.hf_model.parameters():
            p.requires_grad = False

        for p in self.hf_model.score.parameters():
            p.requires_grad = True

    def forward(self, data: MutableMapping) -> torch.Tensor:
        """
        Custom forward function. Handles things like moving the
        input tensor to the correct device inside.

        Args:
            data: A dict-like data structure with `input_ids` inside.
                This is the default data structure assumed by Huggingface
                dataloaders.

        Returns:
            logits: An `(batch_size, n_classes)`-sized tensor of logits.
        """
        device = next(self.parameters()).device
        input_ids = data["input_ids"].to(device)
        output_dict = self.hf_model(input_ids)
        return output_dict.logits


model = MyGPT2(tokenizer).to(torch.bfloat16)

with torch.no_grad():
    logits = model(data)
    print(f"Logits shape: {logits.shape}")


# %%
#
# Curvlinops
# ----------
#
# We are now ready to compute the curvature of this HF model using Curvlinops.
# For this, we need to define a function to tell Curvlinops how to get the
# batch size of the :code:`UserDict` input batch. Everything else is unchanged
# from the standard usage of Curvlinops!


def batch_size_fn(x: MutableMapping):
    return x["input_ids"].shape[0]


params = [p for p in model.parameters() if p.requires_grad]

ggn = GGNLinearOperator(
    model,
    nn.CrossEntropyLoss(),
    params,
    [(data, data["labels"])],  # We still need to input a list of "(X, y)" pairs!
    check_deterministic=False,
    batch_size_fn=batch_size_fn,  # Remember to specify this!
)

G = ggn @ np.eye(ggn.shape[0])

print(f"GGN shape: {G.shape}")


# %%
#
# Conclusion
# ----------
#
# This :code:`UserDict` (or any other dict-like data structure) specification
# is very flexible. This doesn't stop at HF models. You can leverage this
# for any custom models!
