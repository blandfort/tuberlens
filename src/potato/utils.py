import os

import huggingface_hub
import numpy as np
import requests
import torch

from potato.interfaces.dataset import LabelledDataset


def as_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert a torch.Tensor to a numpy array.
    """
    if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]:
        return x.detach().cpu().int().numpy()
    else:
        return x.detach().cpu().float().numpy()


def hf_login():
    HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN"))
    if not HF_TOKEN:
        raise ValueError("No HuggingFace token found")
    try:
        huggingface_hub.login(token=HF_TOKEN)
    except requests.exceptions.HTTPError as e:
        print(
            f"Error logging in to HuggingFace: {e} (Might be fine in case of rate limit error.)"
        )


def create_train_test_split(
    dataset: LabelledDataset,
    test_size: float = 0.2,
    split_field: str | None = None,
) -> tuple[LabelledDataset, LabelledDataset]:
    """Create a train-test split of the dataset.

    Args:
        dataset: Dataset to split
        test_size: Fraction of data to use for test set
        split_field: If provided, ensures examples with the same value for this field
                    are kept together in either train or test set
    """
    if split_field is None:
        # Simple random split
        train_indices = np.random.choice(
            range(len(dataset.ids)),
            size=int(len(dataset.ids) * (1 - test_size)),
            replace=False,
        )
        test_indices = np.random.permutation(
            np.setdiff1d(np.arange(len(dataset.ids)), train_indices)
        )
        train_indices = list(train_indices)
        test_indices = list(test_indices)
    else:
        # Split based on unique values of the field
        assert (
            split_field in dataset.other_fields
        ), f"Field {split_field} not found in dataset"
        unique_values = list(set(dataset.other_fields[split_field]))
        n_test = int(len(unique_values) * test_size)

        test_values = set(np.random.choice(unique_values, size=n_test, replace=False))

        train_indices = [
            i
            for i, val in enumerate(dataset.other_fields[split_field])
            if val not in test_values
        ]
        test_indices = [
            i
            for i, val in enumerate(dataset.other_fields[split_field])
            if val in test_values
        ]

    return dataset[train_indices], dataset[test_indices]  # type: ignore
