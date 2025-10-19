from dataclasses import dataclass

import einops
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from tuberlens.config import global_settings
from tuberlens.interfaces.dataset import BaseDataset


class ActivationDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a batch-wise dataset.

    This dataset can be either
    - per-token (activations shape: (b, s, e), attention_mask shape: (b, s))
    - or per-entry (activations shape: (b * s, e), attention_mask shape: (b * s))

    where b is the batch size, s is the sequence length, and e is the embedding dimension.
    """

    def __init__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        y: torch.Tensor,
        device: torch.device | str = global_settings.DEVICE,
        dtype: torch.dtype = global_settings.DTYPE,
    ):
        self.activations = activations
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.y = y
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the masked activations, attention mask, input ids and label.
        """
        return self.__getitems__([index])[0]

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Get the tensors for the batch indices
        batch_acts = self.activations[indices].to(self.device).to(self.dtype)
        batch_mask = self.attention_mask[indices].to(self.device)
        batch_input_ids = self.input_ids[indices].to(self.device).to(self.dtype)
        batch_y = self.y[indices].to(self.device).to(self.dtype)

        # Return as a list of tuples
        return [
            (batch_acts[i], batch_mask[i], batch_input_ids[i], batch_y[i])
            for i in range(len(indices))
        ]


@dataclass
class Activation:
    activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    attention_mask: Float[torch.Tensor, "batch_size seq_len"]
    input_ids: Float[torch.Tensor, "batch_size seq_len"]

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "Activation":
        return cls(
            activations=dataset.other_fields["activations"],  # type: ignore
            attention_mask=dataset.other_fields["attention_mask"].bool(),  # type: ignore
            input_ids=dataset.other_fields["input_ids"],  # type: ignore
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.activations.shape  # type: ignore

    @property
    def batch_size(self) -> int:
        return self.activations.shape[0]

    @property
    def seq_len(self) -> int:
        return self.activations.shape[1]

    @property
    def embed_dim(self) -> int:
        return self.activations.shape[2]

    def __post_init__(self):
        """Validate shapes after initialization, applies attention mask to activations."""
        shape = (self.batch_size, self.seq_len)
        assert (
            self.attention_mask.shape == shape
        ), f"Attention mask shape {self.attention_mask.shape} doesn't agree with {shape}"
        assert (
            self.input_ids.shape == shape
        ), f"Input ids shape {self.input_ids.shape} doesn't agree with {shape}"

        self.activations *= self.attention_mask[:, :, None]

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "Activation":
        return Activation(
            activations=self.activations.to(device).to(dtype),
            attention_mask=self.attention_mask.to(device),
            input_ids=self.input_ids.to(device).to(dtype),
        )

    def per_token(self) -> "PerTokenActivation":
        activations = einops.rearrange(self.activations, "b s e -> (b s) e")
        attention_mask = einops.rearrange(self.attention_mask, "b s -> (b s)")
        input_ids = einops.rearrange(self.input_ids, "b s -> (b s)")

        # Create entry indices tensor to track which entry each token came from
        entry_indices = torch.arange(self.batch_size, device=activations.device)
        entry_indices = einops.repeat(entry_indices, "b -> (b s)", s=self.seq_len)

        # Use the original attention mask to index the flattened sequence
        mask_indices = torch.where(attention_mask == 1)[0]
        return PerTokenActivation(
            activations=activations[mask_indices],
            attention_mask=attention_mask[mask_indices],
            input_ids=input_ids[mask_indices],
            entry_indices=entry_indices[mask_indices],
        )

    def to_dataset(
        self, y: Float[torch.Tensor, " batch_size"] | None = None
    ) -> ActivationDataset:
        if y is None:
            y = torch.empty(self.batch_size, device=global_settings.DEVICE)
        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )


@dataclass
class PerTokenActivation:
    activations: Float[torch.Tensor, "tokens embed_dim"]
    attention_mask: Float[torch.Tensor, " tokens"]
    input_ids: Float[torch.Tensor, " tokens"]
    entry_indices: Float[torch.Tensor, " tokens"]  # Which entry each token came from

    def to_dataset(
        self, y: Float[torch.Tensor, " batch_size"] | None = None
    ) -> ActivationDataset:
        if y is None:
            num_tokens = self.entry_indices.shape[0]
            y = torch.empty(num_tokens, device=global_settings.DEVICE)
        else:
            # Repeat each label the number of times its entry appears
            entry_counts = torch.bincount(self.entry_indices, minlength=y.shape[0])
            y = torch.repeat_interleave(y, entry_counts.to(y.device))

        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )
