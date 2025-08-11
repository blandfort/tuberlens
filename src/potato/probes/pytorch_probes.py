from dataclasses import dataclass
from typing import Any, Self, Sequence

import numpy as np
import torch
from jaxtyping import Float

from potato.interfaces.activations import (
    Activation,
)
from potato.interfaces.dataset import (
    BaseDataset,
    Input,
    Label,
    LabelledDataset,
    Message,
)
from potato.interfaces.probes import Probe
from potato.model import LLMModel
from potato.probes.pytorch_classifiers import (
    PytorchAdamClassifier,
    PytorchClassifier,
)
from potato.utils import as_numpy


@dataclass(kw_only=True)
class PytorchProbe(Probe):
    hyper_params: dict
    _classifier: PytorchClassifier
    start_turn_index: int | None = None
    end_turn_index: int | None = None

    def __post_init__(self):
        self._classifier.training_args = self.hyper_params

    def fit(
        self,
        dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None = None,
        **train_args: Any,
    ) -> Self:
        """
        Fit the probe to the dataset, return a self object with a trained classifier.
        """
        activations_obj = Activation.from_dataset(dataset)

        print("Training probe...")
        if validation_dataset is not None:
            self._classifier.train(
                activations_obj,
                dataset.labels_torch(),
                validation_activations=Activation.from_dataset(validation_dataset),
                validation_y=validation_dataset.labels_torch(),
                **train_args,
            )
        else:
            self._classifier.train(
                activations_obj, dataset.labels_torch(), **train_args
            )
        return self

    def predict(self, dataset: BaseDataset) -> list[Label]:
        """
        Predict and return the labels of the dataset.
        """

        labels = self.predict_proba(dataset) > 0.5
        return [Label.from_int(label) for label in labels]

    def predict_proba(self, dataset: BaseDataset) -> Float[np.ndarray, " batch_size"]:
        """
        Predict and return the probabilities of the dataset.

        Probabilities are expected from the classifier in the shape (batch_size,)
        """
        activations_obj = Activation.from_dataset(dataset)
        return as_numpy(self._classifier.probs(activations_obj))

    def per_token_predictions(
        self,
        dataset: BaseDataset,
    ) -> (
        Float[np.ndarray, "batch_size seq_len"]
        | tuple[
            Float[np.ndarray, "batch_size seq_len"],
            Float[np.ndarray, "batch_size seq_len"],
        ]
    ):
        """
        Probabilities are expected in the shape (batch_size, seq_len) by the classifier.
        """

        # TODO: Change such that it uses the aggregation framework
        activations_obj = Activation.from_dataset(dataset)

        probs = self._classifier.probs(activations_obj, per_token=True)

        if isinstance(self._classifier, PytorchAdamClassifier):
            return as_numpy(probs[1]), as_numpy(probs[2])
        else:
            return as_numpy(probs)

    def predict_proba_from_inputs(
        self,
        inputs: list[Input],
        model: LLMModel,
        layer: int | None = None,
        start_turn_index: int | None = None,
        end_turn_index: int | None = None,
    ) -> Float[np.ndarray, " batch_size"]:
        """
        Predict probabilities from inputs.

        Args:
            inputs: List of inputs (strings or dialogues)
            model: The language model to extract activations from
            layer: The layer to extract activations from
            start_turn_index: If provided and inputs are dialogues, start filtering from this turn (0-indexed, inclusive)
            end_turn_index: If provided and inputs are dialogues, end filtering at this turn (0-indexed, exclusive)
                            If None, include all turns from start_turn_index onwards
        """
        assert self.layer is not None or layer is not None

        # Get full activations first
        activations = model.get_activations(
            inputs,
            layer=layer if layer is not None else self.layer,  # type: ignore
        )

        # If turn indices are specified, filter activations
        if start_turn_index is None:
            start_turn_index = self.start_turn_index
        if end_turn_index is None:
            end_turn_index = self.end_turn_index

        if start_turn_index is not None or end_turn_index is not None:
            activations = filter_activations_by_turns(
                activations, inputs, model, start_turn_index, end_turn_index
            )

        # Change dtype of attention_mask to bool
        activations.attention_mask = activations.attention_mask.bool()

        return as_numpy(self._classifier.probs(activations))

    def predict_proba_from_activations_tensor(
        self,
        activations: Float[torch.Tensor, " seq_len embed_dim"],
        per_token: bool = False,
    ) -> float | Float[np.ndarray, " seq_len"]:
        """Apply the probe to the activation tensor of a single sample.

        Note that we assume that the activations are already masked."""

        # Add dummy attention mask and input ids (not used by classifier anyway)
        # We also add a batch dimension to the activations
        attention_mask = torch.ones_like(
            activations[:, 0].unsqueeze(0), dtype=torch.bool
        )
        input_ids = torch.ones_like(activations[:, 0].unsqueeze(0), dtype=torch.float16)
        activations_obj = Activation(
            activations=activations.unsqueeze(0),
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        if per_token:
            return as_numpy(self._classifier.probs(activations_obj, per_token=True))[0]
        else:
            return as_numpy(self._classifier.probs(activations_obj))[0]


def filter_activations_by_turns(
    activations: Activation,
    inputs: list[Input],
    model: LLMModel,
    start_turn_index: int | None,
    end_turn_index: int | None,
) -> Activation:
    """
    Filter activations to only include tokens from the specified turn range.

    Args:
        activations: Full activations from all turns
        inputs: Original inputs (some may be dialogues)
        model: The language model used for tokenization
        start_turn_index: Start turn (inclusive, 0-indexed)
        end_turn_index: End turn (exclusive, 0-indexed)

    Returns:
        Filtered activations containing only the specified turn range
    """
    filtered_activations = []
    filtered_attention_masks = []
    filtered_input_ids = []

    for i, input_item in enumerate(inputs):
        if isinstance(input_item, str):
            # For string inputs, use all activations (turn indices are ignored)
            filtered_activations.append(activations.activations[i])
            filtered_attention_masks.append(activations.attention_mask[i])
            filtered_input_ids.append(activations.input_ids[i])
        else:
            # For dialogue inputs, filter by turn range
            dialogue = input_item
            num_turns = len(dialogue)

            # Set default values if not provided
            start_idx = start_turn_index if start_turn_index is not None else 0
            end_idx = end_turn_index if end_turn_index is not None else num_turns

            # Validate indices
            if start_idx < 0 or start_idx >= num_turns:
                raise ValueError(
                    f"Start turn index {start_idx} is out of range for dialogue with {num_turns} turns"
                )
            if end_idx < start_idx or end_idx > num_turns:
                raise ValueError(
                    f"End turn index {end_idx} is invalid for start {start_idx} and dialogue with {num_turns} turns"
                )

            # Get token boundaries for this dialogue
            turn_boundaries = _get_turn_token_boundaries(dialogue, model)

            # Get the start and end token indices for the specified turn range
            start_token = turn_boundaries[start_idx][0]
            end_token = (
                turn_boundaries[end_idx - 1][1]
                if end_idx > start_idx
                else turn_boundaries[start_idx][0]
            )

            # Extract activations for this turn range
            turn_activations = activations.activations[i, start_token:end_token]
            turn_attention_mask = activations.attention_mask[i, start_token:end_token]
            turn_input_ids = activations.input_ids[i, start_token:end_token]

            filtered_activations.append(turn_activations)
            filtered_attention_masks.append(turn_attention_mask)
            filtered_input_ids.append(turn_input_ids)

    # Pad sequences to the same length
    max_len = max(act.shape[0] for act in filtered_activations)

    def pad_sequence(
        tensor_list: list[torch.Tensor], max_len: int
    ) -> list[torch.Tensor]:
        return [
            torch.cat(
                [
                    tensor,
                    torch.zeros(
                        max_len - tensor.shape[0],
                        *tensor.shape[1:],
                        device=tensor.device,
                        dtype=tensor.dtype,
                    ),
                ],
                dim=0,
            )
            if tensor.shape[0] < max_len
            else tensor[:max_len]
            for tensor in tensor_list
        ]

    padded_activations = pad_sequence(filtered_activations, max_len)
    padded_attention_masks = pad_sequence(filtered_attention_masks, max_len)
    padded_input_ids = pad_sequence(filtered_input_ids, max_len)

    return Activation(
        activations=torch.stack(padded_activations, dim=0),
        attention_mask=torch.stack(padded_attention_masks, dim=0),
        input_ids=torch.stack(padded_input_ids, dim=0),
    )


def _get_turn_token_boundaries(
    dialogue: Sequence[Message],
    model: LLMModel,
) -> list[tuple[int, int]]:
    """
    Get the token boundaries for each turn in a dialogue using progressive tokenization.

    This method tokenizes the dialogue progressively (1 message, 2 messages, 3 messages, etc.)
    to accurately determine where each turn starts and ends, accounting for chat template
    formatting and special tokens.

    Args:
        dialogue: List of messages representing the dialogue
        model: The language model used for tokenization

    Returns:
        List of (start_token, end_token) tuples for each turn
    """
    boundaries = []

    # Tokenize progressively: 1 message, then 2 messages, then 3, etc.
    for i in range(len(dialogue)):
        # Tokenize up to and including the current turn
        partial_dialogue = dialogue[: i + 1]
        tokenized = model.tokenize([partial_dialogue])
        current_length = tokenized["input_ids"].shape[1]

        if i == 0:
            # First turn starts at 0
            start_token = 0
        else:
            # Get the length of the previous partial dialogue
            prev_partial_dialogue = dialogue[:i]
            prev_tokenized = model.tokenize([prev_partial_dialogue])
            prev_length = prev_tokenized["input_ids"].shape[1]
            start_token = prev_length

        end_token = current_length
        boundaries.append((start_token, end_token))

    return boundaries
