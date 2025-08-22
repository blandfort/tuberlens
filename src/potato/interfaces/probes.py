from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Self

import numpy as np
import torch
from jaxtyping import Float
from pydantic import BaseModel, JsonValue

from potato.interfaces.dataset import (
    BaseDataset,
    Label,
    LabelledDataset,
)


class ProbeType(str, Enum):
    sklearn = "sklearn"
    difference_of_means = "difference_of_means"
    lda = "lda"
    pre_mean = "pre_mean"
    attention = "attention"
    linear_then_mean = "linear_then_mean"
    linear_then_max = "linear_then_max"
    linear_then_softmax = "linear_then_softmax"
    linear_then_rolling_max = "linear_then_rolling_max"
    linear_then_last = "linear_then_last"

    @property
    def default_hyperparams(self) -> Dict[str, Any]:
        """Return default hyperparameters for each probe type."""
        defaults = {
            ProbeType.sklearn: {
                "C": 1e-3,
                "random_state": 42,
                "fit_intercept": False,
            },
            ProbeType.difference_of_means: {
                "batch_size": 16384,
            },
            ProbeType.lda: {
                "batch_size": 16384,
            },
            ProbeType.pre_mean: {
                "batch_size": 16,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
            ProbeType.attention: {
                "batch_size": 128,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 5e-4,
                "gradient_accumulation_steps": 1,
                "patience": 50,
            },
            ProbeType.linear_then_mean: {
                "batch_size": 16,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
            ProbeType.linear_then_max: {
                "batch_size": 16,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
            ProbeType.linear_then_softmax: {
                "temperature": 5,
                "batch_size": 16,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
            ProbeType.linear_then_rolling_max: {
                "batch_size": 16,
                "window_size": 40,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
            ProbeType.linear_then_last: {
                "batch_size": 16,
                "epochs": 200,
                "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                "final_lr": 1e-4,
                "gradient_accumulation_steps": 4,
                "patience": 50,
            },
        }
        return defaults[self]


class ProbeSpec(BaseModel):
    name: ProbeType
    hyperparams: Optional[Dict[str, JsonValue]] = None

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to merge with defaults."""
        if self.hyperparams is None:
            # If no hyperparams provided, use all defaults
            self.hyperparams = self.name.default_hyperparams
        else:
            # If hyperparams provided, merge with defaults
            defaults = self.name.default_hyperparams.copy()
            defaults.update(self.hyperparams)  # Override defaults with provided values
            self.hyperparams = defaults


@dataclass
class Probe(ABC):
    model_name: str | None
    layer: int | None
    description: str | None = None
    pos_class_label: str = "positive"
    neg_class_label: str = "negative"

    @abstractmethod
    def fit(
        self,
        dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None = None,
    ) -> Self: ...

    @abstractmethod
    def predict(self, dataset: BaseDataset) -> list[Label]: ...

    @abstractmethod
    def predict_proba(
        self, dataset: BaseDataset
    ) -> Float[np.ndarray, " batch_size"]: ...

    @abstractmethod
    def per_token_predictions(
        self,
        dataset: BaseDataset,
    ) -> Float[np.ndarray, "batch_size seq_len"]: ...


class Classifier(Protocol):
    def fit(
        self,
        X: Float[np.ndarray, "batch_size ..."],
        y: Float[np.ndarray, " batch_size"],
    ) -> Self: ...

    def predict(
        self, X: Float[np.ndarray, "batch_size ..."]
    ) -> Float[np.ndarray, " batch_size"]: ...

    def predict_proba(
        self, X: Float[np.ndarray, "batch_size ..."]
    ) -> Float[np.ndarray, "batch_size n_classes"]: ...


class Aggregation(Protocol):
    def __call__(
        self,
        logits: Float[torch.Tensor, "batch_size seq_len"],
        attention_mask: Float[torch.Tensor, "batch_size seq_len"],
        input_ids: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]: ...
