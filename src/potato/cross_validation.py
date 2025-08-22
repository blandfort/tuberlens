"""
This script is used to choose the best layer for a given model and dataset.

It does this by training a probe on the train set and evaluating it on the test set.

It then repeats this process for each layer and reports the best layer.

"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Self, Tuple

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from potato.config import (
    DATA_DIR,
    LOCAL_MODELS,
    RESULTS_DIR,
)
from potato.interfaces.dataset import LabelledDataset
from potato.interfaces.probes import ProbeSpec, ProbeType
from potato.model import LLMModel
from potato.probes.probe_factory import ProbeFactory
from potato.probes.pytorch_probes import filter_activations_by_turns


class ChooseLayerConfig(BaseModel):
    model_name: str
    dataset_path: Path
    cv_folds: int
    batch_size: int
    probe_spec: ProbeSpec
    max_samples: int | None = None
    layers: list[int] | None = None
    output_dir: Path = RESULTS_DIR / "cross_validation"
    layer_batch_size: int = 4
    pos_class_label: str | None = None
    neg_class_label: str | None = None
    # Transformation parameters (matching training.py)
    ending_tokens_to_ignore: int = 0
    start_turn_index: int | None = None
    end_turn_index: int | None = None

    @property
    def output_path(self) -> Path:
        return self.output_dir / "results.jsonl"

    @property
    def temp_output_path(self) -> Path:
        return self.output_dir / "temp_results.jsonl"


class CVIntermediateResults(BaseModel):
    config: ChooseLayerConfig
    layer_results: dict[int, list[float]] = Field(default_factory=dict)
    layer_mean_accuracies: dict[int, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def add_layer_results(self, layer: int, results: list[float]):
        self.layer_results[layer] = results
        self.layer_mean_accuracies[layer] = float(np.mean(results))

    def save(self):
        self.config.temp_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving intermediate results to {self.config.temp_output_path}")
        with open(self.config.temp_output_path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class CVFinalResults(BaseModel):
    results: CVIntermediateResults
    best_layer: int
    best_layer_accuracy: float

    @classmethod
    def from_intermediate(cls, intermediate: CVIntermediateResults) -> Self:
        best_layer = max(
            intermediate.layer_mean_accuracies.keys(),
            key=lambda x: intermediate.layer_mean_accuracies[x],
        )
        best_layer_accuracy = intermediate.layer_mean_accuracies[best_layer]

        return cls(
            results=intermediate,
            best_layer=best_layer,
            best_layer_accuracy=best_layer_accuracy,
        )

    def save(self):
        path = self.results.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving final results to {path}")
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


@dataclass
class CVSplits:
    """
    A class that contains the cross validation splits for a given dataset.

    Note: we're not using scikit-learn's cross validation because we want to
    use the pair IDs to create the splits.
    """

    num_folds: int
    folds: list[list[int]]

    @classmethod
    def create(cls, dataset: LabelledDataset, num_folds: int) -> "CVSplits":
        """Create cross validation splits from the dataset.

        Args:
            dataset: Dataset to split
            num_folds: Number of folds to create

        Returns:
            CVSplits object containing the folds
        """
        # Get unique pair IDs
        pair_ids = list(set(dataset.other_fields["pair_id"]))

        # Randomly shuffle pair IDs
        import numpy as np

        np.random.shuffle(pair_ids)

        # Split pair IDs into num_folds groups
        fold_size = len(pair_ids) // num_folds
        pair_id_folds = [
            pair_ids[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)
        ]

        # Create indices for each fold by finding indices matching pair IDs
        folds = []
        for fold_pair_ids in pair_id_folds:
            fold_indices = [
                i
                for i, x in enumerate(dataset.other_fields["pair_id"])
                if x in fold_pair_ids
            ]
            folds.append(fold_indices)

        return cls(num_folds=num_folds, folds=folds)

    def splits(
        self, dataset: LabelledDataset
    ) -> Iterator[Tuple[LabelledDataset, LabelledDataset]]:
        """Get train/test splits for cross validation.

        Returns:
            Sequence of (train, test) pairs where train is all folds except one
            and test is the held-out fold
        """
        for i in range(self.num_folds):
            # Train indices are all indices except the current fold
            train_indices = [
                idx for fold in self.folds[:i] + self.folds[i + 1 :] for idx in fold
            ]
            # Test set is the current fold
            test_indices = self.folds[i]

            yield dataset[train_indices], dataset[test_indices]


def get_cross_validation_accuracies(
    dataset: LabelledDataset,
    cv_splits: CVSplits,
    probe_spec: ProbeSpec,
    model_name: str,
    layer: int,
) -> list[float]:
    """Get the cross validation accuracies for a given layer.

    Args:
        dataset: Dataset to evaluate (with activations already computed and transformations applied)
        cv_splits: CVSplits
        probe_spec: ProbeSpec
        model_name: Name of the model
        layer: Layer index to use for probing

    Returns:
        List of accuracies, one for each fold
    """
    results = []

    for train, test in cv_splits.splits(dataset):
        probe = ProbeFactory.build(
            probe_spec=probe_spec,
            train_dataset=train,
            model_name=model_name,
            layer=layer,
        )
        test_predictions = probe.predict(test)
        test_labels = test.labels_numpy()

        # Convert Label enum predictions to integers
        test_prediction_ints = [score.to_int() for score in test_predictions]
        accuracy = (np.array(test_prediction_ints) == test_labels).mean()
        results.append(accuracy)

    return results


def choose_best_layer_via_cv(config: ChooseLayerConfig) -> CVFinalResults:
    """Main function to choose the best layer via cross validation.

    This function supports the same transformations as training.py:
    - ending_tokens_to_ignore: Number of tokens to ignore at the end of input
    - start_turn_index/end_turn_index: Turn-based filtering for dialogue inputs

    Activations are computed once per layer with transformations applied,
    then the dataset is split for cross-validation to avoid recomputing activations.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset = LabelledDataset.load_from(
        config.dataset_path,
        pos_class_label=config.pos_class_label,
        neg_class_label=config.neg_class_label,
    )
    dataset = dataset.filter(lambda x: x.other_fields.get("split", "train") == "train")

    if config.max_samples is not None:
        dataset = dataset.sample(config.max_samples)

    llm = LLMModel.load(config.model_name, batch_size=config.batch_size)

    if config.layers is None:
        layers = list(range(llm.n_layers))
    else:
        assert all(0 <= layer < llm.n_layers for layer in config.layers)
        layers = config.layers

    results = CVIntermediateResults(config=config)

    cv_splits = CVSplits.create(dataset, config.cv_folds)

    pbar = tqdm(total=len(layers), desc="Cross-validating layers")

    for layer in layers:
        # Compute activations once for the entire dataset with transformations applied
        print(f"Computing activations for layer {layer}...")
        activations = llm.get_activations(
            dataset.inputs,
            layer=layer,
            ending_tokens_to_ignore=config.ending_tokens_to_ignore,
            show_progress=True,
        )

        # Apply turn-based filtering if specified
        if config.start_turn_index is not None or config.end_turn_index is not None:
            activations = filter_activations_by_turns(
                activations=activations,
                inputs=list(dataset.inputs),
                model=llm,
                start_turn_index=config.start_turn_index,
                end_turn_index=config.end_turn_index,
            )

        # Assign transformed activations to dataset
        dataset = dataset.assign(
            activations=activations.activations,
            attention_mask=activations.attention_mask,
            input_ids=activations.input_ids,
        )

        layer_results = get_cross_validation_accuracies(
            dataset=dataset,
            cv_splits=cv_splits,
            probe_spec=config.probe_spec,
            model_name=config.model_name,
            layer=layer,
        )

        results.add_layer_results(layer, layer_results)
        results.save()

        pbar.update(1)

    pbar.close()
    print(f"Results: {results}")

    # Compute final results
    final_results = CVFinalResults.from_intermediate(results)

    # Save final results
    final_results.save()
    return final_results


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    dataset_path = DATA_DIR / "high-stakes" / "combined_deployment_22_04_25.jsonl"
    pos_class_label = "high-stakes"
    neg_class_label = "low-stakes"

    configs = [
        ChooseLayerConfig(
            model_name=LOCAL_MODELS[model_name],
            dataset_path=dataset_path,
            max_samples=None,
            cv_folds=4,
            layers=list(range(0, max_layer, 2)),
            batch_size=4,
            output_dir=RESULTS_DIR / "cross_validation",
            probe_spec=ProbeSpec(name=ProbeType.sklearn, hyperparams={}),
            pos_class_label=pos_class_label,
            neg_class_label=neg_class_label,
            # Example transformation parameters (optional)
            # ending_tokens_to_ignore=2,
            # start_turn_index=0,
            # end_turn_index=2,
        )
        for model_name, max_layer in [
            # ("gemma-27b", 61),
            # ("gemma-1b", 25),
            # ("gemma-12b", 47),
            ("llama-1b", 16),
        ]
    ]

    for config in configs:
        choose_best_layer_via_cv(config)
