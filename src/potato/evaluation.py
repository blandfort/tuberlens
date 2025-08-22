from typing import Any, Dict

import numpy as np
import pandas as pd
from jaxtyping import Float
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from potato.interfaces.dataset import LabelledDataset
from potato.model import LLMModel
from potato.probes.pytorch_probes import filter_activations_by_turns


def tpr_at_fixed_fpr_score(
    y_true: Float[np.ndarray, " batch_size"],
    y_pred: Float[np.ndarray, " batch_size"],
    fpr: float,
) -> float:
    """Calculate TPR at a fixed FPR threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        fpr: Target false positive rate threshold

    Returns:
        TPR value at the specified FPR threshold
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred)

    # Find the TPR value at the closest FPR to our target
    idx = np.argmin(np.abs(fpr_vals - fpr))
    return float(tpr_vals[idx])


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, fpr: float
) -> dict[str, float]:
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred > 0.5)),
        "tpr_at_fpr": float(tpr_at_fixed_fpr_score(y_true, y_pred, fpr=fpr)),
        "fpr": float(fpr),
    }
    return metrics


def get_performances(
    probe: Any,
    evaluation_datasets: Dict[str, LabelledDataset],
    start_turn_index: int | None = None,
    end_turn_index: int | None = None,
    fpr: float = 0.01,
) -> pd.DataFrame:
    """Calculate performance metrics for a probe across multiple evaluation datasets.

    Args:
        probe: Probe object with predict_proba_from_inputs method
        evaluation_datasets: Dict of LabelledDataset objects for evaluation (dataset names as keys)
        fpr: Target false positive rate for TPR calculation (default: 0.01)

    Returns:
        DataFrame with metrics as columns and datasets as rows, plus a mean row
    """
    results = []

    # Make sure activations are available for all datasets
    model = None

    for name, dataset in evaluation_datasets.items():
        # Make sure activations are available for the dataset
        if "activations" not in dataset.other_fields:
            if model is None:
                model = LLMModel.load(probe.model_name)
            activations = model.get_activations(
                dataset.inputs, layer=probe.layer, show_progress=True
            )

            if start_turn_index is not None or end_turn_index is not None:
                activations = filter_activations_by_turns(
                    activations=activations,
                    inputs=list(dataset.inputs),
                    model=model,
                    start_turn_index=start_turn_index,
                    end_turn_index=end_turn_index,
                )

            dataset = dataset.assign(
                activations=activations.activations,
                attention_mask=activations.attention_mask,
                input_ids=activations.input_ids,
            )
        # Get predictions from probe
        predictions = probe.predict_proba(dataset)

        # Calculate metrics
        metrics = calculate_metrics(
            np.array([label.to_int() for label in dataset.labels]), predictions, fpr=fpr
        )

        results.append(metrics | {"dataset": name})
    del model

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns to put dataset first
    cols = ["dataset"] + [col for col in df.columns if col != "dataset"]
    df = df[cols]

    # Add mean row
    mean_metrics = df.drop("dataset", axis=1).mean()
    mean_row = pd.DataFrame([["mean"] + mean_metrics.tolist()], columns=df.columns)
    df = pd.concat([df, mean_row], ignore_index=True)

    return df


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    import pickle

    from potato.config import DATA_DIR

    probe_path = DATA_DIR / "high-stakes" / "high-stakes_probe.pkl"
    probe = pickle.load(open(probe_path, "rb"))
    print("Probe initialized:")
    print(probe.description)

    # Load the datasets
    dataset_path = DATA_DIR / "high-stakes" / "combined_deployment_22_04_25.jsonl"
    dataset = LabelledDataset.load_from(
        dataset_path, pos_class_label="high-stakes", neg_class_label="low-stakes"
    )

    print(get_performances(probe, {"training": dataset}))
