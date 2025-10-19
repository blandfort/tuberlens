from tuberlens.config import global_settings
from tuberlens.interfaces.dataset import LabelledDataset
from tuberlens.interfaces.probes import Probe, ProbeSpec, ProbeType
from tuberlens.probes.probe_store import FullProbeSpec, ProbeStore
from tuberlens.probes.pytorch_classifiers import (
    PytorchAdamClassifier,
    PytorchDifferenceOfMeansClassifier,
)
from tuberlens.probes.pytorch_modules import (
    AttnLite,
    LinearThenLast,
    LinearThenMax,
    LinearThenMean,
    LinearThenRollingMax,
    LinearThenSoftmax,
    MeanThenLinear,
)
from tuberlens.probes.pytorch_probes import PytorchProbe
from tuberlens.probes.sklearn_probes import SklearnProbe


class ProbeFactory:
    @classmethod
    def load(cls, probe_id: str) -> Probe:
        return ProbeStore().load_from_id(probe_id)

    @classmethod
    def build(
        cls,
        probe_spec: ProbeSpec,
        train_dataset: LabelledDataset,
        model_name: str,
        layer: int,
        validation_dataset: LabelledDataset | None = None,
        use_store: bool = global_settings.USE_PROBE_STORE,
        pos_class_label: str | None = None,
        neg_class_label: str | None = None,
        probe_description: str | None = None,
        start_turn_index: int | None = None,
        end_turn_index: int | None = None,
    ) -> Probe:
        if use_store:
            store = ProbeStore()
            full_spec = FullProbeSpec.from_spec(
                probe_spec,
                model_name=model_name,
                layer=layer,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
            )
            if store.exists(full_spec):
                return store.load(full_spec)

        if not has_activations(train_dataset):
            raise ValueError(
                "Train dataset must contain activations, attention_mask, and input_ids"
            )
        if validation_dataset is not None:
            if not has_activations(validation_dataset):
                raise ValueError(
                    "Validation dataset must contain activations, attention_mask, and input_ids"
                )

        probe_kwargs = {
            "model_name": model_name,
            "layer": layer,
        }
        if pos_class_label is not None:
            probe_kwargs["pos_class_label"] = pos_class_label
        if neg_class_label is not None:
            probe_kwargs["neg_class_label"] = neg_class_label
        if probe_description is not None:
            probe_kwargs["description"] = probe_description
        if start_turn_index is not None:
            probe_kwargs["start_turn_index"] = start_turn_index
        if end_turn_index is not None:
            probe_kwargs["end_turn_index"] = end_turn_index

        match probe_spec.name:
            case ProbeType.sklearn:
                probe = SklearnProbe(
                    hyper_params=probe_spec.hyperparams or {},
                    **probe_kwargs,
                )
                return probe.fit(train_dataset)

            case ProbeType.difference_of_means:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=False,
                    training_args=probe_spec.hyperparams or {},
                )
            case ProbeType.lda:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=True,
                    training_args=probe_spec.hyperparams or {},
                )
            case ProbeType.pre_mean:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=MeanThenLinear,
                )
            case ProbeType.attention:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=AttnLite,
                )
            case ProbeType.linear_then_mean:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=LinearThenMean,
                )
            case ProbeType.linear_then_max:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=LinearThenMax,
                )
            case ProbeType.linear_then_softmax:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=LinearThenSoftmax,
                )
            case ProbeType.linear_then_rolling_max:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=LinearThenRollingMax,
                )
            case ProbeType.linear_then_last:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams or {},
                    probe_architecture=LinearThenLast,
                )
        probe = PytorchProbe(
            hyper_params=probe_spec.hyperparams or {},
            _classifier=classifier,
            **probe_kwargs,
        )

        probe.fit(train_dataset, validation_dataset)
        if use_store:
            store.save(probe, full_spec)

        return probe


def has_activations(dataset: LabelledDataset) -> bool:
    return {"activations", "attention_mask", "input_ids"} <= set(dataset.other_fields)
