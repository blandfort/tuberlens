from potato.interfaces.dataset import LabelledDataset
from potato.interfaces.probes import ProbeSpec
from potato.model import LLMModel
from potato.probes.probe_factory import ProbeFactory
from potato.probes.pytorch_probes import filter_activations_by_turns


def train_probe(
    train_dataset: LabelledDataset,
    validation_dataset: LabelledDataset | None,
    model_name: str,
    layer: int,
    probe_spec: ProbeSpec,
    verbose: bool = True,
    ending_tokens_to_ignore: int = 0,
    start_turn_index: int | None = None,
    end_turn_index: int | None = None,
    apply_transformations_to_validation_dataset: bool = True,
    pos_class_label: str | None = None,
    neg_class_label: str | None = None,
    probe_description: str | None = None,
    use_store: bool = False,
):
    model = LLMModel.load(model_name)
    activations = model.get_activations(
        train_dataset.inputs,
        layer=layer,
        ending_tokens_to_ignore=ending_tokens_to_ignore,
    )
    if start_turn_index is not None or end_turn_index is not None:
        activations = filter_activations_by_turns(
            activations=activations,
            inputs=list(train_dataset.inputs),
            model=model,
            start_turn_index=start_turn_index,
            end_turn_index=end_turn_index,
        )
    train_dataset = train_dataset.assign(
        activations=activations.activations,
        attention_mask=activations.attention_mask,
        input_ids=activations.input_ids,
    )
    print(f"Loaded {len(train_dataset)} training samples")
    print(activations.activations.shape)

    if validation_dataset is not None:
        activations = model.get_activations(
            validation_dataset.inputs,
            layer=layer,
            ending_tokens_to_ignore=ending_tokens_to_ignore
            if apply_transformations_to_validation_dataset
            else 0,
        )
        if apply_transformations_to_validation_dataset and (
            start_turn_index is not None or end_turn_index is not None
        ):
            activations = filter_activations_by_turns(
                activations=activations,
                inputs=list(validation_dataset.inputs),
                model=model,
                start_turn_index=start_turn_index,
                end_turn_index=end_turn_index,
            )
        validation_dataset = validation_dataset.assign(
            activations=activations.activations,
            attention_mask=activations.attention_mask,
            input_ids=activations.input_ids,
        )
        print(f"Loaded {len(validation_dataset)} validation samples")

    if verbose:
        # Dataset example (to make sure special tokens are not added twice)
        import random

        i = random.randint(0, len(train_dataset) - 1)
        print(f"Sample {i}:")
        # Get attention mask for this sample and decode only tokens where mask is 1
        mask = train_dataset.other_fields["attention_mask"][i] == 1
        print(model.tokenizer.decode(train_dataset.other_fields["input_ids"][i][mask]))

    # Train a probe
    print("Creating probe ...")
    probe = ProbeFactory.build(
        layer=layer,
        probe_spec=probe_spec,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_name=model_name,
        use_store=use_store,
        start_turn_index=start_turn_index,
        end_turn_index=end_turn_index,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
    )
    return probe
