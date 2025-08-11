# %%
import pickle

import yaml

from potato.config import DATA_DIR, LOCAL_MODELS
from potato.interfaces.dataset import (
    LabelledDataset,
    Message,
)
from potato.interfaces.probes import ProbeSpec, ProbeType
from potato.model import LLMModel
from potato.probes.probe_factory import ProbeFactory
from potato.probes.pytorch_probes import filter_activations_by_turns
from potato.utils import create_train_test_split

CONCEPT = "policy"
model_name = LOCAL_MODELS["llama-1b"]
layer = 11

CONCEPT_DIR = DATA_DIR / CONCEPT
PROBE_PATH = CONCEPT_DIR / f"{CONCEPT}_probe.pkl"

if CONCEPT == "deception":
    pos_class_label = "deceptive"
    neg_class_label = "honest"
    probe_description = (
        f"A linear probe on {model_name} detecting whether the assistant is deceptive."
    )
elif CONCEPT == "high-stakes":
    pos_class_label = "high-stakes"
    neg_class_label = "low-stakes"
    probe_description = f"A linear probe on {model_name} detecting whether the conversation is high-stakes."
elif CONCEPT == "policy":
    pos_class_label = "violation"
    neg_class_label = "compliant"
    probe_description = f"A linear probe on {model_name} detecting whether the assistant's response is compliant with the content policy."

# %%


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
    # NOTE Turn off wandb tracking by setting WANDB_PROJECT to None in config.py
    print("Creating probe ...")
    probe = ProbeFactory.build(
        layer=layer,
        probe_spec=probe_spec,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_name=model_name,
        use_store=False,
        start_turn_index=start_turn_index,
        end_turn_index=end_turn_index,
        pos_class_label=pos_class_label,
        neg_class_label=neg_class_label,
        probe_description=probe_description,
    )
    return probe


# %%

if CONCEPT == "deception":
    # Load true facts from true_false_facts.csv
    import pandas as pd

    # Read the true facts (where label == 1)
    # NOTE: The source of this CSV file is
    # https://github.com/andyzoujm/representation-engineering/blob/main/data/facts/facts_true_false.csv
    facts_df = pd.read_csv(CONCEPT_DIR / "true_false_facts.csv")
    true_facts = facts_df[facts_df["label"] == 1]["statement"].tolist()

    # Create samples with user prompts for each true fact
    samples = []
    pair_id = 1

    for fact in true_facts:
        # Create deceptive sample
        deceptive_input = [
            Message(
                role="user",
                content="Pretend to be a dishonest person making statements about the world.",
            ),
            Message(role="assistant", content=fact),
        ]
        samples.append(
            {
                "pair_id": pair_id,
                "ids": f"{pair_id}d",
                "inputs": deceptive_input,
                "labels": "deceptive",
            }
        )

        # Create honest sample
        honest_input = [
            Message(
                role="user",
                content="Pretend to be an honest person making statements about the world.",
            ),
            Message(role="assistant", content=fact),
        ]
        samples.append(
            {
                "pair_id": pair_id,
                "ids": f"{pair_id}h",
                "inputs": honest_input,
                "labels": "honest",
            }
        )

        pair_id += 1

    # Create LabelledDataset from the samples
    train_dataset = LabelledDataset(
        ids=[sample["ids"] for sample in samples],
        inputs=[sample["inputs"] for sample in samples],
        other_fields={
            "labels": [sample["labels"] for sample in samples],
            "pair_id": [sample["pair_id"] for sample in samples],
        },
    )

    # Convert labels to high-stakes/low-stakes
    new_labels = [
        "high-stakes" if label == pos_class_label else "low-stakes"
        for label in train_dataset.other_fields["labels"]
    ]
    train_dataset = train_dataset.assign(labels=new_labels)

    train_dataset, validation_dataset = create_train_test_split(
        train_dataset, split_field="pair_id"
    )
elif CONCEPT == "high-stakes":
    stakes_dataset = CONCEPT_DIR / "combined_deployment_22_04_25.jsonl"
    dataset = LabelledDataset.load_from(stakes_dataset)
    train_dataset, validation_dataset = create_train_test_split(
        dataset, split_field="pair_id"
    )
else:
    dataset_path = CONCEPT_DIR / "training.csv"
    dataset = LabelledDataset.load_from(dataset_path)
    train_dataset, validation_dataset = create_train_test_split(
        dataset,  # split_field="pair_id"
    )


# %%

# TODO Why is training stopping after 125 epochs?

probe = train_probe(
    train_dataset,
    validation_dataset,
    model_name,
    layer,
    # ending_tokens_to_ignore=5,
    start_turn_index=1,  # Exclude user message
    apply_transformations_to_validation_dataset=True,
    pos_class_label=pos_class_label,
    neg_class_label=neg_class_label,
    probe_description=probe_description,
    probe_spec=ProbeSpec(
        # name=ProbeType.sklearn,
        # hyperparams={},
        name=ProbeType.linear_then_mean,
        hyperparams={
            "batch_size": 8,
            "epochs": 200,
            "optimizer_args": {"lr": 1e-3, "weight_decay": 1e-2},
            "final_lr": 1e-4,
            "gradient_accumulation_steps": 1,
            "patience": 100,
            "temperature": 0.1,
        },
    ),
)

# %% Storing the probe
pickle.dump(probe, open(PROBE_PATH, "wb"))

# -------------------------------------------------
# DEPLOYMENT
# -------------------------------------------------

# %% Loading the probe
probe = pickle.load(open(PROBE_PATH, "rb"))
assert probe.model_name is not None
assert probe.layer is not None
print("Probe initialized:")
print(probe.description)

# Initialize the model so we can compute activations
model = LLMModel.load(probe.model_name)


# %% Applying the probe to a new sample

# Load test inputs from YAML
with open(CONCEPT_DIR / "test_inputs.yaml") as f:
    raw_inputs = yaml.safe_load(f)
inputs = [[Message(**msg) for msg in pair] for pair in raw_inputs]

preds = probe.predict_proba_from_inputs(inputs, model=model)
for i in range(len(preds)):
    print(f"Sample {i}: {preds[i]}")
    print(f"Input: {inputs[i]}")
    print()


# %%
# Verifying that the probe works with activation tensors

# NOTE To apply the probe to a HF transformer directly, get the activations tensor
# from activations before layer norm. Either process one item at a time or make sure
# to apply the attention mask.

for inp in inputs:
    activations = model.get_activations([inp], layer=probe.layer)

    activations = filter_activations_by_turns(
        activations=activations,
        inputs=[inp],
        model=model,
        start_turn_index=probe.start_turn_index,
        end_turn_index=probe.end_turn_index,
    )
    print(probe.predict_proba_from_activations_tensor(activations.activations[0]))

# %%
