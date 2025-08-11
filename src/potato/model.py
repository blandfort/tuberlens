"""
Core module for working with language models and extracting their activations.

This module provides tools for:
1. Loading and managing language models
2. Extracting model activations at specific layers
3. Computing log likelihoods and generating text
4. Handling different model architectures (LLaMA-style and GPT-style)

The module handles batching and memory management to efficiently process large datasets
while working with potentially very large language models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Self, Sequence, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from potato.config import global_settings
from potato.interfaces.activations import Activation
from potato.interfaces.dataset import (
    Dialogue,
    Input,
    to_dialogue,
)
from potato.utils import hf_login


# type: ignore
class ModelArchitecture(ABC):
    """Base class for handling different model architectures."""

    @abstractmethod
    def get_layer_norm(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        """Get the layer normalization module for a specific layer."""
        pass

    @abstractmethod
    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Get all layers of the model."""
        pass

    @abstractmethod
    def set_layers(self, model: torch.nn.Module, layers: list[torch.nn.Module]) -> None:
        """Set the model's layers."""
        pass


class Gemma3Arch(ModelArchitecture):
    def get_layer_norm(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        return model.language_model.model.layers[layer_idx].input_layernorm  # type: ignore

    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        return model.language_model.model.layers  # type: ignore

    def set_layers(self, model: torch.nn.Module, layers: list[torch.nn.Module]) -> None:
        model.language_model.model.layers = layers  # type: ignore


class LlamaArch(ModelArchitecture):
    def get_layer_norm(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        return model.model.layers[layer_idx].input_layernorm  # type: ignore

    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        return model.model.layers  # type: ignore

    def set_layers(self, model: torch.nn.Module, layers: list[torch.nn.Module]) -> None:
        model.model.layers = layers  # type: ignore


class GPTArch(ModelArchitecture):
    def get_layer_norm(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        return model.transformer.h[layer_idx].ln_1  # type: ignore

    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        return model.transformer.h  # type: ignore

    def set_layers(self, model: torch.nn.Module, layers: list[torch.nn.Module]) -> None:
        model.transformer.h = layers  # type: ignore


class ArchitectureRegistry:
    """Registry for mapping model types to their architecture handlers."""

    _architectures: dict[str, Type[ModelArchitecture]] = {
        "gemma3": Gemma3Arch,
        "llama": LlamaArch,
        "gpt": GPTArch,
    }

    @classmethod
    def get_architecture(cls, model: torch.nn.Module) -> ModelArchitecture:
        """Detect and return the appropriate architecture handler."""
        for _, arch_class in cls._architectures.items():
            try:
                arch = arch_class()
                # Test if this architecture matches by trying to access a layer
                arch.get_layer_norm(model, 0)
                return arch
            except (AttributeError, IndexError):
                continue
        raise ValueError(f"Unsupported model architecture: {type(model)}")


class HookedModel:
    """Context manager for extracting activations from specific model layers."""

    def __init__(self, model: torch.nn.Module, layers: list[int]):
        self.model = model
        self.layers = layers
        self.cache = {}
        self.hooks = []
        self.architecture = ArchitectureRegistry.get_architecture(model)
        self.original_layers = None

    def make_hook(self, layer: int) -> Callable:
        def hook_fn(module, input, output):  # type: ignore
            self.cache[layer] = output.cpu()

        return hook_fn

    def __enter__(self) -> Self:
        max_layer = max(self.layers)
        hook_fns = [self.make_hook(layer) for layer in self.layers]

        # Store original layers
        self.original_layers = self.architecture.get_layers(self.model)

        # Register hooks
        for layer, hook_fn in zip(self.layers, hook_fns):
            resid = self.architecture.get_layer_norm(self.model, layer)
            self.hooks.append(resid.register_forward_hook(hook_fn))

        # Truncate layers
        self.architecture.set_layers(self.model, self.original_layers[: max_layer + 1])

        return self

    def get_acts(
        self,
        batch_inputs: dict[str, torch.Tensor],
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = self.model(**batch_inputs)
        activations = torch.stack([self.cache[layer] for layer in self.layers], dim=0)
        if output_buffer is not None:
            output_buffer[:] = activations
            return output_buffer
        return activations

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        # Restore original layers
        if self.original_layers is not None:
            self.architecture.set_layers(self.model, self.original_layers)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()


def tokenize_inputs(
    tokenizer: PreTrainedTokenizerBase,
    dialogues: Sequence[Input],
    add_generation_prompt: bool = False,
    ending_tokens_to_ignore: int = 0,
    device: torch.device | str = "cpu",
    **tokenize_kwargs: Any,
) -> dict[str, torch.Tensor]:
    dialogues = [to_dialogue(d) for d in dialogues]
    input_dicts = [[d.model_dump() for d in dialogue] for dialogue in dialogues]

    input_str = tokenizer.apply_chat_template(
        input_dicts,
        tokenize=False,  # Return string instead of tokens
        add_generation_prompt=add_generation_prompt,  # Add final assistant prefix for generation
    )

    token_dict = tokenizer(input_str, **tokenize_kwargs)  # type: ignore
    for k, v in token_dict.items():
        if k in ["input_ids", "attention_mask"]:
            token_dict[k] = v[:, 1:]
        if ending_tokens_to_ignore > 0 and k == "attention_mask":
            token_dict[k][:, -ending_tokens_to_ignore:] = 0
        if isinstance(v, torch.Tensor):
            token_dict[k] = v.to(device)

    # Check that attention mask exists in token dict
    if "attention_mask" not in token_dict:
        raise ValueError("Tokenizer output must include attention mask")

    return token_dict  # type: ignore


@dataclass
class LLMModel:
    """
    High-level interface for working with language models.

    Provides unified access to:
    - Model loading and management
    - Tokenization
    - Activation extraction
    - Log likelihood computation
    - Text generation

    Handles architecture differences, tokenization, and memory management.
    """

    name: str
    llm_device: torch.device | str
    dtype: torch.dtype
    batch_size: int
    tokenize_kwargs: dict[str, Any]
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    default_tokenize_kwargs: MappingProxyType[str, Any] = MappingProxyType(
        {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": 2**13,
            "add_special_tokens": False,  # We already add special tokens when applying chat template
        }
    )

    @classmethod
    def load(
        cls,
        model_name: str,
        llm_device: torch.device | str = global_settings.LLM_DEVICE,
        batch_size: int = global_settings.BATCH_SIZE,
        tokenize_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """
        Load a language model and its tokenizer.

        Handles model quantization (bfloat16 on CUDA, float16 otherwise),
        device placement, and memory management.

        Args:
            model_name: Name or path of the model
            llm_device: Device to load the model on
            batch_size: Default batch size
            tokenize_kwargs: Additional tokenization args
            model_kwargs: Additional model init args
            tokenizer_kwargs: Additional tokenizer args

        Returns:
            Initialized LLMModel instance
        """
        hf_login()

        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": llm_device,
            "torch_dtype": global_settings.DTYPE,
            "cache_dir": global_settings.CACHE_DIR,
            "max_memory": global_settings.MODEL_MAX_MEMORY.get(
                global_settings.DEFAULT_MODEL
            ),
            **(model_kwargs or {}),
        }
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "cache_dir": global_settings.CACHE_DIR,
            **(tokenizer_kwargs or {}),
        }
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        tokenize_kwargs = cls.default_tokenize_kwargs | (tokenize_kwargs or {})

        llm_device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        return cls(
            name=model_name,
            llm_device=llm_device,
            dtype=dtype,
            batch_size=batch_size,
            tokenize_kwargs=tokenize_kwargs,
            model=model,
            tokenizer=tokenizer,
        )

    @property
    def n_layers(self) -> int:
        # Use num_hidden_layers for LLaMA models, otherwise n_layers
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers  # type: ignore
        elif hasattr(self.model.config, "n_layers"):
            return self.model.config.n_layers  # type: ignore
        elif hasattr(self.model.config, "num_layers"):
            return self.model.config.num_layers  # type: ignore
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)  # type: ignore
        elif hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config,  # type: ignore
            "num_hidden_layers",
        ):
            return self.model.config.text_config.num_hidden_layers  # type: ignore
        else:
            raise ValueError(
                f"Don't know how to get the number of layers for model {self.model.name_or_path}."
            )

    @property
    def hidden_dim(self) -> int:
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size  # type: ignore
        elif hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config,  # type: ignore
            "hidden_size",
        ):
            return self.model.config.text_config.hidden_size  # type: ignore
        else:
            raise ValueError(
                f"Don't know how to get the hidden dimension for model {self.model.name_or_path}."
            )

    def to(self, llm_device: torch.device) -> Self:
        self.llm_device = llm_device
        self.model.to(llm_device)
        return self

    def tokenize(
        self,
        dialogues: Sequence[Input],
        add_generation_prompt: bool = False,
        ending_tokens_to_ignore: int = 0,
    ) -> dict[str, torch.Tensor]:
        return tokenize_inputs(
            self.tokenizer,
            dialogues,
            add_generation_prompt=add_generation_prompt,
            ending_tokens_to_ignore=ending_tokens_to_ignore,
            device=self.llm_device,
            **self.tokenize_kwargs,
        )

    @torch.no_grad()
    def get_activations(
        self,
        inputs: Sequence[Input],
        layer: int,
        max_length: int = 1024,
        ending_tokens_to_ignore: int = 0,
    ) -> Activation:
        """Simplified version of get_batched_activations for a single layer."""
        batch_size = self.batch_size

        # Initialize result tensors
        activations = []
        attention_masks = []
        input_ids = []

        with HookedModel(self.model, [layer]) as hooked_model:
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]
                batch_inputs = self.tokenize(
                    batch_inputs, ending_tokens_to_ignore=ending_tokens_to_ignore
                )

                # Get activations for this batch
                batch_acts = hooked_model.get_acts(batch_inputs)[0].half().cpu()
                activations.append(batch_acts)
                attention_masks.append(batch_inputs["attention_mask"])
                input_ids.append(batch_inputs["input_ids"])

        # Apply padding or truncate such that all activations have the same length
        max_length = min(max_length, max(act.shape[1] for act in activations))

        def pad_or_truncate(
            tensor_list: list[torch.Tensor], max_len: int
        ) -> list[torch.Tensor]:
            return [
                torch.cat(
                    [
                        tensor,
                        torch.zeros(
                            tensor.shape[0],
                            max_len - tensor.shape[1],
                            *tensor.shape[2:],
                            device=tensor.device,
                            dtype=tensor.dtype,
                        ),
                    ],
                    dim=1,
                )
                if tensor.shape[1] < max_len
                else tensor[:, :max_len]
                for tensor in tensor_list
            ]

        activations = pad_or_truncate(activations, max_length)
        attention_masks = pad_or_truncate(attention_masks, max_length)
        input_ids = pad_or_truncate(input_ids, max_length)

        # Ensure all tensors are on CPU before creating Activation object
        return Activation(
            activations=torch.cat(activations, dim=0),
            attention_mask=torch.cat(attention_masks, dim=0).cpu(),
            input_ids=torch.cat(input_ids, dim=0).cpu(),
        )

    def generate(
        self,
        dialogue: Dialogue,
        max_new_tokens: int = 10,
        temperature: float | None = None,
        do_sample: bool = False,
        top_p: float = 1.0,
        skip_special_tokens: bool = False,
        return_full_output: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        """
        Generate text continuation for a dialogue.

        Handles tokenization, generation prompts, and decoding.
        Supports temperature, top-p sampling, and custom parameters.

        Args:
            dialogue: Input dialogue
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (None for greedy)
            do_sample: Use sampling instead of greedy
            top_p: Top-p sampling parameter
            skip_special_tokens: Skip special tokens in output
            return_full_output: Return full dialogue or just continuation
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        inputs = self.tokenize([dialogue], add_generation_prompt=True)

        # Generate the answer
        outputs = self.model.generate(  # type: ignore
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **generation_kwargs,
        )

        if return_full_output:
            out_tokens = outputs[0]
        else:
            # Only get the newly generated tokens by slicing from the input length
            out_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            out_tokens, skip_special_tokens=skip_special_tokens
        )
