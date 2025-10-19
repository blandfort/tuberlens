from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


class GlobalSettings(BaseSettings):
    LLM_DEVICE: str = "auto"  # Device for the LLM model
    DEVICE: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Device for activations, probes, etc.
    DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    BATCH_SIZE: int = 4
    MODEL_MAX_MEMORY: dict[str, int | None] = Field(default_factory=dict)
    CACHE_DIR: str | None = None
    DEFAULT_MODEL: str = "gpt-4o"
    ACTIVATIONS_DIR: Path = DATA_DIR / "activations"
    DOUBLE_CHECK_CONFIG: bool = True
    PL_DEFAULT_ROOT_DIR: str | None = None
    WANDB_PROJECT: str | None = None  # Default W&B project name (not using W&B if None)
    WANDB_API_KEY: str | None = None
    USE_PROBE_STORE: bool = True


global_settings = GlobalSettings()


LOCAL_MODELS = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-12b": "google/gemma-3-12b-it",
    "gemma-27b": "google/gemma-3-27b-it",
}

# Paths to input files
INPUTS_DIR = DATA_DIR / "inputs"

# Paths to output files
RESULTS_DIR = DATA_DIR / "results"
PROBES_DIR = DATA_DIR / "probes"
TRAIN_DIR = DATA_DIR / "training"
