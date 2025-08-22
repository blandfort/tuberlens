<div align="center">
  <img src="assets/potato_probe.png" alt="Potato Probes Logo" width="200"/>
</div>

# Potato

A Python library for training and evaluating neural network activation probes.
This repository provides tools to detect concepts of interest based on model activations from a single layer.

## Features

- **Multiple probe types**: Support for sklearn classifiers, PyTorch neural networks, difference-of-means, LDA, and attention-based probes
- **Flexible dataset handling**: Easy integration with various data formats and labeling schemes
- **Model integration**: Seamless compatibility with Hugging Face transformers
- **Experiment tracking**: Built-in support for Weights & Biases logging

## Related Repositories

### Origin

This project is based on [models-under-pressure](https://github.com/arrrlex/models-under-pressure),
which focuses on activation probes for the specific purpose of detecting high-stakes situations.
It was created with the purpose of quickly running follow-up experiments with other concepts.

Differences to the original repo:

- High-stakes specific code has been removed, including the dataset generation pipeline
- Removed baseline methods
- Simplified configuration considerably
- Removed activation caching
- Added probe attributes for storing metadata (such as class descriptions and model name), so that loading probes elsewhere is easier and less error-prone
- Added example notebook to illustrate the whole process of training, evaluating and loading probes
- New features to train and run probes on parts of the input (e.g. only the assistant's response)

### Probity

Another open-source library for working with activation probes is [probity](https://github.com/curt-tigges/probity).

The main differences between potato and probity are:

- Potato doesn't rely on TransformerLens but uses HuggingFace models directly
- Applying chat templates isn't directly suppported in probity (to the best of my knowledge; as of August 2025)
- Probity includes more dataset functionality
- Probity has comprehensive tests
- Implemented probe architectures are different

## Installation

### For Development
```bash
# Clone the repository
git clone https://github.com/blandfort/potato.git
cd potato

# Install with uv (recommended)
uv sync && uv run pre-commit install

# Or with pip
pip install -e .[dev]
```

Note: In case 'uv' isn't already installed, see [here](https://docs.astral.sh/uv/getting-started/installation) how to set it up.


### Environment Setup
Add a `.env` file to the project root with the following environment variables:
```
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

## Quick Start

See [notebooks/examples/](notebooks/examples/) for several examples on how to train and load probes.


## Available Probe Types

- `sklearn`: Scikit-learn based classifiers
- `difference_of_means`: Simple difference of means classifier
- `lda`: Linear Discriminant Analysis
- `attention`: Attention-based probing
- `linear_then_mean`: Linear layer followed by mean pooling
- `linear_then_max`: Linear layer followed by max pooling
- `linear_then_softmax`: Linear layer followed by softmax

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this software in your research, please cite both the original research paper and this software:

### Original Research Paper
```bibtex
@misc{mckenzie2025highstakes,
      title={Detecting High-Stakes Interactions with Activation Probes},
      author={Alex McKenzie and Urja Pawar and Phil Blandfort and William Bankes and David Krueger and Ekdeep Singh Lubana and Dmitrii Krasheninnikov},
      year={2025},
      eprint={2506.10805},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10805},
}
```

### This Software
```bibtex
@software{potato,
  author={Phil Blandfort and Alex McKenzie and Urja Pawar and William Bankes},
  title={Potato: A Python Library for Training and Evaluating Neural Network Activation Probes},
  year={2025},
  url={https://github.com/blandfort/potato}
}
```
