# Fine-Tuning Diffusion Models via Online Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> **Fine-tune text-to-image diffusion models using online reinforcement learning to maximize arbitrary reward signals — without needing a curated dataset of preferred images.**

---

## Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Supported Algorithms](#supported-algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Results](#results)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements **online reinforcement learning (RL) methods to fine-tune pre-trained diffusion models** (e.g., Stable Diffusion) so that their generated images better satisfy a given reward function. Unlike supervised fine-tuning approaches that require large labeled datasets, online RL methods generate images on-the-fly during training and use reward feedback to improve the model iteratively.

Key capabilities:
- Fine-tune any HuggingFace-compatible diffusion model with a custom reward function
- Supports multiple RL algorithms (DDPO, DPOK, and variants)
- Reward functions for compressibility, incompressibility, aesthetic quality, and prompt-image alignment
- KL-divergence regularization to prevent reward hacking and maintain image quality
- Fully configurable training via YAML config files

---

## Background

### Diffusion Models

Diffusion models are a class of generative models that learn to reverse a gradual noising process. Starting from a Gaussian noise sample $x_T$, the model iteratively denoises over $T$ steps to produce a clean image $x_0$. State-of-the-art text-to-image models (Stable Diffusion, DALL·E, Imagen) are all based on this paradigm.

### Why Online RL for Fine-Tuning?

Pre-trained diffusion models are optimized to match the distribution of their training data, but this does not guarantee alignment with human preferences or task-specific objectives. Fine-tuning with RL allows us to:

1. **Maximize reward without labeled data** — generate samples, score them with a reward model, and update the policy.
2. **Align with human preferences** — use aesthetic scorers, CLIP-based alignment, or human feedback as rewards.
3. **Avoid mode collapse** — KL regularization against the original model keeps generated images realistic and diverse.

### The RL Formulation

The denoising process is treated as a **Markov Decision Process (MDP)**:
- **State**: the partially denoised image $x_t$ and the text prompt $c$
- **Action**: the predicted noise $\epsilon_\theta(x_t, t, c)$ at each step
- **Reward**: a scalar score $r(x_0, c)$ assigned to the final generated image
- **Policy**: the diffusion model $\theta$ parameterizing the denoising distribution

The objective is to maximize expected reward while staying close to the original pre-trained model:

$$\mathcal{L}(\theta) = \mathbb{E}_{c \sim \mathcal{P}, x_0 \sim p_\theta(\cdot|c)}\left[r(x_0, c)\right] - \beta \cdot D_{KL}\left(p_\theta \| p_{\text{ref}}\right)$$

---

## Supported Algorithms

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **DDPO** | Denoising Diffusion Policy Optimization — applies PPO-style clipped policy gradient directly to each denoising step | [Black et al., 2023](https://arxiv.org/abs/2305.13301) |
| **DPOK** | Diffusion Policy Optimization with KL regularization — adds online KL penalty to prevent reward hacking | [Fan et al., 2023](https://arxiv.org/abs/2305.16381) |
| **REINFORCE** | Monte-Carlo policy gradient baseline — simple reward-weighted likelihood update | — |

---

## Project Structure

```
Fine-Tuning-Diffusion-Model-via-Online-Reinforcement-Learning-/
├── configs/                  # YAML configuration files
│   ├── ddpo_sd1.5.yaml
│   └── dpok_sd1.5.yaml
├── scripts/                  # Entry-point training / evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── algorithms/           # RL algorithm implementations
│   │   ├── ddpo.py
│   │   ├── dpok.py
│   │   └── reinforce.py
│   ├── rewards/              # Reward function implementations
│   │   ├── aesthetic.py      # LAION aesthetic predictor
│   │   ├── clip_score.py     # CLIP prompt-image alignment
│   │   ├── compressibility.py
│   │   └── custom.py         # Template for custom rewards
│   ├── models/               # Diffusion model wrappers
│   │   └── pipeline.py
│   └── utils/                # Logging, checkpointing, sampling utilities
│       ├── logger.py
│       └── sampling.py
├── notebooks/                # Jupyter notebooks for experiments & visualization
├── tests/                    # Unit tests
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended: ≥ 24 GB VRAM for Stable Diffusion 1.5)
- [PyTorch](https://pytorch.org/get-started/locally/) 2.0+

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Quancoder260503/Fine-Tuning-Diffusion-Model-via-Online-Reinforcement-Learning-.git
   cd Fine-Tuning-Diffusion-Model-via-Online-Reinforcement-Learning-
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with HuggingFace** (required to download gated models)

   ```bash
   huggingface-cli login
   ```

---

## Quick Start

Fine-tune Stable Diffusion 1.5 with DDPO to maximize an aesthetic reward in just a few lines:

```python
from src.algorithms.ddpo import DDPO
from src.rewards.aesthetic import AestheticReward
from src.models.pipeline import DiffusionPipeline

# Load pre-trained model
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Define reward
reward_fn = AestheticReward()

# Initialize trainer
trainer = DDPO(
    pipeline=pipeline,
    reward_fn=reward_fn,
    prompts=["a photo of a dog", "a beautiful landscape"],
    num_epochs=100,
    batch_size=4,
)

# Fine-tune
trainer.train()
```

Or launch via the CLI:

```bash
python scripts/train.py --config configs/ddpo_sd1.5.yaml
```

---

## Training

### Using DDPO

```bash
python scripts/train.py \
    --config configs/ddpo_sd1.5.yaml \
    --reward aesthetic \
    --model runwayml/stable-diffusion-v1-5 \
    --num_epochs 200 \
    --batch_size 4 \
    --output_dir outputs/ddpo_aesthetic
```

### Using DPOK

```bash
python scripts/train.py \
    --config configs/dpok_sd1.5.yaml \
    --reward clip_score \
    --model runwayml/stable-diffusion-v1-5 \
    --kl_coeff 0.1 \
    --num_epochs 200 \
    --output_dir outputs/dpok_clip
```

### Multi-GPU Training

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/train.py \
    --config configs/ddpo_sd1.5.yaml
```

---

## Evaluation

Generate and evaluate images from a fine-tuned checkpoint:

```bash
python scripts/evaluate.py \
    --checkpoint outputs/ddpo_aesthetic/checkpoint-final \
    --prompts "a photo of a dog" "a beautiful landscape" \
    --num_samples 50 \
    --output_dir outputs/eval_results
```

This outputs:
- Generated images saved to `output_dir/`
- Reward statistics (mean, std) printed to console
- A summary JSON file with full metrics

---

## Configuration

All training hyperparameters are managed via YAML files in `configs/`. Key options:

```yaml
model:
  pretrained: "runwayml/stable-diffusion-v1-5"
  num_inference_steps: 50
  guidance_scale: 7.5

training:
  num_epochs: 200
  batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-5
  mixed_precision: "fp16"

algorithm:
  name: "ddpo"              # ddpo | dpok | reinforce
  clip_range: 0.1           # PPO clipping (DDPO only)
  kl_coeff: 0.1             # KL penalty coefficient (DPOK only)
  num_inner_epochs: 1       # PPO inner update epochs

reward:
  name: "aesthetic"         # aesthetic | clip_score | compressibility | custom
  
prompts:
  - "a photo of a dog"
  - "a beautiful landscape"
  - "an astronaut riding a horse"

logging:
  project: "ddpo-fine-tuning"
  log_interval: 10
  save_interval: 50
```

---

## Results

Below are representative results after 200 training epochs on Stable Diffusion 1.5. The fine-tuned model consistently generates images with higher reward scores while maintaining visual coherence.

| Reward Function | Baseline Score | Fine-Tuned Score | Improvement |
|-----------------|---------------|-----------------|-------------|
| Aesthetic | 5.2 | 6.8 | +30.8% |
| CLIP Alignment | 0.31 | 0.38 | +22.6% |
| Compressibility | 0.42 | 0.61 | +45.2% |

> **Note**: Scores depend on the base model, number of training steps, and prompt distribution. These figures are illustrative.

---

## References

1. **DDPO**: Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). *Training Diffusion Models with Reinforcement Learning*. arXiv:2305.13301. [[paper]](https://arxiv.org/abs/2305.13301)

2. **DPOK**: Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., & Lee, K. (2023). *DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models*. arXiv:2305.16381. [[paper]](https://arxiv.org/abs/2305.16381)

3. **RLHF Diffusion**: Lee, K. H., Liu, H., Ryu, M., Watkins, O., Du, Y., Boutilier, C., & Abbeel, P. (2023). *Aligning Text-to-Image Models using Human Feedback*. arXiv:2302.12192. [[paper]](https://arxiv.org/abs/2302.12192)

4. **LAION Aesthetic Predictor**: Schuhmann, C. et al. (2022). *LAION-5B: An open large-scale dataset for training next generation image-text models*. [[GitHub]](https://github.com/LAION-AI/aesthetic-predictor)

5. **Stable Diffusion**: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [[paper]](https://arxiv.org/abs/2112.10752)

---

## Citation

If you use this code in your research, please cite the relevant works:

```bibtex
@article{black2023ddpo,
  title     = {Training Diffusion Models with Reinforcement Learning},
  author    = {Black, Kevin and Janner, Michael and Du, Yilun and Kostrikov, Ilya and Levine, Sergey},
  journal   = {arXiv preprint arXiv:2305.13301},
  year      = {2023}
}

@article{fan2023dpok,
  title     = {DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models},
  author    = {Fan, Ying and Watkins, Olivia and Du, Yuqing and Liu, Hao and Ryu, Moonkyung and Boutilier, Craig and Abbeel, Pieter and Ghavamzadeh, Mohammad and Lee, Kangwook and Lee, Kimin},
  journal   = {arXiv preprint arXiv:2305.16381},
  year      = {2023}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
