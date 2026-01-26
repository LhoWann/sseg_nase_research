# SSEG-NASE: Self-Supervised Evolution with Guided Neural Architecture Sparse Evolution

> **A Novel Framework for Hardware-Efficient Few-Shot Learning via Curriculum-Guided Network Evolution and Adaptive Sparsity**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Motivation &amp; Problem Statement](#research-motivation--problem-statement)
3. [Core Contributions](#core-contributions)
4. [Theoretical Framework](#theoretical-framework)
5. [Architecture Deep Dive](#architecture-deep-dive)
6. [Project Structure](#project-structure)
7. [Installation &amp; Setup](#installation--setup)
8. [Quick Start Guide](#quick-start-guide)
9. [Detailed Module Documentation](#detailed-module-documentation)
10. [Experiment Configurations](#experiment-configurations)
11. [Evaluation &amp; Benchmarking](#evaluation--benchmarking)
12. [Ablation Studies](#ablation-studies)
13. [Hardware Optimization](#hardware-optimization)
14. [Visualization &amp; Reporting](#visualization--reporting)
15. [Reproducibility](#reproducibility)
16. [API Reference](#api-reference)
17. [Troubleshooting](#troubleshooting)
18. [Citation](#citation)
19. [Acknowledgments](#acknowledgments)

---

## Executive Summary

**SSEG-NASE** (Self-Supervised Evolution with Guided Neural Architecture Sparse Evolution) is a comprehensive research framework that addresses the critical challenge of developing **hardware-efficient neural architectures** for **few-shot learning** on resource-constrained devices.

### Key Innovation

This framework introduces a novel **three-pronged approach**:

1. **SSEG (Self-Supervised Evolution with Guidance)**: Automatically grows neural network architectures from a minimal seed network using self-supervised learning signals and curriculum-based guidance.
2. **NASE (Neural Architecture Sparse Evolution)**: Implements complementary sparsity masks that allow network pruning while maintaining performance through importance-based channel selection.
3. **Curriculum Learning Integration**: Progressively increases data complexity from simple geometric shapes to adversarial examples, enabling the network to develop robust representations.

### Target Metrics

| Metric                | Target | Hardware                |
| --------------------- | ------ | ----------------------- |
| 5-way 1-shot Accuracy | >50%   | RTX 3060 (12GB), RTX 3050 (4GB) |
| 5-way 5-shot Accuracy | >68%   | RTX 3060 (12GB), RTX 3050 (4GB) |
| Model Parameters      | <1M    | -                       |
| Inference FLOPs       | <1G    | -                       |

---

## Research Motivation & Problem Statement

### The Challenge

Few-shot learning—the ability to learn new concepts from very limited examples—is a crucial capability for real-world AI applications. However, current state-of-the-art approaches face significant challenges:

1. **Computational Cost**: Methods like DINO and SimCLR require massive computational resources (22M+ parameters, 4.6+ GFLOPs)
2. **Hardware Constraints**: Edge devices and consumer GPUs cannot practically deploy these models
3. **Architecture Design**: Manual architecture design requires extensive expertise and experimentation
4. **Feature Quality**: Learning discriminative features from limited data remains difficult

### Our Solution

SSEG-NASE addresses these challenges through:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SSEG-NASE Framework Overview                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                  │
│   │  Curriculum  │────▶│    SSEG      │────▶│    NASE      │                  │
│   │   Learning   │     │  Evolution   │     │   Sparsity   │                  │
│   └──────────────┘     └──────────────┘     └──────────────┘                  │
│         │                     │                     │                        │
│         ▼                     ▼                     ▼                        │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                  │
│   │ Progressive  │     │  Automatic   │     │  Efficient   │                  │
│   │  Difficulty  │     │   Growth     │     │  Inference   │                  │
│   └──────────────┘     └──────────────┘     └──────────────┘                  │
│                                                                              │
│   Result: Hardware-efficient model with strong few-shot performance           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Contributions

### 1. Self-Supervised Evolution with Guidance (SSEG)

SSEG introduces a novel paradigm for **automatic neural architecture growth**:

- **Seed Network**: Starts with a minimal CNN (3 blocks, 16 channels)
- **Plateau Detection**: Monitors training loss to detect learning saturation
- **Distillation Gap Analysis**: Uses teacher-student divergence to guide growth decisions
- **Growth Operations**:
  - **GROW**: Adds new convolutional blocks
  - **WIDEN**: Increases channel dimensions of existing blocks

**Mathematical Formulation:**

The evolution trigger is based on detecting a plateau in the SSL loss:

$$
\text{Plateau} = |\mathcal{L}_{SSL}^{(t)} - \mathcal{L}_{SSL}^{(t-w)}| < \epsilon
$$

where $w$ is the window size and $\epsilon$ is the plateau threshold.

Growth decision is then guided by the distillation gap:

$$
\text{Should Evolve} = \text{Plateau} \land (\mathcal{L}_{distill} > \tau_{gap})
$$

### 2. Neural Architecture Sparse Evolution (NASE)

NASE implements **importance-based sparsification**:

- **Importance Scoring**: Uses Taylor expansion to estimate parameter importance:

$$
I(\theta_i) = |\theta_i \cdot \nabla_{\theta_i} \mathcal{L}|
$$

- **Complementary Masks**: Generates positive (active) and negative (pruned) masks
- **Adaptive Pruning**: Maintains minimum channel counts per layer

### 3. Curriculum Learning Pipeline

Four progressive complexity levels:

| Level | Name                  | Description                                           | Samples |
| ----- | --------------------- | ----------------------------------------------------- | ------- |
| 1     | **BASIC**       | Simple geometric shapes (circles, squares, triangles) | 5,000   |
| 2     | **TEXTURE**     | Sinusoidal patterns with noise                        | 10,000  |
| 3     | **OBJECT**      | Composite objects with backgrounds                    | 20,000  |
| 4     | **ADVERSARIAL** | Visually similar but semantically different           | 10,000  |

---

## Theoretical Framework

### Self-Supervised Learning Foundation

The framework combines two complementary SSL objectives:

#### 1. Contrastive Learning (NT-Xent Loss)

$$
\mathcal{L}_{NT-Xent} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

where:

- $z_i, z_j$ are embeddings of augmented views of the same image
- $\tau$ is the temperature parameter (default: 0.5)
- $\text{sim}(\cdot)$ is cosine similarity

#### 2. Knowledge Distillation

$$
\mathcal{L}_{distill} = \text{MSE}(f_{student}(x), f_{teacher}(x))
$$

The teacher network is an Exponential Moving Average (EMA) of the student:

$$
\theta_{teacher} \leftarrow \alpha \cdot \theta_{teacher} + (1-\alpha) \cdot \theta_{student}
$$

where $\alpha = 0.999$ (EMA decay).

#### Combined Loss

$$
\mathcal{L}_{total} = \mathcal{L}_{NT-Xent} + \lambda \cdot \mathcal{L}_{distill}
$$

where $\lambda = 0.5$ balances the two objectives.

### Evolution Fitness Function

Network evolution is guided by a fitness function that balances performance and efficiency:

$$
F = \text{Performance} - \alpha \cdot \text{Complexity Penalty}
$$

where:

$$
\text{Complexity Penalty} = \frac{\text{FLOPs}}{\text{Target FLOPs}} + \frac{\text{Params}}{\text{Target Params}}
$$

---

## Architecture Deep Dive

### Network Evolution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Network Evolution Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Epoch 0          Epoch N           Epoch M           Epoch K               │
│  ┌─────────┐     ┌─────────┐       ┌─────────┐       ┌─────────┐            │
│  │Block 1  │     │Block 1  │       │Block 1  │       │Block 1  │            │
│  │16 ch    │     │16 ch    │       │24 ch    │       │24 ch    │            │
│  ├─────────┤     ├─────────┤       ├─────────┤       ├─────────┤            │
│  │Block 2  │     │Block 2  │       │Block 2  │       │Block 2  │            │
│  │32 ch    │     │32 ch    │       │48 ch    │       │48 ch    │            │
│  ├─────────┤     ├─────────┤       ├─────────┤       ├─────────┤            │
│  │Block 3  │     │Block 3  │       │Block 3  │       │Block 3  │            │
│  │64 ch    │     │64 ch    │       │96 ch    │       │96 ch    │            │
│  └─────────┘     ├─────────┤       ├─────────┤       ├─────────┤            │
│                  │Block 4  │       │Block 4  │       │Block 4  │            │
│                  │96 ch    │       │144 ch   │       │144 ch   │            │
│                  └─────────┘       └─────────┘       ├─────────┤            │
│                                                      │Block 5  │            │
│       GROW           WIDEN             GROW          │216 ch   │            │
│       ──────▶        ──────▶           ──────▶       └─────────┘          │
│                                                                             │
│  Seed Network    After 1st        After 1st        After 2nd                │
│  (3 blocks)      Growth           Widening         Growth                   │
│                  (4 blocks)       (4 blocks)       (5 blocks)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Convolutional Block Architecture

Each `ConvBlock` consists of:

```python
ConvBlock(
    Conv2d(in_ch, out_ch, kernel=3, padding=1)
    → BatchNorm2d(out_ch)
    → ReLU/GELU
    → MaxPool2d(2, 2)
)
```

### Projection Head for SSL

```python
ProjectionHead(
    Linear(feature_dim → 256)
    → BatchNorm1d(256)
    → ReLU
    → Linear(256 → 128)
)
```

### NASE Sparsity Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NASE Sparsity Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│   │  Importance  │────▶│    Mask      │────▶│   Sparse     │               │
│   │   Scoring    │     │  Generation  │     │   Forward    │                │
│   └──────────────┘     └──────────────┘     └──────────────┘                │
│         │                     │                     │                       │
│         ▼                     ▼                     ▼                       │
│   Taylor: |θ·∇θL|       Top-k selection       θ' = θ ⊙ mask                │
│                         by importance                                       │
│                                                                             │
│   Complementary Masks:                                                      │
│   ┌─────────────────┐   ┌─────────────────┐                                 │
│   │ Positive Mask   │   │ Negative Mask   │                                 │
│   │ (Active paths)  │   │ (Pruned paths)  │                                 │
│   └─────────────────┘   └─────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sseg_nase_research/
├── configs/                          # Configuration Management
│   ├── __init__.py
│   ├── base_config.py               # Master configuration dataclass
│   ├── config_loader.py             # YAML configuration parser
│   ├── curriculum_config.py         # Curriculum learning settings
│   ├── evaluation_config.py         # Few-shot evaluation parameters
│   ├── evolution_config.py          # SSEG/NASE evolution settings
│   ├── hardware_config.py           # GPU-specific optimizations
│   ├── ssl_config.py                # Self-supervised learning config
│   └── experiments/                 # Pre-defined experiment configs
│       ├── exp_ablation_nase_only.yaml
│       ├── exp_ablation_sseg_only.yaml
│       ├── exp_full_pipeline.yaml
│       └── exp_rtx3060_baseline.yaml
│
├── data/                            # Data Pipeline
│   ├── __init__.py
│   ├── augmentations/               # Data augmentation strategies
│   │   ├── few_shot_augmentation.py # Evaluation-time augmentations
│   │   └── ssl_augmentation.py      # SimCLR-style SSL augmentations
│   ├── benchmarks/                  # Standard few-shot datasets
│   │   ├── cifar_fs_dataset.py      # CIFAR-FS implementation (few-shot evaluation)
│   │   ├── episode_sampler.py       # N-way K-shot episode sampling
│   │   └── minimagenet_dataset.py   # Mini-ImageNet implementation (few-shot evaluation)
│   ├── curriculum/                  # Synthetic curriculum data
│   │   ├── curriculum_dataset.py    # Curriculum dataset wrapper
│   │   ├── curriculum_scheduler.py  # Level progression manager
│   │   ├── difficulty_scorer.py     # Image complexity metrics
│   │   └── synthetic_generator.py   # Procedural shape generation
│   └── datamodules/                 # Lightning DataModules
│       ├── curriculum_datamodule.py # Training data pipeline
│       └── fewshot_datamodule.py    # Evaluation data pipeline
│
├── models/                          # Neural Network Components
│   ├── __init__.py
│   ├── backbones/                   # Feature extractors
│   │   ├── conv_block.py            # Basic/Residual conv blocks
│   │   ├── evolvable_cnn.py         # Main evolvable architecture
│   │   └── seed_network.py          # Minimal starting network
│   ├── evolution/                   # Architecture evolution
│   │   ├── architecture_tracker.py  # Mutation history logging
│   │   ├── evolution_operators.py   # GROW/WIDEN operations
│   │   └── mutation_selector.py     # Sensitivity-based selection
│   ├── heads/                       # Task-specific heads
│   │   ├── projection_head.py       # SSL projection MLP
│   │   └── prototype_head.py        # Few-shot prototype classifier
│   ├── nase/                        # Sparsity components
│   │   ├── importance_scorer.py     # Parameter importance metrics
│   │   ├── mask_generator.py        # Complementary mask creation
│   │   └── sparse_router.py         # Sparse forward pass manager
│   └── ssl/                         # Self-supervised learning
│       ├── ema_teacher.py           # EMA teacher network
│       └── ssl_losses.py            # NT-Xent + Distillation losses
│
├── training/                        # Training Infrastructure
│   ├── __init__.py
│   ├── callbacks/                   # PyTorch Lightning callbacks
│   │   ├── architecture_logger.py   # Evolution logging
│   │   ├── curriculum_callback.py   # Level progression handling
│   │   ├── evolution_callback.py    # SSEG trigger management
│   │   ├── nase_callback.py         # Sparsity mask updates
│   │   └── plateau_detector.py      # Loss plateau detection
│   ├── lightning_modules/           # Training modules
│   │   ├── base_ssl_module.py       # Abstract SSL module
│   │   └── sseg_module.py           # Main SSEG training module
│   ├── optimizers/                  # Optimizer factories
│   │   └── optimizer_factory.py     # AdamW/SGD/Adam creation
│   └── schedulers/                  # LR scheduler factories
│       └── lr_scheduler_factory.py  # Cosine/Step/Warmup schedulers
│
├── evaluation/                      # Evaluation Framework
│   ├── __init__.py
│   ├── evaluators/                  # Evaluation engines
│   │   ├── efficiency_evaluator.py  # FLOPs/params/latency metrics
│   │   └── fewshot_evaluator.py     # Episode-based evaluation
│   ├── metrics/                     # Metric implementations
│   │   ├── accuracy_metrics.py      # Top-1/Top-k accuracy
│   │   ├── complexity_metrics.py    # Model complexity analysis
│   │   └── confidence_interval.py   # Statistical intervals
│   └── protocols/                   # Evaluation standards
│       ├── benchmark_protocol.py    # Full benchmark pipeline
│       └── episode_protocol.py      # Single episode evaluation
│
├── experiments/                     # Experiment Management
│   ├── __init__.py
│   ├── ablation_studies/            # Ablation experiments
│   │   ├── ablation_configs.py      # 7 ablation configurations
│   │   └── ablation_runner.py       # Automated ablation executor
│   └── comparisons/                 # Baseline comparisons
│       └── baseline_comparisons.py  # ProtoNet/MAML/SimCLR/DINO
│
├── scripts/                         # Executable Scripts
│   ├── analyze_evolution.py         # Architecture evolution analysis
│   ├── evaluate_fewshot.py          # Few-shot evaluation runner
│   ├── export_results.py            # Result export utilities
│   ├── generate_curriculum.py       # Curriculum data generation
│   ├── run_ablation.py              # Ablation study runner
│   └── train_sseg.py                # Main training script
│
├── utils/                           # Utility Functions
│   ├── __init__.py
│   ├── hardware/                    # Hardware monitoring
│   │   └── gpu_memory_tracker.py    # VRAM usage tracking
│   ├── io/                          # I/O utilities
│   │   ├── checkpoint_manager.py    # Model saving/loading
│   │   └── config_loader.py         # YAML/JSON parsing
│   ├── logging/                     # Logging utilities
│   │   └── custom_logger.py         # Enhanced logging
│   └── reproducibility/             # Reproducibility tools
│       └── seed_everything.py       # Deterministic seeding
│
├── visualization/                   # Visualization Tools
│   ├── __init__.py
│   ├── loggers/                     # Experiment logging
│   ├── plotters/                    # Matplotlib visualizations
│   │   ├── accuracy_plotter.py      # Accuracy curves
│   │   ├── embedding_plotter.py     # t-SNE/UMAP embeddings
│   │   ├── evolution_plotter.py     # Architecture trajectory
│   │   └── loss_plotter.py          # Training loss curves
│   └── reporters/                   # Report generation
│       ├── latex_table_generator.py # LaTeX tables for papers
│       ├── markdown_reporter.py     # Markdown documentation
│       └── result_formatter.py      # Result formatting
│
├── tests/                           # Test Suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_data/                   # Data pipeline tests
│   ├── test_evaluation/             # Evaluation tests
│   ├── test_models/                 # Model architecture tests
│   └── test_training/               # Training pipeline tests
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU training)
- 12GB+ GPU VRAM (RTX 3060 or better)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/sseg_nase_research.git
cd sseg_nase_research
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```


### Step 4: Prepare Datasets (Optional)

For few-shot evaluation on standard benchmarks (**Mini-ImageNet** and **CIFAR-FS**):

```bash
# Download Mini-ImageNet
mkdir -p datasets/minimagenet
# Follow instructions at: https://github.com/twitter/meta-learning-lstm

# Download CIFAR-FS
mkdir -p datasets/cifar_fs
# Download from: https://github.com/bertinetto/r2d2
```

Both datasets must be organized as:

```
datasets/
  minimagenet/
    train/
    val/
    test/
  cifar_fs/
    train/
    val/
    test/
```

---

## Quick Start Guide

### Running the Full Pipeline with CLI Runner

You can run the entire pipeline (curriculum, training, evaluation, etc.) using the CLI runner script:

```bash
# For RTX 3050
python scripts_cli_run.py configs_experiments_exp_full_pipeline_rtx3050.json

# For RTX 3060
python scripts_cli_run.py configs_experiments_exp_full_pipeline_rtx3060.json
```

Pastikan file konfigurasi JSON sudah sesuai hardware dan kebutuhan Anda. Pipeline akan berjalan otomatis sesuai urutan dan tahapan yang diaktifkan di file config.

### Training with Default Configuration


```bash
# For RTX 3050 (4GB)
python scripts/train_sseg.py \
  --experiment-name my_first_experiment \
  --hardware rtx3050 \
  --max-epochs 100 \
  --seed 42

# For RTX 3060 (12GB)
python scripts/train_sseg.py \
  --experiment-name my_first_experiment \
  --hardware rtx3060 \
  --max-epochs 100 \
  --seed 42
```

### Training with Custom Configuration

```bash
python scripts/train_sseg.py \
    --config configs/experiments/exp_full_pipeline.yaml \
    --output-dir ./outputs \
    --data-dir ./datasets
```


### Evaluating a Trained Model (Mini-ImageNet or CIFAR-FS)

#### Mini-ImageNet
```bash
python scripts/evaluate_fewshot.py \
  --checkpoint outputs/my_experiment/checkpoints/best.ckpt \
  --data-dir datasets/minimagenet \
  --num-ways 5 \
  --num-shots 1 5 \
  --num-episodes 600
```

#### CIFAR-FS
```bash
python scripts/evaluate_fewshot.py \
  --checkpoint outputs/my_experiment/checkpoints/best.ckpt \
  --data-dir datasets/cifar_fs \
  --num-ways 5 \
  --num-shots 1 5 \
  --num-episodes 600
```

You can also configure both datasets for evaluation in the pipeline JSON config (see Evaluation & Benchmarking section).

### Running Ablation Studies


```bash
# For RTX 3050 (4GB)
python scripts/run_ablation.py \
  --output-dir outputs/ablation \
  --hardware rtx3050 \
  --max-epochs 100

# For RTX 3060 (12GB)
python scripts/run_ablation.py \
  --output-dir outputs/ablation \
  --hardware rtx3060 \
  --max-epochs 100
```

---

## Detailed Module Documentation

### 1. Configuration System (`configs/`)

The configuration system uses Python dataclasses for type safety and validation.

#### BaseConfig

The master configuration that aggregates all sub-configurations:

```python
@dataclass
class BaseConfig:
    experiment_name: str           # Unique experiment identifier
    seed: int                      # Random seed for reproducibility
    paths: PathConfig              # Directory structure
    hardware: HardwareConfig       # GPU-specific settings
    curriculum: CurriculumConfig   # Curriculum learning settings
    evolution: EvolutionConfig     # SSEG parameters
    ssl: SSLConfig                 # Self-supervised learning config
    evaluation: EvaluationConfig   # Few-shot evaluation settings
```

#### Hardware Configurations

Pre-defined profiles for different GPUs:

| Profile  | Batch Size | Precision  | Accumulation | Memory |
| -------- | ---------- | ---------- | ------------ | ------ |
| RTX 3060 | 48         | 16-mixed   | 3            | 12 GB  |
| RTX 3050 | 24         | 16-mixed   | 4            | 4 GB   |

#### Evolution Configuration

```python
@dataclass
class EvolutionConfig:
    seed_network: SeedNetworkConfig  # Initial architecture
    growth: GrowthConfig             # Growth constraints
    nase: NASEConfig                 # Sparsity settings
    fitness: FitnessConfig           # Evolution fitness function
```

Key parameters:

- `max_blocks`: Maximum number of conv blocks (default: 12)
- `max_channels`: Maximum channels per layer (default: 256)
- `channel_expansion_ratio`: Growth rate for widening (default: 1.5)
- `sparsity_ratio`: Target sparsity level (default: 0.3)
- `plateau_threshold`: Loss change threshold (default: 1e-4)

### 2. Data Pipeline (`data/`)

#### Curriculum Dataset

The `CurriculumDataset` generates synthetic training data with progressive difficulty:

```python
# Level 1: Basic Shapes
generator.generate_basic_shape()  # Circles, squares, triangles

# Level 2: Textures
generator.generate_texture()  # Sinusoidal patterns

# Level 3: Complex Objects
generator.generate_complex_object()  # Multi-shape compositions

# Level 4: Adversarial
generator.generate_adversarial()  # Perturbation-based examples
```

#### Difficulty Scoring

Images are scored based on three metrics:

1. **Edge Density**: Sobel filter magnitude
2. **Color Variance**: Per-channel variance
3. **Spatial Frequency**: FFT-based high-frequency content

```python
score = 0.4 * edge_density + 0.3 * color_variance + 0.3 * spatial_frequency
```

#### Episode Sampling

For few-shot evaluation:

```python
sampler = EpisodeSampler(
    dataset=dataset,
    num_ways=5,        # Classes per episode
    num_shots=5,       # Support samples per class
    num_queries=15,    # Query samples per class
    num_episodes=600   # Total evaluation episodes
)
```

### 3. Model Architecture (`models/`)

#### EvolvableCNN

The core evolvable architecture:

```python
model = EvolvableCNN(
    seed_config=SeedNetworkConfig(
        architecture="cnn",
        initial_channels=16,
        initial_blocks=3,
        kernel_size=3,
        activation="relu",
        use_batch_norm=True,
        use_pooling=True
    ),
    evolution_config=evolution_config
)

# Architecture operations
model.grow(out_channels=96)      # Add new block
model.widen(block_idx=1)         # Widen existing block
model.get_architecture_summary() # Get current state
```

#### NASE Sparse Router

```python
router = SparseRouter(nase_config)

# Update importance masks
router.update_masks(model)

# Forward with sparsity
output = router.apply_sparse_forward(model, x, use_negative_path=False)

# Permanent pruning
router.prune_permanently(model)
```

### 4. Training Pipeline (`training/`)

#### SSEGModule

The main Lightning module orchestrating training:

```python
class SSEGModule(BaseSSLModule):
    def training_step(self, batch, batch_idx):
        # 1. Generate augmented views
        view1, view2 = self._augmentation(image)

        # 2. Extract features
        features1 = self._backbone(view1)
        features2 = self._backbone(view2)

        # 3. Project to embedding space
        z1 = self._projection_head(features1)
        z2 = self._projection_head(features2)

        # 4. Get teacher predictions
        teacher_features = self._teacher(view1)

        # 5. Compute combined loss
        loss = self._ssl_loss(z1, z2, features1, teacher_features)

        # 6. Update EMA teacher
        self._teacher.update(self._backbone)

        return loss
```

#### Evolution Callback

Monitors training and triggers architecture evolution:

```python
class EvolutionCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 1. Check for plateau
        plateau_status = self._plateau_detector.check_plateau()

        # 2. Decide evolution action
        if plateau_status.should_evolve:
            self._trigger_evolution(trainer, pl_module)
        elif plateau_status.should_advance_level:
            self._trigger_level_advance(trainer, pl_module)
```

### 5. Evaluation System (`evaluation/`)


#### Few-Shot Evaluation (Mini-ImageNet & CIFAR-FS)

The evaluation system supports both **Mini-ImageNet** and **CIFAR-FS**. The evaluation script will automatically use the correct dataset class based on the `--data-dir` argument (or config). For Mini-ImageNet, it uses `MiniImageNetDataset`; for CIFAR-FS, it uses `CIFARFSDataset`.

```python
evaluator = FewShotEvaluator(
  model=model,
  config=FewShotConfig(
    num_ways=5,
    num_shots=(1, 5),
    num_queries_per_class=15,
    num_episodes=600,
    distance_metric="cosine",
    normalize_features=True
  ),
  device="cuda"
)

# Run evaluation
confidence_interval, accuracies = evaluator.run_evaluation(episode_sampler)
print(f"Accuracy: {confidence_interval}")  # e.g., "68.42±0.35"
```

**Note:** To evaluate on CIFAR-FS, simply set `--data-dir datasets/cifar_fs` or the equivalent config. The protocol and metrics are identical for both datasets.

#### Efficiency Metrics

```python
efficiency_evaluator = EfficiencyEvaluator(
    model=model,
    device="cuda",
    input_size=(1, 3, 84, 84)
)

metrics = efficiency_evaluator.evaluate()
# metrics.params_millions    → e.g., 0.85
# metrics.flops_giga         → e.g., 0.72
# metrics.inference_time_ms  → e.g., 2.3
# metrics.memory_mb          → e.g., 3.4
```

---

## Experiment Configurations


### Full Pipeline (`exp_full_pipeline.yaml`)

Complete SSEG-NASE with all components:

```yaml
experiment_name: sseg_nase_full_pipeline
seed: 42


# Example hardware config for RTX 3050 (4GB)
hardware:
  name: rtx3050
  batch_size: 24
  gradient_accumulation_steps: 4
  num_workers: 2
  precision: 16-mixed

# Example hardware config for RTX 3060 (12GB)
hardware:
  name: rtx3060
  batch_size: 48
  gradient_accumulation_steps: 3
  num_workers: 4
  precision: 16-mixed

curriculum:
  image_size: 84
  transition_strategy: gradual
  gradual_mixing_ratio: 0.2

evolution:
  seed_network:
    initial_channels: 16
    initial_blocks: 3
  growth:
    max_blocks: 12
    max_channels: 256
    plateau_threshold: 0.0001
  nase:
    sparsity_ratio: 0.3
    importance_metric: taylor

ssl:
  contrastive:
    temperature: 0.5
  distillation:
    ema_decay: 0.999
    distillation_weight: 0.5
```


#### Evaluation on Both Datasets (Config Example)

In your pipeline config (e.g., `configs_experiments_exp_full_pipeline_rtx3050.json`), you can specify both datasets for evaluation:

```json
  "evaluate": {
    "datasets": [
      {"name": "minimagenet", "root_dir": "datasets/minimagenet", "split": "test"},
      {"name": "cifar_fs", "root_dir": "datasets/cifar_fs", "split": "test"}
    ],
    "num_ways": 5,
    "num_shots": [1, 5],
    "num_episodes": 600,
    "device": "cuda",
    "seed": 42
  }
```

This will run evaluation on both Mini-ImageNet and CIFAR-FS sequentially.

Network evolution without NASE sparsity:

```yaml
evolution:
  nase:
    sparsity_ratio: 0.0 # No sparsity
    pruning_interval_epochs: 999999 # Never prune
```

### NASE-Only Ablation (`exp_ablation_nase_only.yaml`)

Fixed architecture with sparsity:

```yaml
evolution:
  seed_network:
    initial_channels: 32
    initial_blocks: 4
  growth:
    plateau_window_size: 999999 # Never evolve
```

---

## Evaluation & Benchmarking

### Standard Few-Shot Benchmarks


The framework supports evaluation on **both** standard few-shot benchmarks:

1. **Mini-ImageNet**: 100 classes, 600 images/class, 84×84 resolution
2. **CIFAR-FS**: 100 classes, 600 images/class, 32×32 (resized to 84×84)

You can evaluate on either or both datasets by specifying the appropriate `--data-dir` (for CLI) or by configuring the evaluation section in your pipeline JSON (see below). Both datasets are fully supported in the evaluation pipeline and config files.

#### Example: Evaluating on Mini-ImageNet

```bash
python scripts/evaluate_fewshot.py \
    --checkpoint outputs/my_experiment/checkpoints/best.ckpt \
    --data-dir datasets/minimagenet \
    --num-ways 5 \
    --num-shots 1 5 \
    --num-episodes 600
```

#### Example: Evaluating on CIFAR-FS

```bash
python scripts/evaluate_fewshot.py \
    --checkpoint outputs/my_experiment/checkpoints/best.ckpt \
    --data-dir datasets/cifar_fs \
    --num-ways 5 \
    --num-shots 1 5 \
    --num-episodes 600
```

#### Example: Config-based Evaluation (Full Pipeline)

In `configs_experiments_exp_full_pipeline_rtx3050.json`:

```json
  "evaluate": {
    "datasets": [
      {"name": "minimagenet", "root_dir": "datasets/minimagenet", "split": "test"},
      {"name": "cifar_fs", "root_dir": "datasets/cifar_fs", "split": "test"}
    ],
    "num_ways": 5,
    "num_shots": [1, 5],
    "num_episodes": 600,
    "device": "cuda",
    "seed": 42
  }
```

This will run evaluation on both datasets sequentially and output results for each.

**Note:** The evaluation protocol (N-way K-shot, 600 episodes, etc.) is identical for both datasets. The only difference is the dataset root directory and split.

### Evaluation Protocol

Following standard few-shot learning protocols:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Few-Shot Evaluation Protocol                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each of 600 episodes:                                                  │
│                                                                             │
│  1. Sample 5 random classes (N-way = 5)                                     │
│  2. For each class:                                                         │
│     - Sample K support images (K-shot ∈ {1, 5})                             │
│     - Sample 15 query images                                                │
│  3. Compute class prototypes from support set                               │
│  4. Classify queries using nearest prototype                                │
│  5. Record episode accuracy                                                 │
│                                                                             │
│  Report: Mean accuracy ± 95% confidence interval                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Baseline Comparisons

| Method              | Backbone          | Params        | FLOPs         | 1-shot        | 5-shot        |
| ------------------- | ----------------- | ------------- | ------------- | ------------- | ------------- |
| ProtoNet            | Conv-4            | 0.11M         | 50M           | 49.42         | 68.20         |
| MatchingNet         | Conv-4            | 0.11M         | 50M           | 43.56         | 55.31         |
| MAML                | Conv-4            | 0.11M         | 50M           | 48.70         | 63.11         |
| RelationNet         | Conv-4            | 0.23M         | 90M           | 50.44         | 65.32         |
| SimCLR+Linear       | ResNet-18         | 11.2M         | 1.82G         | 51.23         | 69.45         |
| DINO+Linear         | ViT-S             | 22M           | 4.61G         | 53.12         | 71.20         |
| **SSEG-NASE** | **Evolved** | **<1M** | **<1G** | **TBD** | **TBD** |

---

## Ablation Studies

### Ablation Configurations

| ID | Configuration       | SSEG | NASE | Curriculum | Distillation |
| -- | ------------------- | ---- | ---- | ---------- | ------------ |
| A1 | Seed Only           | ✗   | ✗   | ✗         | ✗           |
| A2 | SSL Only            | ✗   | ✗   | ✗         | ✗           |
| A3 | SSEG Only           | ✓   | ✗   | ✗         | ✗           |
| A4 | SSEG + Curriculum   | ✓   | ✗   | ✓         | ✗           |
| A5 | SSEG + Distillation | ✓   | ✗   | ✗         | ✓           |
| A6 | SSEG + NASE         | ✓   | ✓   | ✗         | ✗           |
| A7 | Full Pipeline       | ✓   | ✓   | ✓         | ✓           |

### Running Ablations

```bash
# Run all ablations
python scripts/run_ablation.py --ablations all

# Run specific ablations
python scripts/run_ablation.py --ablations SSEG_ONLY FULL_PIPELINE
```

---

## Hardware Optimization

### RTX 3060 & 3050 Optimization Strategy

#### RTX 3060 (12GB)

1. **Mixed Precision (FP16)**: Reduces memory footprint by ~50%
2. **Gradient Accumulation**: Simulates larger batch sizes (48 × 3 = 144 effective)
3. **Gradient Checkpointing**: Trades compute for memory
4. **Efficient Data Loading**: Persistent workers, pin memory

```python
@dataclass
class RTX3050Config(HardwareConfig):
  device: str = "cuda"
  precision: str = "16-mixed"
  batch_size: int = 24
  gradient_accumulation_steps: int = 4
  num_workers: int = 2
  pin_memory: bool = True
  persistent_workers: bool = True
  max_memory_gb: float = 4.0

@dataclass
class RTX3060Config(HardwareConfig):
  device: str = "cuda"
  precision: str = "16-mixed"
  batch_size: int = 48
  gradient_accumulation_steps: int = 3
  num_workers: int = 4
  pin_memory: bool = True
  persistent_workers: bool = True
  max_memory_gb: float = 12.0
```

class RTX3050Config(HardwareConfig):
  device: str = "cuda"
  precision: str = "16-mixed"
  batch_size: int = 24
  gradient_accumulation_steps: int = 4
  num_workers: int = 2
  pin_memory: bool = True
  persistent_workers: bool = True
  max_memory_gb: float = 4.0
```

### Memory Monitoring

```python
from utils.hardware.gpu_memory_tracker import GPUMemoryTracker

tracker = GPUMemoryTracker(
    device_id=0,
    warning_threshold_percent=85.0,
    critical_threshold_percent=95.0
)

# Take snapshots during training
snapshot = tracker.take_snapshot("after_forward")
print(f"Memory: {snapshot.allocated_mb:.1f} MB / {snapshot.total_mb:.1f} MB")
```

---

## Visualization & Reporting

### Evolution Trajectory Plot

```python
from visualization.plotters.evolution_plotter import EvolutionPlotter

plotter = EvolutionPlotter(output_dir=Path("./plots"))

plotter.plot_architecture_trajectory(
    epochs=epochs,
    num_params=params_history,
    num_blocks=blocks_history,
    mutation_epochs=mutation_points,
    level_boundaries=level_transitions
)
```

### Markdown Reports

```python
from visualization.reporters.markdown_reporter import MarkdownReporter

reporter = MarkdownReporter(output_dir=Path("./reports"))

# Generate experiment report
report = reporter.generate_experiment_report(summary)

# Generate comparison table
table = reporter.generate_comparison_table(
    methods=["ProtoNet", "MAML", "SSEG-NASE"],
    backbones=["Conv-4", "Conv-4", "Evolved"],
    # ...
)
```

### LaTeX Tables

For academic papers:

```python
from visualization.reporters.latex_table_generator import LatexTableGenerator

generator = LatexTableGenerator()
latex = generator.generate_results_table(results)
```

---

## Reproducibility

### Deterministic Training

```python
from utils.reproducibility.seed_everything import seed_everything

# Set all random seeds
seed_everything(seed=42, deterministic=True)
```

This sets:

- Python's `random` module
- NumPy's random state
- PyTorch's random generators (CPU + CUDA)
- CUDNN deterministic mode
- Environment variables for hash reproducibility

### Saving/Loading Random State

```python
from utils.reproducibility.seed_everything import get_random_state, set_random_state

# Save state
state = get_random_state()
state_dict = state.to_dict()

# Restore state
set_random_state(state)
```

---

## API Reference

### Core Classes

#### `EvolvableCNN`

- `grow(out_channels: Optional[int]) -> bool`: Add new block
- `widen(block_idx: int) -> bool`: Widen existing block
- `forward(x: Tensor) -> Tensor`: Feature extraction
- `get_architecture_summary() -> dict`: Architecture statistics

#### `SparseRouter`

- `update_masks(model: nn.Module) -> None`: Compute importance masks
- `apply_sparse_forward(model, x, use_negative_path) -> Tensor`: Masked forward
- `prune_permanently(model: nn.Module) -> None`: Apply permanent pruning
- `get_statistics() -> SparsityStatistics`: Sparsity metrics

#### `SSEGModule`

- `training_step(batch, batch_idx) -> Tensor`: Training logic
- `evolve_network(mutation_type, target_idx) -> bool`: Apply evolution
- `get_loss_history() -> tuple[list, list]`: Training history

#### `FewShotEvaluator`

- `evaluate_episode(episode: Episode) -> float`: Single episode
- `run_evaluation(sampler) -> tuple[ConfidenceInterval, list]`: Full evaluation

---

## Troubleshooting


### Evaluation Dataset Selection

**Symptom:** Evaluation fails or loads the wrong dataset.

**Solutions:**
1. Ensure you specify the correct `--data-dir` (e.g., `datasets/minimagenet` or `datasets/cifar_fs`).
2. For config-based runs, check that the `evaluate.datasets` field lists the correct root directories and splits for each dataset.
3. The evaluation script will use `MiniImageNetDataset` for Mini-ImageNet and `CIFARFSDataset` for CIFAR-FS automatically.
4. If you add new datasets, ensure the evaluation script is updated to support them.

---

### Common Issues

#### Out of Memory (OOM)

**Symptom**: CUDA out of memory error

**Solutions**:

1. Reduce batch size in hardware config
2. Increase gradient accumulation steps
3. Enable gradient checkpointing
4. Use FP16 precision

```python
# Reduce memory usage (example for RTX3050)
hardware_config = RTX3050Config()
hardware_config.batch_size = 16  # Reduce from 24 if OOM
hardware_config.gradient_accumulation_steps = 6  # Increase if needed
```

#### Training Not Converging

**Symptom**: Loss stays flat or increases

**Solutions**:

1. Check learning rate (try 1e-3 to 1e-4)
2. Verify data augmentation is not too aggressive
3. Ensure curriculum progression is working
4. Check SSL loss temperature

#### Evolution Not Triggering

**Symptom**: Network never grows

**Solutions**:

1. Reduce `plateau_threshold` (e.g., 1e-5)
2. Reduce `plateau_window_size` (e.g., 5)
3. Lower `distillation_gap_threshold`

---

## Citation

If you use SSEG-NASE in your research, please cite:

```bibtex
@article{sseg_nase_2024,
  title={SSEG-NASE: Self-Supervised Evolution with Guided Neural Architecture Sparse Evolution for Hardware-Efficient Few-Shot Learning},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## Acknowledgments

This research was conducted at **Telkom University** as part of ongoing research in efficient machine learning systems.

### Frameworks & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Lightning](https://lightning.ai/) - Training orchestration
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Experiment tracking

### Inspirations

- SimCLR: A Simple Framework for Contrastive Learning (Chen et al., 2020)
- Prototypical Networks (Snell et al., 2017)
- Network Morphism (Wei et al., 2016)
- DINO: Self-Supervised Vision Transformers (Caron et al., 2021)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>SSEG-NASE</strong> - Efficient Few-Shot Learning Through Neural Architecture Evolution
</p>
