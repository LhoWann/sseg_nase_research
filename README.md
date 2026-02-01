# SSEG-NASE

**Self-Supervised Evolution-Guided Neural Architecture Search for Few-Shot Learning**

A novel research framework that combines self-supervised learning, neural architecture evolution, and curriculum learning to achieve efficient few-shot image classification.

---

## Overview

SSEG-NASE introduces a unified approach to few-shot learning by dynamically evolving network architectures during self-supervised pretraining. The framework starts with a minimal seed network and progressively grows its capacity based on learning signals, guided by a curriculum of increasing difficulty.

### Key Innovations

| Component | Description |
|-----------|-------------|
| **Evolvable CNN** | Dynamic backbone that supports grow and widen mutations |
| **NASE Module** | Neural Architecture Search via Sparsity-based routing |
| **DINO-based SSL** | Self-distillation with multi-crop augmentation strategy |
| **Curriculum Learning** | Synthetic data generation with progressive difficulty levels |
| **Evolution Operators** | Gradient-sensitivity-guided network mutations |

---

## Architecture

```
sseg_nase_research/
├── configs/                    # Configuration management
│   ├── base_config.py          # Base configuration dataclass
│   ├── curriculum_config.py    # Curriculum learning settings
│   ├── evolution_config.py     # Network evolution parameters
│   ├── ssl_config.py           # Self-supervised learning config
│   └── experiments/            # Experiment-specific YAML configs
│
├── models/                     # Neural network components
│   ├── backbones/              # EvolvableCNN, SeedNetwork
│   ├── evolution/              # MutationSelector, EvolutionOperators
│   ├── nase/                   # SparseRouter, ImportanceScorer
│   ├── ssl/                    # EMATeacher, DINOLoss, NTXentLoss
│   └── heads/                  # ProjectionHead, PrototypeHead, RotationHead
│
├── training/                   # Training infrastructure
│   ├── lightning_modules/      # SSEGModule (main training module)
│   ├── callbacks/              # Evolution, checkpoint, logging callbacks
│   ├── optimizers/             # AdamW optimizer factory
│   └── schedulers/             # Cosine + warmup LR scheduler
│
├── data/                       # Data handling
│   ├── curriculum/             # SyntheticGenerator, DifficultyScorer
│   ├── benchmarks/             # MiniImageNet, CIFAR-FS loaders
│   ├── datamodules/            # PyTorch Lightning data modules
│   └── augmentations/          # SSL augmentation pipeline
│
├── evaluation/                 # Evaluation framework
│   ├── protocols/              # ProtoNet, MatchingNet protocols
│   ├── metrics/                # Accuracy, confidence intervals
│   └── evaluators/             # Few-shot episode evaluator
│
├── experiments/                # Experiment utilities
│   ├── ablation_studies/       # Component ablation runners
│   └── comparisons/            # Baseline comparison tools
│
├── visualization/              # Visualization tools
│   ├── plotters/               # Training curves, architecture evolution
│   ├── reporters/              # LaTeX, Markdown, CSV exporters
│   └── loggers/                # TensorBoard integration
│
├── scripts/                    # Executable scripts
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit and integration tests
└── utils/                      # Utility functions
```

---

## Method

### 1. Self-Supervised Pretraining

The framework employs DINO-style self-distillation with an exponential moving average (EMA) teacher:

- **Multi-crop strategy**: 2 global crops + 6 local crops
- **Contrastive loss**: NTXent with temperature scaling
- **Distillation loss**: MSE/Cosine/KL between student-teacher features
- **Rotation prediction**: Auxiliary task for representation learning

### 2. Network Evolution

The evolvable backbone starts minimal and grows based on gradient sensitivity:

```
Seed Network (32 channels, 2 blocks)
        ↓ [sensitivity analysis]
Mutation Selection (GROW / WIDEN)
        ↓ [apply mutation]
Evolved Network (expanded capacity)
        ↓ [continue training]
        ... (repeat based on plateau detection)
```

**Evolution Operators:**
- `GROW`: Add a new convolutional block
- `WIDEN`: Increase channel count of existing layers
- Fitness function balances accuracy vs. computational cost

### 3. NASE Sparsity Routing

The Sparse Router dynamically masks network parameters:

- **Importance scoring** based on gradient magnitude or Taylor expansion
- **Complementary masks**: positive path (important) + negative path (pruned)
- Enables efficient architecture search without full network enumeration

### 4. Curriculum Learning

Synthetic data with progressive difficulty levels:

| Level | Description |
|-------|-------------|
| 1 | Basic geometric shapes (circle, square, triangle) |
| 2 | Multi-frequency textures with noise |
| 3 | Complex multi-object compositions |
| 4 | Adversarial samples with occlusion and perturbations |

---

## Installation

```bash
git clone https://github.com/your-repo/sseg_nase_research.git
cd sseg_nase_research

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- PyTorch Lightning >= 2.2.0
- CUDA-compatible GPU (recommended)

---

## Quick Start

### Full Pipeline

Run the complete training and evaluation pipeline:

```bash
python scripts_cli_run.py configs_experiments_exp_full_pipeline.json
```

This executes:
1. Curriculum data generation (optional)
2. Self-supervised training with evolution
3. Few-shot evaluation on benchmarks
4. Ablation studies
5. Evolution analysis
6. Result export

### Individual Scripts

```bash
# Generate curriculum data
python scripts/generate_curriculum.py --config configs/experiments/exp_full_pipeline.yaml

# Train the model
python scripts/train_sseg.py --config configs/experiments/exp_full_pipeline.yaml

# Evaluate few-shot performance
python scripts/evaluate_fewshot.py --checkpoint outputs/checkpoints/best.ckpt

# Run ablation studies
python scripts/run_ablation.py --config configs/experiments/exp_full_pipeline.yaml

# Analyze evolution history
python scripts/analyze_evolution.py --output_dir outputs/full_pipeline

# Export results
python scripts/export_results.py --format latex markdown csv json
```

---

## Configuration

### Experiment Configuration (YAML)

```yaml
evolution:
  seed_network:
    initial_channels: 32
    num_blocks: 2
  growth:
    max_blocks: 6
    max_channels: 256
    sensitivity_method: taylor

ssl:
  contrastive:
    temperature: 0.1
  distillation:
    ema_decay: 0.996
  rotation:
    enabled: true

curriculum:
  image_size: 84
  levels: [1, 2, 3, 4]
```

### Pipeline Configuration (JSON)

```json
{
  "stages": {
    "generate_curriculum": false,
    "train": true,
    "evaluate": true,
    "ablation": true,
    "analyze": true,
    "export": true
  }
}
```

---

## Evaluation Benchmarks

| Dataset | Classes | Images | Split |
|---------|---------|--------|-------|
| MiniImageNet | 100 | 60,000 | 64/16/20 |
| CIFAR-FS | 100 | 60,000 | 64/16/20 |

### Evaluation Protocol

- **N-way K-shot**: 5-way 1-shot, 5-way 5-shot
- **Episodes**: 600 test episodes
- **Metric**: Mean accuracy with 95% confidence interval

---

## Notebooks

Interactive Jupyter notebooks for exploration:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Curriculum data visualization |
| `02_architecture_visualization.ipynb` | Network evolution visualization |
| `03_evolution_analysis.ipynb` | Mutation history analysis |
| `04_result_visualization.ipynb` | Performance plots and comparisons |

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models/
pytest tests/test_training/
pytest tests/test_evaluation/
```

---

## Project Structure Details

### Models

- **EvolvableCNN**: Backbone with dynamic block addition and channel expansion
- **SparseRouter**: NASE sparsity-based parameter routing
- **EMATeacher**: Exponential moving average for self-distillation
- **ProjectionHead**: MLP projector for contrastive learning
- **PrototypeHead**: Prototype-based classification for few-shot

### Training

- **SSEGModule**: Main PyTorch Lightning module integrating all components
- **EvolutionCallback**: Triggers network mutations based on plateau detection
- **Optimizers**: AdamW with weight decay
- **Schedulers**: Cosine annealing with linear warmup

### Data

- **SyntheticGenerator**: Procedural curriculum data generation
- **CurriculumScheduler**: Difficulty progression management
- **SSLAugmentation**: Multi-crop augmentation for DINO training

---

## Citation

```bibtex
@article{sseg_nase_2026,
  title={SSEG-NASE: Self-Supervised Evolution-Guided Neural Architecture Search for Few-Shot Learning},
  author={Author Name},
  year={2026}
}
```

---

## License

This project is for research purposes.

---

## Acknowledgments

- DINO (Self-Distillation with No Labels)
- ProtoNet (Prototypical Networks for Few-Shot Learning)
- NetAdapt and NAS literature for architecture search inspiration
