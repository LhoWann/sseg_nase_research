---

## Research Design & Methodology

### Research Approach Overview

This research applies an **experimental approach** through the development of a **Scalable Self-Supervised Pretraining Pipeline** framework. The system is specifically designed to overcome computational inefficiencies in static architectures for **Few-Shot Image Classification (FSIC)** tasks by utilizing **automatic network evolution mechanisms**.

The research methodology is carried out in **two interconnected main phases**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                        SSEG-NASE: TWO-PHASE RESEARCH METHODOLOGY                            │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         PHASE 1: EVOLUTIONARY PRETRAINING                             ║  │
│  ║                                                                                       ║  │
│  ║   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           ║  │
│  ║   │   LEVEL 1   │───▶│   LEVEL 2   │───▶│   LEVEL 3   │───▶│   LEVEL 4   │           ║  │
│  ║   │   Basic     │    │   Texture   │    │   Object    │    │ Adversarial │           ║  │
│  ║   │   Shapes    │    │   Patterns  │    │   Scenes    │    │   Examples  │           ║  │
│  ║   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘           ║  │
│  ║          │                  │                  │                  │                  ║  │
│  ║          ▼                  ▼                  ▼                  ▼                  ║  │
│  ║   ┌──────────────────────────────────────────────────────────────────────────┐       ║  │
│  ║   │                    SSEG + NASE Iterative Loop                            │       ║  │
│  ║   │  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │       ║  │
│  ║   │  │  SSL Train │───▶│  Plateau   │───▶│   SSEG     │───▶│   NASE     │   │       ║  │
│  ║   │  │            │    │  Detect    │    │   Evolve   │    │   Sparsify │   │       ║  │
│  ║   │  └────────────┘    └────────────┘    └────────────┘    └────────────┘   │       ║  │
│  ║   │        ▲                                                      │          │       ║  │
│  ║   │        └──────────────────────────────────────────────────────┘          │       ║  │
│  ║   └──────────────────────────────────────────────────────────────────────────┘       ║  │
│  ║                                        │                                              ║  │
│  ║                                        ▼                                              ║  │
│  ║                            ┌───────────────────────┐                                  ║  │
│  ║                            │   EVOLVED BACKBONE    │                                  ║  │
│  ║                            │   (Optimized CNN)     │                                  ║  │
│  ║                            └───────────────────────┘                                  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════╝  │
│                                        │                                                    │
│                                        ▼                                                    │
│  ╔═══════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         PHASE 2: FEW-SHOT ADAPTATION                                  ║  │
│  ║                                                                                       ║  │
│  ║   ┌───────────────────────────────────────────────────────────────────────────────┐   ║  │
│  ║   │                        FROZEN BACKBONE                                        │   ║  │
│  ║   │                     (No Weight Updates)                                       │   ║  │
│  ║   └───────────────────────────────────────────────────────────────────────────────┘   ║  │
│  ║                                        │                                              ║  │
│  ║              ┌─────────────────────────┼─────────────────────────┐                   ║  │
│  ║              ▼                         ▼                         ▼                   ║  │
│  ║   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐        ║  │
│  ║   │   Mini-ImageNet    │   │     CIFAR-FS        │   │  Tiered-ImageNet   │        ║  │
│  ║   │   (100 classes)    │   │   (100 classes)     │   │   (608 classes)    │        ║  │
│  ║   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘        ║  │
│  ║              │                         │                         │                   ║  │
│  ║              ▼                         ▼                         ▼                   ║  │
│  ║   ┌─────────────────────────────────────────────────────────────────────────────┐    ║  │
│  ║   │                    N-WAY K-SHOT EPISODE EVALUATION                         │    ║  │
│  ║   │                                                                            │    ║  │
│  ║   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │    ║  │
│  ║   │   │  5-way 1-shot│    │  5-way 5-shot│    │  Efficiency  │                │    ║  │
│  ║   │   │  Accuracy    │    │  Accuracy    │    │  Metrics     │                │    ║  │
│  ║   │   └──────────────┘    └──────────────┘    └──────────────┘                │    ║  │
│  ║   └─────────────────────────────────────────────────────────────────────────────┘    ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Evolutionary Pretraining

The first phase focuses on **training a seed network and dynamically growing it** using a synthetic data curriculum. This phase is where the core innovation of SSEG-NASE operates.

#### 1.1 Synthetic Data Curriculum

The curriculum provides progressively complex training data to guide network evolution:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CURRICULUM COMPLEXITY PROGRESSION                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Complexity                                                                     │
│     ▲                                                                           │
│     │                                              ┌─────────────────┐          │
│  1.0│                                              │   LEVEL 4:      │          │
│     │                                              │   Adversarial   │          │
│     │                         ┌────────────────────┤   Perturbations │          │
│  0.75                         │    LEVEL 3:       │   (10K samples) │          │
│     │                         │    Complex        └─────────────────┘          │
│     │       ┌─────────────────┤    Objects                                      │
│  0.50│       │    LEVEL 2:    │   (20K samples)                                 │
│     │       │    Texture      └────────────────────                             │
│     │       │    Patterns                                                       │
│  0.25├───────┤   (10K samples)                                                  │
│     │ LEVEL 1:                                                                  │
│     │ Basic Shapes                                                              │
│     │ (5K samples)                                                              │
│  0.0└───────┴───────────────────────────────────────────────────────────▶ Time │
│                                                                                 │
│     ├──────────┼──────────┼──────────┼──────────┤                              │
│     0         25%        50%        75%       100%  Training Progress           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Difficulty Scoring Algorithm:**

Each synthetic image is assigned a difficulty score based on three visual complexity metrics:

$$D(I) = w_1 \cdot E(I) + w_2 \cdot C(I) + w_3 \cdot F(I)$$

where:

- $E(I)$ = Edge density (Sobel filter magnitude)
- $C(I)$ = Color variance (per-channel variance)
- $F(I)$ = Spatial frequency (FFT high-frequency content)
- $(w_1, w_2, w_3) = (0.4, 0.3, 0.3)$ (default weights)

#### 1.2 SSEG Algorithm: Self-Supervised Evolutionary Growth

The SSEG algorithm iteratively optimizes network topology based on learning dynamics:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SSEG ALGORITHM FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   START                                                                         │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     Initialize Seed Network                              │   │
│   │              (3 blocks, 16→32→64 channels, ~30K params)                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    Initialize EMA Teacher                                │   │
│   │                     (Copy of student network)                            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ╔═════════════════════════════════════════════════════════════════════════╗   │
│   ║              FOR EACH CURRICULUM LEVEL (1 → 4):                         ║   │
│   ╠═════════════════════════════════════════════════════════════════════════╣   │
│   ║  │                                                                      ║   │
│   ║  ▼                                                                      ║   │
│   ║ ┌───────────────────────────────────────────────────────────────────┐   ║   │
│   ║ │                    TRAINING EPOCH LOOP                            │   ║   │
│   ║ │  ┌─────────────────────────────────────────────────────────────┐  │   ║   │
│   ║ │  │  1. Sample batch from current curriculum level              │  │   ║   │
│   ║ │  │  2. Generate augmented views (view₁, view₂)                 │  │   ║   │
│   ║ │  │  3. Forward pass through student → features                 │  │   ║   │
│   ║ │  │  4. Forward pass through teacher → targets                  │  │   ║   │
│   ║ │  │  5. Compute SSL loss: L = L_contrastive + λ·L_distillation  │  │   ║   │
│   ║ │  │  6. Backward pass & optimizer step                          │  │   ║   │
│   ║ │  │  7. Update EMA teacher: θ_t ← α·θ_t + (1-α)·θ_s            │  │   ║   │
│   ║ │  └─────────────────────────────────────────────────────────────┘  │   ║   │
│   ║ └───────────────────────────────────────────────────────────────────┘   ║   │
│   ║    │                                                                    ║   │
│   ║    ▼                                                                    ║   │
│   ║ ┌───────────────────────────────────────────────────────────────────┐   ║   │
│   ║ │               PLATEAU DETECTION (every epoch)                     │   ║   │
│   ║ │  ┌─────────────────────────────────────────────────────────────┐  │   ║   │
│   ║ │  │  Δ_SSL = |L_SSL(t) - L_SSL(t-w)|                            │  │   ║   │
│   ║ │  │  is_plateau = (Δ_SSL < ε)                                   │  │   ║   │
│   ║ │  │  has_capacity_gap = (L_distill > τ)                         │  │   ║   │
│   ║ │  └─────────────────────────────────────────────────────────────┘  │   ║   │
│   ║ └───────────────────────────────────────────────────────────────────┘   ║   │
│   ║    │                                                                    ║   │
│   ║    ▼                                                                    ║   │
│   ║ ┌───────────────────────────────────────────────────────────────────┐   ║   │
│   ║ │                    EVOLUTION DECISION                             │   ║   │
│   ║ │                                                                   │   ║   │
│   ║ │   if is_plateau AND has_capacity_gap:                            │   ║   │
│   ║ │       → TRIGGER NETWORK EVOLUTION (GROW or WIDEN)                │   ║   │
│   ║ │       → Synchronize EMA teacher with new architecture            │   ║   │
│   ║ │       → Reset optimizer state                                    │   ║   │
│   ║ │                                                                   │   ║   │
│   ║ │   elif is_plateau AND NOT has_capacity_gap:                      │   ║   │
│   ║ │       → ADVANCE TO NEXT CURRICULUM LEVEL                         │   ║   │
│   ║ │                                                                   │   ║   │
│   ║ │   else:                                                          │   ║   │
│   ║ │       → CONTINUE TRAINING                                        │   ║   │
│   ║ └───────────────────────────────────────────────────────────────────┘   ║   │
│   ╚═════════════════════════════════════════════════════════════════════════╝   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      OUTPUT: Evolved Backbone                            │   │
│   │           (Automatically grown architecture, ~500K-1M params)            │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Evolution Operations:**

| Operation | Trigger Condition            | Action                       | Effect          |
| --------- | ---------------------------- | ---------------------------- | --------------- |
| **GROW**  | `num_blocks < max_blocks`    | Add new ConvBlock at end     | Increases depth |
| **WIDEN** | `channels[i] < max_channels` | Expand channels of block `i` | Increases width |

#### 1.3 NASE Algorithm: Negative-Aware Sparse Evolution

NASE operates in parallel with SSEG to maintain computational efficiency:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            NASE ALGORITHM FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │               EVERY N EPOCHS (pruning_interval = 10):                   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │               STEP 1: Compute Parameter Importance                       │   │
│   │                                                                          │   │
│   │   For each parameter θᵢ in the network:                                 │   │
│   │                                                                          │   │
│   │       I(θᵢ) = |θᵢ · ∇_θᵢ L|    (Taylor importance score)                │   │
│   │                                                                          │   │
│   │   Methods available:                                                     │   │
│   │   • "magnitude": I = |θ|                                                 │   │
│   │   • "gradient":  I = |∇L|                                                │   │
│   │   • "taylor":    I = |θ · ∇L|  ← (default, most effective)              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │             STEP 2: Generate Complementary Masks                         │   │
│   │                                                                          │   │
│   │   threshold = quantile(I, sparsity_ratio)  # e.g., 30th percentile      │   │
│   │                                                                          │   │
│   │   ┌─────────────────────┐    ┌─────────────────────┐                    │   │
│   │   │   POSITIVE MASK     │    │   NEGATIVE MASK     │                    │   │
│   │   │   M⁺ = (I ≥ τ)      │    │   M⁻ = (I < τ)      │                    │   │
│   │   │   (Active paths)    │    │   (Pruned paths)    │                    │   │
│   │   │   ~70% connections  │    │   ~30% connections  │                    │   │
│   │   └─────────────────────┘    └─────────────────────┘                    │   │
│   │                                                                          │   │
│   │   Constraint: Maintain min_channels_per_layer (default: 8)              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │               STEP 3: Apply Sparse Forward Pass                          │   │
│   │                                                                          │   │
│   │   During training/inference:                                             │   │
│   │       θ' = θ ⊙ M⁺    (element-wise multiplication with mask)            │   │
│   │                                                                          │   │
│   │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│   │   │  Original Network          Sparse Network (30% pruned)           │  │   │
│   │   │  ●●●●●●●●●●              ●●●●●●●○○○                              │  │   │
│   │   │  ●●●●●●●●●●      →       ●●●●●●●○○○                              │  │   │
│   │   │  ●●●●●●●●●●              ●●●●●●●○○○                              │  │   │
│   │   │  (100% active)           (70% active, 30% zeroed)                │  │   │
│   │   └──────────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│     │                                                                           │
│     ▼                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │               STEP 4: Optional Permanent Pruning                         │   │
│   │                                                                          │   │
│   │   At end of training:                                                    │   │
│   │       θ_final = θ ⊙ M⁺    (permanently remove pruned weights)           │   │
│   │                                                                          │   │
│   │   Benefits:                                                              │   │
│   │   • Reduced model size (memory footprint)                                │   │
│   │   • Faster inference (fewer computations)                                │   │
│   │   • Maintained accuracy (importance-based selection)                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 1.4 SSEG-NASE Integration Loop

The complete integration of SSEG and NASE in Phase 1:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SSEG-NASE INTEGRATED TRAINING LOOP                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │                     OUTER LOOP: Curriculum Levels                   │     │
│     │                     (Level 1 → Level 2 → Level 3 → Level 4)        │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                       │                                         │
│                                       ▼                                         │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │                     MIDDLE LOOP: Training Epochs                    │     │
│     │                     (until plateau + level advance)                 │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                       │                                         │
│              ┌────────────────────────┼────────────────────────┐                │
│              ▼                        ▼                        ▼                │
│     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│     │   SSL Training  │     │  SSEG Check     │     │  NASE Update    │        │
│     │   (every batch) │     │  (every epoch)  │     │  (every N epochs)│        │
│     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘        │
│              │                       │                       │                  │
│              ▼                       ▼                       ▼                  │
│     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐        │
│     │ • Augmentation  │     │ • Plateau detect│     │ • Importance    │        │
│     │ • Forward pass  │     │ • Gap analysis  │     │   scoring       │        │
│     │ • Contrastive L │     │ • Grow/Widen?   │     │ • Mask update   │        │
│     │ • Distillation L│     │ • Level advance?│     │ • Sparsity log  │        │
│     │ • Backward pass │     │ • Teacher sync  │     │                 │        │
│     │ • EMA update    │     │                 │     │                 │        │
│     └─────────────────┘     └─────────────────┘     └─────────────────┘        │
│              │                       │                       │                  │
│              └───────────────────────┴───────────────────────┘                  │
│                                       │                                         │
│                                       ▼                                         │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │                         LOGGING & TRACKING                          │     │
│     │  • Architecture mutations                                          │     │
│     │  • Loss curves (SSL, distillation)                                 │     │
│     │  • Sparsity statistics                                             │     │
│     │  • GPU memory usage                                                │     │
│     │  • Curriculum progression                                          │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 2: Few-Shot Adaptation

The second phase is an **evaluation stage** where the evolved backbone is **frozen** to test its ability to extract features for N-way K-shot tasks using standard benchmark datasets.

#### 2.1 Evaluation Protocol

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FEW-SHOT EVALUATION PROTOCOL                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   INPUT: Trained & Frozen SSEG-NASE Backbone                                    │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 1: Episode Sampling                             │   │
│   │                                                                          │   │
│   │   For episode = 1 to 600:                                               │   │
│   │       1. Randomly select N classes from test set (N = 5)                │   │
│   │       2. For each class:                                                │   │
│   │          - Sample K images for support set (K ∈ {1, 5})                 │   │
│   │          - Sample Q images for query set (Q = 15)                       │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │   Episode Structure (5-way 5-shot example):                     │   │   │
│   │   │                                                                  │   │   │
│   │   │   SUPPORT SET              QUERY SET                            │   │   │
│   │   │   (5 classes × 5 shots)    (5 classes × 15 queries)             │   │   │
│   │   │   ┌───┬───┬───┬───┬───┐   ┌───┬───┬───┬...┬───┐                │   │   │
│   │   │   │C1 │C1 │C1 │C1 │C1 │   │C1 │C1 │C1 │...│C1 │ (15 queries)   │   │   │
│   │   │   ├───┼───┼───┼───┼───┤   ├───┼───┼───┼...┼───┤                │   │   │
│   │   │   │C2 │C2 │C2 │C2 │C2 │   │C2 │C2 │C2 │...│C2 │                │   │   │
│   │   │   ├───┼───┼───┼───┼───┤   ├───┼───┼───┼...┼───┤                │   │   │
│   │   │   │C3 │C3 │C3 │C3 │C3 │   │C3 │C3 │C3 │...│C3 │                │   │   │
│   │   │   ├───┼───┼───┼───┼───┤   ├───┼───┼───┼...┼───┤                │   │   │
│   │   │   │C4 │C4 │C4 │C4 │C4 │   │C4 │C4 │C4 │...│C4 │                │   │   │
│   │   │   ├───┼───┼───┼───┼───┤   ├───┼───┼───┼...┼───┤                │   │   │
│   │   │   │C5 │C5 │C5 │C5 │C5 │   │C5 │C5 │C5 │...│C5 │                │   │   │
│   │   │   └───┴───┴───┴───┴───┘   └───┴───┴───┴...┴───┘                │   │   │
│   │   │   25 support samples       75 query samples                      │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 2: Feature Extraction                           │   │
│   │                                                                          │   │
│   │   ┌───────────────┐                ┌───────────────┐                    │   │
│   │   │ Support Images│    Backbone    │Support Features│                    │   │
│   │   │  (N×K × C×H×W)│ ────────────▶ │  (N×K × D)     │                    │   │
│   │   └───────────────┘    (frozen)    └───────────────┘                    │   │
│   │                                                                          │   │
│   │   ┌───────────────┐                ┌───────────────┐                    │   │
│   │   │ Query Images  │    Backbone    │ Query Features │                    │   │
│   │   │  (N×Q × C×H×W)│ ────────────▶ │  (N×Q × D)     │                    │   │
│   │   └───────────────┘    (frozen)    └───────────────┘                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 3: Prototype Computation                        │   │
│   │                                                                          │   │
│   │   For each class c ∈ {1, ..., N}:                                       │   │
│   │                                                                          │   │
│   │       p_c = (1/K) Σ f(x_i)    where x_i ∈ support set of class c        │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │   Class 1 Prototype: p₁ = mean(f₁₁, f₁₂, f₁₃, f₁₄, f₁₅)       │   │   │
│   │   │   Class 2 Prototype: p₂ = mean(f₂₁, f₂₂, f₂₃, f₂₄, f₂₅)       │   │   │
│   │   │   Class 3 Prototype: p₃ = mean(f₃₁, f₃₂, f₃₃, f₃₄, f₃₅)       │   │   │
│   │   │   Class 4 Prototype: p₄ = mean(f₄₁, f₄₂, f₄₃, f₄₄, f₄₅)       │   │   │
│   │   │   Class 5 Prototype: p₅ = mean(f₅₁, f₅₂, f₅₃, f₅₄, f₅₅)       │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 4: Query Classification                         │   │
│   │                                                                          │   │
│   │   For each query q with feature f_q:                                    │   │
│   │                                                                          │   │
│   │       prediction = argmax_c  sim(f_q, p_c)                              │   │
│   │                                                                          │   │
│   │   where sim() is cosine similarity (default) or negative Euclidean      │   │
│   │                                                                          │   │
│   │   ┌────────────────────────────────────────────────────────────────┐    │   │
│   │   │   Query: f_q = [0.2, 0.8, 0.1, ...]                            │    │   │
│   │   │                                                                 │    │   │
│   │   │   Similarities:                                                 │    │   │
│   │   │   • sim(f_q, p₁) = 0.32                                        │    │   │
│   │   │   • sim(f_q, p₂) = 0.91  ← HIGHEST                             │    │   │
│   │   │   • sim(f_q, p₃) = 0.45                                        │    │   │
│   │   │   • sim(f_q, p₄) = 0.28                                        │    │   │
│   │   │   • sim(f_q, p₅) = 0.15                                        │    │   │
│   │   │                                                                 │    │   │
│   │   │   Prediction: Class 2                                          │    │   │
│   │   └────────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 5: Compute Episode Accuracy                     │   │
│   │                                                                          │   │
│   │       accuracy = (# correct predictions) / (# total queries)            │   │
│   │                = (# correct) / (N × Q)                                   │   │
│   │                = (# correct) / 75  (for 5-way with 15 queries)          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                       │                                         │
│                                       ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    STEP 6: Aggregate Over Episodes                      │   │
│   │                                                                          │   │
│   │   Mean Accuracy = (1/600) Σ accuracy_i                                  │   │
│   │                                                                          │   │
│   │   95% Confidence Interval = mean ± 1.96 × (std / √600)                  │   │
│   │                                                                          │   │
│   │   Final Result: 68.42% ± 0.35%                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.2 Benchmark Datasets

| Dataset             | Classes (Train/Val/Test) | Images/Class | Resolution     | Source                  |
| ------------------- | ------------------------ | ------------ | -------------- | ----------------------- |
| **Mini-ImageNet**   | 64 / 16 / 20             | 600          | 84×84          | Vinyals et al., 2016    |
| **CIFAR-FS**        | 64 / 16 / 20             | 600          | 32×32 (→84×84) | Bertinetto et al., 2019 |
| **Tiered-ImageNet** | 351 / 97 / 160           | ~1300        | 84×84          | Ren et al., 2018        |

#### 2.3 Efficiency Evaluation

In addition to accuracy, Phase 2 also evaluates computational efficiency:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EFFICIENCY METRICS                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    COMPUTATIONAL COMPLEXITY                             │   │
│   │                                                                          │   │
│   │   • Parameters: Count of trainable weights                              │   │
│   │   • FLOPs: Floating-point operations per forward pass                   │   │
│   │   • Memory: GPU memory footprint                                        │   │
│   │   • Latency: Inference time per image                                   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    TARGET CONSTRAINTS (RTX 3060)                        │   │
│   │                                                                          │   │
│   │   │ Metric       │ Target    │ Reason                       │           │   │
│   │   │──────────────│───────────│──────────────────────────────│           │   │
│   │   │ Parameters   │ < 1M      │ Edge deployment feasibility  │           │   │
│   │   │ FLOPs        │ < 1G      │ Real-time inference          │           │   │
│   │   │ Memory       │ < 4GB     │ Consumer GPU compatibility   │           │   │
│   │   │ Latency      │ < 10ms    │ Interactive applications     │           │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Complete Research Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SSEG-NASE COMPLETE RESEARCH FLOW                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                              PROBLEM DEFINITION                                           │ │
│   │   "How to develop hardware-efficient neural architectures for few-shot learning?"        │ │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                  │
│                                              ▼                                                  │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                              PROPOSED SOLUTION                                            │ │
│   │   Scalable Self-Supervised Pretraining Pipeline with Automatic Network Evolution         │ │
│   │   • SSEG: Self-Supervised Evolutionary Growth                                            │ │
│   │   • NASE: Negative-Aware Sparse Evolution                                                │ │
│   │   • Curriculum Learning: Progressive data complexity                                     │ │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                  │
│                  ┌───────────────────────────┴───────────────────────────┐                      │
│                  ▼                                                       ▼                      │
│   ┌────────────────────────────────────────┐   ┌────────────────────────────────────────────┐   │
│   │     PHASE 1: EVOLUTIONARY PRETRAINING  │   │      PHASE 2: FEW-SHOT ADAPTATION         │   │
│   ├────────────────────────────────────────┤   ├────────────────────────────────────────────┤   │
│   │                                        │   │                                            │   │
│   │  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────────────────┐  │   │
│   │  │    Synthetic Curriculum Data     │  │   │  │     Standard Benchmark Datasets      │  │   │
│   │  │    (Basic→Texture→Object→Adv)    │  │   │  │   (Mini-ImageNet, CIFAR-FS, etc.)   │  │   │
│   │  └──────────────────────────────────┘  │   │  └──────────────────────────────────────┘  │   │
│   │                  │                     │   │                     │                      │   │
│   │                  ▼                     │   │                     ▼                      │   │
│   │  ┌──────────────────────────────────┐  │   │  ┌──────────────────────────────────────┐  │   │
│   │  │       SSEG + NASE Loop           │  │   │  │        Episode Sampling              │  │   │
│   │  │  • SSL Training                  │  │   │  │   (N-way K-shot episodes)           │  │   │
│   │  │  • Plateau Detection             │  │   │  └──────────────────────────────────────┘  │   │
│   │  │  • Network Evolution             │  │   │                     │                      │   │
│   │  │  • Sparsity Optimization         │  │   │                     ▼                      │   │
│   │  └──────────────────────────────────┘  │   │  ┌──────────────────────────────────────┐  │   │
│   │                  │                     │   │  │     Feature Extraction               │  │   │
│   │                  ▼                     │   │  │   (Frozen backbone)                  │  │   │
│   │  ┌──────────────────────────────────┐  │   │  └──────────────────────────────────────┘  │   │
│   │  │     Evolved Backbone Output      │  │   │                     │                      │   │
│   │  │  • Optimized architecture        │──────▶                     ▼                      │   │
│   │  │  • Learned representations       │  │   │  ┌──────────────────────────────────────┐  │   │
│   │  │  • Efficient computation         │  │   │  │     Prototype Classification         │  │   │
│   │  └──────────────────────────────────┘  │   │  │   (Nearest prototype assignment)     │  │   │
│   │                                        │   │  └──────────────────────────────────────┘  │   │
│   └────────────────────────────────────────┘   │                     │                      │   │
│                                                │                     ▼                      │   │
│                                                │  ┌──────────────────────────────────────┐  │   │
│                                                │  │          EVALUATION METRICS          │  │   │
│                                                │  │  • 5-way 1-shot accuracy            │  │   │
│                                                │  │  • 5-way 5-shot accuracy            │  │   │
│                                                │  │  • 95% confidence intervals         │  │   │
│                                                │  │  • FLOPs, parameters, latency       │  │   │
│                                                │  └──────────────────────────────────────┘  │   │
│                                                └────────────────────────────────────────────┘   │
│                                                                   │                             │
│                                                                   ▼                             │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                              VALIDATION & ANALYSIS                                        │ │
│   │   • Ablation Studies: Validate contribution of each component                            │ │
│   │   • Baseline Comparisons: Compare with ProtoNet, MAML, SimCLR, DINO                     │ │
│   │   • Efficiency Analysis: Verify hardware constraints are met                             │ │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                              │                                                  │
│                                              ▼                                                  │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │
│   │                              EXPECTED OUTCOMES                                            │ │
│   │   • Hardware-efficient backbone (< 1M params, < 1G FLOPs)                                │ │
│   │   • Competitive few-shot accuracy (>50% 1-shot, >68% 5-shot on Mini-ImageNet)           │ │
│   │   • Automated architecture discovery (no manual design required)                         │ │
│   │   • Reproducible experimental framework                                                  │ │
│   └───────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Research Variables

#### Independent Variables

| Variable          | Description                         | Values/Range   |
| ----------------- | ----------------------------------- | -------------- |
| Curriculum Level  | Synthetic data complexity           | {1, 2, 3, 4}   |
| SSEG Enabled      | Whether network evolution is active | {True, False}  |
| NASE Enabled      | Whether sparsification is active    | {True, False}  |
| Sparsity Ratio    | Target pruning ratio                | [0.0, 0.5]     |
| Plateau Threshold | SSL loss change threshold           | [1e-5, 1e-3]   |
| EMA Decay         | Teacher update rate                 | [0.99, 0.9999] |

#### Dependent Variables

| Variable              | Description          | Target |
| --------------------- | -------------------- | ------ |
| 5-way 1-shot Accuracy | Few-shot performance | > 50%  |
| 5-way 5-shot Accuracy | Few-shot performance | > 68%  |
| Model Parameters      | Complexity measure   | < 1M   |
| FLOPs                 | Computational cost   | < 1G   |
| Inference Latency     | Speed measure        | < 10ms |

#### Control Variables

| Variable      | Value         | Rationale              |
| ------------- | ------------- | ---------------------- |
| Random Seed   | 42            | Reproducibility        |
| Image Size    | 84×84         | Standard few-shot size |
| Optimizer     | AdamW         | Stable training        |
| Learning Rate | 1e-3          | Balanced convergence   |
| Batch Size    | 48 (RTX 3060) | Hardware constraint    |
