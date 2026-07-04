# Learned Initializations for NeRF

PyTorch implementation of Learned Initializations for Optimizing Coordinate-Based Neural Representations.

Paper: https://arxiv.org/abs/2012.02189

## What's Implemented

Meta-learning initialization for NeRF using **Reptile** (first-order MAML):

1. **Reptile Meta-Learning**: Learn initial weights φ such that for any task (representing a scene from a subset of images), a few steps of SGD from φ produces a good model
2. **Inner Loop**: K=16 steps of SGD on 4 support images (40K rays), adapting the cloned model to the task
3. **Outer Loop**: Reptile update `φ ← φ - α(φ - φ̃)` pulls the meta-initialization toward the adapted parameters
4. **Few-Shot Evaluation**: Fine-tune meta-initialization on 4 test views for 5 epochs, render remaining 18 views

## How It Works

### Reptile Algorithm

```
For each meta-iteration:
  1. Sample 4 images from training set as support
  2. Save initial parameters φ
  3. Inner loop: K=16 steps SGD on support rays (batch_size=1024)
  4. Update: φ ← φ - α(φ - θ_K)
```

This is a first-order meta-learning algorithm — no second derivatives needed. The Reptile update pulls the initialization toward parameters that adapt well to the task:

```
φ_grad = φ - θ_K  # direction from adapted params back to initial
φ ← φ - α * φ_grad  # move initialization toward adapted params
```

### Why In-Place (No Clone)

Most MAML implementations `copy.deepcopy` the model for each task, which is O(100ms) per iteration. Instead, we:
1. Save `initial_params = [p.data.clone() for p in model.parameters()]`
2. Run inner loop in-place on the same model
3. Reptile update using saved initial params — no deep copy needed

This reduces meta-iteration time from ~8s to ~2s on MPS.

### Evaluation Protocol

| Model | Initialization | Fine-Tuning |
|-------|---------------|-------------|
| Random init | Random weights | 5 epochs on 4 test views |
| Meta init | Reptile (200 iters on 84 train images) | 5 epochs on 4 test views |

Both tested on the same 18 held-out test views from the Lego scene.

## Why It Works

- **Shared structure**: Different views of the same scene share underlying geometry and appearance patterns
- **Fast adaptation**: The meta-initialization is already "close" to a good solution, so few gradient steps suffice
- **First-order approximation**: Reptile's simplified update avoids expensive second-order gradients while retaining most of MAML's benefit
- **Prior over network weights**: The learned initialization encodes a prior over the scene's radiance field distribution

## Results

```
Random init avg MSE:  0.142031
Meta init avg MSE:    0.060991
Improvement:          57.1%
```

The meta-initialization produces **2.3× better** reconstruction quality than random initialization after the same 5 epochs of fine-tuning on only 4 test views. The meta-initialized model starts closer to the optimal parameters, so limited fine-tuning goes further.

## Commands

```bash
# Install dependencies
just install

# Download and prepare training data
just prepare

# Run meta-learning + evaluation
just train

# Or run everything at once
just all
```

## Output

- `novel_views/random_img_*.png` — Novel views rendered from random init (fine-tuned on 4 test views)
- `novel_views/meta_img_*.png` — Novel views rendered from meta init (fine-tuned on 4 test views)
- 18 novel views per model (test images 4–21)

## Limitations

- **Single scene only**: The paper demonstrates cross-scene generalization (meta-train on chairs, test on cars). With only one Lego scene, we meta-learn within the same scene distribution.
- **MPS slowdowns**: On Apple Silicon, MPS memory management causes intermittent slowdowns (iterations can take 1–300s). Performance is inconsistent.
- **Small inner loop**: K=16 with 4 support images limits the diversity of adaptation trajectories. The paper uses larger inner loops and more tasks.
