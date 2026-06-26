# FreeNeRF

PyTorch implementation of FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization.

Paper: https://arxiv.org/abs/2303.07418

## What's Implemented

Two "free lunch" regularization techniques for few-shot NeRF:

1. **Frequency Regularization**: Progressively unmask positional encoding frequencies during training
   - Early training: only low frequencies enabled (coarse structure)
   - Later training: high frequencies gradually enabled (fine details)
   - Prevents overfitting to high-frequency noise when training views are scarce

2. **Occlusion Regularization**: Penalize density near the camera
   - Prevents "floater" artifacts that appear close to the camera
   - Simple L1 penalty on near-camera density
   - Zero additional parameters or forward passes

## How It Works

### Frequency Regularization

Standard positional encoding: `γ(x) = [x, sin(2⁰x), cos(2⁰x), ..., sin(2^(L-1)x), cos(2^(L-1)x)]`

FreeNeRF masks high frequencies based on training progress:
```
progress = step / T
active_dims = int(3 + progress * (total_dims - 3))
encoding[:, active_dims:] = 0
```

This is a "one line of code change" that significantly improves few-shot performance.

### Occlusion Regularization

```
near_mask = (t < near_threshold)
occ_loss = (sigma * near_mask).mean()
```

Penalizes density in the near-camera region to prevent floaters.

## Why It Works

- **Spectral bias**: Neural networks learn low frequencies first
- **Overfitting path**: With few views, NeRF can fit training views with spurious high-frequency geometry
- **Solution**: Limit available frequencies early to force learning robust low-frequency structure first

## Commands

```bash
# Install dependencies
just install

# Download and prepare training data
just prepare

# Train the model (16 epochs)
just train

# Or run everything at once
just all
```

## Output

Trained model renders 22 novel views to `novel_views/img_*.png` (100x100 images of the Lego scene).

## Few-Shot Setting

The paper demonstrates FreeNeRF on extremely limited training views (3-9 images). With the full tiny_nerf dataset (84 training images), the benefit of frequency regularization is less pronounced, but it still helps prevent overfitting and produces cleaner results.
