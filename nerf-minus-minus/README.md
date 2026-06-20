# NeRF-- (NeRF Minus Minus)

Neural Radiance Fields without known camera parameters. Jointly optimizes camera intrinsics, extrinsics, and the NeRF model from images alone.

Paper: https://arxiv.org/abs/2102.07064

## Overview

NeRF-- removes the requirement for pre-computed camera parameters (from COLMAP/SfM) by learning them jointly with the radiance field through photometric reconstruction loss.

**Key insight**: Camera parameters are differentiable - gradients flow from RGB loss through volume rendering, ray generation, and back to camera poses and focal length.

## Architecture

Camera parameters are represented as:
- **Intrinsics**: Single learnable focal length parameter (f = param^2 * W)
- **Extrinsics**: 6-DoF pose per image (3D translation + axis-angle rotation)

The NeRF model is standard: positional encoding + MLP predicting density and view-dependent color.

## Training

Two-stage training process:
1. **Joint optimization**: Train NeRF + camera parameters together
2. **Refinement**: Freeze cameras, continue training NeRF only

## Usage

```bash
# Prepare data (downloads tiny_nerf dataset)
uv run python prepare_data.py

# Train
just train
```

## Configuration

In `main.py`, set `USE_GROUND_TRUTH_CAMERAS`:
- `True`: Standard NeRF with known cameras (works on any scene)
- `False`: NeRF-- mode, learns cameras from scratch (forward-facing scenes only)

## Limitations

**NeRF-- is designed for forward-facing scenes only.**

The tiny_nerf dataset is a 360-degree scene (cameras on a hemisphere), which makes camera pose recovery from scratch extremely difficult due to:
- No natural reference direction
- Many local minima in the optimization landscape
- High degrees of freedom

For 360-degree scenes, use `USE_GROUND_TRUTH_CAMERAS = True` or use a forward-facing dataset (LLFF).
