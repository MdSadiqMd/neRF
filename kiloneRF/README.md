# KiloNeRF

PyTorch implementation of KiloNeRF — Speeding up Neural Radiance Fields with Thousands of Tiny MLPs.

Paper: https://arxiv.org/abs/2103.13744

## What's Implemented

- **Grid of tiny MLPs**: The scene is divided into an N×N×N grid (default N=16 → 4096 independent MLPs)
- **Per-cell MLP architecture** (from Figure 2 of the paper):
  - Position branch: positional encoding → 32-dim hidden → 32+1 (features + density)
  - Direction branch: concatenate direction encoding → 32-dim hidden → RGB
- **Xavier initialization** for all per-cell weight matrices
- **Bounding box masking**: only queries inside the scene volume are processed
- **Batched grid lookup**: each 3D point is routed to its grid cell's MLP via index mapping
- Volume rendering with accumulated transmittance
- White background regularization

## How It Differs from FastNeRF

| Aspect | FastNeRF | KiloNeRF |
|---|---|---|
| Architecture | 1 large MLP factorized into Fpos + Fdir | N³ independent tiny MLPs (one per grid cell) |
| Speedup strategy | Factorized radiance caching | Spatial decomposition with simple MLPs |
| Parameters | Shared across all positions | Each grid cell has its own weights |
| Inference | Cache position/direction grids | Direct per-cell forward pass |

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

Trained model renders 22 novel views to `novel_views/img_*.png` (100×100 images of the Lego scene).

## Known Limitations

This implementation produces **poor quality images** and is **slower than expected**. This is because:

The paper states: *"using teacher-student distillation for training, we show that this speed-up can be achieved without sacrificing visual quality."*

| Issue | Current Implementation | Paper's Approach |
|-------|----------------------|------------------|
| Training | Direct from RGB images | Teacher-student distillation from pre-trained NeRF |
| Architecture | 32-dim tiny MLPs learning from scratch | Tiny MLPs distilled from 256-dim teacher |
| Grid boundaries | Hard boundaries, no interpolation | Occupancy-aware sampling |
| Performance | Python indexed matmul (slow) | Custom CUDA kernels (fast) |

**Why it fails without distillation**: Each tiny MLP only sees sparse samples from its grid cell and cannot learn a good representation. The blocky artifacts come from hard cell boundaries with no interpolation between neighboring MLPs.

**Why it's slow**: KiloNeRF requires custom CUDA kernels to achieve the claimed 3 orders of magnitude speedup. The Python implementation with indexed matrix multiplications is not optimized for this architecture.
