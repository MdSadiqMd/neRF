# PlenOctrees

PyTorch implementation of PlenOctrees for Real-time Rendering of Neural Radiance Fields.

Paper: https://arxiv.org/abs/2103.14024

## What's Implemented

This implements the **NeRF-SH training phase** where the network outputs spherical harmonic (SH) coefficients instead of view-dependent RGB colors.

- **Spherical Harmonic output**: Network outputs (deg+1)² = 9 SH coefficients per color channel (degree 2)
- **No view direction input**: Removes viewing direction as network input - view dependence is encoded in SH coefficients
- **SH evaluation at render time**: Colors are computed by evaluating SH basis functions with the view direction
- **Standard volume rendering**: Same ray marching and alpha compositing as NeRF

## How It Differs from Standard NeRF

| Aspect | Standard NeRF | PlenOctrees (NeRF-SH) |
|--------|---------------|----------------------|
| Network input | Position + direction | Position only |
| Network output | RGB + density | SH coefficients + density |
| View dependence | Encoded in network weights | Encoded in SH coefficients |
| Pre-tabulation | Not possible (view-dependent) | Possible (SH can be cached) |

## Why Spherical Harmonics?

The key insight is **factorization of view-dependent appearance**:

1. Standard NeRF requires the view direction as input, so the network must be evaluated for every ray
2. By outputting SH coefficients, the view-dependent color is factored into:
   - **Position-dependent SH coefficients** (can be pre-computed and cached)
   - **Direction-dependent SH basis functions** (cheap closed-form evaluation)

This enables pre-tabulating the trained network into an octree data structure (PlenOctree) for real-time rendering at 150+ FPS.

## Limitations

This implementation only includes the NeRF-SH training phase. The full PlenOctrees pipeline also requires:
- Octree construction from the trained NeRF-SH
- Octree optimization (direct finetuning of cached values)
- Custom CUDA renderer for real-time octree traversal

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
