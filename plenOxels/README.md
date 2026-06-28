# Plenoxels

PyTorch implementation of Plenoxels: Radiance Fields without Neural Networks.

Paper: https://arxiv.org/abs/2112.05131

## What's Implemented

A radiance field represented as a dense voxel grid without any neural network:

1. **Voxel Grid**: A `128 × 128 × 128` grid where each voxel stores density (1) and spherical harmonic coefficients (27 = 3 RGB channels × 9 SH bases)
2. **Trilinear Interpolation**: Smooth continuous sampling via 8-point trilinear interpolation instead of nearest-neighbor
3. **Spherical Harmonics (Degree 2)**: View-dependent color computed in closed form — no view direction network needed
4. **Volume Rendering**: Standard NeRF-style ray marching with stratified sampling and alpha compositing

## How It Works

### Voxel Representation

Each voxel stores 28 values:
```
[σ, SH_0_R, SH_0_G, SH_0_B, SH_1_R, ..., SH_8_R, SH_8_G, SH_8_B]
```

- `σ` (density): activated with `softplus`
- Color at each point: `sigmoid(eval_sh(deg=2, coeffs, direction))`

### Trilinear Interpolation

Given a continuous coordinate, sample the 8 surrounding voxels and interpolate:
```
c000 = grid[x0, y0, z0]
c001 = grid[x0, y0, z1]
... 8 corners ...
value = lerp(lerp(lerp(c000, c100, xd), lerp(c010, c110, xd), yd), ... , zd)
```

### Ray Marching

Standard stratified sampling along rays, same as NeRF:
- `nb_bins=192` samples per ray between `hn=2` and `hf=6`
- Perturb sampling with uniform jitter
- Alpha compositing: `C = ∑ T_i α_i c_i` where `α_i = 1 − exp(−σ_i δ_i)`

### Optimization

No neural network — just the voxel grid tensor:
- Loss: MSE between rendered and ground truth pixels
- Optimizer: Adam (lr=1e-2) with MultiStepLR at milestones [5, 10]
- 58.7 million parameters optimized directly

## Why It Works

- **No MLP bottleneck**: Each spatial region has independent parameters — no shared representation to limit capacity
- **View-dependent appearance**: SH degree-2 captures smooth view-dependent effects (Lambertian + one lobe of specular)
- **Fast optimization**: Direct density/color storage avoids expensive MLP forward passes
- However, the dense grid cannot adapt to scene complexity — empty space wastes capacity

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

Trained model renders 22 novel test views to `novel_views/img_*.png` (100x100 images of the Lego scene).

## Limitations vs Full Plenoxels

| Feature | This Implementation | Full Plenoxels (Paper) |
|---------|-------------------|----------------------|
| Grid structure | Dense 128³ | Sparse octree |
| Memory efficiency | 58.7M params always active | Adaptive allocation |
| Regularization | None | TV regularization |
| Optimization | Single-scale | Coarse-to-fine |
| Resolution | Fixed 128³ | Multi-scale adaptive |

The paper's sparse octree is the critical missing piece — it allocates voxels only near surfaces, enabling higher effective resolution without wasting memory on empty space.
