# FastNeRF

PyTorch implementation of FastNeRF for high-fidelity neural rendering.

Paper: https://arxiv.org/abs/2103.10380

## What's implemented

- **Factorized architecture**: two independent networks decouple position and direction
  - `Fpos`: maps 3D position → sigma (density) + U/V/W feature vectors (`D`-dim each)
  - `Fdir`: maps ray direction → beta weights (softmax mixing coefficients)
- **Compact radiance representation**: color is computed as `sum(beta * uvw)` — a dot product between direction weights and position features
- Softplus activation for density, sigmoid for UV weights, softmax for beta
- **Cache module**: precomputes the full `Fpos` and `Fdir` outputs over a voxel grid at inference time, reducing rendering to fast table lookups
- Volume rendering with accumulated transmittance
- White background regularization

## Commands

```bash
# Install dependencies
just install

# Download and prepare training data
just prepare

# Train the model (16 epochs, ~30 min on MPS)
just train

# Or run everything at once
just all
```

## Output

Trained model renders 22 novel views to `novel_views/img_*.png` (100x100 images of the Lego scene).
