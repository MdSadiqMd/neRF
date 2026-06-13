# NeRF

PyTorch implementation of Neural Radiance Fields for view synthesis.

Paper: https://arxiv.org/abs/2003.08934

## What's implemented

- Positional encoding for 3D coordinates and view directions
- MLP network with density and color prediction
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
