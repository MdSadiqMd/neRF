# FastNeRF

PyTorch implementation of FastNeRF for high-fidelity neural rendering.

Paper: https://arxiv.org/abs/2103.10380

## What's implemented

- Factorized architecture: separate position (Fpos) and direction (Fdir) networks
- Position network outputs sigma + UV weights for compact radiance caching
- Direction network outputs beta weights for view-dependent color mixing
- Softplus activation for density, sigmoid for UV weights, softmax for beta
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
