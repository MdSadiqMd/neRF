# InfoNeRF

PyTorch implementation of InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering.

Paper: https://arxiv.org/abs/2112.15399

## What's Implemented

Standard NeRF with two information-theoretic regularizations for few-shot learning:

1. **Ray Entropy Minimization**: Regularize density distribution along each ray to be compact
   - Computes entropy H(p) = -∑ p_k log(p_k) where p_k = α_k / ∑ α_k
   - Penalizes spread-out density (floaters) by encouraging the model to concentrate density at surfaces
   - Only applied where total alpha exceeds a threshold T (avoids penalizing empty rays)

2. **Spatial Smoothness Constraint** (optional): KL divergence between nearby rays
   - Perturbs each ray with small random noise
   - Enforces consistent density distributions across neighboring viewpoints
   - Prevents degenerate solutions when training views are nearly redundant

## How It Differs from Standard NeRF

| Aspect | Standard NeRF | InfoNeRF |
|--------|---------------|----------|
| Loss | RGB MSE only | RGB MSE + ray entropy + spatial smoothness |
| Training data | Full set | Few-shot (4 images for tiny_nerf) |
| Regularization | None | Information-theoretic |
| Floaters | Common with few views | Suppressed by entropy penalty |

## Why Ray Entropy?

With limited training views, standard NeRF often produces "floaters" - semi-transparent density clouds floating in space. These happen because the model can explain training pixels with spurious geometry.

Ray entropy minimization fixes this by:
- Making density distributions along each ray **peak sharply** at the true surface
- **Penalizing** spread-out, uncertain density assignments
- Being a **"free lunch"** - no extra forward passes or network parameters

## Commands

```bash
# Install dependencies
just install

# Download and prepare training data
just prepare

# Train the model (32 epochs, few-shot: 4 images)
just train

# Or run everything at once
just all
```

## Few-Shot Setup

This implementation uses only **4 images** from the tiny_nerf training set (indices 0, 20, 40, 60). The entropy regularization helps prevent overfitting with this limited data.

## Output

Trained model renders 22 novel views to `novel_views/img_*.png` (100x100 images of the Lego scene).

## Known Limitations

The rendered images have **poor quality** - blurry, lacking fine detail, and with visible artifacts. This is **expected** due to:

1. **Extreme few-shot setting**: Only 4 training images (vs 84 for standard NeRF = 20x less data). The model must reconstruct a full 360° 3D scene from just 4 viewpoints, which is fundamentally ambiguous.

2. **Information-theoretic regularization trade-off**: The entropy penalty prevents floaters but also makes the model produce overly smooth density distributions, suppressing fine details that would appear with more training views.

3. **Per-element MSE is ~0.019** vs ~0.003 for standard NeRF with 84 images - the model simply hasn't seen enough of the scene.

The paper's experiments use the full NeRF synthetic dataset (100 training images per scene at 800x800) and evaluate on 3-shot to 10-shot settings. With tiny_nerf's small 100x100 images, the limited resolution compounds the few-shot challenge.
