# Fourier Feature NeRF (Inverse Rendering)

PyTorch implementation of NeRF using Random Fourier Features instead of deterministic positional encoding.

Paper: https://arxiv.org/abs/2006.10739

## What's Implemented

- **Random Fourier Features**: Instead of deterministic `sin(2^j * x)` positional encoding, uses random Gaussian projection `sin(x @ B)` and `cos(x @ B)` where B is sampled from N(0, σ²)
- **Tunable bandwidth**: The scale parameter (σ = 6.05) controls the frequency spectrum the network can learn
- **Simpler architecture**: No view-direction dependency - outputs density and color directly from position
- **Volume rendering**: Same ray marching and alpha compositing as standard NeRF

## How It Differs from Standard NeRF

| Aspect | Standard NeRF | Fourier Feature NeRF |
|--------|---------------|---------------------|
| Position encoding | Deterministic `sin/cos(2^j * x)` for j=0..L | Random `sin/cos(x @ B)` with Gaussian B |
| View dependency | Separate direction encoding | No view-direction input |
| Frequency control | Fixed by L (number of octaves) | Tunable via scale σ |
| Network input | 2L*3 + 3 = 63 dims (L=10) | 2*NB_features = 512 dims |

## Why Random Fourier Features?

Standard MLPs have **spectral bias** - they learn low frequencies first and struggle with high-frequency details. The paper shows:

1. **Neural Tangent Kernel (NTK) analysis**: MLP's effective kernel has limited bandwidth
2. **Fourier features fix this**: Mapping inputs through `sin/cos(x @ B)` makes the NTK stationary with tunable bandwidth
3. **Scale matters**: σ controls the frequency range - too low = blurry, too high = noisy

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

## Known Limitations

Training is fast but image quality is poor compared to standard NeRF. The images have significant noise/"floater" artifacts in the background.

**Why the quality is worse:**

1. **No view-direction input**: Standard NeRF uses view direction to model specular reflections and view-dependent effects. This implementation only takes position as input, so it cannot represent view-dependent appearance - the network tries to average all viewing angles into a single color.

2. **Random vs deterministic encoding**: While random Fourier features can learn high frequencies, the specific random matrix B may not be optimal for the scene. The deterministic NeRF encoding with powers of 2 is well-suited for multi-scale scene representation.

3. **Scale sensitivity**: The bandwidth parameter σ=6.05 was chosen for 2D image regression tasks in the paper. For 3D volume rendering, the optimal scale may differ.

4. **Simpler architecture**: 4-layer MLP with 256 hidden dims is smaller than standard NeRF's 8-layer architecture with skip connections.

5. **No coarse-fine sampling**: Standard NeRF uses hierarchical sampling (coarse network guides fine network). This implementation uses uniform sampling only.

The paper's main contribution is theoretical (NTK analysis of spectral bias) - the Fourier feature insight was incorporated into NeRF's positional encoding, not meant as a replacement architecture.
