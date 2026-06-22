# K-Planes

PyTorch implementation of K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.

Paper: https://arxiv.org/abs/2301.10241

## What's Implemented

This implements the **static (d=3) version** of K-Planes using 3 axis-aligned feature planes.

- **Three feature planes**: XY, YZ, and XZ planes store learned feature vectors
- **Bilinear interpolation**: Features are sampled smoothly from planes
- **Hadamard product**: F = F_xy * F_yz * F_xz combines features multiplicatively
- **Small MLP decoder**: Decodes combined features to density and view-dependent color
- **Explicit representation**: No coordinate-based MLP - features stored directly in planes

## How It Works

For a 3D point (x, y, z):
1. Project onto XY plane → sample F_xy at (x, y)
2. Project onto YZ plane → sample F_yz at (y, z)
3. Project onto XZ plane → sample F_xz at (x, z)
4. Combine: F = F_xy ⊙ F_yz ⊙ F_xz (element-wise product)
5. Decode F to density σ and color c(d)

## How It Differs from Other Methods

| Aspect | NeRF | K-Planes |
|--------|------|----------|
| Representation | MLP weights | 2D feature planes |
| Memory scaling | O(1) per point | O(N²) per plane |
| Query cost | Full MLP forward | Bilinear sample + small MLP |
| Optimization | Slow (deep MLP) | Fast (direct plane updates) |
| Extensibility | Hard to add time | Add 3 more planes for d=4 |

## Key Advantages

- **1000x compression** vs full 4D grid
- **Fast optimization** with pure PyTorch (no custom CUDA)
- **Natural factorization** of static/dynamic components
- **Easy to extend** to d=4 (dynamic scenes) by adding time planes

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
