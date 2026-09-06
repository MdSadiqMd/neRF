# PixelNeRF

PyTorch implementation of PixelNeRF: Neural Radiance Fields from One or Few Images.

Paper: https://arxiv.org/abs/2012.02190

## What's Implemented

Image-conditioned NeRF that generalizes to novel scenes from a single input view:

1. **Image Encoder**: ResNet34 backbone extracting dense 512-dim pixel-aligned features (64+64+128+256). Features are upsampled to H×W via bilinear interpolation.
2. **Pixel-Aligned Conditioning**: Each 3D point is projected onto the source view using the source camera intrinsics/extrinsics; features are sampled via `F.grid_sample`.
3. **Source-Frame NeRF**: Points and view directions are expressed in the source camera coordinate frame (`pts_c = W2C_src * pts_w`). Positional encoding (L=6) of points plus view direction are concatenated with sampled features.
4. **Residual MLP**: `hidden=256`, 4 ResNet blocks where each block adds projected image features (`h = h + proj(feat) + block(h)`) before predicting `rgb` (sigmoid) and `sigma` (ReLU).
5. **Volumetric Rendering**: Same stratified sampling (64 samples/ray, near=2 far=6 at `H=W=100`) as NeRF, but conditioned on source-view features.

## How It Works

### Feature Extraction

```
src_img [3,H,W] -> ResNet34 -> f0(64,32,32) f1(64,32,32) f2(128,16,16) f3(256,8,8)
               -> upsample each to H,W -> concat -> feat [512,H,W]
```

ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) is applied before the backbone.

### Projection & Sampling

For each sample point `p_w` along a target ray:

```
p_c = R_src^T * p_w + t_src          # to source camera frame
d_c = normalize(R_src^T * d_w)       # direction in source frame

z = p_c.z,  u = fx * p_c.x / z + cx,  v = fy * p_c.y / z + cy
grid = (2*(u-0.5)/(W-1)-1, 2*(v-0.5)/(H-1)-1)   # to [-1,1] for grid_sample
feat = grid_sample(feat_map, grid)            # [C] pixel-aligned feature
```

### Conditional NeRF

```
x_enc = posenc(p_c, L=6)          # 3 + 3*2*6 = 39
h = ReLU(Linear([x_enc, d_c] -> hidden))
for i in 0..n_blocks-1:
    h = h + Linear(feat -> hidden)  # inject image feature
    h = h + MLP_block(h)
rgb, sigma = Sigmoid(Linear(h)), ReLU(Linear(h))
```

The repeated addition of image features at every block lets the network modulate geometry/appearance based on the source view.

### Training

Single-scene adaptation of the original multi-scene setup: the 106 tiny_nerf lego views are split 84/22 (seed 42). Each training step samples a random `src`/`tgt` pair from the 84 training views, renders `2*1024` random rays from the target view conditioned on the source image, and minimizes MSE. 6000 steps, Adam 5e-5, 64 samples/ray.

### Rendering

At test time, the first test view is encoded once (`feat = Encoder(src_img)`), then all 22 test views are rendered via `render_image` conditioned on that single source feature map. Outputs in `novel_views/img_*.png` plus `img_source.png`.

## Why It Works

- **Pixel-aligned prior**: The feature at the projected location tells the MLP "what the scene looks like at that 3D point from the input view", allowing generalization without per-scene optimization.
- **Source-frame reasoning**: Expressing points in the source camera frame makes the representation canonical — the same 3D point always maps to the same feature regardless of target view.
- **Multi-scale ResNet features**: Low-res features capture global shape, high-res features capture texture — concatenated to give the MLP both.
- **View-direction in source frame**: Lets the MLP model view-dependent effects relative to the input view.

## Commands

```bash
# Install dependencies
just install

# Download and prepare training data (tiny_nerf 100x100)
just prepare

# Train the model (6000 steps)
just train

# Or run everything at once
just all
```

## Output

Trained model renders 22 novel test views to `novel_views/img_*.png` (100×100) conditioned on the first test image (`novel_views/img_source.png` is the source). Example `novel_views/img_0.png` is the same pose as the source and should closely match it; farther views degrade gracefully.

## Limitations vs Full PixelNeRF

| Feature | This Implementation | Full PixelNeRF |
|---------|-------------------|----------------|
| Dataset | Single synthetic lego scene (84 train views) | Multiple ShapeNet categories (chairs/cars) with thousands of scenes |
| Generalization | Within-scene novel view synthesis (same object) | Cross-scene (unseen objects) |
| Views per scene | 1 source view | 1-3 source views (paper shows 1-2-3 view conditioning) |
| Training | 6000 steps, batch 2, 64 samples/ray | 400k steps, batch 4, 64-128 samples/ray |
| Backbone | ResNet34 truncated at layer3 | ResNet34 with full feature pyramid |
| Background | Black (tiny_nerf) vs white (ShapeNet) | White (synthetic) |
| Performance | MPS ~30 min training, ~1s per 100×100 view | GPU days training, real-time inference |

The single-scene setup cannot demonstrate cross-category generalization — the core contribution of the original paper — but it faithfully reproduces the image-conditioned rendering pipeline on the same tiny_nerf data used by the other NeRF variants in this repo.
