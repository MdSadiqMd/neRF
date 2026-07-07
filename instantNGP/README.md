# Instant NGP (Neural Graphics Primitives)

PyTorch implementation of Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.

Paper: https://arxiv.org/abs/2201.05989

## What's Implemented

A compact neural radiance field using a multi-resolution hash encoding:

1. **Multi-Resolution Hash Grid**: 16 levels of hash tables (T=2^19, F=2) with exponentially increasing resolution from 16³ to 2048³
2. **Trilinear Interpolation**: Manual interpolation across 8 corner features at each level (no `F.grid_sample` — MPS-compatible)
3. **Hash Function**: XOR-based spatial hashing with large primes to map 3D grid coordinates to table indices
4. **Tiny MLP**: 32-dim hash features → 64 → 16 (density branch) + 27-dim direction encoding → 64 → 64 → 3 (color branch)
5. **No Hierarchical Sampling**: Single-resolution ray marching (192 samples/ray), unlike NeRF's coarse-to-fine

## How It Works

### Hash Encoding

```
For each level l in [0, L-1]:
    scale coordinate: p_l = x * N_l
    get cell: (i, j, k) = floor(p_l)
    For each of 8 corners:
        hash_idx = (i*π1 ⊕ j*π2 ⊕ k*π3) mod T
        feat = hash_table[hash_idx]
    trilinear_interpolate(8 corner feats, offset = p_l - cell)
Concatenate L level features → [B, F*L]
```

The multi-resolution structure is key: low-res levels capture coarse structure with few collisions, high-res levels capture fine details but have more hash collisions. The tiny MLP learns to disambiguate collisions.

### Tiny MLP

Instead of NeRF's 8-layer 256-dim MLP (1.2M params):
```
Hash features [32] → 64 → ReLU → 16 → splits → σ = exp(H[0])
                                                └→ [16] + dir_enc [27] → 64 → ReLU → 64 → ReLU → RGB
```

Total MLP params: ~21K — 60× smaller than standard NeRF.

### Hash Function

```
hash(x, y, z) = (x·1 ⊕ y·2654435761 ⊕ z·805459861) mod T
```

The large primes (≈2³²) ensure that nearby grid points get very different hash indices, spreading collisions uniformly.

## Why It Works

- **Adaptive resolution**: Low resolutions provide coarse spatial structure (few collisions), high resolutions capture fine detail (more collisions but the MLP disambiguates)
- **Learned hash table**: Features are optimized end-to-end via SGD — the hash table learns which features are useful for the scene
- **Compact network**: The encoding does the heavy lifting, so the MLP can be tiny (3 layers vs 8)
- **Explicit vs implicit memory**: The hash tables are like trainable lookup tables that adapt to the scene's spatial distribution

## Results

- Final loss: 0.00453 (MSE on tiny_nerf, 84 train / 22 test, 16 epochs)
- Training time: ~8 minutes on MPS (vs ~30 min for standard NeRF)
- MLP params: ~21K (vs ~1.2M for standard NeRF)
- Hash table params: 16.8M (but sparse — only accessed at query points)

Quality is comparable to NeRF with fewer MLP parameters, but the hash grid adds memory.

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

## Limitations vs Full Instant NGP

| Feature | This Implementation | Full Instant NGP |
|---------|-------------------|------------------|
| Rendering | Python stratified sampling | Fully fused CUDA kernel with occupancy skipping |
| Training | Standard PyTorch training loop | Custom CUDA kernels with fused operations |
| Performance | ~8 min training, ~2s per 100×100 view | <1 min training, <10ms per 1920×1080 view |
| Occupancy grid | None | Used for empty space skipping |
| Distortion loss | None | Used for improved quality |
