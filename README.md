# neRF

PyTorch implementations of Neural Radiance Fields variants for view synthesis.

## Implementations

1. **[nerf/](./nerf/)** - Original NeRF architecture
   
   Full MLP network with positional encoding for 3D coordinates and view directions. Predicts density and view-dependent color at each point, then uses volume rendering to composite rays into pixels. Produces high-quality novel view synthesis.
   
   Paper: https://arxiv.org/abs/2003.08934

2. **[fastneRF/](./fastneRF/)** - Factorized NeRF for fast inference
   
   Decomposes the radiance field into separate position (Fpos) and direction (Fdir) networks. Position network outputs density + UV basis weights; direction network outputs mixing coefficients. Enables 3000x faster inference via caching, but produces lower quality images.
   
   Paper: https://arxiv.org/abs/2103.10380
   
   **Why FastNeRF has lower quality:**
   - **D=8 bottleneck**: Only 8 basis functions to represent view-dependent radiance, limiting expressiveness
   - **Smaller direction network**: 128 hidden dim, 3 layers vs NeRF's deeper architecture
   - **Factorization trade-off**: Separating position/direction networks reduces capacity for modeling complex view-dependent effects
   
   FastNeRF prioritizes real-time inference (200fps) over image quality - this is the expected trade-off from the paper.
