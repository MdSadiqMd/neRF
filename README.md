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

3. **[kiloneRF/](./kiloneRF/)** - Grid of thousands of tiny MLPs
   
   Partitions the scene into an N×N×N grid where each cell has its own tiny MLP. Points are routed to their cell's network, enabling massive parallelism. Designed for real-time rendering with custom CUDA kernels.
   
   Paper: https://arxiv.org/abs/2103.13744
   
   **Why KiloNeRF produces poor quality (and is slow):**
   
   The current implementation is fundamentally incomplete. The paper states: "using teacher-student distillation for training, we show that this speed-up can be achieved without sacrificing visual quality."
   
   | Issue | Current Implementation | Paper's Approach |
   |-------|----------------------|------------------|
   | Training | Direct from RGB images | Teacher-student distillation from pre-trained NeRF |
   | Architecture | 32-dim tiny MLPs learning from scratch | Tiny MLPs distilled from 256-dim teacher |
   | Grid boundaries | Hard boundaries, no interpolation | Occupancy-aware sampling |
   | Performance | Python indexed matmul (slow) | Custom CUDA kernels (fast) |
   
   Without distillation, each tiny MLP only sees sparse samples from its grid cell and cannot learn a good representation. The blocky artifacts are from hard cell boundaries. The slowness is because KiloNeRF requires custom CUDA kernels to achieve the claimed 3 orders of magnitude speedup.

4. **[inverseRendering/](./inverseRendering/)** - Fourier Feature NeRF
   
   Uses Random Fourier Features instead of deterministic positional encoding. Maps inputs through `sin/cos(x @ B)` where B is a random Gaussian matrix, making the neural tangent kernel stationary with tunable bandwidth. No view-direction dependency.
   
   Paper: https://arxiv.org/abs/2006.10739
   
   **Why quality is poor despite fast training:**
   - **No view-direction input**: Cannot model view-dependent effects (specular, reflections)
   - **Random encoding**: The random matrix B may not be optimal; deterministic powers-of-2 encoding is better suited for multi-scale scenes
   - **Simpler architecture**: 4-layer MLP vs NeRF's 8-layer with skip connections
   - **No hierarchical sampling**: Uses uniform sampling instead of coarse-to-fine
   
   The paper's contribution is theoretical (NTK analysis) - the Fourier feature insight was incorporated into NeRF's positional encoding design, not meant as a standalone replacement.

5. **[nerf-minus-minus/](./nerf-minus-minus/)** - NeRF without known camera parameters
   
   Jointly optimizes camera intrinsics (focal length), extrinsics (6-DoF poses), and the NeRF model through photometric loss. Removes the need for COLMAP/SfM preprocessing.
   
   Paper: https://arxiv.org/abs/2102.07064
   
   **Limitation: Forward-facing scenes only.**
   
   The joint optimization can recover accurate cameras for forward-facing scenes where cameras share a roughly consistent viewing direction. For 360-degree scenes (like tiny_nerf), camera pose estimation from scratch fails due to too many degrees of freedom and local minima. Use ground truth cameras for 360-degree scenes.

6. **[freeneRF/](./freeneRF/)** - Few-shot NeRF with frequency regularization
   
   Two "free lunch" techniques for few-shot neural rendering: (1) progressively unmask positional encoding frequencies during training, and (2) penalize near-camera density to prevent floaters. Achieves state-of-the-art few-shot performance with minimal code changes.
   
   Paper: https://arxiv.org/abs/2303.07418
   
   **Key insight**: Limit high-frequency encoding early in training to force learning robust low-frequency structure first, preventing overfitting when training views are scarce.

7. **[plenOctrees/](./plenOctrees/)** - Spherical Harmonic NeRF for real-time rendering
   
   Network outputs spherical harmonic (SH) coefficients instead of view-dependent RGB. Removes viewing direction as network input - view dependence is encoded in SH coefficients that are evaluated at render time. Enables pre-tabulation into an octree for 150+ FPS rendering.
   
   Paper: https://arxiv.org/abs/2103.14024
   
   **Key insight**: Factorize view-dependent appearance into position-dependent SH coefficients (cacheable) and direction-dependent SH basis functions (cheap closed-form). This implementation covers the NeRF-SH training phase only.
