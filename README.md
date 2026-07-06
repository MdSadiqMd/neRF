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

8. **[kplanes/](./kplanes/)** - Explicit radiance fields with feature planes
   
   Uses 3 axis-aligned 2D feature planes (XY, YZ, XZ) instead of an MLP. Features are sampled via bilinear interpolation and combined via Hadamard product before decoding to density/color. Achieves 1000x compression over a full 4D grid with fast pure-PyTorch optimization.
   
   Paper: https://arxiv.org/abs/2301.10241
   
   **Key insight**: Factorize 3D space into 2D planes. Easy to extend to d=4 (dynamic scenes) by adding time-dependent planes.

9. **[infoneRF/](./infoneRF/)** - Few-shot NeRF with ray entropy regularization
   
   Standard NeRF with an information-theoretic regularizer: minimizes entropy of the normalized alpha weights along each ray. This penalizes spread-out density (floaters) and encourages compact surface representations. Uses only 4 training images.
   
   Paper: https://arxiv.org/abs/2112.15399
   
   **Key insight**: H(p) = -∑ p_k log(p_k) where p_k = α_k / ∑ α_k. Minimizing ray entropy makes density distributions peak sharply at surfaces, preventing floaters in few-shot settings.

10. **[plenOxels/](./plenOxels/)** - Plenoxels: Radiance Fields without Neural Networks

   Dense 3D voxel grid storing density and spherical harmonic (SH) coefficients. Trilinear interpolation for smooth sampling, SH degree-2 for view-dependent color. Pure gradient optimization — no MLP at all. 58.7M parameters (128³ × 28 channels). Faster training than NeRF but blockier quality due to fixed grid resolution.

   Paper: https://arxiv.org/abs/2112.05131

   **Why quality is lower than NeRF:**
   - Dense grid wastes capacity on empty space (paper uses sparse octree)
   - Fixed resolution cannot adapt to scene complexity
   - No hierarchical sampling
   - No TV regularization or coarse-to-fine

11. **[learnedInitializations/](./learnedInitializations/)** - Meta-learned initialization for NeRF
   
   Uses Reptile (first-order MAML) to learn weight initializations for NeRF that adapt quickly to novel views from few images. Meta-trains across random subsets of training views, then fine-tunes on only 4 test views to render the remaining 18. Achieves 57% lower MSE than random init after the same 5 epochs of fine-tuning.
   
   Paper: https://arxiv.org/abs/2012.02189
   
   **Key insight**: Learning a good initialization prior (φ) means that a few steps of SGD on limited data produces a much better model than starting from random weights. The Reptile update φ ← φ − α(φ − θ_K) pulls the initialization toward parameters that adapt successfully.
   
   **Note**: The paper demonstrates cross-scene meta-learning (different object categories). With a single Lego scene, this implementation demonstrates within-scene meta-learning (adapting from 4 views to 18 novel views).
