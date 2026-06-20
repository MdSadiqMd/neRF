import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import glob
from matplotlib.image import imread
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch


@torch.no_grad()
def test(
    model,
    camera_intrinsics,
    camera_extrinsics,
    hn,
    hf,
    images,
    chunk_size=10,
    img_index=0,
    nb_bins=192,
    H=400,
    W=400,
):
    ray_origins, ray_directions, _ = sample_batch(
        camera_extrinsics,
        camera_intrinsics,
        images,
        None,
        H,
        W,
        img_index=img_index,
        sample_all=True,
    )
    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(
            camera_intrinsics.device
        )
        ray_directions_ = ray_directions[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(camera_intrinsics.device)
        regenerated_px_values = render_rays(
            model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins
        )
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    plt.imshow(img)
    import os

    os.makedirs("novel_views", exist_ok=True)
    plt.savefig(f"novel_views/novel_view.png", bbox_inches="tight")
    plt.close()


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        tmp = self.block2(torch.cat((self.block1(emb_x), emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        c = self.block4(self.block3(torch.cat((h, emb_d), dim=1)))
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)  # Perturb sampling along each ray.
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(
        1
    )  # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(
        nb_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(x.shape[:-1]) * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    c = (weights * colors.reshape(x.shape)).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def train(
    nerf_model,
    optimizers,
    schedulers,
    training_images,
    camera_extrinsics,
    camera_intrinsics,
    batch_size,
    nb_iterations,
    hn=0.0,
    hf=1.0,
    nb_bins=192,
):
    H, W = training_images.shape[1:3]

    training_loss = []
    num_images = training_images.shape[0]

    for _ in tqdm(range(nb_iterations)):
        img_index = np.random.randint(num_images)
        rays_o, rays_d, samples_idx = sample_batch(
            camera_extrinsics,
            camera_intrinsics,
            training_images,
            batch_size,
            H,
            W,
            img_index=img_index,
        )
        # training_images is already a tensor on device
        gt_px_values = training_images[samples_idx]

        regenerated_px_values = render_rays(
            nerf_model, rays_o, rays_d, hn=hn, hf=hf, nb_bins=nb_bins
        )
        loss = ((gt_px_values - regenerated_px_values) ** 2).mean()

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(nerf_model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()

        training_loss.append(loss.item())

        for scheduler in schedulers:
            scheduler.step()

    return training_loss


def load_ground_truth_cameras(data_path, train_indices, device="cpu"):
    """Load ground truth camera parameters from the original dataset."""
    from scipy.spatial.transform import Rotation

    data = np.load(data_path)
    focal = data["focal"]
    poses = data["poses"]  # [N, 4, 4] c2w matrices
    W = data["images"].shape[2]

    # Convert focal to our parameterization: focal = param^2 * W
    # So param = sqrt(focal / W)
    camera_intrinsics = torch.tensor(
        [np.sqrt(focal / W)], device=device, dtype=torch.float32, requires_grad=False
    )

    # Extract translation and rotation from c2w matrices for training images only
    train_poses = poses[train_indices]
    N = len(train_indices)
    camera_extrinsics = torch.zeros((N, 6), device=device, dtype=torch.float32)

    for i in range(N):
        c2w = train_poses[i]
        t = c2w[:3, 3]
        R = c2w[:3, :3]
        r = Rotation.from_matrix(R).as_rotvec()
        camera_extrinsics[i, :3] = torch.from_numpy(t).float()
        camera_extrinsics[i, 3:] = torch.from_numpy(r).float()

    camera_extrinsics.requires_grad = False
    return camera_intrinsics, camera_extrinsics


def initialize_camera_parameters(images, device="cpu"):
    from scipy.spatial.transform import Rotation

    # Initialize focal length closer to the true value (138.88 for W=100) -> sqrt(1.3888) = ~1.178
    camera_intrinsics = torch.tensor([1.178], device=device, requires_grad=True)

    N = images.shape[0] if isinstance(images, np.ndarray) else images.shape[0]
    camera_extrinsics = torch.zeros((N, 6), device=device, dtype=torch.float32)

    # Initialize cameras uniformly distributed on upper hemisphere at radius ~4.0
    phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.random.uniform(0.3, 1.2, N)
    radius = 4.0 + np.random.randn(N) * 0.2

    t = torch.zeros(N, 3)
    t[:, 0] = torch.from_numpy(radius * np.sin(theta) * np.cos(phi)).float()
    t[:, 1] = torch.from_numpy(radius * np.sin(theta) * np.sin(phi)).float()
    t[:, 2] = torch.from_numpy(radius * np.cos(theta)).float()

    z_cam = -t / t.norm(dim=1, keepdim=True)
    up = torch.tensor([0.0, 0.0, 1.0]).expand(N, 3).clone()

    parallel_mask = torch.abs(torch.sum(z_cam * up, dim=1)) > 0.99
    up[parallel_mask] = torch.tensor([0.0, 1.0, 0.0])

    x_cam = torch.cross(up, z_cam, dim=1)
    x_cam = x_cam / (x_cam.norm(dim=1, keepdim=True) + 1e-8)
    y_cam = torch.cross(z_cam, x_cam, dim=1)
    y_cam = y_cam / (y_cam.norm(dim=1, keepdim=True) + 1e-8)

    R = torch.stack([x_cam, y_cam, z_cam], dim=2)
    r = torch.from_numpy(Rotation.from_matrix(R.numpy()).as_rotvec()).float()

    camera_extrinsics[:, :3] = t.to(device)
    camera_extrinsics[:, 3:] = r.to(device)
    camera_extrinsics.requires_grad = True

    return camera_intrinsics, camera_extrinsics


def load_images(data_path):
    image_paths = glob.glob(data_path)
    images = None
    for i, image_path in enumerate(image_paths):
        img = np.expand_dims(imread(image_path), 0)
        images = np.concatenate((images, img)) if images is not None else img
    return images


def get_ndc_rays(H, W, focal, rays_o, rays_d, near=1.0):
    # We shift o to the ray’s intersection with the near plane at z = −n (before the NDC conversion)
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    rays_o = torch.stack(
        [
            -focal / W / 2.0 * rays_o[..., 0] / rays_o[..., 2],
            -focal / H / 2.0 * rays_o[..., 1] / rays_o[..., 2],
            1.0 + 2.0 * near / rays_o[..., 2],
        ],
        -1,
    )  # Eq 25 https://arxiv.org/pdf/2003.08934.pdf
    rays_d = torch.stack(
        [
            -focal
            / W
            / 2.0
            * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]),
            -focal
            / H
            / 2.0
            * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]),
            -2.0 * near / rays_o[..., 2],
        ],
        -1,
    )  # Eq 26 https://arxiv.org/pdf/2003.08934.pdf
    return rays_o, rays_d


def sample_batch(
    camera_extrinsics,
    camera_intrinsics,
    images,
    batch_size,
    H,
    W,
    img_index=0,
    sample_all=False,
):
    if sample_all:
        image_indices = (
            torch.zeros(W * H, device=camera_intrinsics.device) + img_index
        ).type(torch.long)
        u, v = np.meshgrid(
            np.linspace(0, W - 1, W, dtype=int), np.linspace(0, H - 1, H, dtype=int)
        )
        u = torch.from_numpy(u.reshape(-1)).to(camera_intrinsics.device)
        v = torch.from_numpy(v.reshape(-1)).to(camera_intrinsics.device)
    else:
        image_indices = (
            torch.zeros(batch_size, device=camera_intrinsics.device) + img_index
        ).type(torch.long)  # Sample random images
        u = torch.randint(
            W, (batch_size,), device=camera_intrinsics.device
        )  # Sample random pixels
        v = torch.randint(H, (batch_size,), device=camera_intrinsics.device)

    focal = camera_intrinsics[0] ** 2 * W
    t = camera_extrinsics[img_index, :3]
    r = camera_extrinsics[img_index, -3:]

    # Creating the c2w matrix, Section 4.1 from the paper
    phi_skew = torch.stack(
        [
            torch.cat([torch.zeros(1, device=r.device), -r[2:3], r[1:2]]),
            torch.cat([r[2:3], torch.zeros(1, device=r.device), -r[0:1]]),
            torch.cat([-r[1:2], r[0:1], torch.zeros(1, device=r.device)]),
        ],
        dim=0,
    )
    alpha = r.norm() + 1e-15
    R = (
        torch.eye(3, device=r.device)
        + (torch.sin(alpha) / alpha) * phi_skew
        + ((1 - torch.cos(alpha)) / alpha**2) * (phi_skew @ phi_skew)
    )
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
    c2w = torch.cat(
        [c2w, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=c2w.device)], dim=0
    )

    rays_d_cam = torch.cat(
        [
            ((u.to(camera_intrinsics.device) - 0.5 * W) / focal).unsqueeze(-1),
            (-(v.to(camera_intrinsics.device) - 0.5 * H) / focal).unsqueeze(-1),
            -torch.ones_like(u).unsqueeze(-1),
        ],
        dim=-1,
    )
    rays_d_world = torch.matmul(
        c2w[:3, :3].view(1, 3, 3), rays_d_cam.unsqueeze(2)
    ).squeeze(2)
    rays_o_world = c2w[:3, 3].view(1, 3).expand_as(rays_d_world)
    # Removed NDC conversion for 360-degree scenes
    return (
        rays_o_world,
        F.normalize(rays_d_world, p=2, dim=1),
        (image_indices, v, u),
    )


if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    nb_iterations = 20000

    # Set to True to use ground truth cameras (standard NeRF)
    # Set to False to learn cameras from scratch (NeRF-- mode, only works for forward-facing)
    USE_GROUND_TRUTH_CAMERAS = True

    import os

    if not os.path.exists("training_images.npy"):
        raise FileNotFoundError(
            "training_images.npy not found. Please run prepare_data.py first."
        )

    # Load data and move to GPU once
    training_images_np = np.load("training_images.npy")
    training_images = torch.from_numpy(training_images_np).to(device)

    # Get the training indices used in prepare_data.py
    n_images = 106  # total images in tiny_nerf
    indices = np.random.RandomState(42).permutation(n_images)
    train_indices = indices[: int(n_images * 0.8)]

    if USE_GROUND_TRUTH_CAMERAS:
        print("Using ground truth camera parameters")
        camera_intrinsics, camera_extrinsics = load_ground_truth_cameras(
            "tiny_nerf_data.npz", train_indices, device=device
        )
    else:
        print("Learning camera parameters from scratch (NeRF-- mode)")
        camera_intrinsics, camera_extrinsics = initialize_camera_parameters(
            training_images_np, device=device
        )

    batch_size = 1024

    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler_model = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer,
        [2000 * (i + 1) for i in range(nb_iterations // 2000)],
        gamma=0.9,
    )

    if USE_GROUND_TRUTH_CAMERAS:
        # Standard NeRF training with known cameras
        train(
            model,
            [model_optimizer],
            [scheduler_model],
            training_images,
            camera_extrinsics,
            camera_intrinsics,
            batch_size,
            nb_iterations,
            hn=2.0,
            hf=6.0,
            nb_bins=192,
        )
    else:
        # NeRF-- two-stage training
        optimizer_camera_parameters = torch.optim.Adam([camera_extrinsics], lr=3e-3)
        optimizer_focal = torch.optim.Adam([camera_intrinsics], lr=1e-3)

        scheduler_camera_parameters = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_camera_parameters,
            [10000 * (i + 1) for i in range(nb_iterations // 10000)],
            gamma=0.81,
        )
        scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_focal,
            [10000 * (i + 1) for i in range(nb_iterations // 10000)],
            gamma=0.9,
        )

        # Part 1: Joint optimization
        train(
            model,
            [model_optimizer, optimizer_camera_parameters, optimizer_focal],
            [scheduler_model, scheduler_camera_parameters, scheduler_focal],
            training_images,
            camera_extrinsics,
            camera_intrinsics,
            batch_size,
            nb_iterations,
            hn=2.0,
            hf=6.0,
            nb_bins=192,
        )

        # Part 2: Continue training with frozen camera parameters
        camera_extrinsics = camera_extrinsics.detach()
        camera_intrinsics = camera_intrinsics.detach()

        model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler_model = torch.optim.lr_scheduler.MultiStepLR(
            model_optimizer,
            [2000 * (i + 1) for i in range(nb_iterations // 2000)],
            gamma=0.9,
        )
        train(
            model,
            [model_optimizer],
            [scheduler_model],
            training_images,
            camera_extrinsics,
            camera_intrinsics,
            batch_size,
            nb_iterations,
            hn=2.0,
            hf=6.0,
            nb_bins=192,
        )

    # Test: render from first training view
    test(
        model,
        camera_intrinsics,
        camera_extrinsics,
        2.0,
        6.0,
        training_images,
        img_index=0,
        nb_bins=192,
        H=training_images.shape[1],
        W=training_images.shape[2],
    )
