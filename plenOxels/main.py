"""
Plenoxels: Radiance Fields without Neural Networks

Key ideas:
1. Dense 3D voxel grid storing density and spherical harmonic coefficients
2. Trilinear interpolation for smooth sampling
3. Spherical harmonics for view-dependent color (degree 2 = 9 coefficients per channel)
4. Pure optimization - no neural network at all

Paper: https://arxiv.org/abs/2112.05131
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Path("novel_views").mkdir(exist_ok=True)


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions.

    Args:
        deg: SH degree (0, 1, or 2)
        sh: [..., (deg+1)^2, 3] SH coefficients per RGB channel
        dirs: [..., 3] unit direction vectors

    Returns:
        [..., 3] RGB colors
    """
    out = SH_C0 * sh[..., 0, :]
    if deg >= 1:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        out = out + SH_C1 * (-y * sh[..., 1, :] + z * sh[..., 2, :] - x * sh[..., 3, :])
        if deg >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            out = out + SH_C2[0] * xy * sh[..., 4, :]
            out = out + SH_C2[1] * yz * sh[..., 5, :]
            out = out + SH_C2[2] * (2 * zz - xx - yy) * sh[..., 6, :]
            out = out + SH_C2[3] * xz * sh[..., 7, :]
            out = out + SH_C2[4] * (xx - yy) * sh[..., 8, :]
    return out


class PlenoxelsModel(nn.Module):
    """
    Plenoxels: A dense voxel grid with spherical harmonic coefficients.

    Each voxel stores:
    - 1 density value (σ)
    - 27 SH coefficients (9 per RGB channel, degree 2)

    No neural network - just a voxel grid with trilinear interpolation.
    """

    def __init__(self, N=256, scale=3.0, sh_degree=2):
        super(PlenoxelsModel, self).__init__()
        n_coeffs = (sh_degree + 1) ** 2  # 9 for degree 2
        self.voxel_grid = nn.Parameter(torch.zeros((N, N, N, n_coeffs * 3 + 1)))
        self.scale = scale
        self.N = N
        self.sh_degree = sh_degree

    def forward(self, x, d):
        """
        Args:
            x: [B, 3] 3D positions
            d: [B, 3] view directions (will be normalized)

        Returns:
            colors: [B, 3] RGB colors
            sigma: [B] density values
        """
        B = x.shape[0]
        color = torch.zeros(B, 3, device=x.device)
        sigma = torch.zeros(B, device=x.device)

        mask = (
            (x[:, 0].abs() < self.scale)
            & (x[:, 1].abs() < self.scale)
            & (x[:, 2].abs() < self.scale)
        )

        if mask.sum() == 0:
            return color, sigma

        x_masked = x[mask]

        # Normalize coordinates to [0, N-1] range for grid sampling
        coords = (x_masked / (2 * self.scale) + 0.5) * (self.N - 1)

        # Trilinear interpolation indices
        x0 = coords[:, 0].long().clamp(0, self.N - 2)
        y0 = coords[:, 1].long().clamp(0, self.N - 2)
        z0 = coords[:, 2].long().clamp(0, self.N - 2)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        xd = (coords[:, 0] - x0.float()).unsqueeze(-1)
        yd = (coords[:, 1] - y0.float()).unsqueeze(-1)
        zd = (coords[:, 2] - z0.float()).unsqueeze(-1)

        # Sample 8 corners
        c000 = self.voxel_grid[x0, y0, z0]  # [B_masked, 28]
        c001 = self.voxel_grid[x0, y0, z1]
        c010 = self.voxel_grid[x0, y1, z0]
        c011 = self.voxel_grid[x0, y1, z1]
        c100 = self.voxel_grid[x1, y0, z0]
        c101 = self.voxel_grid[x1, y0, z1]
        c110 = self.voxel_grid[x1, y1, z0]
        c111 = self.voxel_grid[x1, y1, z1]

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        grid_out = c0 * (1 - zd) + c1 * zd

        # Split into density and SH coefficients
        sigma_masked = F.softplus(grid_out[:, 0])
        sh_coeffs = grid_out[:, 1:].reshape(-1, (self.sh_degree + 1) ** 2, 3)

        # Normalize view directions
        d_norm = d[mask] / (d[mask].norm(dim=1, keepdim=True) + 1e-8)
        color_masked = torch.sigmoid(eval_sh(self.sh_degree, sh_coeffs, d_norm))

        sigma[mask] = sigma_masked
        color[mask] = color_masked

        return color, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions = ray_directions.expand(
        nb_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    colors, sigma = model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def test(
    model,
    device,
    hn,
    hf,
    dataset,
    chunk_size=10,
    img_index=0,
    nb_bins=192,
    H=100,
    W=100,
):
    ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(
            device
        )
        ray_directions_ = ray_directions[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(device)
        regenerated_px_values = render_rays(
            model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins
        )
        data.append(regenerated_px_values)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    img = np.clip(img, 0, 1)
    plt.imsave(f"novel_views/img_{img_index}.png", img)


def train(
    model,
    optimizer,
    scheduler,
    data_loader,
    testing_dataset,
    device="cpu",
    hn=0,
    hf=1,
    nb_epochs=16,
    nb_bins=192,
    H=100,
    W=100,
):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_loss = 0
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(
                model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        training_loss.append(avg_loss)
        scheduler.step()
        print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

        if epoch == nb_epochs - 1:
            n_test_images = testing_dataset.shape[0] // (H * W)
            for img_index in range(n_test_images):
                test(
                    model,
                    device,
                    hn,
                    hf,
                    testing_dataset,
                    img_index=img_index,
                    nb_bins=nb_bins,
                    H=H,
                    W=W,
                )

    return training_loss


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    training_dataset = torch.from_numpy(np.load("training_data.pkl", allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load("testing_data.pkl", allow_pickle=True))

    model = PlenoxelsModel(N=128, scale=3.0, sh_degree=2).to(device)
    print(f"Voxel grid params: {model.voxel_grid.numel()}")
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[5, 10], gamma=0.5
    )

    data_loader = DataLoader(training_dataset, batch_size=2048, shuffle=True)
    train(
        model,
        model_optimizer,
        scheduler,
        data_loader,
        testing_dataset,
        nb_epochs=16,
        device=device,
        hn=2,
        hf=6,
        nb_bins=192,
        H=100,
        W=100,
    )
