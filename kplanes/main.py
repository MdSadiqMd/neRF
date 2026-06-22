"""
K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

This implements the static (d=3) version of K-Planes using 3 axis-aligned feature planes:
XY, YZ, and XZ. Features are sampled via bilinear interpolation and combined via
Hadamard product before being decoded to density and color.

Paper: https://arxiv.org/abs/2301.10241
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


class KPlanesModel(nn.Module):
    """
    K-Planes for static 3D scenes using 3 feature planes.

    For a 3D point (x, y, z):
    - Sample from XY plane at (x, y)
    - Sample from YZ plane at (y, z)
    - Sample from XZ plane at (x, z)
    - Combine via Hadamard product: F = F_xy * F_yz * F_xz
    - Decode to density and color
    """

    def __init__(
        self,
        embedding_dim_direction=4,
        hidden_dim=64,
        N=256,
        num_features=32,
        scale=1.5,
    ):
        super(KPlanesModel, self).__init__()

        self.xy_plane = nn.Parameter(torch.randn((N, N, num_features)) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn((N, N, num_features)) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn((N, N, num_features)) * 0.1)

        self.density_head = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
        )

        self.color_head = nn.Sequential(
            nn.Linear(15 + embedding_dim_direction * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_direction = embedding_dim_direction
        self.scale = scale
        self.N = N

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def sample_plane(self, plane, coords):
        """
        Bilinear interpolation from a feature plane (MPS-compatible).

        Args:
            plane: [N, N, F] feature plane
            coords: [B, 2] normalized coordinates in [-1, 1]

        Returns:
            [B, F] sampled features
        """
        N = plane.shape[0]
        coords_pixel = (coords + 1) * 0.5 * (N - 1)
        coords_pixel = coords_pixel.clamp(0, N - 1 - 1e-4)

        x0 = coords_pixel[:, 0].long()
        y0 = coords_pixel[:, 1].long()
        x1 = (x0 + 1).clamp(max=N - 1)
        y1 = (y0 + 1).clamp(max=N - 1)

        xd = (coords_pixel[:, 0] - x0.float()).unsqueeze(-1)
        yd = (coords_pixel[:, 1] - y0.float()).unsqueeze(-1)

        c00 = plane[x0, y0]
        c01 = plane[x0, y1]
        c10 = plane[x1, y0]
        c11 = plane[x1, y1]

        c0 = c00 * (1 - xd) + c10 * xd
        c1 = c01 * (1 - xd) + c11 * xd

        return c0 * (1 - yd) + c1 * yd

    def forward(self, x, d):
        sigma = torch.zeros(x.shape[0], device=x.device)
        c = torch.zeros(x.shape[0], 3, device=x.device)

        mask = (
            (x[:, 0].abs() < self.scale)
            & (x[:, 1].abs() < self.scale)
            & (x[:, 2].abs() < self.scale)
        )

        if mask.sum() == 0:
            return c, sigma

        x_masked = x[mask]
        d_masked = d[mask]

        xy_coords = x_masked[:, :2] / self.scale
        yz_coords = x_masked[:, 1:] / self.scale
        xz_coords = x_masked[:, [0, 2]] / self.scale

        F_xy = self.sample_plane(self.xy_plane, xy_coords)
        F_yz = self.sample_plane(self.yz_plane, yz_coords)
        F_xz = self.sample_plane(self.xz_plane, xz_coords)

        features = F_xy * F_yz * F_xz

        h = self.density_head(features)
        sigma_out = F.softplus(h[:, -1])
        h = h[:, :-1]

        emb_d = self.positional_encoding(d_masked, self.embedding_dim_direction)
        c_out = self.color_head(torch.cat([h, emb_d], dim=1))

        sigma[mask] = sigma_out
        c[mask] = c_out

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

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
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
    nerf_model,
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
                nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
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
                    nerf_model,
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

    model = KPlanesModel(hidden_dim=128, N=256, num_features=32, scale=3.0).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[5, 10], gamma=0.5
    )

    data_loader = DataLoader(training_dataset, batch_size=4096, shuffle=True)
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
