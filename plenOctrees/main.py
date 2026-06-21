"""
PlenOctrees: Real-time Rendering of Neural Radiance Fields

This implements the NeRF-SH training phase where the network outputs spherical
harmonic coefficients instead of view-dependent colors. This removes the viewing
direction as input to the neural network, enabling pre-tabulation into an octree.

Paper: https://arxiv.org/abs/2103.14024
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
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


def eval_sh(deg, sh_coeffs, dirs):
    """
    Evaluate spherical harmonics at unit directions for RGB.

    Args:
        deg: SH degree (0, 1, or 2)
        sh_coeffs: [N, C, 3] SH coefficients where C = (deg+1)^2
        dirs: [N, 3] unit direction vectors

    Returns:
        [N, 3] RGB colors
    """
    C = (deg + 1) ** 2
    assert sh_coeffs.shape[1] == C

    result = SH_C0 * sh_coeffs[:, 0]

    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result + SH_C1 * (
            -y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2] - x * sh_coeffs[:, 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            result = result + SH_C2[0] * xy * sh_coeffs[:, 4]
            result = result + SH_C2[1] * yz * sh_coeffs[:, 5]
            result = result + SH_C2[2] * (2 * zz - xx - yy) * sh_coeffs[:, 6]
            result = result + SH_C2[3] * xz * sh_coeffs[:, 7]
            result = result + SH_C2[4] * (xx - yy) * sh_coeffs[:, 8]

    return result


class PlenOctreeNeRF(nn.Module):
    """
    NeRF-SH: Network outputs spherical harmonic coefficients for color.

    Key difference from standard NeRF:
    - No view direction input to the network
    - Outputs SH coefficients instead of RGB
    - SH coefficients are evaluated with view direction at render time

    This factorization enables pre-tabulating the network into an octree.
    """

    def __init__(self, embedding_dim_pos=10, hidden_dim=256, sh_degree=2):
        super(PlenOctreeNeRF, self).__init__()

        self.sh_degree = sh_degree
        self.sh_coeffs = (sh_degree + 1) ** 2

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

        self.sh_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.sh_coeffs * 3),
        )

        self.embedding_dim_pos = embedding_dim_pos
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
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])

        sh_coeffs = self.sh_head(h)
        sh_coeffs = sh_coeffs.reshape(-1, self.sh_coeffs, 3)

        d_norm = d / (torch.norm(d, dim=1, keepdim=True) + 1e-8)
        c = eval_sh(self.sh_degree, sh_coeffs, d_norm)
        c = torch.sigmoid(c)

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
    model = PlenOctreeNeRF(hidden_dim=256, sh_degree=2).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[2, 4, 8], gamma=0.5
    )

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
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
