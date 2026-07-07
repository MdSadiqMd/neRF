"""
Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

Key ideas:
1. Multi-resolution hash grid: L=16 levels with exponentially increasing resolution
2. Each level: hash table (T=2^19 entries, F=2 features each), trilinear interpolation
3. Concatenated features fed to tiny MLP (no large 8-layer network)
4. View-dependence via positional encoding of directions (L=4)
5. No hierarchical sampling, no coarse network — just hash + tiny MLP

Paper: https://arxiv.org/abs/2201.05989
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Path("novel_views").mkdir(exist_ok=True)


CORNER_OFFSETS = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.long,
)


class InstantNGP(nn.Module):
    def __init__(
        self, L=16, F=2, T=2**19, N_min=16, N_max=2048, dir_enc_L=4, aabb_scale=3.0
    ):
        super().__init__()
        self.L = L
        self.F = F
        self.T = T
        self.aabb_scale = aabb_scale
        self.dir_enc_L = dir_enc_L

        b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
        self.Nl = [int(np.floor(N_min * b**l)) for l in range(L)]

        self.hash_tables = nn.ParameterList(
            [nn.Parameter((torch.rand(T, F) * 2 - 1) * 1e-4) for _ in range(L)]
        )

        self.pi1 = 1
        self.pi2 = 2654435761
        self.pi3 = 805459861

        self.density_mlp = nn.Sequential(
            nn.Linear(F * L, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

        dir_dim = 3 + 2 * 3 * dir_enc_L
        self.color_mlp = nn.Sequential(
            nn.Linear(16 + dir_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def hash(self, coords):
        h = (
            (coords[:, 0] * self.pi1)
            ^ (coords[:, 1] * self.pi2)
            ^ (coords[:, 2] * self.pi3)
        )
        return h % self.T

    def encode_direction(self, d):
        out = [d]
        for j in range(self.dir_enc_L):
            out.append(torch.sin(2**j * d))
            out.append(torch.cos(2**j * d))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        B = x.shape[0]

        x = x / self.aabb_scale + 0.5

        in_bounds = (
            (x[:, 0] >= 0)
            & (x[:, 0] <= 1)
            & (x[:, 1] >= 0)
            & (x[:, 1] <= 1)
            & (x[:, 2] >= 0)
            & (x[:, 2] <= 1)
        )

        color = torch.zeros(B, 3, device=x.device)
        sigma = torch.zeros(B, device=x.device)

        if in_bounds.sum() == 0:
            return color, sigma

        x_in = x[in_bounds]

        feats = torch.zeros(x_in.shape[0], self.F * self.L, device=x.device)
        offsets = CORNER_OFFSETS.to(x_in.device)

        for l_idx, N in enumerate(self.Nl):
            scaled = x_in * N
            cell = torch.floor(scaled).long()
            offset = scaled - cell.float()

            corners = cell.unsqueeze(0) + offsets.unsqueeze(1)
            h_in = self.hash(corners.reshape(-1, 3)).reshape(8, -1)

            table = self.hash_tables[l_idx]
            corner_feats = table[h_in]

            ox, oy, oz = offset[:, 0], offset[:, 1], offset[:, 2]
            w = torch.stack(
                [
                    (1 - ox) * (1 - oy) * (1 - oz),
                    ox * (1 - oy) * (1 - oz),
                    (1 - ox) * oy * (1 - oz),
                    ox * oy * (1 - oz),
                    (1 - ox) * (1 - oy) * oz,
                    ox * (1 - oy) * oz,
                    (1 - ox) * oy * oz,
                    ox * oy * oz,
                ],
                dim=0,
            )

            feat = (corner_feats * w.unsqueeze(-1)).sum(dim=0)
            feats[:, l_idx * self.F : (l_idx + 1) * self.F] = feat

        h = self.density_mlp(feats)
        sigma_in = torch.exp(h[:, 0])

        d_enc = self.encode_direction(d[in_bounds])
        color_in = self.color_mlp(torch.cat([h, d_enc], dim=1))

        sigma[in_bounds] = sigma_in
        color[in_bounds] = color_in

        return color, sigma


def compute_accumulated_transmittance(alphas):
    acc = torch.cumprod(alphas, 1)
    return torch.cat(
        (torch.ones((acc.shape[0], 1), device=acc.device), acc[:, :-1]), dim=-1
    )


def render_rays(model, ray_origins, ray_directions, hn=2, hf=6, nb_bins=192):
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
        ro = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(device)
        rd = ray_directions[i * W * chunk_size : (i + 1) * W * chunk_size].to(device)
        pred = render_rays(model, ro, rd, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(pred)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    img = np.clip(img, 0, 1)
    plt.imsave(f"novel_views/img_{img_index}.png", img)


def train(
    model,
    optimizer,
    data_loader,
    device="cpu",
    hn=2,
    hf=6,
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
            gt_px_values = batch[:, 6:].to(device)

            pred_px_values = render_rays(
                model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )
            loss = ((gt_px_values - pred_px_values) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

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

    model = InstantNGP(
        L=16, F=2, T=2**19, N_min=16, N_max=2048, dir_enc_L=4, aabb_scale=3.0
    ).to(device)
    print(
        f"Hash grid: {model.L} levels x {model.T} x {model.F} = {model.L * model.T * model.F} params"
    )
    print(f"Resolutions: {model.Nl}")

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.hash_tables.parameters(),
                "lr": 1e-2,
                "betas": (0.9, 0.99),
                "eps": 1e-15,
            },
            {
                "params": model.density_mlp.parameters(),
                "lr": 1e-2,
                "betas": (0.9, 0.99),
                "eps": 1e-15,
                "weight_decay": 1e-6,
            },
            {
                "params": model.color_mlp.parameters(),
                "lr": 1e-2,
                "betas": (0.9, 0.99),
                "eps": 1e-15,
                "weight_decay": 1e-6,
            },
        ]
    )

    data_loader = DataLoader(training_dataset, batch_size=16384, shuffle=True)
    train(model, optimizer, data_loader, nb_epochs=16, device=device, H=100, W=100)

    n_test_images = testing_dataset.shape[0] // (100 * 100)
    for img_index in range(n_test_images):
        test(model, device, 2, 6, testing_dataset, img_index=img_index, H=100, W=100)
