import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Path("novel_views").mkdir(exist_ok=True)


class KiloNerf(nn.Module):
    """
    KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs.

    Instead of one large MLP, KiloNeRF partitions the scene into an N×N×N grid
    of independent tiny MLPs. Each query point is routed to its grid cell's MLP,
    enabling massive parallelism and faster inference.

    Architecture per cell (from Figure 2 of the paper):
        Layer 1: pos_encoding(x) → 32  (position features)
        Layer 2: 32 → 33              (32 hidden + 1 density)
        Layer 3: 32 → 32              (skip connection prep)
        Layer 4: [32 + dir_encoding(d)] → 32  (direction-dependent)
        Layer 5: 32 → 3               (RGB output)
    """

    def __init__(
        self, N, embedding_dim_pos=10, embedding_dim_direction=4, scene_scale=3
    ):
        super(KiloNerf, self).__init__()

        # Each layer stores independent weights for all N^3 cells.
        # Xavier uniform initialization for stable training.
        self.layer1_w = nn.Parameter(
            torch.zeros((N, N, N, 63, 32)).uniform_(
                -np.sqrt(6.0 / 85), np.sqrt(6.0 / 85)
            )
        )
        self.layer1_b = nn.Parameter(torch.zeros((N, N, N, 1, 32)))

        self.layer2_w = nn.Parameter(
            torch.zeros((N, N, N, 32, 33)).uniform_(
                -np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)
            )
        )
        self.layer2_b = nn.Parameter(torch.zeros((N, N, N, 1, 33)))

        self.layer3_w = nn.Parameter(
            torch.zeros((N, N, N, 32, 32)).uniform_(
                -np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)
            )
        )
        self.layer3_b = nn.Parameter(torch.zeros((N, N, N, 1, 32)))

        self.layer4_w = nn.Parameter(
            torch.zeros((N, N, N, 27 + 32, 32)).uniform_(
                -np.sqrt(6.0 / 64), np.sqrt(6.0 / 64)
            )
        )
        self.layer4_b = nn.Parameter(torch.zeros((N, N, N, 1, 32)))

        self.layer5_w = nn.Parameter(
            torch.zeros((N, N, N, 32, 3)).uniform_(
                -np.sqrt(6.0 / 35), np.sqrt(6.0 / 35)
            )
        )
        self.layer5_b = nn.Parameter(torch.zeros((N, N, N, 1, 3)))

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.N = N
        self.scale = scene_scale

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0]), device=x.device)

        # Only process points inside the scene bounding box
        mask = (
            (x[:, 0].abs() < (self.scale / 2))
            & (x[:, 1].abs() < (self.scale / 2))
            & (x[:, 2].abs() < (self.scale / 2))
        )
        # Map continuous positions to discrete grid indices
        idx = (x[mask] / (self.scale / self.N) + self.N / 2).long().clip(0, self.N - 1)

        emb_x = self.positional_encoding(x[mask], self.embedding_dim_pos)
        emb_d = self.positional_encoding(d[mask], self.embedding_dim_direction)

        # Forward pass through per-cell MLPs (batched matmuls)
        # Layer 1: position encoding → hidden
        h = torch.relu(
            emb_x.unsqueeze(1) @ self.layer1_w[idx[:, 0], idx[:, 1], idx[:, 2]]
            + self.layer1_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        )
        # Layer 2: hidden → hidden + density (last channel)
        h = torch.relu(
            h @ self.layer2_w[idx[:, 0], idx[:, 1], idx[:, 2]]
            + self.layer2_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        )
        h, density = h[:, :, :-1], h[:, :, -1]
        # Layer 3: hidden → hidden (pre-direction)
        h = (
            h @ self.layer3_w[idx[:, 0], idx[:, 1], idx[:, 2]]
            + self.layer3_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        )
        # Layer 4: concat direction encoding, hidden → hidden
        h = torch.relu(
            torch.cat((h, emb_d.unsqueeze(1)), dim=-1)
            @ self.layer4_w[idx[:, 0], idx[:, 1], idx[:, 2]]
            + self.layer4_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        )
        # Layer 5: hidden → RGB
        c = torch.sigmoid(
            h @ self.layer5_w[idx[:, 0], idx[:, 1], idx[:, 2]]
            + self.layer5_b[idx[:, 0], idx[:, 1], idx[:, 2]]
        )
        color[mask] = c.squeeze(1)
        sigma[mask] = density.squeeze(1)
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


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
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
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def test(
    model, device, hn, hf, dataset, img_index=0, nb_bins=192, H=100, W=100, chunk_size=5
):
    """Render a test image in chunks to avoid OOM."""
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
    nb_epochs=int(1e5),
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
    model = KiloNerf(16).to(device)
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
