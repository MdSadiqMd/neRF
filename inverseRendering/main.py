import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Path("novel_views").mkdir(exist_ok=True)


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


class FourierFeatureNeRF(nn.Module):
    """
    NeRF with Random Fourier Features instead of deterministic positional encoding.

    From "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    Uses random Gaussian projection followed by sin/cos to overcome spectral bias of MLPs.
    """

    def __init__(self, NB_features=256, hidden_dim=128, scale=6.05, device="cpu"):
        super(FourierFeatureNeRF, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(NB_features * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.register_buffer("B", torch.randn((3, NB_features)) * 2 * np.pi * scale)

    def positional_encoding(self, x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)

    def forward(self, o):
        emb_x = self.positional_encoding(o)
        out = self.net(emb_x)
        c, sigma = self.sigmoid(out[:, :-1]), self.relu(out[:, -1])
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
    colors, sigma = nerf_model(x.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)


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
    model = FourierFeatureNeRF(hidden_dim=256, device=device).to(device)
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
