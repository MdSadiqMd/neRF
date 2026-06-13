import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

Path("novel_views").mkdir(exist_ok=True)


class FastNerf(nn.Module):
    def __init__(
        self,
        embedding_dim_pos=10,
        embedding_dim_direction=4,
        hidden_dim_pos=384,
        hidden_dim_dir=128,
        D=8,
    ):
        super(FastNerf, self).__init__()

        self.Fpos = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, hidden_dim_pos),
            nn.ReLU(),
            nn.Linear(hidden_dim_pos, 3 * D + 1),
        )

        self.Fdir = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + 3, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, hidden_dim_dir),
            nn.ReLU(),
            nn.Linear(hidden_dim_dir, D),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.D = D

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim_pos))
        sigma = torch.nn.functional.softplus(
            sigma_uvw[:, 0][..., None]
        )  # [batch_size, 1]
        uvw = torch.sigmoid(
            sigma_uvw[:, 1:].reshape(-1, 3, self.D)
        )  # [batch_size, 3, D]

        beta = torch.softmax(
            self.Fdir(self.positional_encoding(d, self.embedding_dim_direction)), -1
        )
        color = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
        return color, sigma


class Cache(nn.Module):
    def __init__(self, model, scale, device, Np, Nd):
        super(Cache, self).__init__()

        with torch.no_grad():
            # Position
            x, y, z = torch.meshgrid(
                [
                    torch.linspace(-scale / 2, scale / 2, Np).to(device),
                    torch.linspace(-scale / 2, scale / 2, Np).to(device),
                    torch.linspace(-scale / 2, scale / 2, Np).to(device),
                ]
            )
            xyz = torch.cat(
                (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1
            )
            sigma_uvw = model.Fpos(
                model.positional_encoding(xyz, model.embedding_dim_pos)
            )
            self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
            # Direction
            xd, yd = torch.meshgrid(
                [
                    torch.linspace(-scale / 2, scale / 2, Nd).to(device),
                    torch.linspace(-scale / 2, scale / 2, Nd).to(device),
                ]
            )
            xyz_d = torch.cat(
                (
                    xd.reshape(-1, 1),
                    yd.reshape(-1, 1),
                    torch.sqrt((1 - xd**2 - yd**2).clip(0, 1)).reshape(-1, 1),
                ),
                dim=1,
            )
            beta = model.Fdir(
                model.positional_encoding(xyz_d, model.embedding_dim_direction)
            )
            self.beta = beta.reshape((Nd, Nd, -1))

        self.scale = scale
        self.Np = Np
        self.Nd = Nd
        self.D = model.D

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0], 1), device=x.device)

        mask = (
            (x[:, 0].abs() < (self.scale / 2))
            & (x[:, 1].abs() < (self.scale / 2))
            & (x[:, 2].abs() < (self.scale / 2))
        )
        # Position
        idx = (
            (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1)
        )
        sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]]
        # Direction
        idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1)
        beta = torch.softmax(self.beta[idx[:, 0], idx[:, 1]], -1)

        sigma[mask] = torch.nn.functional.softplus(
            sigma_uvw[:, 0][..., None]
        )  # [batch_size, 1]
        uvw = torch.sigmoid(
            sigma_uvw[:, 1:].reshape(-1, 3, self.D)
        )  # [batch_size, 3, D]
        color[mask] = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
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
    u = torch.rand(t.shape, device=ray_origins.device)
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
def test(model, device, hn, hf, dataset, img_index=0, nb_bins=192, H=100, W=100):
    ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]
    regenerated_px_values = render_rays(
        model,
        ray_origins.to(device),
        ray_directions.to(device),
        hn=hn,
        hf=hf,
        nb_bins=nb_bins,
    )

    img = regenerated_px_values.data.cpu().numpy().reshape(H, W, 3)
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
    model = FastNerf().to(device)
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
