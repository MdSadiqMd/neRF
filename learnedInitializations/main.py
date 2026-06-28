"""
Learned Initializations for Optimizing Coordinate-Based Neural Representations

Meta-learns initial weights for NeRF using Reptile, enabling faster convergence
and better few-shot generalization from limited views.

Core idea: Learn initialization phi such that for any task (fitting a scene from
a sparse subset of images), K steps of SGD from phi produces a good model.

Algorithm (Reptile):
  1. Sample task: pick support images from training set
  2. Save initial params
  3. Inner loop: K steps SGD on support set rays (in-place)
  4. Update meta-params: phi <- phi - lr * (phi - theta_final)

Paper: https://arxiv.org/abs/2012.02189
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy

Path("novel_views").mkdir(exist_ok=True)

H, W = 100, 100
N_SUPPORT = 4
N_QUERY = 2
META_ITERS = 200
INNER_K = 16
INNER_LR = 0.01
BATCH_SIZE = 1024
META_LR = 1.0


class NerfModel(nn.Module):
    """Standard NeRF MLP with positional encoding."""

    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=256):
        super().__init__()

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
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


def compute_accumulated_transmittance(alphas):
    acc = torch.cumprod(alphas, 1)
    return torch.cat(
        (torch.ones((acc.shape[0], 1), device=acc.device), acc[:, :-1]), dim=-1
    )


def render_rays(nerf_model, ray_origins, ray_directions, hn=2, hf=6, nb_bins=192):
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


def compute_loss(model, batch, device, hn=2, hf=6, nb_bins=192):
    ray_origins = batch[:, :3].to(device)
    ray_directions = batch[:, 3:6].to(device)
    gt = batch[:, 6:].to(device)
    pred = render_rays(
        model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
    )
    return F.mse_loss(pred, gt)


def inner_loop(model, task_data, K, lr=INNER_LR, batch_size=BATCH_SIZE, device="cpu"):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    n = task_data.shape[0]
    for _ in range(K):
        idx = torch.randint(0, n, (batch_size,))
        loss = compute_loss(model, task_data[idx].to(device), device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def reptile(meta_model, train_images, nb_meta_iter, K, device):
    n_images = train_images.shape[0]

    for meta_iter in tqdm(range(nb_meta_iter), desc="Meta-training"):
        support_idx = torch.randperm(n_images)[:N_SUPPORT]
        support_data = train_images[support_idx].reshape(-1, 9)

        initial_params = [p.data.clone() for p in meta_model.parameters()]

        inner_loop(meta_model, support_data, K, device=device)

        with torch.no_grad():
            for p, p0 in zip(meta_model.parameters(), initial_params):
                p.data = p0 - META_LR * (p0 - p.data)


@torch.no_grad()
def render_view(model, ray_origins, ray_directions, device, chunk_size=10):
    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ro = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(device)
        rd = ray_directions[i * W * chunk_size : (i + 1) * W * chunk_size].to(device)
        pred = render_rays(model, ro, rd)
        data.append(pred)
    return torch.cat(data).cpu().numpy().reshape(H, W, 3).clip(0, 1)


def fine_tune_and_render(model, views, label, device):
    """Fine-tune model on first 4 test views, render the remaining 18."""
    support = views[:N_SUPPORT].reshape(-1, 9)
    query = views[N_SUPPORT:]

    ft_model = copy.deepcopy(model)
    ft_optim = torch.optim.SGD(ft_model.parameters(), lr=INNER_LR)

    ft_loader = DataLoader(support, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(5):
        for batch in ft_loader:
            loss = compute_loss(ft_model, batch, device)
            ft_optim.zero_grad()
            loss.backward()
            ft_optim.step()

    losses = []
    for q_idx in range(len(query)):
        rays_o = query[q_idx, :, :3]
        rays_d = query[q_idx, :, 3:6]
        img = render_view(ft_model, rays_o, rays_d, device)
        gt = query[q_idx, :, 6:].numpy().reshape(H, W, 3)
        losses.append(np.mean((img - gt) ** 2))
        plt.imsave(f"novel_views/{label}_img_{q_idx + N_SUPPORT}.png", img)

    avg = np.mean(losses)
    print(f"  {label} avg MSE: {avg:.6f}")
    return avg


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    training_data = torch.from_numpy(np.load("training_data.pkl", allow_pickle=True))
    testing_data = torch.from_numpy(np.load("testing_data.pkl", allow_pickle=True))

    n_train = training_data.shape[0] // (H * W)
    n_test = testing_data.shape[0] // (H * W)
    train_images = training_data.reshape(n_train, H * W, 9)
    test_images = testing_data.reshape(n_test, H * W, 9)

    print(f"Training images: {n_train}, Test images: {n_test}")

    # Phase 1: Meta-learn initialization on training images
    print("\nPhase 1: Meta-learning initialization with Reptile...")
    meta_model = NerfModel(hidden_dim=256).to(device)
    reptile(meta_model, train_images, META_ITERS, INNER_K, device)

    # Phase 2: Compare random init vs meta init when fine-tuned on 4 test views
    print("\nPhase 2: Few-shot evaluation on test views...")
    print("  Both models get 5 epochs fine-tuning on the same 4 test views.")

    random_model = NerfModel(hidden_dim=256).to(device)
    mse_random = fine_tune_and_render(random_model, test_images, "random", device)
    mse_meta = fine_tune_and_render(meta_model, test_images, "meta", device)

    print(f"\n=== Comparison (fine-tuned on 4 test views, tested on 18) ===")
    print(f"Random init avg MSE:  {mse_random:.6f}")
    print(f"Meta init avg MSE:    {mse_meta:.6f}")
    print(f"Ratio (meta/random):  {mse_meta / mse_random:.3f}x")
    if mse_meta < mse_random:
        print(f"Improvement: {((1 - mse_meta / mse_random) * 100):.1f}%")
    else:
        print("No improvement (random init was better in this run)")
