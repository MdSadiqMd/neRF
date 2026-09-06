"""
PixelNeRF: Neural Radiance Fields from One or Few Images (CVPR 2021)

Key ideas:
1. Image-conditioned NeRF: CNN encoder extracts pixel-aligned features from
   source view(s), NeRF MLP is conditioned on those features
2. ResNet34 backbone giving dense 512-dim feature map per pixel
3. For each 3D query point, project onto source view, sample feature via
   bilinear interpolation, feed to MLP with positional encoding of point
   and view direction in source frame
4. Volumetric rendering same as NeRF but in source-camera space

Paper: https://arxiv.org/abs/2012.02190
"""

import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoaderg

Path("novel_views").mkdir(exist_ok=True)


class TinyPixelDataset(Dataset):
    """Single-scene PixelNeRF dataset from tiny_nerf_data.npz (lego, 100x100)."""

    def __init__(self, npz_path="tiny_nerf_data.npz", train=True, H=100, W=100):
        data = np.load(npz_path)
        images = data["images"]  # [106, H, W, 3]
        poses = data["poses"]  # [106, 4, 4]
        focal = float(data["focal"])

        n = images.shape[0]
        n_train = int(n * 0.8)
        indices = np.random.RandomState(42).permutation(n)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        idx = train_idx if train else test_idx

        self.images = images[idx].astype(np.float32)  # already 0-1
        self.poses = poses[idx].astype(np.float32)
        self.focal = focal
        self.H, self.W = H, W
        self.N = self.images.shape[0]

        s = 2.0 * focal / W
        self.camera_mats = np.tile(np.eye(4, dtype=np.float32)[None], (self.N, 1, 1))
        self.camera_mats[:, 0, 0] = s

        self.is_train = train

    def __len__(self):
        return max(self.N * 20, 500) if self.is_train else self.N

    def __getitem__(self, i):
        if self.is_train:
            src, tgt = random.sample(range(self.N), 2)
        else:
            src = 0
            tgt = i % self.N
            if tgt == src:
                tgt = (tgt + 1) % self.N

        return {
            "src_img": torch.from_numpy(self.images[src]).permute(2, 0, 1),
            "tgt_img": torch.from_numpy(self.images[tgt]).permute(2, 0, 1),
            "source_c2w": torch.from_numpy(self.poses[src]),
            "target_c2w": torch.from_numpy(self.poses[tgt]),
            "source_cam": torch.from_numpy(self.camera_mats[src]),
            "target_cam": torch.from_numpy(self.camera_mats[tgt]),
        }


def intrinsics_to_fxfycxcy(camera_mat, H, W):
    s = float(camera_mat[0, 0])
    return s * W / 2.0, s * H / 2.0, W / 2.0, H / 2.0


class ImageEncoder(nn.Module):
    """ResNet34 backbone giving a multi-scale feature map, one vector per pixel."""

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import resnet34, ResNet34_Weights

            try:
                net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                net = resnet34(weights=None)
        except Exception:
            from torchvision.models import resnet34

            try:
                net = resnet34(weights="IMAGENET1K_V1")
            except Exception:
                net = resnet34(weights=None)
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.feat_dim = 64 + 64 + 128 + 256
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        x = (x - self.mean) / self.std
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        feats = [
            F.interpolate(f, size=(H, W), mode="bilinear", align_corners=True)
            for f in (f0, f1, f2, f3)
        ]
        return torch.cat(feats, dim=1)


def positional_encoding(x, n_freqs=6):
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    xb = x[..., None] * freqs
    enc = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
    return torch.cat([x, enc.reshape(*x.shape[:-1], -1)], dim=-1)


class PixelNeRF(nn.Module):
    def __init__(self, feat_dim, n_freqs=6, hidden=128, n_blocks=3):
        super().__init__()
        self.n_freqs = n_freqs
        self.inp = nn.Linear(3 + 3 * 2 * n_freqs + 3, hidden)
        self.feat_proj = nn.ModuleList(
            nn.Linear(feat_dim, hidden) for _ in range(n_blocks)
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(n_blocks)
        )
        self.out = nn.Linear(hidden, 4)

    def forward(self, pts, dirs, feat):
        h = F.relu(
            self.inp(torch.cat([positional_encoding(pts, self.n_freqs), dirs], -1))
        )
        for proj, block in zip(self.feat_proj, self.blocks):
            h = h + proj(feat)
            h = h + block(h)
        h = self.out(F.relu(h))
        return torch.sigmoid(h[..., :3]), F.relu(h[..., 3])


def get_rays(c2w, K, H, W, device):
    fx, fy, cx, cy = K
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dirs = torch.stack(
        [(xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy, torch.ones_like(xs)], dim=-1
    )
    rays_d = dirs @ c2w[:3, :3].t()
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def render_rays(model, feat, src_w2c, src_K, rays_o, rays_d, near, far, n_samples=64):
    device = rays_o.device
    R = rays_o.shape[0]
    t = torch.linspace(near, far, n_samples, device=device).expand(R, n_samples).clone()
    mid = 0.5 * (t[:, 1:] + t[:, :-1])
    lower = torch.cat([t[:, :1], mid], -1)
    upper = torch.cat([mid, t[:, -1:]], -1)
    t = lower + (upper - lower) * torch.rand_like(t)
    pts_w = rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]

    Rsrc, tsrc = src_w2c[:3, :3], src_w2c[:3, 3]
    pts_c = pts_w @ Rsrc.t() + tsrc
    dirs_c = F.normalize(rays_d @ Rsrc.t(), dim=-1)[:, None, :].expand_as(pts_c)

    fx, fy, cx, cy = src_K
    H, W = feat.shape[-2:]
    z = pts_c[..., 2].clamp(min=1e-4)
    u = fx * pts_c[..., 0] / z + cx
    v = fy * pts_c[..., 1] / z + cy
    grid = torch.stack(
        [2 * (u - 0.5) / (W - 1) - 1, 2 * (v - 0.5) / (H - 1) - 1], dim=-1
    )
    sampled = F.grid_sample(feat, grid[None], align_corners=True, padding_mode="border")
    sampled = sampled[0].permute(1, 2, 0)

    color, sigma = model(pts_c, dirs_c, sampled)
    delta = torch.cat([t[:, 1:] - t[:, :-1], torch.full_like(t[:, :1], 1e10)], -1)
    alpha = 1 - torch.exp(-sigma * delta)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1
    )[:, :-1]
    weights = trans * alpha
    rgb = (weights[..., None] * color).sum(1)
    return rgb


@torch.no_grad()
def render_image(model, feat, src_w2c, src_K, c2w, tgt_K, H, W, near, far, chunk=4096):
    rays_o, rays_d = get_rays(c2w, tgt_K, H, W, feat.device)
    out = [
        render_rays(
            model,
            feat,
            src_w2c,
            src_K,
            rays_o[i : i + chunk],
            rays_d[i : i + chunk],
            near,
            far,
        )
        for i in range(0, rays_o.shape[0], chunk)
    ]
    return torch.cat(out, 0).reshape(H, W, 3)


@torch.no_grad()
def render_test_views(encoder, model, test_set, H, W, near, far, device):
    n = test_set.N
    src_idx = 0
    src_img = (
        torch.from_numpy(test_set.images[src_idx]).permute(2, 0, 1)[None].to(device)
    )
    feat = encoder(src_img)
    src_w2c = torch.inverse(torch.from_numpy(test_set.poses[src_idx]).to(device))
    src_K = (test_set.focal, test_set.focal, W / 2.0, H / 2.0)

    for i in range(n):
        tgt_c2w = torch.from_numpy(test_set.poses[i]).to(device)
        tgt_K = src_K
        img = render_image(model, feat, src_w2c, src_K, tgt_c2w, tgt_K, H, W, near, far)
        arr = (img.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(f"novel_views/img_{i}.png")
        if i == 0:
            src_arr = (test_set.images[src_idx].clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(src_arr).save("novel_views/img_source.png")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    H = W = 100
    near, far = 2.0, 6.0
    batch_size, n_rays = 2, 1024
    n_samples = 64
    total_steps = 6000

    train_set = TinyPixelDataset("tiny_nerf_data.npz", train=True, H=H, W=W)
    test_set = TinyPixelDataset("tiny_nerf_data.npz", train=False, H=H, W=W)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    print(f"Train views: {train_set.N}, Test views: {test_set.N}")

    encoder = ImageEncoder().to(device)
    model = PixelNeRF(encoder.feat_dim, n_freqs=6, hidden=256, n_blocks=4).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(model.parameters()), lr=5e-5
    )

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, total_steps + 1))
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        feats = encoder(batch["src_img"].to(device))
        loss = 0.0
        for b in range(batch_size):
            src_w2c = torch.inverse(batch["source_c2w"][b]).to(device)
            src_K = intrinsics_to_fxfycxcy(batch["source_cam"][b], H, W)
            tgt_K = intrinsics_to_fxfycxcy(batch["target_cam"][b], H, W)
            rays_o, rays_d = get_rays(
                batch["target_c2w"][b].to(device), tgt_K, H, W, device
            )
            idx = torch.randint(0, H * W, (n_rays,), device=device)
            rgb = render_rays(
                model,
                feats[b : b + 1],
                src_w2c,
                src_K,
                rays_o[idx],
                rays_d[idx],
                near,
                far,
                n_samples=n_samples,
            )
            tgt = batch["tgt_img"][b].permute(1, 2, 0).reshape(-1, 3).to(device)[idx]
            loss = loss + F.mse_loss(rgb, tgt)
        loss = loss / batch_size

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            print(f"Step {step}: loss={loss.item():.6f}")

    encoder.eval()
    model.eval()
    print("Rendering test views conditioned on first test image...")
    render_test_views(encoder, model, test_set, H, W, near, far, device)
    print("Done. Outputs in novel_views/")
