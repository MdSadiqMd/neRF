import urllib.request
import numpy as np
from pathlib import Path


def download_tiny_nerf_data():
    url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    output_path = Path("tiny_nerf_data.npz")

    if output_path.exists():
        print(f"Data already exists at {output_path}")
        return output_path

    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    return output_path


def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )

    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1
    )

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def prepare_dataset(images, poses, focal):
    N, H, W, _ = images.shape
    all_rays_o = []
    all_rays_d = []
    all_rgbs = []

    for i in range(N):
        rays_o, rays_d = get_rays(H, W, focal, poses[i])
        rgb = images[i].reshape(-1, 3)

        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgbs.append(rgb)

    rays_o = np.concatenate(all_rays_o, axis=0)
    rays_d = np.concatenate(all_rays_d, axis=0)
    rgbs = np.concatenate(all_rgbs, axis=0)

    dataset = np.concatenate([rays_o, rays_d, rgbs], axis=1).astype(np.float32)
    return dataset


def main():
    npz_path = download_tiny_nerf_data()

    print("Loading data...")
    data = np.load(npz_path)

    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])

    print(f"Images shape: {images.shape}")
    print(f"Poses shape: {poses.shape}")
    print(f"Focal length: {focal}")

    n_images = images.shape[0]
    n_train = int(n_images * 0.8)

    indices = np.random.RandomState(42).permutation(n_images)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Training images: {len(train_indices)}")
    print(f"Testing images: {len(test_indices)}")

    print("Preparing training dataset...")
    training_data = prepare_dataset(images[train_indices], poses[train_indices], focal)

    print("Preparing testing dataset...")
    testing_data = prepare_dataset(images[test_indices], poses[test_indices], focal)

    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape: {testing_data.shape}")

    print("Saving training_data.pkl...")
    training_data.dump("training_data.pkl")

    print("Saving testing_data.pkl...")
    testing_data.dump("testing_data.pkl")

    print("Done!")


if __name__ == "__main__":
    main()
