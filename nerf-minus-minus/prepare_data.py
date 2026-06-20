"""
Download and prepare Tiny NeRF data for NeRF--.

This script downloads the tiny_nerf dataset and extracts ONLY the images,
saving them as training and testing numpy arrays. We ignore the camera poses
because NeRF-- is designed to learn them jointly with the NeRF representation.
"""

import urllib.request
import numpy as np
from pathlib import Path


def download_tiny_nerf_data():
    """Download the tiny_nerf dataset from the original NeRF authors."""
    url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    output_path = Path("tiny_nerf_data.npz")

    if output_path.exists():
        print(f"Data already exists at {output_path}")
        return output_path

    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    return output_path


def main():
    npz_path = download_tiny_nerf_data()

    print("Loading data...")
    data = np.load(npz_path)

    images = data["images"]  # [N, H, W, 3]

    print(f"Images shape: {images.shape}")

    n_images = images.shape[0]
    n_train = int(n_images * 0.8)

    # Use a fixed random seed for reproducibility
    indices = np.random.RandomState(42).permutation(n_images)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    print(f"Training images: {len(train_indices)}")
    print(f"Testing images: {len(test_indices)}")

    training_images = images[train_indices]
    testing_images = images[test_indices]

    print("Saving training_images.npy...")
    np.save("training_images.npy", training_images)

    print("Saving testing_images.npy...")
    np.save("testing_images.npy", testing_images)

    print("Done! Files created:")
    print("  - training_images.npy")
    print("  - testing_images.npy")
    print("\nYou can now run: just train")


if __name__ == "__main__":
    main()
