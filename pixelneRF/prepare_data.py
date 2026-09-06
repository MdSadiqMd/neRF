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
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
    print(f"Downloaded to {output_path}")
    return output_path


def main():
    npz_path = download_tiny_nerf_data()
    print("Loading data...")
    data = np.load(npz_path)
    images, poses, focal = data["images"], data["poses"], float(data["focal"])
    print(f"Images: {images.shape}, Poses: {poses.shape}, Focal: {focal}")
    print(
        "Done! Ready for PixelNeRF training (single-scene lego, 84 train / 22 test views)."
    )


if __name__ == "__main__":
    main()
