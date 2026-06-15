"""
Toy model: linear regression with natural images (FFHQ).

Tests whether OLS / Ridge / Lasso can recover a ground-truth
sparse filter (circle mask) from face images under varying
sample sizes and noise levels.

This is the "accentuation" theory toy model — with correlated
natural image statistics, OLS accentuates spurious correlations
while Ridge/Lasso regularize toward the true filter.

Converted from: notebooks/20250528_toy_model_adv_generation.ipynb
"""

import os
import pickle as pkl
from os.path import join
from time import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FFHQ_DIR  = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256/ffhq256"
EXPROOT   = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/AdvExampleLinearRegr/circ_mask_weights"
IMG_SIZE  = 100
RADIUS    = 0.3
N_SAMPLES_SWEEP = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
NOISE_LEVELS    = [0.000, 0.01, 0.1, 1.0, 10.0]
TEST_SIZE        = 1000
RUN_LASSO        = True

os.makedirs(EXPROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FFHQDataset(Dataset):
    def __init__(self, root_dir, n_images=70000):
        self.img_paths = [join(root_dir, f"{i:05d}.png") for i in range(n_images)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).resize((IMG_SIZE, IMG_SIZE)).convert("L")
        return torch.from_numpy(np.array(img)).float() / 255.0


def load_ffhq(root_dir=FFHQ_DIR, n_images=70000, batch_size=500, num_workers=8):
    """Load FFHQ images into a (N, H*W) float tensor."""
    dataset    = FFHQDataset(root_dir, n_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    batches = []
    for batch in tqdm(dataloader, desc="Loading FFHQ"):
        batches.append(batch)
    imgs = torch.cat(batches, dim=0)          # (N, H, W)
    return imgs.view(imgs.shape[0], -1)       # (N, H*W)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_circle_mask(img_size=IMG_SIZE, radius=RADIUS):
    """Binary circle mask in image center, shape (img_size, img_size)."""
    x = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, x)
    return ((X**2 + Y**2) < radius**2).astype(float)


def run_regressors(train_X, train_y, test_X, test_y, img_size=IMG_SIZE, run_lasso=RUN_LASSO):
    """Fit OLS, Ridge, optionally Lasso; return coef maps and R² scores."""
    regressors = {
        "OLS":   LinearRegression(),
        "Ridge": RidgeCV(alphas=list(np.logspace(-4, 5, 10))),
    }
    if run_lasso:
        regressors["Lasso"] = LassoCV(alphas=list(np.logspace(-4, 5, 10)), n_jobs=-1, max_iter=5000)

    results = {}
    for name, model in regressors.items():
        t0 = time()
        model.fit(train_X, train_y)
        elapsed = time() - t0
        r2_train = model.score(train_X, train_y)
        r2_test  = model.score(test_X,  test_y)
        print(f"  {name:6s}  R²_train={r2_train:.4f}  R²_test={r2_test:.4f}  ({elapsed:.1f}s)")
        results[f"coef_{name}"]    = model.coef_.reshape(img_size, img_size)
        results[f"r2_train_{name}"] = r2_train
        results[f"r2_test_{name}"]  = r2_test
        results[f"time_{name}"]    = elapsed
    return results


def visualize_results(results, title="", img_size=IMG_SIZE):
    """Plot ground truth vs recovered coefficient maps."""
    names = [k.replace("coef_", "") for k in results if k.startswith("coef_")]
    n_cols = 1 + len(names)
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.25))

    axs[0].imshow(results["ground_truth"], cmap="coolwarm")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    for ax, name in zip(axs[1:], names):
        coef = results.get(f"coef_{name}")
        if coef is None:
            ax.text(0.5, 0.5, f"{name}\nnot run", ha="center", va="center")
        else:
            ax.imshow(coef, cmap="coolwarm")
            r2_tr = results.get(f"r2_train_{name}", float("nan"))
            r2_te = results.get(f"r2_test_{name}",  float("nan"))
            ax.set_title(f"{name}\n1-R²_tr={1-r2_tr:.2e}\n1-R²_te={1-r2_te:.2e}")
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    # 1. Load images
    print("Loading FFHQ images …")
    img_mat = load_ffhq()          # (N, 10000) float32 on CPU
    N_total = img_mat.shape[0]
    print(f"Loaded {N_total} images, shape {img_mat.shape}")

    # 2. Ground truth filter
    circle_mask   = create_circle_mask()
    gt_coef       = torch.from_numpy(circle_mask.flatten()).float()
    y_all         = img_mat @ gt_coef               # (N,) dot product

    # 3. Fixed test split (last TEST_SIZE images)
    test_X  = img_mat[-TEST_SIZE:].numpy()
    test_y  = y_all[-TEST_SIZE:].numpy()

    # 4. Sweep
    for n_samples in N_SAMPLES_SWEEP:
        if n_samples + TEST_SIZE > N_total:
            print(f"Skipping n_samples={n_samples}: not enough images")
            continue

        train_X_t = img_mat[:n_samples]
        train_y_t = y_all[:n_samples]
        train_X   = train_X_t.numpy()

        for noise_level in NOISE_LEVELS:
            out_path = join(EXPROOT, f"regression_results_n_samp{n_samples}_noise{noise_level}.pkl")
            if os.path.exists(out_path):
                print(f"[skip] {out_path} already exists")
                continue

            print(f"\n=== n_samples={n_samples}, noise={noise_level} ===")
            NSR = noise_level**2 / float(train_y_t.std()**2)
            train_y_noisy = (train_y_t + torch.randn(n_samples) * noise_level).numpy()

            results = run_regressors(train_X, train_y_noisy, test_X, test_y)
            results["ground_truth"] = circle_mask
            results["n_samples"]    = n_samples
            results["noise_level"]  = noise_level
            results["NSR"]          = NSR

            pkl.dump(results, open(out_path, "wb"))
            print(f"  Saved {out_path}")

            title = f"n_samples={n_samples}, noise={noise_level}, NSR={NSR:.2e}"
            fig = visualize_results(results, title=title)
            fig_path = join(EXPROOT, f"regression_results_n_samp{n_samples}_noise{noise_level}.png")
            fig.savefig(fig_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
