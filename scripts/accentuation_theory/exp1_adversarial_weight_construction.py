"""
Experiment 1: Adversarial Weight Construction

Demonstrates that a weight vector can look visually very different from
the ground truth (circle mask) yet make nearly identical predictions on
natural images — because the perturbation lives in the low-variance
(near-null) eigenspace of the image covariance matrix.

Two datasets: FFHQ faces (100x100) and van Hateren natural image patches.

Saves figures to the Github repo notebooks folder (summary) and to
DL_Projects/AdvExampleLinearRegr/exp1/ (full outputs).

Usage:
    python exp1_adversarial_weight_construction.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import torch
from PIL import Image
from os.path import join
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FFHQ_DIR   = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256/ffhq256"
VH_DIR     = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/vanhateren_natural_stimuli"
EXPROOT    = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/AdvExampleLinearRegr/exp1"
FIGROOT    = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/notebooks"

DISCORD_TOKEN      = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = "1488321710573883502"

IMG_SIZE    = 100
N_SAMPLES   = 12000  # images used to estimate covariance / test predictions
N_PATCHES   = 12000  # van Hateren patches
PATCH_SIZE  = (100, 100)
N_VH_IMAGES = 500    # how many VH images to sample patches from
ALPHA_VALUES = [10.0, 100.0, 300.0, 1000.0, 10000.0]  # strength of low-eigenspace perturbation
K_LOW       = 100    # number of low-variance eigenvectors to use for perturbation

os.makedirs(EXPROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class FFHQDataset(Dataset):
    def __init__(self, root_dir, n_images=70000):
        self.img_paths = [join(root_dir, f"{i:05d}.png") for i in range(n_images)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).resize((IMG_SIZE, IMG_SIZE)).convert("L")
        return torch.from_numpy(np.array(img)).float() / 255.0


def load_ffhq(n_samples=N_SAMPLES):
    print(f"Loading {n_samples} FFHQ images …")
    dataset    = FFHQDataset(FFHQ_DIR, n_images=n_samples)
    dataloader = DataLoader(dataset, batch_size=500, num_workers=8, shuffle=False)
    batches = [b for b in tqdm(dataloader, desc="FFHQ")]
    imgs = torch.cat(batches, dim=0)
    return imgs.view(imgs.shape[0], -1).numpy()  # (N, 10000) float32


def load_vanhateren_patches(n_patches=N_PATCHES, patch_size=PATCH_SIZE, n_images=N_VH_IMAGES):
    """Sample random patches from van Hateren natural images (reads .iml files directly)."""
    print(f"Loading {n_patches} van Hateren patches …")
    rng = np.random.RandomState(42)
    imshape = (1024, 1536)
    ph, pw = patch_size

    # List available .iml files directly in VH_DIR
    import glob
    iml_files = sorted(glob.glob(join(VH_DIR, "imk*.iml")))
    assert len(iml_files) > 0, f"No .iml files found in {VH_DIR}"

    chosen = rng.choice(len(iml_files), size=min(n_images, len(iml_files)), replace=False)
    chosen_paths = [iml_files[i] for i in chosen]

    patches = np.zeros((n_patches, ph, pw), dtype=np.float32)
    idxs = rng.randint(0, n_images, size=n_patches)
    ii   = rng.randint(0, imshape[0] - ph, size=n_patches)
    jj   = rng.randint(0, imshape[1] - pw, size=n_patches)

    # Cache loaded images
    imgs_cache = {}
    for p_idx in tqdm(range(n_patches), desc="VH patches"):
        img_idx = idxs[p_idx]
        if img_idx not in imgs_cache:
            path = chosen_paths[img_idx]
            with open(path, 'rb') as f:
                s = f.read()
            img = np.frombuffer(s, dtype='>u2').astype(np.float32)  # big-endian uint16
            img = img.reshape(imshape)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            imgs_cache[img_idx] = img
        img = imgs_cache[img_idx]
        patches[p_idx] = img[ii[p_idx]:ii[p_idx]+ph, jj[p_idx]:jj[p_idx]+pw]

    return patches.reshape(n_patches, -1)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def create_circle_mask(img_size=IMG_SIZE, radius=0.3):
    x = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, x)
    mask = ((X**2 + Y**2) < radius**2).astype(np.float32)
    return mask


def compute_svd(X, k_low=K_LOW):
    """
    Compute truncated SVD of centered X.
    Returns:
      U, S, Vt  — full thin SVD  (expensive but necessary for low-eigenvectors)
      V_low     — (p, k_low) matrix of the k_low *lowest* variance right singular vectors
      V_high    — (p, n-k_low) matrix of the remaining (high-variance) right singular vectors
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0, keepdims=True)
    print(f"  Computing SVD of ({n}, {p}) matrix on CUDA …")
    Xt = torch.from_numpy(X_centered).float().cuda()
    U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
    S  = S.cpu().numpy()
    Vh = Vh.cpu().numpy()  # (min(n,p), p)
    V  = Vh.T        # (p, min(n,p))
    # Singular values are in descending order → low-variance = last k_low columns
    V_high = V[:, :-k_low]   # (p, n-k_low)
    V_low  = V[:, -k_low:]   # (p, k_low)
    return S, V_high, V_low, X_centered


def build_adversarial_weight(w_true, V_low, alpha):
    """
    w_adv = w_true + alpha * perturbation_in_low_eigenspace
    The perturbation direction is the unit-norm projection of w_true
    onto V_low (or if that's zero, just the first low-eigenvector).
    """
    # Project w_true onto low eigenspace
    coords = V_low.T @ w_true                      # (k_low,)
    norm   = np.linalg.norm(coords)
    if norm > 1e-8:
        direction = V_low @ (coords / norm)         # unit vector in low eigenspace
    else:
        direction = V_low[:, 0]                     # fallback
    w_adv = w_true + alpha * direction
    return w_adv, direction


def prediction_correlation(y1, y2):
    """Pearson r between two prediction vectors."""
    r = np.corrcoef(y1, y2)[0, 1]
    return r

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def savefig_both(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    print(f"  Saved {path_no_ext}.png / .pdf")


def send_to_discord(filepath, caption=""):
    """Send a file to the Discord channel via REST API."""
    if not DISCORD_TOKEN:
        print("  [Discord] No token — skipping upload")
        return
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    with open(filepath, "rb") as f:
        filename = os.path.basename(filepath)
        resp = requests.post(
            url,
            headers=headers,
            data={"content": caption},
            files={"files[0]": (filename, f, "image/png")},
        )
    if resp.status_code in (200, 201):
        print(f"  [Discord] Sent {filename}")
    else:
        print(f"  [Discord] Failed: {resp.status_code} {resp.text[:200]}")


def plot_experiment(dataset_name, X, S, V_high, V_low, w_true, alpha_values, img_size=IMG_SIZE):
    """Full figure suite for one dataset."""
    p = X.shape[1]

    # ---- Figure 1: Eigenspectrum ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(S, lw=1.5)
    ax.axvspan(len(S) - K_LOW, len(S), alpha=0.15, color="red", label=f"Low eigenspace (k={K_LOW})")
    ax.set_xlabel("Singular value index (high → low variance)")
    ax.set_ylabel("Singular value")
    ax.set_title(f"{dataset_name}: Image singular value spectrum")
    ax.legend()
    plt.tight_layout()
    fname = join(EXPROOT, f"{dataset_name}_eigenspectrum")
    savefig_both(fig, fname)
    send_to_discord(fname + ".png", f"*{dataset_name}* — eigenspectrum")
    plt.close(fig)

    # ---- Figure 2: Weight visualisation + prediction scatter (grid over alpha) ----
    n_alpha = len(alpha_values)
    fig, axs = plt.subplots(3, n_alpha + 1, figsize=(3.5 * (n_alpha + 1), 9))

    # Column 0: ground truth
    axs[0, 0].imshow(w_true.reshape(img_size, img_size), cmap="coolwarm")
    axs[0, 0].set_title("w_true\n(circle mask)", fontsize=9)
    axs[0, 0].axis("off")
    axs[1, 0].axis("off")
    axs[2, 0].axis("off")

    for col, alpha in enumerate(alpha_values, start=1):
        w_adv, direction = build_adversarial_weight(w_true, V_low, alpha)

        y_true = X @ w_true
        y_adv  = X @ w_adv
        r      = prediction_correlation(y_true, y_adv)
        rmse   = np.sqrt(np.mean((y_true - y_adv)**2))
        rel    = rmse / y_true.std()
        r2     = 1 - rel**2

        perturbation = w_adv - w_true

        # Row 0: w_adv
        vmax = max(np.abs(w_adv).max(), np.abs(w_true).max())
        axs[0, col].imshow(w_adv.reshape(img_size, img_size), cmap="coolwarm",
                           vmin=-vmax, vmax=vmax)
        axs[0, col].set_title(f"w_adv  α={alpha}\n||Δw||={alpha:.1f}", fontsize=9)
        axs[0, col].axis("off")

        # Row 1: perturbation
        axs[1, col].imshow(perturbation.reshape(img_size, img_size), cmap="RdBu_r")
        axs[1, col].set_title(f"Perturbation\n(low eigenspace)", fontsize=9)
        axs[1, col].axis("off")

        # Row 2: prediction scatter
        axs[2, col].scatter(y_true, y_adv, s=2, alpha=0.4, rasterized=True)
        lims = [min(y_true.min(), y_adv.min()), max(y_true.max(), y_adv.max())]
        axs[2, col].plot(lims, lims, "r--", lw=1)
        axs[2, col].set_xlabel("y_true", fontsize=8)
        axs[2, col].set_ylabel("y_adv", fontsize=8)
        axs[2, col].set_title(f"r={r:.4f}  R²={r2:.4f}\nrel_RMSE={rel:.2e}", fontsize=9)

    fig.suptitle(f"{dataset_name}: Adversarial weight vs ground truth\n"
                 f"(perturbation = low eigenspace of image covariance, k={K_LOW})",
                 fontsize=11)
    plt.tight_layout()
    fname = join(EXPROOT, f"{dataset_name}_adversarial_weights")
    savefig_both(fig, fname)
    send_to_discord(fname + ".png", f"*{dataset_name}* — adversarial weights vs ground truth")
    plt.close(fig)

    # ---- Figure 3: Energy in eigenspace bands ----
    # Decompose w_true and w_adv (at max alpha) in the SVD basis
    alpha_max = max(alpha_values)
    w_adv_max, _ = build_adversarial_weight(w_true, V_low, alpha_max)
    V_all = np.hstack([V_high, V_low])  # (p, n_sv)  ordered high→low variance
    coords_true = V_all.T @ w_true
    coords_adv  = V_all.T @ w_adv_max
    n_sv = V_all.shape[1]

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(n_sv)
    ax.semilogy(idx, coords_true**2 + 1e-12, lw=1, label="w_true", alpha=0.8)
    ax.semilogy(idx, coords_adv**2  + 1e-12, lw=1, label=f"w_adv (α={alpha_max})", alpha=0.8)
    ax.axvspan(n_sv - K_LOW, n_sv, alpha=0.15, color="red", label="Low eigenspace")
    ax.set_xlabel("Eigenspace index (high → low variance)")
    ax.set_ylabel("Coefficient energy (squared projection)")
    ax.set_title(f"{dataset_name}: Weight energy in each eigenspace component")
    ax.legend()
    plt.tight_layout()
    fname = join(EXPROOT, f"{dataset_name}_weight_energy")
    savefig_both(fig, fname)
    send_to_discord(fname + ".png", f"*{dataset_name}* — weight energy in eigenspace")
    plt.close(fig)

    # ---- Summary stats ----
    print(f"\n  === {dataset_name} summary ===")
    print(f"  Singular value range: {S.min():.3f} … {S.max():.3f}  (ratio {S.max()/S.min():.1e})")
    print(f"  Condition number (low-k cutoff): {S[-K_LOW]:.4f} / {S[0]:.4f} = {S[-K_LOW]/S[0]:.2e}")
    for alpha in alpha_values:
        w_adv, _ = build_adversarial_weight(w_true, V_low, alpha)
        y_true = X @ w_true
        y_adv  = X @ w_adv
        r      = prediction_correlation(y_true, y_adv)
        rel    = np.sqrt(np.mean((y_true - y_adv)**2)) / y_true.std()
        print(f"  α={alpha:6.1f}: ||Δw||={alpha:.1f}  r={r:.6f}  rel_RMSE={rel:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    w_true_2d = create_circle_mask(IMG_SIZE)
    w_true    = w_true_2d.flatten()

    datasets = {}

    # --- FFHQ ---
    X_ffhq = load_ffhq(n_samples=N_SAMPLES)
    S_ffhq, Vh_ffhq, Vl_ffhq, _ = compute_svd(X_ffhq, k_low=K_LOW)
    datasets["FFHQ"] = (X_ffhq, S_ffhq, Vh_ffhq, Vl_ffhq)

    # --- van Hateren ---
    X_vh = load_vanhateren_patches(n_patches=N_PATCHES)
    S_vh, Vh_vh, Vl_vh, _ = compute_svd(X_vh, k_low=K_LOW)
    datasets["vanHateren"] = (X_vh, S_vh, Vh_vh, Vl_vh)

    # --- Figures ---
    for name, (X, S, V_high, V_low) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        plot_experiment(name, X, S, V_high, V_low, w_true, ALPHA_VALUES)

    # --- Also copy summary figure to Github repo ---
    import shutil
    for name in datasets:
        src = join(EXPROOT, f"{name}_adversarial_weights.png")
        dst = join(FIGROOT, f"exp1_{name}_adversarial_weights.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            shutil.copy2(src.replace(".png", ".pdf"), dst.replace(".png", ".pdf"))
    print("\nDone.")


if __name__ == "__main__":
    main()
