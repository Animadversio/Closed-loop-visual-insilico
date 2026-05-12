"""
Supplementary: Eigenmode gallery for van Hateren (and FFHQ).

Composite figure:
  - Top panel: singular value spectrum with marked modes
  - Bottom panels: 5 top eigenmodes (high variance) + 5 bottom eigenmodes (low variance)

Usage:
    python exp1_supp_eigenmode_gallery.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from PIL import Image
from os.path import join
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
FFHQ_DIR   = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256/ffhq256"
VH_DIR     = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/vanhateren_natural_stimuli"
EXPROOT    = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/AdvExampleLinearRegr/exp1"
FIGROOT    = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/notebooks"

DISCORD_TOKEN      = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = "1488321710573883502"

IMG_SIZE   = 100
N_SAMPLES  = 12000
N_PATCHES  = 12000
PATCH_SIZE = (100, 100)
N_VH_IMGS  = 500
TOP_IDXS   = [0, 1, 4, 16, 64]    # high-variance modes (0-based)
BOT_OFFSETS = [1, 4, 16, 64, 256] # low-variance modes: n_sv - offset

os.makedirs(EXPROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loaders (same as exp1)
# ---------------------------------------------------------------------------

class FFHQDataset(Dataset):
    def __init__(self, root_dir, n_images):
        self.img_paths = [join(root_dir, f"{i:05d}.png") for i in range(n_images)]
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).resize((IMG_SIZE, IMG_SIZE)).convert("L")
        return torch.from_numpy(np.array(img)).float() / 255.0


def load_ffhq(n_samples=N_SAMPLES):
    print(f"Loading {n_samples} FFHQ images …")
    ds = FFHQDataset(FFHQ_DIR, n_samples)
    dl = DataLoader(ds, batch_size=500, num_workers=8, shuffle=False)
    imgs = torch.cat([b for b in tqdm(dl, desc="FFHQ")], dim=0)
    return imgs.view(imgs.shape[0], -1).numpy()


def load_vanhateren_patches(n_patches=N_PATCHES, n_images=N_VH_IMGS):
    print(f"Loading {n_patches} van Hateren patches …")
    rng = np.random.RandomState(42)
    imshape = (1024, 1536)
    ph, pw = PATCH_SIZE
    iml_files = sorted(glob.glob(join(VH_DIR, "imk*.iml")))
    chosen = rng.choice(len(iml_files), size=min(n_images, len(iml_files)), replace=False)
    chosen_paths = [iml_files[i] for i in chosen]
    patches = np.zeros((n_patches, ph, pw), dtype=np.float32)
    idxs = rng.randint(0, n_images, size=n_patches)
    ii   = rng.randint(0, imshape[0] - ph, size=n_patches)
    jj   = rng.randint(0, imshape[1] - pw, size=n_patches)
    cache = {}
    for p in tqdm(range(n_patches), desc="VH patches"):
        k = idxs[p]
        if k not in cache:
            with open(chosen_paths[k], 'rb') as f:
                s = f.read()
            img = np.frombuffer(s, dtype='>u2').astype(np.float32).reshape(imshape)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            cache[k] = img
        patches[p] = cache[k][ii[p]:ii[p]+ph, jj[p]:jj[p]+pw]
    return patches.reshape(n_patches, -1)

# ---------------------------------------------------------------------------
# SVD
# ---------------------------------------------------------------------------

def compute_svd(X):
    X_c = X - X.mean(axis=0, keepdims=True)
    print(f"  SVD of ({X_c.shape[0]}, {X_c.shape[1]}) on CUDA …")
    Xt = torch.from_numpy(X_c).float().cuda()
    U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
    return S.cpu().numpy(), Vh.cpu().numpy()  # S: descending, Vh: (n_sv, p)

# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------

def savefig_both(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    print(f"  Saved {path_no_ext}.png / .pdf")


def send_to_discord(filepath, caption=""):
    if not DISCORD_TOKEN:
        print("  [Discord] No token — skipping")
        return
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    with open(filepath, "rb") as f:
        resp = requests.post(url, headers=headers,
                             data={"content": caption},
                             files={"files[0]": (os.path.basename(filepath), f, "image/png")})
    status = "OK" if resp.status_code in (200, 201) else f"FAIL {resp.status_code}"
    print(f"  [Discord] {status}: {os.path.basename(filepath)}")


def plot_eigenmode_gallery(dataset_name, S, Vh, img_size=IMG_SIZE,
                           top_idxs=None, bot_offsets=None):
    """
    Composite figure:
      Row 0 (tall): singular value spectrum, marked modes
      Row 1: high-variance mode images (top_idxs)
      Row 2: low-variance mode images (n_sv - bot_offsets)
    """
    if top_idxs is None:
        top_idxs = TOP_IDXS
    if bot_offsets is None:
        bot_offsets = BOT_OFFSETS

    n_sv     = len(S)
    bot_idxs = [n_sv - o for o in bot_offsets]
    n_cols   = max(len(top_idxs), len(bot_idxs))

    # ---- layout — 70% width ----
    fig = plt.figure(figsize=(9.8, 10))
    gs  = gridspec.GridSpec(3, n_cols,
                            height_ratios=[2.2, 1, 1],
                            hspace=0.50, wspace=0.12)

    # ---- spectrum panel ----
    ax_spec = fig.add_subplot(gs[0, :])
    ax_spec.semilogy(np.arange(n_sv), S, lw=1.2, color="steelblue", zorder=1)

    colors_top = plt.cm.Reds(np.linspace(0.45, 0.90, len(top_idxs)))
    colors_bot = plt.cm.Greens(np.linspace(0.55, 0.90, len(bot_idxs)))

    for i, idx in enumerate(top_idxs):
        ax_spec.scatter(idx, S[idx], color=colors_top[i], s=70, zorder=3, marker="v")
        ax_spec.annotate(f"k={idx}", (idx, S[idx]),
                         textcoords="offset points", xytext=(3, 5), fontsize=7,
                         color=colors_top[i])

    for i, (idx, off) in enumerate(zip(bot_idxs, bot_offsets)):
        ax_spec.scatter(idx, S[idx], color=colors_bot[i], s=70, zorder=3, marker="^")
        ax_spec.annotate(f"k=-{off}", (idx, S[idx]),
                         textcoords="offset points", xytext=(-32, 5), fontsize=7,
                         color=colors_bot[i])

    ax_spec.set_xlabel("Singular value index  (high → low variance)", fontsize=10)
    ax_spec.set_ylabel("Singular value", fontsize=10)
    ax_spec.set_title(f"{dataset_name}: Singular value spectrum", fontsize=12)

    # ---- mode images ----
    def _plot_mode(ax, idx, color_border, label):
        mode = Vh[idx].reshape(img_size, img_size)
        vmax = np.abs(mode).max()
        ax.imshow(mode, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=8, pad=2)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color_border)
            spine.set_linewidth(2.5)

    # Row 1: high-variance modes
    for i, idx in enumerate(top_idxs):
        ax = fig.add_subplot(gs[1, i])
        _plot_mode(ax, idx, colors_top[i], f"k={idx}\nσ={S[idx]:.1f}")

    # Row 2: low-variance modes
    for i, (idx, off) in enumerate(zip(bot_idxs, bot_offsets)):
        ax = fig.add_subplot(gs[2, i])
        _plot_mode(ax, idx, colors_bot[i], f"k=-{off}\nσ={S[idx]:.3f}")

    # Row labels
    fig.text(0.005, 0.38, "High-var\nmodes", va="center", ha="left",
             fontsize=8, rotation=90, color="darkred")
    fig.text(0.005, 0.13, "Low-var\nmodes", va="center", ha="left",
             fontsize=8, rotation=90, color="darkgreen")

    fig.suptitle(f"{dataset_name}: Eigenmode gallery",
                 fontsize=13, y=0.98)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    datasets = {
        "vanHateren": load_vanhateren_patches,
        "FFHQ":       lambda: load_ffhq(),
    }

    for name, loader in datasets.items():
        X = loader()
        S, Vh = compute_svd(X)

        fig  = plot_eigenmode_gallery(name, S, Vh)
        path = join(EXPROOT, f"{name}_eigenmode_gallery")
        savefig_both(fig, path)
        plt.close(fig)
        send_to_discord(path + ".png",
                        f"*{name}* — eigenmode gallery (k=0,1,4,16,64 + last mode)")

        # copy summary to repo
        import shutil
        repo_path = join(FIGROOT, f"exp1_supp_{name}_eigenmode_gallery")
        shutil.copy2(path + ".png", repo_path + ".png")
        shutil.copy2(path + ".pdf", repo_path + ".pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
