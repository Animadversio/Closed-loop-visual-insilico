"""
Experiment 2: Accentuation

Three weights:
  w*  = circle mask (ground truth)
  w1  = w* + alpha=1  perturbation in MID-LOW eigenspace  (indices -500..-100)
  w2  = w* + alpha=1000 perturbation in VERY-BOTTOM eigenspace (indices -100..-1)

Part A: All three agree on natural images (van Hateren)
Part B: Gradient accentuation — push image toward target response level using each model
Part C: Cross-evaluate — accentuated images from w1/w2 evaluated by all three models
"""

import os, glob, numpy as np, matplotlib
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
import sys
sys.path.append("/n/home12/binxuwang/Github/circuit_toolkit")
from circuit_toolkit.plot_utils import saveallforms

# ---------------------------------------------------------------------------
FFHQ_DIR   = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/ffhq256/ffhq256"
VH_DIR     = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/vanhateren_natural_stimuli"
EXPROOT    = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/AdvExampleLinearRegr/exp2"
FIGROOT    = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/notebooks"
TOKEN      = os.environ.get("DISCORD_BOT_TOKEN", "")
CHAN       = "1488321710573883502"
IMG_SIZE   = 100
N_SAMPLES  = 12000
N_PATCHES  = 12000
N_VH_IMGS  = 500
N_TARGETS  = 13     # target response levels
N_STARTS   = 20     # starting images for accentuation
ALPHA1     = 1.0
ALPHA2     = 100.0
K_MID_LO   = 500    # w1 perturbation: last 500..last 100
K_MID_HI   = 100
K_BOT      = 100    # w2 perturbation: last 100
os.makedirs(EXPROOT, exist_ok=True)

# ---------------------------------------------------------------------------
def send(path, caption=""):
    if not TOKEN: return
    url = f"https://discord.com/api/v10/channels/{CHAN}/messages"
    with open(path, "rb") as f:
        r = requests.post(url, headers={"Authorization": f"Bot {TOKEN}"},
                          data={"content": caption},
                          files={"files[0]": (os.path.basename(path), f, "image/png")})
    print("  [Discord]", "OK" if r.status_code in (200,201) else f"FAIL {r.status_code}")

def savefig(fig, path):
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  Saved {path}.png/.pdf")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_vanhateren(n_patches=N_PATCHES, n_images=N_VH_IMGS):
    print(f"Loading {n_patches} van Hateren patches …")
    rng = np.random.RandomState(42)
    ph, pw = IMG_SIZE, IMG_SIZE
    iml_files = sorted(glob.glob(join(VH_DIR, "imk*.iml")))
    chosen_paths = [iml_files[i] for i in rng.choice(len(iml_files), size=min(n_images, len(iml_files)), replace=False)]
    patches = np.zeros((n_patches, ph * pw), dtype=np.float32)
    idxs = rng.randint(0, len(chosen_paths), size=n_patches)
    ii   = rng.randint(0, 1024 - ph, size=n_patches)
    jj   = rng.randint(0, 1536 - pw, size=n_patches)
    cache = {}
    for p in tqdm(range(n_patches), desc="VH"):
        k = idxs[p]
        if k not in cache:
            with open(chosen_paths[k], 'rb') as f:
                img = np.frombuffer(f.read(), dtype='>u2').astype(np.float32).reshape(1024, 1536)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            cache[k] = img
        patch = cache[k][ii[p]:ii[p]+ph, jj[p]:jj[p]+pw]
        patches[p] = patch.flatten()
    return patches - 0.15

# ---------------------------------------------------------------------------
# SVD + weight construction
# ---------------------------------------------------------------------------
def compute_svd(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    print(f"  SVD of {Xc.shape} on CUDA …")
    Xt = torch.from_numpy(Xc).float().cuda()
    U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
    return S.cpu().numpy(), Vh.cpu().numpy()  # S descending, Vh (n_sv, p)

def create_circle_mask(img_size=IMG_SIZE, radius=0.3):
    x = np.linspace(-1, 1, img_size)
    X, Y = np.meshgrid(x, x)
    return ((X**2 + Y**2) < radius**2).astype(np.float32).flatten()

def build_weight(w_star, Vh, lo_idx, hi_idx, alpha):
    """Perturb w_star by alpha * unit direction in eigenspace [lo_idx:hi_idx] (from bottom)."""
    n_sv = Vh.shape[0]
    V_sub = Vh[n_sv - lo_idx : n_sv - hi_idx].T  # (p, k)  low-variance subspace slice
    coords = V_sub.T @ w_star
    norm = np.linalg.norm(coords)
    direction = V_sub @ (coords / norm) if norm > 1e-8 else V_sub[:, 0]
    return w_star + alpha * direction

# ---------------------------------------------------------------------------
# Accentuation (closed-form for linear model)
# ---------------------------------------------------------------------------
def accentuate(I0, w, target):
    """Closed-form: I = I0 + lambda * w  s.t. w^T I = target."""
    f0  = w @ I0
    lam = (target - f0) / (w @ w)
    return I0 + lam * w

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_weights(w_star, w1, w2):
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, w, title in zip(axs,
                             [w_star, w1, w2],
                             ["w* (circle mask)",
                              f"w₁ (w* + α={ALPHA1}, mid-low eigenspace)",
                              f"w₂ (w* + α={ALPHA2}, bottom eigenspace)"]):
        vmax = np.abs(w).max()
        ax.imshow(w.reshape(IMG_SIZE, IMG_SIZE), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig

def fig_natural_scatter(X, w_star, w1, w2):
    fs = X @ w_star
    f1 = X @ w1
    f2 = X @ w2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # panels 1 & 2: f* on y, f1/f2 on x; panel 3: f1 vs f2
    pairs = [(f1, fs, "f₁", "f*"), (f2, fs, "f₂", "f*"), (f1, f2, "f₁", "f₂")]
    for ax, (fx, fy, xlbl, ylbl) in zip(axs, pairs):
        r2 = 1 - np.var(fx - fy) / (np.var(fy) + 1e-12)
        ax.scatter(fx, fy, s=2, alpha=0.2, rasterized=True)
        lims = [min(fx.min(), fy.min()), max(fx.max(), fy.max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel(xlbl, fontsize=9)
        ax.set_ylabel(ylbl, fontsize=9)
        ax.set_title(f"{ylbl} vs {xlbl}\nR²={r2:.5f}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Natural image predictions — van Hateren", fontsize=12)
    plt.tight_layout()
    return fig, fs

def hicontr(arr2d):
    """Percentile contrast stretch for display only (1%–99%)."""
    lo, hi = np.percentile(arr2d, 1), np.percentile(arr2d, 99)
    return np.clip((arr2d - lo) / (hi - lo + 1e-8), 0, 1)

def fig_accentuated_images(X_starts, w_star, w1, w2, targets, n_show=None, high_contrast=False):
    """Grid: rows=weights (w1, w2), cols=targets. Plus original images."""
    n_tgt   = len(targets)
    weights  = [w1, w2]
    wlabels  = [f"w₁ (α={ALPHA1})", f"w₂ (α={ALPHA2})"]
    if n_show is None:
        n_show = len(X_starts)
    vis = hicontr if high_contrast else (lambda x: np.clip(x + 0.15, 0, 1))

    figs = []
    for si in range(min(n_show, len(X_starts))):
        I0 = X_starts[si]
        fig, axs = plt.subplots(2, n_tgt + 1, figsize=(1.6 * (n_tgt + 1), 4.2))
        for row, (w, wlbl) in enumerate(zip(weights, wlabels)):
            ax0 = axs[row, 0]
            ax0.imshow(vis(I0.reshape(IMG_SIZE, IMG_SIZE)), cmap="gray", vmin=0, vmax=1)
            ax0.set_title(f"I₀\nf*={w_star@I0:.3f}", fontsize=7)
            ax0.axis("off")
            ax0.set_ylabel(wlbl, fontsize=8)
            for col, t in enumerate(targets, start=1):
                I_acc = accentuate(I0, w, t)
                axs[row, col].imshow(vis(I_acc.reshape(IMG_SIZE, IMG_SIZE)), cmap="gray", vmin=0, vmax=1)
                fs_t = w_star @ I_acc
                axs[row, col].set_title(f"t={t:.2f}\nf*={fs_t:.3f}", fontsize=7)
                axs[row, col].axis("off")
        fig.suptitle(f"Accentuated images — starting image #{si+1}", fontsize=11)
        plt.tight_layout()
        figs.append(fig)
    return figs

def fig_cross_eval(X_starts, w_star, w1, w2, targets):
    """
    For images accentuated by w1 and w2 across all starting images and targets:
    Plot f_k(I_acc) vs f*(I_acc).
    """
    fs_on_w1, f1_on_w1, f2_on_w1 = [], [], []
    fs_on_w2, f1_on_w2, f2_on_w2 = [], [], []

    for I0 in X_starts:
        for t in targets:
            I1 = accentuate(I0, w1, t)
            I2 = accentuate(I0, w2, t)
            fs_on_w1.append(w_star @ I1);  f1_on_w1.append(w1 @ I1);  f2_on_w1.append(w2 @ I1)
            fs_on_w2.append(w_star @ I2);  f1_on_w2.append(w1 @ I2);  f2_on_w2.append(w2 @ I2)

    fs_on_w1 = np.array(fs_on_w1); f1_on_w1 = np.array(f1_on_w1); f2_on_w1 = np.array(f2_on_w1)
    fs_on_w2 = np.array(fs_on_w2); f1_on_w2 = np.array(f1_on_w2); f2_on_w2 = np.array(f2_on_w2)

    def clean_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    # Row 0: images accentuated by w1
    for ax, (fy, lbl) in zip(axs[0], [(f1_on_w1, "f₁"), (f2_on_w1, "f₂"), (fs_on_w1, "f*")]):
        r2 = 1 - np.var(fy - fs_on_w1) / (np.var(fs_on_w1) + 1e-12) if lbl != "f*" else 1.0
        sc = ax.scatter(np.tile(targets, len(X_starts)), fy,
                        c=fs_on_w1, cmap="RdBu_r", s=15, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="f*(I_acc)")
        ax.set_xlabel("Target t  (w₁ accentuation)", fontsize=9)
        ax.set_ylabel(f"{lbl}(I_acc)", fontsize=9)
        title = f"w₁-accentuated: {lbl} vs target" if lbl != "f*" else "w₁-accentuated: f* vs target"
        if lbl != "f*": title += f"\nR²(vs f*)={r2:.4f}"
        ax.set_title(title, fontsize=9); clean_ax(ax)
    # Row 1: images accentuated by w2
    for ax, (fy, lbl) in zip(axs[1], [(f1_on_w2, "f₁"), (f2_on_w2, "f₂"), (fs_on_w2, "f*")]):
        r2 = 1 - np.var(fy - fs_on_w2) / (np.var(fs_on_w2) + 1e-12) if lbl != "f*" else 1.0
        sc = ax.scatter(np.tile(targets, len(X_starts)), fy,
                        c=fs_on_w2, cmap="RdBu_r", s=15, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="f*(I_acc)")
        ax.set_xlabel("Target t  (w₂ accentuation)", fontsize=9)
        ax.set_ylabel(f"{lbl}(I_acc)", fontsize=9)
        title = f"w₂-accentuated: {lbl} vs target" if lbl != "f*" else "w₂-accentuated: f* vs target"
        if lbl != "f*": title += f"\nR²(vs f*)={r2:.4f}"
        ax.set_title(title, fontsize=9); clean_ax(ax)
    fig.suptitle("Cross-evaluation: model responses on accentuated images\n"
                 "Key: w₂-accentuated images fail to change f* (bottom row, right)",
                 fontsize=11)
    plt.tight_layout()

    # ---- Second figure: 6-panel  (2 rows × 3 cols) ----
    t_tile = np.tile(targets, len(X_starts))
    rows = [
        (fs_on_w1, f1_on_w1, f2_on_w1, "w₁-accentuated"),
        (fs_on_w2, f1_on_w2, f2_on_w2, "w₂-accentuated"),
    ]
    fig2, axs2 = plt.subplots(2, 3, figsize=(13, 8))
    for row_i, (fs_vals, f1_vals, f2_vals, row_lbl) in enumerate(rows):
        cols = [
            (t_tile,   "target t",  "k"),
            (f1_vals,  "f₁(I_acc)", "steelblue"),
            (f2_vals,  "f₂(I_acc)", "tomato"),
        ]
        for col_i, (xvals, xlbl, color) in enumerate(cols):
            ax = axs2[row_i, col_i]
            ax.scatter(xvals, fs_vals, s=10, alpha=0.35, color=color, rasterized=True)
            diag = np.array([xvals.min(), xvals.max()])
            ax.plot(diag, diag, "k--", lw=1)
            r2 = 1 - np.var(fs_vals - xvals) / (np.var(xvals) + 1e-12)
            ax.set_xlabel(xlbl, fontsize=9)
            ax.set_ylabel("f*(I_acc)", fontsize=9)
            ax.set_title(f"{row_lbl}\nf* vs {xlbl}  R²={r2:.4f}", fontsize=9)
            clean_ax(ax)
    fig2.suptitle("f* (ground truth) on accentuated images\n"
                  "Key: w₂ row — f* decouples from target/f₂; w₁ row — f* tracks well",
                  fontsize=11)
    plt.tight_layout()

    return fig, fig2

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X = load_vanhateren()
    S, Vh = compute_svd(X)
    n_sv = len(S)
    print(f"  n_sv={n_sv}, S range: {S[-1]:.4f} … {S[0]:.1f}")

    w_star = create_circle_mask()
    w1 = build_weight(w_star, Vh, lo_idx=K_MID_LO, hi_idx=K_MID_HI, alpha=ALPHA1)
    w2 = build_weight(w_star, Vh, lo_idx=K_BOT,    hi_idx=0,         alpha=ALPHA2)

    print(f"  ||w*||={np.linalg.norm(w_star):.3f}  ||w1||={np.linalg.norm(w1):.3f}  ||w2||={np.linalg.norm(w2):.3f}")

    # Target range: span natural f* distribution ± 1.5σ
    fs_nat = X @ w_star
    mu, sig = fs_nat.mean(), fs_nat.std()
    t_min = mu - 3.0 * sig
    t_max = mu + 3.0 * sig
    targets = np.linspace(t_min, t_max, N_TARGETS)
    print(f"  f* natural: mean={mu:.3f} std={sig:.3f}  targets: [{t_min:.3f}, {t_max:.3f}]")

    # Starting images for accentuation
    rng = np.random.RandomState(7)
    start_idxs = rng.choice(len(X), size=N_STARTS, replace=False)
    X_starts = X[start_idxs]

    # Fig 1: weights
    fig = fig_weights(w_star, w1, w2)
    savefig(fig, join(EXPROOT, "exp2_weights"))
    send(join(EXPROOT, "exp2_weights.png"), "*Exp2* — weight vectors w*, w₁, w₂")
    plt.close(fig)

    # Fig 2: natural scatter
    fig, fs_nat2 = fig_natural_scatter(X, w_star, w1, w2)
    savefig(fig, join(EXPROOT, "exp2_natural_scatter"))
    send(join(EXPROOT, "exp2_natural_scatter.png"), "*Exp2* — natural image predictions (Part A)")
    plt.close(fig)

    # Fig 3: accentuated images — normal contrast
    figs = fig_accentuated_images(X_starts, w_star, w1, w2, targets, n_show=8, high_contrast=False)
    for i, fig in enumerate(figs):
        savefig(fig, join(EXPROOT, f"exp2_accentuated_images_start{i}"))
        plt.close(fig)

    # Fig 3b: high-contrast versions
    figs_hc = fig_accentuated_images(X_starts, w_star, w1, w2, targets, n_show=8, high_contrast=True)
    for i, fig in enumerate(figs_hc):
        savefig(fig, join(EXPROOT, f"exp2_accentuated_images_start{i}_highcontr"))
        send(join(EXPROOT, f"exp2_accentuated_images_start{i}_highcontr.png"),
             f"*Exp2 high-contrast* — accentuated images, starting image #{i+1}")
        plt.close(fig)

    # Fig 4: cross-evaluation
    fig, fig2 = fig_cross_eval(X_starts, w_star, w1, w2, targets)
    savefig(fig, join(EXPROOT, "exp2_cross_eval"))
    send(join(EXPROOT, "exp2_cross_eval.png"), "*Exp2* — cross-evaluation on accentuated images (Part C)")
    plt.close(fig)
    savefig(fig2, join(EXPROOT, "exp2_fstar_vs_target"))
    send(join(EXPROOT, "exp2_fstar_vs_target.png"), "*Exp2* — f* vs target level for w₁ vs w₂ accentuation")
    plt.close(fig2)

    # Repo copies
    import shutil
    for fname in ["exp2_weights", "exp2_natural_scatter", "exp2_cross_eval"]:
        for ext in [".png", ".pdf"]:
            src = join(EXPROOT, fname + ext)
            if os.path.exists(src):
                shutil.copy2(src, join(FIGROOT, fname + ext))

    print("\nDone.")

if __name__ == "__main__":
    main()
