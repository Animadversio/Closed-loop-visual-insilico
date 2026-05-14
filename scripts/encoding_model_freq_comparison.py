"""
Encoding model gradient & accentuated-image frequency comparison.

  Panel A — gradient maps (side by side, 10 seed images)
  Panel B — gradient radial frequency profiles (side-by-side + overlay)
  Panel C — accentuated image spectral fold-change vs level

Usage:
    # defaults: red, resnet50 vs resnet50_robust
    python scripts/encoding_model_freq_comparison.py

    # custom subject / models
    python scripts/encoding_model_freq_comparison.py \\
        --subject red_20250428-20250430 \\
        --models resnet50 resnet50_robust

    # full 10-model list
    python scripts/encoding_model_freq_comparison.py \\
        --subject red_20250428-20250430 \\
        --models resnet50 resnet50_robust resnet50_clip resnet50_dino \\
                 regnety_640 AlexNet_training_seed_01 clipag_vitb32 \\
                 dinov2_vitb14_reg siglip2_vitb16 radio_v2.5-b
"""

import argparse
import os
import pickle as pkl
import sys
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from circuit_toolkit.plot_utils import saveallforms

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_SUBJECT      = "red_20250428-20250430"
DEFAULT_FOCUS_MODELS = ["resnet50", "resnet50_robust"]
DEFAULT_UNIT_ID      = 0

# Color palette: first two are blue/orange; extras use tab10
_BASE_COLORS = ["#4878CF", "#FF8C00", "#6ACC65", "#D65F5F", "#B47CC7",
                "#C4AD66", "#77BEDB", "#F7910F", "#8EBA42", "#FFB5B8"]

ENCODING_ROOT = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"
FIGURE_ROOT   = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/figures"
STIMULI_ROOT  = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli"

# The 10 seed images used in gradient computation (relative to STIMULI_ROOT)
SEED_IMAGE_PATHS = [
    "shared1000/shared0575_nsd43157.png",
    "shared1000/shared0850_nsd61798.png",
    "shared1000/shared0968_nsd70194.png",
    "shared1000/shared0241_nsd20065.png",
    "shared1000/shared0160_nsd13231.png",
    "shared1000/shared0070_nsd07008.png",
    "shared1000/shared0055_nsd05879.png",
    "shared1000/shared0668_nsd48623.png",
    "shared1000/shared0488_nsd36979.png",
    "shared1000/shared0940_nsd68312.png",
]


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default=DEFAULT_SUBJECT)
    p.add_argument("--models",  nargs="+", default=DEFAULT_FOCUS_MODELS)
    p.add_argument("--unit",    type=int,  default=DEFAULT_UNIT_ID)
    return p.parse_args()


# ── Derived paths ────────────────────────────────────────────────────────────

def make_figdir(subject_id, model_list):
    """Create a per-comparison subfolder under figures/."""
    tag = "_vs_".join(model_list)
    figdir = join(FIGURE_ROOT, f"{subject_id}_{tag}_freq_comparison")
    os.makedirs(figdir, exist_ok=True)
    return figdir


def fft_dir(subject_id):
    return join(ENCODING_ROOT, subject_id, "posthoc_model_predict",
                "encoding_gradient_map_fourier_spectra")


def acc_spectra_pkl(subject_id):
    return join(ENCODING_ROOT, subject_id, "posthoc_model_predict",
                "accentuated_images_fourier_spectra",
                "accentuated_images_fourier_spectra_db.pkl")


# ── Helpers ──────────────────────────────────────────────────────────────────

def model_color(model_list, model_name):
    idx = model_list.index(model_name)
    return _BASE_COLORS[idx % len(_BASE_COLORS)]


def model_label(model_name):
    _LABELS = {
        "resnet50":                "ResNet50",
        "resnet50_robust":         "ResNet50-Robust",
        "resnet50_clip":           "ResNet50-CLIP",
        "resnet50_dino":           "ResNet50-DINO",
        "regnety_640":             "RegNetY-640",
        "AlexNet_training_seed_01":"AlexNet",
        "clipag_vitb32":           "CLIP-ViT-B/32",
        "dinov2_vitb14_reg":       "DINOv2-ViT-B/14",
        "siglip2_vitb16":          "SigLIP2-ViT-B/16",
        "radio_v2.5-b":            "RADIO-v2.5-B",
    }
    return _LABELS.get(model_name, model_name)


def load_grad_data(subject_id, model_name, unit_id):
    fname = f"{subject_id}_unit_{unit_id}_model_{model_name}_grad_maps_freq_profiles.pkl"
    with open(join(fft_dir(subject_id), fname), "rb") as f:
        return pkl.load(f)


def normalize_grad(grad_img):
    """(3,H,W) -> (H,W,3) normalized for display."""
    std = grad_img.std()
    if std < 1e-10:
        return np.zeros_like(grad_img).transpose(1, 2, 0)
    return np.clip(0.5 + grad_img / (3.0 * std), 0, 1).transpose(1, 2, 0)


def _comparison_tag(model_list):
    return "_vs_".join(model_list)


# ── Panel A: gradient maps ───────────────────────────────────────────────────

def load_seed_images(n_seeds=10):
    """Load the seed images used in gradient computation as (H,W,3) uint8 arrays."""
    imgs = []
    for rel_path in SEED_IMAGE_PATHS[:n_seeds]:
        img = np.array(Image.open(join(STIMULI_ROOT, rel_path)).convert("RGB"))
        imgs.append(img)
    return imgs


def fig_gradient_maps(subject_id, model_list, unit_id, figdir, n_seeds=10):
    n_models = len(model_list)
    grad_data = {m: load_grad_data(subject_id, m, unit_id) for m in model_list}
    seed_imgs = load_seed_images(n_seeds)

    n_cols = 1 + n_models  # original + one per model
    fig, axes = plt.subplots(n_seeds, n_cols,
                             figsize=(n_cols * 2.2, n_seeds * 2.2))
    if n_cols == 1:
        axes = axes[:, None]

    # Column 0: original images
    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold", color="k")
    for row in range(n_seeds):
        ax = axes[row, 0]
        ax.imshow(seed_imgs[row])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"seed {row}", fontsize=8)

    # Columns 1+: gradient maps per model
    for col, model in enumerate(model_list):
        c = model_color(model_list, model)
        axes[0, col + 1].set_title(model_label(model), fontsize=11,
                                    fontweight="bold", color=c)
        grad_imgs = grad_data[model]["grad_img"]
        for row in range(n_seeds):
            ax = axes[row, col + 1]
            ax.imshow(normalize_grad(grad_imgs[row]))
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{subject_id}  unit {unit_id}\nGradient maps", fontsize=12)
    fig.tight_layout()
    tag = _comparison_tag(model_list)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_gradmaps_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: gradmaps")


# ── Panel B: gradient radial frequency profiles ──────────────────────────────

def fig_grad_freq_profiles(subject_id, model_list, unit_id, figdir):
    """Side-by-side subplots, one per model."""
    n = len(model_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, model_list):
        data = load_grad_data(subject_id, model, unit_id)
        profiles, freqs = data["profiles"], data["freqs"]
        c = model_color(model_list, model)
        f = freqs[1:]
        for prof in profiles:
            ax.plot(f, prof[1:], color=c, alpha=0.25, lw=0.9)
        ax.plot(f, profiles.mean(axis=0)[1:], color=c, lw=2.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(model_label(model), fontsize=11, fontweight="bold", color=c)
        ax.set_xlabel("Spatial frequency (cycles/image)")
        ax.grid(True, alpha=0.25, which="both")
        if ax is axes[0]:
            ax.set_ylabel("Power")

    fig.suptitle(f"{subject_id}  unit {unit_id}\nRadial Fourier power of gradient maps",
                 fontsize=11)
    fig.tight_layout()
    tag = _comparison_tag(model_list)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_grad_freq_profiles_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: grad freq profiles")


def _bootstrap_ci(data, n_boot=2000, ci=90, rng=None):
    """Return (lo, hi) bootstrap CI arrays for the mean of data (n_samples, n_freqs)."""
    rng = np.random.default_rng(rng)
    n = data.shape[0]
    boot_means = np.stack([
        data[rng.integers(0, n, n)].mean(axis=0) for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (100 - ci) / 2,     axis=0)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    return lo, hi


def fig_grad_freq_overlay(subject_id, model_list, unit_id, figdir):
    """Single-axes overlay of all models' mean gradient power spectra."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for model in model_list:
        data = load_grad_data(subject_id, model, unit_id)
        profiles, freqs = data["profiles"], data["freqs"]
        f = freqs[1:]
        mean_prof = profiles.mean(axis=0)
        lo, hi = _bootstrap_ci(profiles)
        c = model_color(model_list, model)
        ax.plot(f, mean_prof[1:], color=c, lw=2.2, label=model_label(model))
        ax.fill_between(f, lo[1:], hi[1:], color=c, alpha=0.2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (cycles/image)")
    ax.set_ylabel("Power")
    ax.set_title(f"{subject_id}  unit {unit_id}\nGradient radial power (mean ± std)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    tag = _comparison_tag(model_list)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_grad_freq_overlay_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: grad freq overlay")


# ── Panel C: accentuated image spectral fold-change ─────────────────────────

def fig_acc_freq_foldchange(subject_id, model_list, figdir):
    import pandas as pd
    db = pd.read_pickle(acc_spectra_pkl(subject_id))
    n = len(model_list)
    freqs = np.arange(len(db["spectrum_foldchange"].iloc[0]))
    # skip freq=0 for log-x axis
    freqs_nz = freqs[1:]
    cmap = plt.cm.RdBu_r
    level_min = db["level"].min()
    level_max = db["level"].max()
    tag = _comparison_tag(model_list)

    for xscale in ("linear", "log"):
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, model in zip(axes, model_list):
            df_m = db[db["model_name"] == model]
            for lv_val in sorted(df_m["level"].unique()):
                rows = df_m[df_m["level"] == lv_val]
                specs = np.stack(rows["spectrum_foldchange"].values)
                mean_fc = specs.mean(axis=0)
                sem_fc  = specs.std(axis=0) / max(np.sqrt(len(specs)), 1)
                lv_norm = (lv_val - level_min) / (level_max - level_min)
                c = cmap(lv_norm)
                x = freqs_nz if xscale == "log" else freqs
                y      = mean_fc[1:] if xscale == "log" else mean_fc
                y_lo   = (mean_fc - sem_fc)[1:] if xscale == "log" else (mean_fc - sem_fc)
                y_hi   = (mean_fc + sem_fc)[1:] if xscale == "log" else (mean_fc + sem_fc)
                ax.plot(x, y, color=c, lw=1.2, alpha=0.45)
                ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.06)

            ax.axhline(1.0, color="k", lw=1.0, ls="--")
            ax.set_xscale(xscale)
            ax.set_xlabel("Spatial frequency (cycles/image)")
            if ax is axes[0]:
                ax.set_ylabel("Spectral fold-change")
            ax.set_title(model_label(model), fontsize=11, fontweight="bold",
                         color=model_color(model_list, model))
            ax.set_yscale("log")
            ax.grid(True, alpha=0.2, which="both")

        fig.suptitle(f"{subject_id}\nAccentuated image spectral fold-change"
                     f" (mean ± SEM across units & seeds)  [{xscale} x]", fontsize=10)
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(level_min, level_max))
        fig.colorbar(sm, cax=cbar_ax, label="Accentuation level (normalized score)")
        saveallforms(figdir, f"{subject_id}_acc_freq_foldchange_{xscale}x_{tag}", figh=fig)
        plt.close(fig)
        print(f"Saved: acc freq foldchange [{xscale} x]")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    subject_id  = args.subject
    model_list  = args.models
    unit_id     = args.unit
    figdir      = make_figdir(subject_id, model_list)

    print(f"Subject : {subject_id}")
    print(f"Models  : {model_list}")
    print(f"Unit    : {unit_id}")
    print(f"Figures : {figdir}\n")

    fig_gradient_maps(subject_id, model_list, unit_id, figdir)
    fig_grad_freq_profiles(subject_id, model_list, unit_id, figdir)
    fig_grad_freq_overlay(subject_id, model_list, unit_id, figdir)
    fig_acc_freq_foldchange(subject_id, model_list, figdir)
    print(f"\nDone. All figures saved to: {figdir}")
