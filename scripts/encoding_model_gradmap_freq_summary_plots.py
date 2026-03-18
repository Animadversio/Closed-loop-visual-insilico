"""
Summary plots for encoding model gradient map frequency analysis.

Reads precomputed pkl outputs from encoding_model_gradmap_freq_analysis.py
and generates:
  1. Average frequency power spectrum per model (2x5 layout)
  2. Representative gradient maps per model (2x5 layout)
  3. Cross-model comparison overlay of grand-mean profiles
"""

import glob
import os
import re
import pickle as pkl
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
ENCODING_MODEL_ROOT = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"

SUBJECTS = [
    "leap_250426-250501",
    "paul_20250428-20250430",
    "red_20250428-20250430",
    "three0_250426-250501",
    "venus_250426-250429",
]

MODEL_NAMES = [
    "resnet50",
    "resnet50_robust",
    "resnet50_clip",
    "resnet50_dino",
    "regnety_640",
    "AlexNet_training_seed_01",
    "clipag_vitb32",
    "dinov2_vitb14_reg",
    "siglip2_vitb16",
    "radio_v2.5-b",
]

# Short display names for plot titles
MODEL_SHORT_NAMES = {
    "resnet50": "ResNet50",
    "resnet50_robust": "ResNet50-Robust",
    "resnet50_clip": "ResNet50-CLIP",
    "resnet50_dino": "ResNet50-DINO",
    "regnety_640": "RegNetY-640",
    "AlexNet_training_seed_01": "AlexNet",
    "clipag_vitb32": "CLIP-ViT-B/32",
    "dinov2_vitb14_reg": "DINOv2-ViT-B/14",
    "siglip2_vitb16": "SigLIP2-ViT-B/16",
    "radio_v2.5-b": "RADIO-v2.5-B",
}

SUMMARY_DIR = join(ENCODING_MODEL_ROOT, "summary_gradmap_freq_analysis")
os.makedirs(SUMMARY_DIR, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────

def get_fft_dir(subject):
    return join(
        ENCODING_MODEL_ROOT, subject,
        "posthoc_model_predict", "encoding_gradient_map_fourier_spectra",
    )


def find_pkl_files(subject, model_name):
    """Find all pkl files for a given subject and model."""
    fft_dir = get_fft_dir(subject)
    pattern = join(fft_dir, f"*_model_{model_name}_grad_maps_freq_profiles.pkl")
    return sorted(glob.glob(pattern))


def load_all_data():
    """Load profiles and grad_img for every (subject, unit, model) combination.

    Returns:
        model_profiles: dict  model_name -> list of (10, 158) arrays
        model_gradimgs: dict  model_name -> list of (10, 3, 224, 224) arrays
        freqs: (158,) array
        model_metadata: dict  model_name -> list of (subject, unit_id) tuples
    """
    model_profiles = {m: [] for m in MODEL_NAMES}
    model_gradimgs = {m: [] for m in MODEL_NAMES}
    model_metadata = {m: [] for m in MODEL_NAMES}
    freqs = None

    for subject in SUBJECTS:
        for model_name in MODEL_NAMES:
            pkl_files = find_pkl_files(subject, model_name)
            for pf in pkl_files:
                data = pkl.load(open(pf, "rb"))
                model_profiles[model_name].append(data["profiles"])
                model_gradimgs[model_name].append(data["grad_img"])
                # Parse unit id from filename
                m = re.search(r"_unit_(\d+)_model_", pf)
                unit_id = int(m.group(1)) if m else -1
                model_metadata[model_name].append((subject, unit_id))
                if freqs is None:
                    freqs = data["freqs"]

    return model_profiles, model_gradimgs, freqs, model_metadata


# ── Plot 1: Average frequency power spectrum per model (2x5) ──────────────

def plot_avg_freq_spectra(model_profiles, freqs):
    """2x5 grid: each subplot shows all unit profiles (faint) + grand mean (bold)."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[idx]
        all_profiles = model_profiles[model_name]  # list of (10, 158)
        if len(all_profiles) == 0:
            ax.set_title(MODEL_SHORT_NAMES[model_name])
            continue

        # Stack: (n_units, 10_seeds, 158_freq)
        stacked = np.stack(all_profiles, axis=0)
        # Mean across seed images -> (n_units, 158)
        unit_means = stacked.mean(axis=1)
        # Grand mean across units
        grand_mean = unit_means.mean(axis=0)
        grand_std = unit_means.std(axis=0)

        f = freqs[1:]  # skip DC
        for i in range(unit_means.shape[0]):
            ax.plot(f, unit_means[i, 1:], alpha=0.15, color="steelblue", linewidth=0.7)

        ax.plot(f, grand_mean[1:], color="crimson", linewidth=2, label="Grand mean")
        ax.fill_between(
            f,
            (grand_mean - grand_std)[1:],
            (grand_mean + grand_std)[1:],
            alpha=0.2, color="crimson",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MODEL_SHORT_NAMES[model_name], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        if idx >= 5:
            ax.set_xlabel("Spatial frequency (cycles/image)")
        if idx % 5 == 0:
            ax.set_ylabel("Power")

    fig.suptitle(
        "Radial Fourier Power of Encoding-Model Gradient Maps\n"
        f"(aggregated across {len(SUBJECTS)} subjects, per-unit means shown faintly)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(join(SUMMARY_DIR, "avg_freq_spectra_per_model_2x5.png"), dpi=200, bbox_inches="tight")
    fig.savefig(join(SUMMARY_DIR, "avg_freq_spectra_per_model_2x5.pdf"), bbox_inches="tight")
    fig.show()
    print(f"Saved: avg_freq_spectra_per_model_2x5.png/pdf")


# ── Plot 2: Representative gradient maps per model (2x5) ──────────────────

def _normalize_grad_for_display(grad_img):
    """Normalize gradient image to [0, 1] for display (per-image std normalization)."""
    # grad_img: (3, H, W)
    std = grad_img.std()
    if std < 1e-10:
        return np.zeros_like(grad_img).transpose(1, 2, 0)
    normed = 0.5 + grad_img / (3.0 * std)
    return np.clip(normed, 0, 1).transpose(1, 2, 0)  # (H, W, 3)


def plot_representative_gradmaps(model_gradimgs, model_metadata):
    """2x5 grid: each subplot shows one representative gradient map per model.

    Picks the first unit of the first subject as the representative,
    showing one seed image's gradient per subplot (square aspect).
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[idx]
        gradimgs = model_gradimgs[model_name]
        metadata = model_metadata[model_name]

        if len(gradimgs) == 0:
            ax.set_title(MODEL_SHORT_NAMES[model_name])
            ax.axis("off")
            continue

        # Pick the first entry as representative, first seed image
        grad_batch = gradimgs[0]  # (10, 3, 224, 224)
        subj, unit_id = metadata[0]

        disp = _normalize_grad_for_display(grad_batch[0])  # (224, 224, 3)
        ax.imshow(disp)
        ax.set_aspect("equal")
        ax.set_title(MODEL_SHORT_NAMES[model_name], fontsize=10, fontweight="bold")
        subj_short = subj.split("_")[0]
        ax.set_xlabel(f"{subj_short} unit {unit_id}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Representative Gradient Maps per Encoding Model\n"
        "(single seed image, normalized per-image)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(join(SUMMARY_DIR, "representative_gradmaps_per_model_2x5.png"), dpi=200, bbox_inches="tight")
    fig.savefig(join(SUMMARY_DIR, "representative_gradmaps_per_model_2x5.pdf"), bbox_inches="tight")
    fig.show()
    print(f"Saved: representative_gradmaps_per_model_2x5.png/pdf")


# ── Plot 3: Cross-model comparison overlay ─────────────────────────────────

def plot_crossmodel_overlay(model_profiles, freqs):
    """Single plot overlaying the grand-mean profile of each model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))

    for idx, model_name in enumerate(MODEL_NAMES):
        all_profiles = model_profiles[model_name]
        if len(all_profiles) == 0:
            continue
        stacked = np.stack(all_profiles, axis=0)  # (n_units, 10, 158)
        grand_mean = stacked.mean(axis=(0, 1))  # (158,)

        f = freqs[1:]
        ax.plot(
            f, grand_mean[1:],
            color=colors[idx], linewidth=2,
            label=MODEL_SHORT_NAMES[model_name],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (cycles/image)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title("Grand-Mean Radial Fourier Power: Cross-Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(join(SUMMARY_DIR, "crossmodel_freq_overlay.png"), dpi=200, bbox_inches="tight")
    fig.savefig(join(SUMMARY_DIR, "crossmodel_freq_overlay.pdf"), bbox_inches="tight")
    fig.show()
    print(f"Saved: crossmodel_freq_overlay.png/pdf")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data from all subjects and models...")
    model_profiles, model_gradimgs, freqs, model_metadata = load_all_data()

    for m in MODEL_NAMES:
        print(f"  {MODEL_SHORT_NAMES[m]:>22s}: {len(model_profiles[m]):3d} units loaded")

    print("\nGenerating plots...")
    plot_avg_freq_spectra(model_profiles, freqs)
    plot_representative_gradmaps(model_gradimgs, model_metadata)
    plot_crossmodel_overlay(model_profiles, freqs)
    print(f"\nAll plots saved to: {SUMMARY_DIR}")
