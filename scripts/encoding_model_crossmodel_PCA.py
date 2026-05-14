"""
Cross-model PCA analysis of accentuated image activations.

For a given subject + unit, shows how accentuated images from one model
(accentuator) are represented in another model's PCA space (reviewer).

Three figure types:
  A — 2×2 PC1/PC2 scatter:  NSD cloud (gray) + accentuated trajectories colored by level
  B — 2×2 PC activation profile:  each line = one image, x = PC index, color = level (750 dims)
  C — 2×2 PC-level correlation:   Pearson r between each PC and accentuation level (750 dims)

Usage:
    python scripts/encoding_model_crossmodel_PCA.py
    python scripts/encoding_model_crossmodel_PCA.py --subject red_20250428-20250430 --unit 15 \\
        --accentuators resnet50 resnet50_robust \\
        --reviewers    resnet50 resnet50_robust
"""

import argparse
import os
import pickle
import sys
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from circuit_toolkit.plot_utils import saveallforms

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SUBJECT      = "red_20250428-20250430"
DEFAULT_UNIT         = 15
DEFAULT_ACCENTUATORS = ["resnet50", "resnet50_robust"]
DEFAULT_REVIEWERS    = ["resnet50", "resnet50_robust"]

ENCODING_ROOT = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"
FIGURE_ROOT   = "/n/home12/binxuwang/Github/Closed-loop-visual-insilico/figures"

_BASE_COLORS = ["#4878CF", "#FF8C00", "#6ACC65", "#D65F5F", "#B47CC7",
                "#C4AD66", "#77BEDB", "#F7910F", "#8EBA42", "#FFB5B8"]

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_label(name):
    return _LABELS.get(name, name)


def model_color(model_list, name):
    return _BASE_COLORS[model_list.index(name) % len(_BASE_COLORS)]


def pca_dir(subject_id):
    return join(ENCODING_ROOT, subject_id, "posthoc_model_predict_PCA_popul_unit")


def load_reviewer_data(subject_id, unit_id, reviewer):
    """PCA activations for all 5500 accentuated images in reviewer's PCA space."""
    path = join(pca_dir(subject_id),
                f"posthoc_prediction_PCA_pop_unit_{subject_id}_unit{unit_id}_{reviewer}.pkl")
    return pickle.load(open(path, "rb"))


def load_nsd_data(subject_id, unit_id, reviewer):
    """PCA activations for ~969 NSD encoding images in reviewer's PCA space."""
    path = join(pca_dir(subject_id),
                f"posthoc_prediction_NSDencimg_PCA_pop_unit_{subject_id}_unit{unit_id}_{reviewer}.pkl")
    return pickle.load(open(path, "rb"))


def filter_acc(d_rev, accentuator, unit_id):
    """Return (pca_arr, df, levels) filtered to one accentuator × unit."""
    df_all  = d_rev["df"]
    pca_all = d_rev["PCA_resp"].numpy()
    mask    = (df_all["model_name"] == accentuator) & (df_all["unit_id"] == unit_id)
    pca_acc = pca_all[mask.values]
    df_acc  = df_all[mask]
    return pca_acc, df_acc, df_acc["level"].values


def make_figdir(subject_id, accentuators, reviewers):
    tag = "_vs_".join(dict.fromkeys(accentuators + reviewers))
    figdir = join(FIGURE_ROOT, f"{subject_id}_{tag}_freq_comparison")
    os.makedirs(figdir, exist_ok=True)
    return figdir


def _comparison_tag(accentuators, reviewers):
    return "_vs_".join(dict.fromkeys(accentuators + reviewers))


# ── Figure A: PC1/PC2 scatter ─────────────────────────────────────────────────

def fig_pca_scatter(subject_id, unit_id, accentuators, reviewers, figdir):
    nrows, ncols = len(reviewers), len(accentuators)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    all_models = accentuators + reviewers
    lv_min = lv_max = None

    for ri, reviewer in enumerate(reviewers):
        d_rev = load_reviewer_data(subject_id, unit_id, reviewer)
        d_nsd = load_nsd_data(subject_id, unit_id, reviewer)
        pca_nsd = d_nsd["PCA_resp"].numpy()

        for ci, accentuator in enumerate(accentuators):
            ax = axes[ri, ci]
            pca_acc, df_acc, levels = filter_acc(d_rev, accentuator, unit_id)
            if lv_min is None:
                lv_min, lv_max = levels.min(), levels.max()

            # NSD reference
            ax.scatter(pca_nsd[:, 0], pca_nsd[:, 1],
                       c="lightgray", s=8, alpha=0.4, label="NSD (~1k)",
                       zorder=1, rasterized=True)

            # Accentuated trajectories per seed image
            cmap = plt.cm.RdBu_r
            for img_id, grp in df_acc.groupby("img_id"):
                pos = df_acc.index.get_indexer(grp.index)
                pca_grp = pca_acc[pos]
                lvs = grp["level"].values
                order = np.argsort(lvs)
                pca_s, lvs_s = pca_grp[order], lvs[order]
                ax.scatter(pca_s[:, 0], pca_s[:, 1],
                           c=lvs_s, cmap=cmap, vmin=lv_min, vmax=lv_max,
                           s=30, alpha=0.8, zorder=3)
                ax.plot(pca_s[:, 0], pca_s[:, 1],
                        color="gray", lw=0.5, alpha=0.3, zorder=2)

            ax.set_xlabel(f"PC1 ({model_label(reviewer)} space)", fontsize=9)
            ax.set_ylabel(f"PC2 ({model_label(reviewer)} space)", fontsize=9)
            ax.set_title(f"Accentuator: {model_label(accentuator)}\n"
                         f"Reviewer PCA: {model_label(reviewer)}",
                         fontsize=10, color=model_color(all_models, accentuator))
            ax.tick_params(labelsize=8)

    fig.subplots_adjust(right=0.88, hspace=0.4, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                norm=plt.Normalize(lv_min, lv_max))
    fig.colorbar(sm, cax=cbar_ax, label="Accentuation level")
    fig.suptitle(f"{subject_id}  unit {unit_id}\n"
                 "Cross-model PCA: accentuated image activations\n"
                 "(gray = NSD ~1k reference)", fontsize=12, fontweight="bold")

    tag = _comparison_tag(accentuators, reviewers)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_crossmodel_PCA_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: PCA scatter")


# ── Figure B: PC activation profile ──────────────────────────────────────────

def fig_pc_profile(subject_id, unit_id, accentuators, reviewers, figdir, n_pcs=750):
    nrows, ncols = len(reviewers), len(accentuators)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=False)
    axes = np.atleast_2d(axes)

    all_models = accentuators + reviewers
    lv_min = lv_max = None

    for ri, reviewer in enumerate(reviewers):
        d_rev = load_reviewer_data(subject_id, unit_id, reviewer)

        for ci, accentuator in enumerate(accentuators):
            ax = axes[ri, ci]
            pca_acc, df_acc, levels = filter_acc(d_rev, accentuator, unit_id)
            if lv_min is None:
                lv_min, lv_max = levels.min(), levels.max()

            cmap = plt.cm.RdBu_r
            pcs = np.arange(min(n_pcs, pca_acc.shape[1]))
            for lv, row in zip(levels, pca_acc):
                lv_norm = (lv - lv_min) / (lv_max - lv_min)
                ax.plot(pcs, row[:len(pcs)], color=cmap(lv_norm),
                        lw=0.5, alpha=0.25)

            ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
            ax.set_xlabel("PC index", fontsize=9)
            ax.set_ylabel("Activation", fontsize=9)
            ax.set_title(f"Accentuator: {model_label(accentuator)}\n"
                         f"Reviewer: {model_label(reviewer)}",
                         fontsize=10, color=model_color(all_models, accentuator))
            ax.tick_params(labelsize=8)
            ax.set_xlim(0, len(pcs) - 1)

    fig.subplots_adjust(right=0.88, hspace=0.42, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                norm=plt.Normalize(lv_min, lv_max))
    fig.colorbar(sm, cax=cbar_ax, label="Accentuation level")
    fig.suptitle(f"{subject_id}  unit {unit_id}\n"
                 f"PC activation profile (all {n_pcs} dims, each line = one image)",
                 fontsize=12, fontweight="bold")

    tag = _comparison_tag(accentuators, reviewers)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_crossmodel_PC_profile_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: PC profile")


# ── Figure C: PC-level correlation ───────────────────────────────────────────

def fig_pc_level_corr(subject_id, unit_id, accentuators, reviewers, figdir, n_pcs=750):
    nrows, ncols = len(reviewers), len(accentuators)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=False)
    axes = np.atleast_2d(axes)

    all_models = accentuators + reviewers

    for ri, reviewer in enumerate(reviewers):
        d_rev = load_reviewer_data(subject_id, unit_id, reviewer)

        for ci, accentuator in enumerate(accentuators):
            ax = axes[ri, ci]
            pca_acc, df_acc, levels = filter_acc(d_rev, accentuator, unit_id)
            n = min(n_pcs, pca_acc.shape[1])
            corrs = np.array([pearsonr(pca_acc[:, k], levels)[0] for k in range(n)])

            c = model_color(all_models, accentuator)
            ax.bar(np.arange(n), corrs, color=c, alpha=0.6, width=1.0)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xlabel("PC index", fontsize=9)
            ax.set_ylabel("Pearson r with level", fontsize=9)
            ax.set_title(f"Accentuator: {model_label(accentuator)}\n"
                         f"Reviewer: {model_label(reviewer)}",
                         fontsize=10, color=c)
            ax.set_xlim(-1, n)
            ax.tick_params(labelsize=8)

    fig.suptitle(f"{subject_id}  unit {unit_id}\n"
                 f"PC correlation with accentuation level (all {n_pcs} dims)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    tag = _comparison_tag(accentuators, reviewers)
    saveallforms(figdir, f"{subject_id}_U{unit_id}_crossmodel_PC_level_corr_{tag}", figh=fig)
    plt.close(fig)
    print("Saved: PC-level correlation")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",      default=DEFAULT_SUBJECT)
    p.add_argument("--unit",         type=int, default=DEFAULT_UNIT)
    p.add_argument("--accentuators", nargs="+", default=DEFAULT_ACCENTUATORS)
    p.add_argument("--reviewers",    nargs="+", default=DEFAULT_REVIEWERS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subject_id   = args.subject
    unit_id      = args.unit
    accentuators = args.accentuators
    reviewers    = args.reviewers
    figdir       = make_figdir(subject_id, accentuators, reviewers)

    print(f"Subject     : {subject_id}")
    print(f"Unit        : {unit_id}")
    print(f"Accentuators: {accentuators}")
    print(f"Reviewers   : {reviewers}")
    print(f"Figures     : {figdir}\n")

    fig_pca_scatter(subject_id, unit_id, accentuators, reviewers, figdir)
    fig_pc_profile( subject_id, unit_id, accentuators, reviewers, figdir)
    fig_pc_level_corr(subject_id, unit_id, accentuators, reviewers, figdir)
    print(f"\nDone. Figures saved to: {figdir}")
