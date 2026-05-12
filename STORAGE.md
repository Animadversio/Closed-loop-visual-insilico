# Storage & Data Paths

Quick reference for where everything lives on the HPC cluster.

## Roots

| Variable | Path |
|----------|------|
| `$HOME` | `/n/home12/binxuwang/` |
| `$STORE` | `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/` |
| Code repo | `$HOME/Github/Closed-loop-visual-insilico/` |

---

## Neural Data

```
$STORE/Projects/VVS_Accentuation/          # 271 GB total
├── paul_*/                                 # Monkey "Paul" recordings
├── red_*/                                  # Monkey "Red" recordings
├── baby1_*/                                # Monkey "Baby1" recordings
├── baby5_*/                                # Monkey "Baby5" recordings
└── vvs-accentuate-day{1,2,3}_*.h5/.hdf5  # HDF5 neural response tensors
                                            # normalized and raw versions

$STORE/step1_results/
└── nsd_shared1000_6monkeys_2024.h5        # 6.8 GB — consolidated NSD responses
                                            # 6 monkeys × 1000 shared images
```

Also see `$STORE/Data/` and `$STORE/Datasets/NSD_N3/` for additional neural/behavioral data.

---

## Image Stimuli

```
$STORE/Datasets/
├── ffhq256/ffhq256/                       # 70k face images (256px PNG)
│                                          # Used: resized to 100×100 grayscale, float [0,1]
├── vanhateren_natural_stimuli/            # van Hateren natural scenes
│   └── imk*.iml                           # Raw uint16, shape (1024, 1536)
│                                          # Normalize per image to [0,1]; mean ≈ 0.15
│                                          # Subtract 0.15 before linear models
├── MacaqueITBench/                        # Macaque IT benchmark images
├── THINGS_database/                       # 4.9 GB object image database
├── imagenet-valid/                        # ImageNet validation set
└── COCO_dataset/                          # COCO images

$STORE/Projects/VVS_Accentuation/Stimuli/
├── shared1000/                            # 1000 natural images used in experiments
├── results_fa2/                           # Feature accentuation output stimuli
└── stimuli_pilot_20241119/                # Pilot experiment stimulus set

$STORE/Projects/VVS_Accentuation/NeuralData_raw/
└── {monkey}_{date}_hp{h}_bs{b}_zs{z}_rw{start}-{end}_sessdata.pkl
    # Preprocessing flags: hp=highpass, bs=baseline-subtract, zs=zscore, rw=reward window
```

---

## Encoding Models & Checkpoints

```
$STORE/Projects/VVS_Accentuation/model_outputs/   # 189+ .pkl/.pth files
├── paul_20241119*/                                 # Per-monkey × session
├── red_20241212-20241220*/                         # Ridge and Lasso variants
├── *_Lasso/                                        # Sparse regression models
└── (grid search results, layer sweeps, robustness variants)

$STORE/Projects/VVS_Accentuation/model_backbones/  # Base model architectures

$STORE/Projects/VVS_Accentuation/Encoding_models/  # YAML-exported model configs
├── leap_250426-250501/
├── red_20250428-20250430/
└── venus_250426-250429/

$HOME/Github/Closed-loop-visual-insilico/checkpoints/
└── imagenet_linf_8_pure.pt                # 98 MB — adversarially robust ResNet-50
```

---

## Theory Experiment Outputs (Accentuation)

```
$STORE/DL_Projects/AdvExampleLinearRegr/
├── exp1/                                  # Exp1: adversarial weight construction
│   ├── FFHQ_eigenspectrum.{png,pdf}
│   ├── FFHQ_eigenmode_gallery.{png,pdf}
│   ├── FFHQ_adversarial_weights.{png,pdf}
│   ├── FFHQ_weight_energy.{png,pdf}
│   ├── FFHQ_samples.{png,pdf}
│   ├── vanHateren_*.{png,pdf}             # Same set for van Hateren
│   └── ...
├── exp2/                                  # Exp2: accentuation divergence
│   ├── exp2_weights.{png,pdf}
│   ├── exp2_natural_scatter.{png,pdf}
│   ├── exp2_accentuated_images_start{0-7}.{png,pdf}
│   ├── exp2_accentuated_images_start{0-7}_highcontr.{png,pdf}
│   ├── exp2_cross_eval.{png,pdf}
│   └── exp2_fstar_vs_target.{png,pdf}
└── circ_mask_weights/                     # Regression sweep: circle mask recovery
    └── regression_results_n_samp{N}_noise{sigma}.{pkl,png}
```

---

## Analysis Results & Figures

```
$STORE/step1_results/                      # 1.2 GB regression outputs
├── mlp_dinov2/                            # DINOv2 MLP outputs
├── per_sigma/, per_sigma_v2/              # Sigma-parameterized results
└── pilot_sweep/                           # Initial parameter searches

$STORE/brain_score_cache/                  # BrainScore model evaluations

$HOME/Github/Closed-loop-visual-insilico/figures/
├── older/
├── red_20250123/
├── leap_250426/
└── three0_250426/
```

---

## Code Organization

```
$HOME/Github/Closed-loop-visual-insilico/
├── core/                   # Model loading, data utils, post-hoc prediction
├── neural_regress/         # Regression pipeline (Ridge, Lasso, PCA, SRP)
├── scripts/                # Batch HPC scripts (regression, accentuation, export)
├── notebooks/              # Analysis notebooks + theory experiment scripts
│   ├── exp1_adversarial_weight_construction.py
│   ├── exp1_supp_eigenmode_gallery.py
│   └── exp2_accentuation.py
├── pytorch-lasso/          # Custom GPU Lasso implementation
├── checkpoints/            # Model weight files
└── figures/                # Exported publication figures
```

---

## Figure Conventions

- Save as **both PNG (dpi=150) and PDF**
- Always set `matplotlib.rcParams["pdf.fonttype"] = 42` (vector fonts in PDF)
- `import circuit_toolkit.plot_utils` → auto-removes top/right spines
- Summary figures → `notebooks/` in repo
- Bulk/large figures → corresponding `DL_Projects/` subfolder
- Send to Discord channel `1488321710573883502` via Bot API after saving

---

## Key Dataset Details

| Dataset | Format | Size | Notes |
|---------|--------|------|-------|
| FFHQ | PNG 256px | ~7 GB | Resize to 100×100 grayscale, [0,1] |
| van Hateren | uint16 `.iml` | — | 1024×1536, normalize per image to [0,1], subtract mean 0.15 |
| NSD shared1000 | HDF5 | 6.8 GB | 6 monkeys, 1000 images |
| VVS Accentuation | HDF5/pkl | 271 GB | Multi-session primate recordings |

SVD convention: load 12000 samples → CUDA SVD → 10000 singular values (full pixel space for 100×100 images).
