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

### Encoding Session Neural Responses (for training encoding models)

```
/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data/
  red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5   # 153 MB
```

HDF5 structure:
- `repavg/response_peak`      — (969, 64) float64, stimulus-averaged peak responses
- `repavg/response_temporal`  — (60, 969, 64), temporal profile
- `repavg/stimulus_name`      — stimulus filenames (row index for joining)
- `trials/response_peak`      — (8241, 64) float32, trial-by-trial responses
- `neuron_metadata/reliability`, `ncsnr`, `brain_area` — per-unit quality metrics
- `stimulus_meta/xy_deg`, `size_px` — stimulus position and size

Loading:
```python
from core.data_utils import load_from_hdf5, extract_neural_data_dict_2025apr
data = load_from_hdf5("<path>.h5")
data_dict = extract_neural_data_dict_2025apr(data)
# data_dict['resp_mat']       → (969, 64) peak responses
# data_dict['stimulus_names'] → stimulus filenames (join key)
# data_dict['image_fps']      → full paths to stimulus images
```

Model predictions for same stimuli (post-hoc, all models × units):
```
/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/
  red_20250428-20250430/
  ├── posthoc_model_predict/
  │   ├── encoding_stim_info_w_pred_resp_{subject}.pkl      # 969×58 — NSD stims + pred_resp all models×units
  │   ├── accentuated_stim_info_w_pred_resp_{subject}.pkl   # 5500×N — accentuated stims + pred_resp
  │   └── accentuated_stim_info_{subject}.pkl/.csv          # 5500×6 — metadata only (no pred_resp)
  │
  ├── posthoc_model_predict_PCA_popul_unit/
  │   ├── df_accentuated_{subject}.pkl                      # 5500-row metadata df (shared)
  │   ├── posthoc_prediction_PCA_pop_unit_{subject}_unit{u}_{model}.pkl
  │   │   # Per unit×model pkl (5 units × 10 models = 50 files); keys:
  │   │   #   PCA_resp        (5500, 750)  — PCA-projected activations for accentuated imgs
  │   │   #   population_resp (5500, 64)   — full population response
  │   │   #   target_unit_resp(5500, 1)    — predicted response for target unit
  │   │   #   readout_vec     (750,)       — encoding model weight in PCA space
  │   │   #   readout_bias    scalar
  │   │   #   cosine_sims     (5500,)
  │   │   #   df              (5500, 8)    — model/unit/img_id/level metadata
  │   └── posthoc_prediction_NSDencimg_PCA_pop_unit_{subject}_unit{u}_{model}.pkl
  │       # Same structure but for NSD encoding images (969 rows) — for encoding scatter plots
  │
  ├── model_outputs_pca4all/
  │   └── {subject}_{model}_layer_sweep_synopisis*.{png,pdf}  # Layer sweep synopsis figures
  │
  └── encoding_gradient_map_fourier_spectra/
      ├── {subject}_unit_{u}_model_{model}_grad_maps.png
      └── {subject}_unit_{u}_model_{model}_grad_maps_freq_profiles.pkl
          # keys: profiles (10, 158), freqs (158,), grad_img (10, 3, 224, 224)
          # Computed on 224×224 RGB images; x-axis = cycles/224px image
```

Accentuated image frequency spectra (1024×1024 PNG inputs):
```
  red_20250428-20250430/posthoc_model_predict/accentuated_images_fourier_spectra/
    accentuated_images_fourier_spectra_db.pkl   # DataFrame: model_name, unit_id, img_id,
                                                #   level, score, filepath,
                                                #   spectrum (158,), spectrum_foldchange (158,)
    # x-axis = cycles/1024px image (not same scale as gradient map profiles above)
```

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
