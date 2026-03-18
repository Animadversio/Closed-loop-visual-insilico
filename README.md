# Closed-Loop Visual In-Silico Experiments

A computational neuroscience research project for predicting primate visual cortex responses using deep neural network encoding models and generating feature-accentuated stimuli through gradient-based optimization.

## Project Overview

This codebase implements a complete pipeline for:

1. **Neural Encoding**: Train linear regression models to predict neural responses from deep network features
2. **Feature Accentuation**: Generate synthetic images that maximally activate specific neurons
3. **Closed-Loop Experiments**: Test model predictions with physiologically recorded responses to optimized stimuli
4. **Cross-Model Analysis**: Compare encoding performance across vision model architectures

## Repository Structure

```
.
├── core/                           # Core utilities and model loading
│   ├── model_load_utils.py        # Unified interface for ~10 vision models
│   ├── brainscore_model_utils.py  # BrainScore model loaders (ReAlNet, AlexNet)
│   ├── posthoc_prediction_utils.py # Load trained models and predict on new images
│   ├── data_utils.py               # Data loading and preprocessing
│   └── GAN_utils_hf.py             # GAN models for feature visualization
│
├── neural_regress/                 # Neural regression library
│   ├── regress_lib.py              # Main regression pipeline with CV
│   ├── PCA_dual_solver_lib.py      # GPU-accelerated dual PCA solver
│   ├── SRP_torch_lib.py            # GPU-accelerated sparse random projection
│   ├── feature_reduction_lib.py    # Dimensionality reduction methods
│   ├── regress_eval_lib.py         # Model evaluation utilities
│   └── posthoc_analysis_lib.py     # Analysis of trained models
│
├── scripts/                        # Batch processing scripts
│   ├── neural_regression_massprod_*.py  # Mass production regression runs
│   ├── feature_accentuation_*.py        # Generate accentuated stimuli
│   ├── posthoc_prediction_*.py          # Apply models to new images
│   └── neural_regress_yaml_export.py    # Export models to YAML configs
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── neural_regression_*.ipynb   # Regression experiments
│   ├── feature_accentuate_*.ipynb  # Feature accentuation notebooks
│   ├── data_explorer_*.ipynb       # Data exploration and visualization
│   └── *_posthoc_synopsis.ipynb    # Results summaries
│
└── figures/                        # Generated figures and visualizations
```

## Data Organization

Experimental data is stored at: `/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/`

### Stimuli Directory Structure

```
VVS_Accentuation/Stimuli/
│
├── encodingstimuli_apr2025/           # Natural images for encoding (971 images, 232MB)
│   ├── apple_04.jpg
│   ├── cat_06.jpg
│   └── ... (objects from various datasets)
│
├── encoding_stimuli_split_seed0_swapped.csv  # Metadata for train/test splits
│   # Columns: stimulus_name, is_nsd, is_floc, is_OO, is_train, is_normalizer, is_test
│
├── results_22-11-2024/                # Feature accentuated stimuli (Level sweep)
│   └── r50_unit_{id}_img_{id}_level_{value}_score_{value}.png
│       # Images at different accentuation levels (-1.0 to 4.8)
│
├── results_12-01-2025/                # Controversial stimuli (3,221 images, 5.7GB)
│   └── controversial_max_{model}_MultiLassoCV_unit_{id}_img_{id}_srobust_{score}_sr50_{score}.png
│       # Stimuli that maximize one model while minimizing another
│
├── shared1000/                        # Pilot study images (symlinks)
│   └── clip_{pos|neg}_{id}_{dim}_im.png
│       # CLIP-generated images with 512/2048 dimensions
│
└── stimuli_pilot_20241119/           # Initial pilot experiment
```

### Neural Data Structure

```
VVS_Accentuation/NeuralData_raw/      # Raw neural recordings (6.1GB)
│
└── {monkey}_{date}_hp{h}_bs{b}_zs{z}_rw{start}-{end}_sessdata.pkl
    # Naming convention encodes preprocessing parameters:
    #   monkey: leap, paul, red, three0, venus
    #   date: YYMMDD format
    #   hp: highpass filter flag (0/1)
    #   bs: baseline subtraction (0/1)
    #   zs: z-score normalization (0/1)
    #   rw: reward window timing (e.g., 100-400ms)
```

### Encoding Models

```
VVS_Accentuation/Encoding_models/
│
├── {monkey}_{daterange}/             # Models organized by subject and date
│   ├── *.pt                          # Saved PyTorch models (readouts, transforms)
│   ├── *.pkl                         # Scikit-learn model pickles
│   └── *.yaml                        # Model configuration files

Examples:
├── leap_250426-250501/
├── red_20250428-20250430/
└── venus_250426-250429/
```

### Other Data Directories

- `accentuation_configs/`: YAML configs for optimization experiments
- `Encoding_model_outputs/`: Regression results (DataFrames, stats)
- `Ephys_Data/`: Processed electrophysiology data
- `Reduced_Data/`: PCA/dimensionality-reduced features
- `cluster_run_outputs/`: HPC SLURM job logs

## Experimental Workflow

### 1. Train Encoding Models

```python
from core.model_load_utils import load_model_transform
from circuit_toolkit.layer_hook_utils import featureFetcher
from neural_regress.regress_lib import perform_regression_sweeplayer_RidgeCV

# Load vision model
model, transforms = load_model_transform("resnet50_robust", device="cuda")

# Extract features from target layer
fetcher = featureFetcher(model, input_size=(3, 224, 224))
fetcher.record("layer4.Bottleneck2", ingraph=False)

# Run images through model
model(images)
feat_dict = {"layer4": fetcher["layer4.Bottleneck2"]}

# Fit regression model with PCA reduction
result_df, models, Xdict, tfm_dict = perform_regression_sweeplayer_RidgeCV(
    feat_dict,
    neural_responses,  # shape: (n_images, n_units)
    dimred_list=["pca1000"],
    alpha_list=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5],
)
```

### 2. Generate Feature-Accentuated Stimuli

```python
from core.posthoc_prediction_utils import get_predictor_from_config

# Load trained predictor
pred_fn, target_fn, model, transforms, fetcher, Xtransform, readout = \
    get_predictor_from_config("path/to/model_config.yaml")

# Optimize images via gradient ascent
img_opt = torch.randn(1, 3, 224, 224, requires_grad=True, device="cuda")
optimizer = torch.optim.Adam([img_opt], lr=0.05)

for step in range(1000):
    optimizer.zero_grad()
    prediction = target_fn(img_opt)  # Predict target unit response
    loss = -prediction.sum()         # Maximize prediction
    loss.backward()
    optimizer.step()
```

### 3. Analyze Results

```python
from neural_regress.regress_lib import evaluate_prediction

# Evaluate on held-out test set
df, eval_dict, y_pred_dict = evaluate_prediction(
    fit_models=models,
    Xfeat_dict=Xdict_test,
    y_true=neural_responses_test,
    label="test_set",
    savedir="results/"
)

# Statistics saved: Pearson r, Spearman ρ, D² (explained variance)
```

## Supported Vision Models

Models available via `load_model_transform(modelname)`:

| Model Name | Architecture | Notes |
|------------|-------------|-------|
| `resnet50` | ResNet-50 | Standard ImageNet pretrained |
| `resnet50_robust` | ResNet-50 | Adversarially trained (L∞ robust) |
| `resnet50_clip` | ResNet-50 | CLIP visual encoder |
| `resnet50_dino` | ResNet-50 | DINO self-supervised |
| `clipag_vitb32` | ViT-B/32 | CLIP attention-guided |
| `siglip2_vitb16` | ViT-B/16 | SigLIP v2 |
| `dinov2_vitb14_reg` | ViT-B/14 | DINOv2 with registers |
| `radio_v2.5-b` | ViT-B/16 | RADIO v2.5 |
| `ReAlnet01`-`10` | CORnet-S | BrainScore-trained models |
| `AlexNet_training_seed_01` | AlexNet | Custom trained variant |
| `regnety_640` | RegNetY | SEER pretrained |

## Feature Reduction Methods

Specify in `dimred_list` parameter:

- `pca{N}`: PCA with N components (e.g., `pca1000`)
- `srp{N}`: Sparse Random Projection with N components
- `sp_avg`: Spatial average pooling (ConvNets)
- `sp_cent`: Center position only
- `avgtoken`: Average token pooling (Transformers)
- `clstoken`: CLS token only (Transformers)
- `full`: Flattened features (no reduction)

## Key Configuration Files

### Model Export YAML Format

```yaml
model_name: resnet50_robust
layer_name: .layer4.Bottleneck2
xtransform_path: /path/to/pca_transform.pt
readout_path: /path/to/ridge_readout.pt
unit_ids: [0, 5, 10, 15]  # Target neural units
fit_method_name: RidgeCV
meta_path: /path/to/metadata.pkl
```

### Stimulus Metadata CSV

```csv
stimulus_name,is_nsd,is_floc,is_OO,is_train,is_normalizer,is_test
apple_04.jpg,False,False,True,True,False,False
cat_06.jpg,False,True,False,False,True,False
```

Flags indicate:
- `is_nsd`: From Natural Scenes Dataset
- `is_floc`: Functional localizer image
- `is_OO`: From object dataset
- `is_train`: Used for training encoding model
- `is_normalizer`: Used for response normalization
- `is_test`: Held-out test image

## Computational Resources

### HPC Paths
- Model checkpoints: `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones/`
- BrainScore cache: `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/brain_score_cache/`
- Shared data: `/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/`

### GPU Acceleration
- PCA: Dual solver on GPU (default with `use_pca_dual=True`)
- SRP: PyTorch implementation on GPU (default with `use_srp_torch=True`)
- Feature extraction: Set `store_device="cuda"` in `featureFetcher.record()`

## Common Tasks

### Run Layer Sweep Regression
```bash
python scripts/neural_regression_massprod_VVS_20250428-20250429.py
```

### Export Model to YAML
```bash
python scripts/neural_regress_yaml_export.py \
    --model_path /path/to/model.pkl \
    --output_dir /path/to/configs/
```

### Generate Accentuated Stimuli
```bash
python scripts/feature_accentuation_model_posthoc_prediction.py \
    --config /path/to/model_config.yaml \
    --output_dir /path/to/stimuli/
```

## Data Format Details

### Neural Response Matrix
- Shape: `(n_images, n_units)`
- Units: Typically firing rates (spikes/sec) or z-scored responses
- Missing data: Marked as NaN

### Feature Tensors
- ConvNets: `(batch, channels, height, width)`
- Transformers: `(batch, seq_len, embed_dim)`
- After reduction: `(batch, n_features)`

### Saved Model Components
1. **Feature Transform** (`.pt`): PyTorch module (PCA, SRP, or spatial pooling)
2. **Readout Layer** (`.pt`): Linear layer with learned weights
3. **Metadata** (`.pkl` or `.yaml`): Model configuration and statistics

## Citation

If you use this code or data, please cite:

```
[Citation to be added]
```

## Authors

Alvarez Lab, Harvard University

## License

[To be specified]
