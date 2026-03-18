# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**See README.md for comprehensive project documentation, data organization, and experimental workflows.**

## Project Overview

This is a computational neuroscience research codebase for closed-loop visual in-silico experiments. The project focuses on:
- Neural encoding models that predict primate visual cortex responses from deep neural network features
- Feature accentuation and gradient-based optimization to generate stimuli that drive neural responses
- Posthoc analysis of neural regression models across different architectures and layers

## Key Architecture

### Core Workflow Pipeline

1. **Model Loading** (`core/model_load_utils.py`): Centralized loading of vision models (ResNet50, ViT, DINOv2, CLIP variants, etc.) with consistent preprocessing
2. **Feature Extraction**: Use `featureFetcher` from `circuit_toolkit` to hook and extract intermediate layer activations
3. **Neural Regression** (`neural_regress/regress_lib.py`): Fit linear models (Ridge, Lasso, RidgeCV) to predict neural responses from model features
4. **Feature Reduction**: Apply PCA/SRP dimensionality reduction before regression (GPU-accelerated implementations in `neural_regress/PCA_dual_solver_lib.py` and `neural_regress/SRP_torch_lib.py`)
5. **Posthoc Prediction** (`core/posthoc_prediction_utils.py`): Load trained readout weights and transforms to predict neural responses on new images
6. **Feature Accentuation**: Gradient-based optimization to maximize predicted neural responses (generates synthetic stimuli)

### Module Organization

- `core/`: Core utilities for model loading, data processing, and posthoc prediction
  - `model_load_utils.py`: Unified model loading with ~10 vision models (ResNet, ViT, CLIP, DINOv2, RADIO, etc.)
  - `brainscore_model_utils.py`: Special loaders for BrainScore models (ReAlNet, AlexNet variants) with S3 downloads
  - `posthoc_prediction_utils.py`: Load saved regression models and predict responses on new images
  - `GAN_utils_hf.py`: GAN models for feature visualization (upconvGAN, Caffenet)

- `neural_regress/`: Neural regression and feature processing library
  - `regress_lib.py`: Main regression pipeline with cross-validation, feature transforms, and evaluation
  - `PCA_dual_solver_lib.py`: GPU-accelerated dual PCA solver for high-dimensional features
  - `SRP_torch_lib.py`: GPU-accelerated sparse random projection
  - `feature_reduction_lib.py`: Feature dimensionality reduction methods
  - `posthoc_analysis_lib.py`: Analysis functions for trained models

- `scripts/`: Batch processing scripts for mass production analysis
  - `neural_regression_massprod_*.py`: Run neural regression across multiple sessions/models/layers
  - `feature_accentuation_*.py`: Generate feature-accentuated stimuli
  - `posthoc_prediction_*.py`: Apply saved models to new image sets
  - `neural_regress_yaml_export.py`: Export regression models to YAML configs

- `notebooks/`: Jupyter notebooks for interactive analysis and visualization

## Common Development Commands

### Running Neural Regression

```python
# Typical workflow in notebooks or scripts
from core.model_load_utils import load_model_transform
from circuit_toolkit.layer_hook_utils import featureFetcher
from neural_regress.regress_lib import (
    transform_features2Xdict,
    sweep_regressors,
    perform_regression_sweeplayer_RidgeCV
)

# Load model
model, transforms = load_model_transform("resnet50_robust", device="cuda")

# Extract features from specific layers
fetcher = featureFetcher(model, input_size=(3, 224, 224))
fetcher.record(layer_name, ingraph=False)

# Run model on images to populate activations
model(images)
feat_dict = {layer: fetcher[layer] for layer in layers}

# Perform regression with feature reduction + cross-validation
result_df, models, Xdict, tfm_dict = perform_regression_sweeplayer_RidgeCV(
    feat_dict,
    resp_mat,  # neural responses (n_images x n_units)
    layer_names=["layer4.Bottleneck2"],
    dimred_list=["pca1000", "sp_avg", "sp_cent"],
)
```

### Loading and Using Saved Models

```python
from core.posthoc_prediction_utils import get_predictor_from_config

# Load from YAML config
pred_fn, target_unit_fn, model, transforms, fetcher, Xtransform, readout = \
    get_predictor_from_config("path/to/config.yaml")

# Predict on new images
predictions = pred_fn(images)  # Full population
unit_predictions = target_unit_fn(images)  # Specific units
```

## Model Naming Conventions

Models supported in `load_model_transform()`:
- ResNet variants: `resnet50`, `resnet50_robust`, `resnet50_clip`, `resnet50_dino`
- Vision Transformers: `clipag_vitb32`, `siglip2_vitb16`, `dinov2_vitb14_reg`, `radio_v2.5-b`
- Custom models: `ReAlnet01`-`ReAlnet10`, `AlexNet_training_seed_01`, `regnety_640`

Layer filtering patterns are defined in `MODEL_LAYER_FILTERS` dict. Use `LAYER_ABBREVIATION_MAPS` for shortened layer names.

## Feature Reduction Methods

Specified in `dimred_list` parameter:
- `pca{N}`: PCA with N components (e.g., `pca1000`)
- `srp{N}`: Sparse random projection with N components (e.g., `srp1000`)
- `sp_avg`: Spatial average pooling (for conv layers)
- `sp_cent`: Center spatial position only
- `avgtoken`: Average all tokens (for transformers)
- `clstoken`: CLS token only (for transformers)
- `full`: Flattened full feature tensor

## Data Paths and Organization

**Primary experimental data location**: `/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/`

Key directories:
- **Stimuli/**: All experimental images organized by type
  - `encodingstimuli_apr2025/`: Natural images for encoding (971 images)
  - `results_22-11-2024/`: Feature accentuated stimuli at multiple levels
  - `results_12-01-2025/`: Controversial stimuli (maximizing one model, minimizing another)
  - `encoding_stimuli_split_seed0_swapped.csv`: Metadata with train/test splits
- **NeuralData_raw/**: Neural recordings from 5 monkeys (leap, paul, red, three0, venus)
  - Naming: `{monkey}_{date}_hp{h}_bs{b}_zs{z}_rw{start}-{end}_sessdata.pkl`
- **Encoding_models/**: Saved regression models organized by monkey and date range
- **Ephys_Data/**: Processed electrophysiology data
- **Reduced_Data/**: PCA/dimensionality-reduced features

Model checkpoints and cache:
- Model checkpoints: `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/model_backbones/`
- BrainScore cache: `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/brain_score_cache/`

**See README.md for detailed data structure documentation.**

## Important Implementation Details

### GPU Acceleration
- PCA uses dual solver on GPU by default (`use_pca_dual=True` in `transform_features2Xdict`)
- SRP uses PyTorch implementation on GPU (`use_srp_torch=True`)
- Feature extraction stores activations on GPU when `store_device="cuda"` in `featureFetcher.record()`

### Regression Methods
- `RidgeCV`: Cross-validated Ridge with `alpha_per_target` for per-unit regularization
- `LassoCV`: Cross-validated Lasso (slower, use `MultiTaskLassoCV` for multi-output)
- `MultiOutputSeparateLassoCV`: Custom estimator fitting separate Lasso per output channel

### Configuration Export
Trained models can be exported to YAML configs containing:
- `model_name`, `layer_name`: Feature extraction configuration
- `xtransform_path`: Path to saved feature transform (PCA/SRP)
- `readout_path`: Path to saved linear readout weights
- `unit_ids`: Target neural unit indices
- `fit_method_name`: Regression method used

## Working with Scripts

Scripts in `scripts/` typically:
1. Load experimental data (neural recordings + image paths)
2. Extract features from multiple models/layers
3. Fit regression models with cross-validation
4. Save results (dataframes, models, predictions) to disk
5. Generate summary plots and statistics

To adapt existing scripts:
- Modify `modelname` and `layer_names` for different architectures
- Adjust `dimred_list` for different feature reductions
- Change `savedir` to control output location
- Update data paths to point to your experimental session
