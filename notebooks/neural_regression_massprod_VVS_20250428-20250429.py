# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import sys
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
import timm
import time
import torch
import torch as th
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.auto import tqdm
from os.path import join
import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from horama import maco, plot_maco
import torchvision.transforms as T
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from torchvision.models import resnet50
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.GAN_utils import upconvGAN, Caffenet
from circuit_toolkit.plot_utils import to_imgrid, show_imgrid, save_imgrid, saveallforms
from circuit_toolkit.layer_hook_utils import featureFetcher_module, featureFetcher, get_module_names
from circuit_toolkit.dataset_utils import ImagePathDataset
from torch.utils.data import DataLoader
from neural_regress.regress_lib import sweep_regressors
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, LinearRegression_torch, SpatialAvg_torch

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
# %%
import sys
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from core.model_load_utils import load_model_transform, MODEL_LAYER_FILTERS, LAYER_ABBREVIATION_MAPS
from neural_regress.feature_reduction_lib import FEATURE_REDUCTION_DEFAULTS, LAYER_TRANSFORM_FILTERS
from neural_regress.regress_lib import record_features, perform_regression_sweeplayer, perform_regression_sweeplayer_RidgeCV
from neural_regress.regress_lib import sweep_regressors, transform_features2Xdict, RidgeCV
from neural_regress.regress_eval_lib import format_result_df, plot_result_df_per_layer, construct_result_df_masked, \
    compute_pred_dict_D2_per_unit
#%% Utility Functions
from core.data_utils import load_from_hdf5, load_neural_data, load_neural_trial_resp_tensor, create_response_tensor, parse_image_fullpaths

dataroot = r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data"
data_path = join(dataroot, "red_20250428-20240429_vvs-encodingstimuli_z1_rw100-400.h5")
data = load_from_hdf5(data_path)
subject_id = "red_20250428-20240429"

encoding_stim_dir = r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli/encodingstimuli_apr2025"
data_dict = {}
data_dict['image_fps'] = parse_image_fullpaths(data['repavg']["stimulus_name"], [encoding_stim_dir], arbitrary_format=True)
data_dict['resp_mat'] = data['repavg']["response_peak"]
data_dict['resp_temp_mat'] = data['repavg']["response_temporal"]
data_dict['reliability'] = data['neuron_metadata']["reliability"]
data_dict['ncsnr'] = data['neuron_metadata']["ncsnr"]
data_dict['brain_area'] = data['neuron_metadata']["brain_area"]
data_dict['stim_pos'] = data['stimulus_meta']['xy_deg']
data_dict['stim_size'] = data["stimulus_meta"]["size_px"]

image_fps = data_dict['image_fps']
resp_mat = data_dict['resp_mat']
reliability = data_dict['reliability']
ncsnr = data_dict['ncsnr']

df_stim = pd.read_csv(join(encoding_stim_dir, "encoding_stimuli_split_seed0.csv"), )
train_idx = df_stim[df_stim["is_train"]].index

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 96

model_names = [
    "dinov2_vitb14_reg",
    "clipag_vitb32",
    "siglip2_vitb16",
    "radio_v2.5-b",
    "resnet50_robust",
    "resnet50_clip",
    "resnet50_dino",
    "resnet50",
]


def apply_feature_transforms(feat_dict_lyrswp, module_names2record, dimred_transform_dict, 
                            layer_transform_filter=None):
    """Apply feature transforms to each layer using the provided transform dictionary.
    
    Args:
        feat_dict_lyrswp: Dictionary of features for each layer
        module_names2record: List of layer names to process
        dimred_transform_dict: Dictionary of transform modules
        layer_transform_filter: Optional function that takes (layer_name, transform_name) 
                               and returns True if the transform should be applied to that layer
        
    Returns:
        Xdict_lyrswp: Dictionary of transformed features
        Xtfmer_lyrswp: Dictionary of transform modules used
    """
    Xdict_lyrswp = {}
    Xtfmer_lyrswp = {}
    
    # Default filter that allows all combinations
    if layer_transform_filter is None:
        # For backward compatibility with siglip2_vitb16 case
        layer_transform_filter = lambda layer, dimred_str: not (("attn_pool" in layer) != ("full" in dimred_str))
    
    for layer in module_names2record:
        for dimred_str, transform_module in dimred_transform_dict.items():
            # Skip if the filter returns False
            if not layer_transform_filter(layer, dimred_str):
                continue
                
            t0 = time.time()
            Xdict_lyrswp[f"{layer}_{dimred_str}"] = transform_module(feat_dict_lyrswp[layer])
            Xtfmer_lyrswp[f"{layer}_{dimred_str}"] = transform_module
            print(f"Time taken to transform {layer} x {dimred_str}: {time.time() - t0:.3f}s")
    
    return Xdict_lyrswp, Xtfmer_lyrswp


outputroot = r"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"
output_name = "model_outputs"
figdir = join(outputroot, subject_id, output_name)
os.makedirs(figdir, exist_ok=True)
for modelname in model_names:
    print(f"\nTesting model: {modelname}")
    try:
        # Load model and transforms
        model, transforms_pipeline = load_model_transform(modelname, device)
        print(f"Successfully loaded model: {modelname}")
        dataset = ImagePathDataset(image_fps, scores=None, transform=transforms_pipeline)
        print(f"Successfully created dataset for {modelname} with {len(dataset)} images")
        # Create feature fetcher
        fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
        print(f"Successfully created feature fetcher for {modelname}")
        # Get all module names
        all_module_names = list(fetcher.module_names.values())
        print(f"Total number of modules: {len(all_module_names)}")
        # Get layer filter for this model
        layer_filter = MODEL_LAYER_FILTERS[modelname]
        layer_abbrev = LAYER_ABBREVIATION_MAPS[modelname]
        # Count layers that pass the filter
        module_names2record = [name for name in all_module_names if layer_filter(name)]
        print(f"Number of layers passing filter: {len(module_names2record)}")
        print(f"Filtered layer names: {module_names2record}")
        module_names2record_abbrev = [layer_abbrev(name) for name in module_names2record]
        print(f"Filtered layer names (abbreviated): {module_names2record_abbrev}")
    except Exception as e:
        print(f"Error loading model {modelname}: {str(e)}")
        continue
    
    # # hook the layers
    for name in module_names2record: 
        fetcher.record(name, store_device='cpu', ingraph=False, )
    
    # Record features
    feat_dict_lyrswp = record_features(model, fetcher, dataset, batch_size=batch_size, device=device)
    # Cleanup
    print(f"{modelname} done!!!")
    
    fetcher.cleanup()
    print(f"Applying feature reduction: {dimred_list}")
    if modelname in FEATURE_REDUCTION_DEFAULTS and ("resnet" not in modelname):
        dimred_transform_dict = FEATURE_REDUCTION_DEFAULTS[modelname](model)
        layer_transform_filter = LAYER_TRANSFORM_FILTERS[modelname]
        dimred_list = list(dimred_transform_dict.keys())
        Xdict_lyrswp, Xtfmer_lyrswp = apply_feature_transforms(
            feat_dict_lyrswp, module_names2record, dimred_transform_dict, layer_transform_filter)
    else:
        dimred_list = ["pca750", "srp", ]
        Xdict_lyrswp, Xtfmer_lyrswp = transform_features2Xdict(feat_dict_lyrswp, module_names2record, 
                            dimred_list=dimred_list, pretrained_Xtransforms={}, #  "srp"
                            use_pca_dual=True, use_srp_torch=True, train_split_idx=train_idx)
    
    print(f"{len(Xdict_lyrswp)} features computed! ")
    
    resp_mat_sel = resp_mat[:, :] # Select all channels, no mask
    print(f"Fitting models for All channels N={resp_mat_sel.shape[1]}")
    regressors = [RidgeCV(alphas=[1E-4, 1E-3, 1E-2, 1E-1, 1, 10, 100, 1E3, 1E4, 1E5, 1E6, 1E7, 1E8, 1E9], 
                        alpha_per_target=True,),
                # MultiTaskLassoCV(cv=5, n_alphas=100, n_jobs=-1, max_iter=10000, tol=1E-4), 
                # MultiOutputSeparateLassoCV(cv=5, n_alphas=100, n_jobs=-1, max_iter=10000, tol=1E-4), 
                ] 
    regressor_names = ["RidgeCV"]
    print(f"Sweeping regressors: {regressor_names}")
    result_df_lyrswp, fit_models_lyrswp = sweep_regressors(Xdict_lyrswp, resp_mat_sel, regressors, regressor_names, 
                                                        verbose=True, train_split_idx=train_idx)
    pred_D2_dict = compute_pred_dict_D2_per_unit(fit_models_lyrswp, Xdict_lyrswp, resp_mat_sel)
    pkl.dump(pred_D2_dict, open(join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_pred_meta.pkl"), "wb"))
    result_df_lyrswp.to_csv(join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_sweep_RidgeCV.csv"))
    th.save(fit_models_lyrswp, join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_fitmodels_RidgeCV.pth")) 
    # th.save(Xtfmer_lyrswp, join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_Xtfmer_RidgeCV.pth"))
    # pkl.dump(Xtfmer_lyrswp, open(join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_Xtfmer_RidgeCV.pkl"), "wb"))
    # %%
    figh = plot_result_df_per_layer(result_df_lyrswp, dimred_list=dimred_list, shorten_func=layer_abbrev, sharey=True, grid=True)
    figh.suptitle(f"{subject_id} {modelname} layer sweep")
    figh.tight_layout()
    figh.show()
    saveallforms(figdir, f"{subject_id}_{modelname}_layer_sweep_synopisis", figh=figh)
    # %%
    # Mask out unreliable channels and plot again
    for thresh in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        channel_count = (reliability > thresh).sum()
        result_df_masked = construct_result_df_masked(pred_D2_dict['D2_per_unit_train_dict'], 
                                                    pred_D2_dict['D2_per_unit_test_dict'], 
                                                    mask=reliability > thresh)
        figh = plot_result_df_per_layer(result_df_masked, dimred_list=dimred_list, shorten_func=layer_abbrev, sharey=True, grid=True)
        figh.suptitle(f"{subject_id} {modelname} layer sweep | reliable channels > {thresh} (N={channel_count})")
        figh.tight_layout()
        figh.show()
        saveallforms(figdir, f"{subject_id}_{modelname}_layer_sweep_synopisis_reliable_thresh{thresh}_masked", figh=figh)
    plt.close("all")
    if "resnet50" in modelname:
        # if so, save the sparsified Xtfmer_lyrswp, not every layer 
        Xtfmer_lyrswp_sparse = {k: v for k, v in Xtfmer_lyrswp.items() if "layer4" in k and "pca" in k}
        th.save(Xtfmer_lyrswp_sparse, join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_Xtfmer_RidgeCV.pth"))
    try:
        pkl.dump(Xtfmer_lyrswp, open(join(figdir, f"{subject_id}_{modelname}_sweep_regressors_layers_Xtfmer_RidgeCV.pkl"), "wb"))
    except:
        pass
    
# %%



