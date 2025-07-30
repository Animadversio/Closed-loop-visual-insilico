import os
import sys
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
import yaml
import re
import glob
import timm
import torch
import torch as th
import torch.nn as nn
from tqdm.auto import tqdm
from os.path import join
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from circuit_toolkit.CNN_scorers import TorchScorer
from circuit_toolkit.GAN_utils import upconvGAN, Caffenet
from circuit_toolkit.plot_utils import to_imgrid, show_imgrid, save_imgrid, saveallforms
from circuit_toolkit.layer_hook_utils import featureFetcher_module, featureFetcher, get_module_names
from circuit_toolkit.dataset_utils import ImagePathDataset, DataLoader
from neural_regress.regress_lib import sweep_regressors, perform_regression_sweeplayer_RidgeCV, perform_regression_sweeplayer, record_features
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, LinearRegression_torch, SpatialAvg_torch, LinearLayer_from_sklearn
from core.data_utils import load_neural_data, load_from_hdf5, load_neural_trial_resp_tensor, create_response_tensor, parse_image_fullpaths
from core.model_load_utils import load_model_transform
from horama import maco, plot_maco
from sklearn.metrics.pairwise import cosine_similarity

def check_gradient(objective_fn):
    """Check if gradients can flow through the objective function."""
    img_opt = th.randn(1, 3, 224, 224).cuda()
    img_opt.requires_grad_(True)
    resp = objective_fn(img_opt)
    resp.mean().backward()
    print(resp.shape)
    assert img_opt.grad is not None


def get_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform             {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path).to(device)
    Xtransform = th.load(xtransform_path).to(device)
    def predict_population_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)
    
    
    def predict_target_unit_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)[:, unit_ids]

    check_gradient(predict_population_response)
    print("Gradient check passed!")
    return predict_population_response, predict_target_unit_response, \
           model, transforms_pipeline, fetcher, Xtransform, readout
           

def get_unit_population_PCA_basis_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path).to(device)
    Xtransform = th.load(xtransform_path).to(device)
    def predict_population_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)
    
    
    def predict_target_unit_response(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return readout(feat_vec)[:, unit_ids]


    def predict_reduced_feature(images):
        """Predict PCA features responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted PCA features (no readout)
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return feat_vec
    
    check_gradient(predict_reduced_feature)
    print("Gradient check passed!")
    return predict_reduced_feature, predict_population_response, predict_target_unit_response, \
           model, transforms_pipeline, fetcher, Xtransform, readout


def get_PCA_basis_predictor_from_config(acc_config, device="cuda"):
    """Create a function that predicts neural population responses for images.
    
    Args:
        subject_id (str): ID of the subject
        modelname (str): Name of the model to use (e.g. "resnet50_robust") 
        layer_name (str): Name of layer to extract features from
        device (str): Device to run model on ("cuda" or "cpu")
        
    Returns:
        function: A function that takes images as input and returns predicted population responses
    """
    if isinstance(acc_config, str):
        acc_config = yaml.safe_load(open(acc_config))
        
    # Construct paths
    xtransform_path = acc_config['xtransform_path']
    meta_path = acc_config['meta_path']
    readout_path = acc_config['readout_path']
    model_name = acc_config['model_name']
    unit_ids = acc_config['unit_ids']
    layer_name = acc_config['layer_name']
    regressor = acc_config['fit_method_name']
    print(f"Loading model and set up feature extraction: model {model_name}, layer {layer_name}")
    print(f"Target unit ids: {unit_ids}")
    print(f"Loading readout layer and PCA transform: readout {readout_path}, xtransform {xtransform_path}")
    # Load model and set up feature extraction
    model, transforms_pipeline = load_model_transform(model_name, device=device)
    model = model.eval().to(device)
    model.requires_grad_(False)
    fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
    fetcher.record(layer_name, ingraph=True, store_device=device)
    # Load readout layer and PCA transform
    readout = th.load(readout_path).to(device)
    Xtransform = th.load(xtransform_path).to(device)
    def predict_reduced_feature(images):
        """Predict neural population responses for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted neural responses
        """
        model(images)
        feat_tsr = fetcher[layer_name]
        feat_vec = Xtransform(feat_tsr)
        return feat_vec
    
    check_gradient(predict_reduced_feature)
    print("Gradient check passed!")
    return predict_reduced_feature, \
           model, transforms_pipeline, fetcher, Xtransform, readout
           

def get_prediction_responses(pred_fn, transforms_pipeline, image_fps, device="cuda", batch_size=100, num_workers=16):
    """
    Get the prediction responses for a set of images
    """
    dataset = ImagePathDataset(image_fps, transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_resp = []
    for batch in tqdm(dataloader):
        batch_img = batch[0].to(device)
        with th.no_grad():
            batch_resp = pred_fn(batch_img).cpu()
        pred_resp.append(batch_resp)
    return torch.cat(pred_resp, dim=0)


def find_sort_png_files_by_subfolder(target_subfolder, suffix="png"):
    """
    List all png files in each subfolder (one level deep)
    """
    all_png_files = {}
    for subfolder in target_subfolder:
        subfolder_name = os.path.basename(subfolder)
        png_files = glob.glob(join(subfolder, f"*.{suffix}"))
        # Sort the files to ensure consistent order
        png_files.sort()
        all_png_files[subfolder_name] = png_files
        print(f"Found {len(png_files)} {suffix} files in {subfolder_name}")
        # Print first few files as example
        for file in png_files[:5]:
            print(f"  - {os.path.basename(file)}")
        if len(png_files) > 5:
            print(f"  - ... and {len(png_files)-5} more files")
        print()
    return all_png_files



def parse_accentuated_filename(filename):
    """
    Parse accentuated image filenames to extract metadata.
    
    Example filename format:
    radio_v2.5-b_RidgeCV_unit_0_img_0_level_0.580872533679007_score_0.5726814270019531.png
    
    Returns:
        dict: Dictionary containing extracted metadata (model_name, unit_id, img_id, level, score)
    """
    # Get just the filename without path
    basename = os.path.basename(filename)
    # Extract components using regex
    pattern = r'(.+?)_RidgeCV_unit_(\d+)_img_(\d+)_level_([-\d\.]+)_score_([-\d\.]+)\.png'
    match = re.search(pattern, basename)
    if match:
        model_name, unit_id, img_id, level, score = match.groups()
        return {
            'model_name': model_name,
            'unit_id': int(unit_id),
            'img_id': int(img_id),
            'level': float(level),
            'score': float(score),
            'filepath': filename
        }
    else:
        return None

def parse_accentuated_filenames_to_df(filenames):
    """
    Parse a list of accentuated image filenames and return as a DataFrame.
    
    Args:
        filenames (list): List of filenames to parse
        
    Returns:
        pd.DataFrame: DataFrame containing parsed metadata for all files
    """
    parsed_data = []
    for filename in filenames:
        parsed = parse_accentuated_filename(filename)
        if parsed:
            parsed_data.append(parsed)
    
    if parsed_data:
        return pd.DataFrame(parsed_data)
    else:
        return pd.DataFrame()


acc_stim_root = r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs"
config_root = r"/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_configs"
subject_id = "red_20250428-20250430"
    
for subject_id, filename in [
    ("red_20250428-20250430", "red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    ("paul_20250428-20250430", "paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5"), 
    ("venus_250426-250429", "venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5"),
    ("three0_250426-250501", "three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
    ("leap_250426-250501", "leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5"),
]:
    posthoc_PCA_dir = f"/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models/{subject_id}/posthoc_model_predict_PCA_popul_unit"
    os.makedirs(posthoc_PCA_dir, exist_ok=True)
    # config_files
    target_subfolder = glob.glob(join(acc_stim_root, f"*{subject_id}*_accentuation"))
    all_png_files = find_sort_png_files_by_subfolder(target_subfolder)
    png_files_list = sum(all_png_files.values(), [])
    df_accentuated = parse_accentuated_filenames_to_df(png_files_list)
    df_accentuated.to_pickle(join(posthoc_PCA_dir, f"df_accentuated_{subject_id}.pkl"))
    config_dir = join(config_root, subject_id)
    config_files = sorted(glob.glob(join(config_dir, "*.yaml")))
    chan_pattern = "_Ch19_"
    config_pre_chan = [f for f in config_files if chan_pattern in f]
    # cosine_results = []
    for config_file in config_pre_chan:
        config_acc = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        model_name = config_acc["model_name"]
        unit_id = config_acc["unit_ids"][0]
        layer_name = config_acc["layer_name"]
        print(f"Model: {model_name}, Unit: {unit_id}, Layer: {layer_name}")
        acc_split_df = df_accentuated.copy()
        # acc_split_df = df_accentuated.query("model_name == @model_name and unit_id == @unit_id")
        # assert len (acc_split_df) == 110 

        predict_PCA_feature, predict_population_response, predict_target_unit_response, \
            model, transforms_pipeline, fetcher, Xtransform, readout \
                = get_unit_population_PCA_basis_predictor_from_config(config_file, device="cuda")
        # accentuated_dataset = ImagePathDataset(acc_split_df["filepath"], transform=transforms_pipeline)
        # accentuated_dataloader = DataLoader(accentuated_dataset, batch_size=60, shuffle=False, num_workers=10)
        acc_img_PCA_resp = get_prediction_responses(predict_PCA_feature, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        acc_img_population_resp = get_prediction_responses(predict_population_response, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        acc_model_target_unit_resp = get_prediction_responses(predict_target_unit_response, transforms_pipeline, 
                                            acc_split_df["filepath"].tolist(), batch_size=120, num_workers=16)
        readout_vec = readout.weight.data[unit_id, :].cpu()
        readout_bias = readout.bias.data[unit_id].cpu()
        PCA_norm = acc_img_PCA_resp.norm(dim=1)
        cosine_sims = cosine_similarity(acc_img_PCA_resp.cpu().numpy(), readout_vec.numpy().reshape(1, -1))
        cosine_sims = cosine_sims.flatten() 
        acc_split_df_cos = acc_split_df.copy()
        acc_split_df_cos["cosine_similarity"] = cosine_sims
        acc_split_df_cos["PCA_norm"] = PCA_norm
        
        print(f"Min cosine similarity: {cosine_sims.min():.4f}")
        print(f"Max cosine similarity: {cosine_sims.max():.4f}")
        print(f"PCA norm max {PCA_norm.max():.4f}")
        print(f"PCA norm min {PCA_norm.min():.4f}")
        
        stats_dict = {"config": config_acc, 
                            "df": acc_split_df_cos.copy(), 
                            "PCA_resp": acc_img_PCA_resp, 
                            "population_resp": acc_img_population_resp,
                            "target_unit_resp": acc_model_target_unit_resp,
                            "readout_vec": readout_vec, 
                            "readout_bias": readout_bias,
                            "cosine_sims": cosine_sims, 
                            "PCA_norm": PCA_norm}
        pkl.dump(stats_dict, open(join(posthoc_PCA_dir, f"posthoc_prediction_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl"), "wb"))
        print(f"Saved to {join(posthoc_PCA_dir, f'posthoc_prediction_PCA_pop_unit_{subject_id}_unit{unit_id}_{model_name}.pkl')} completed!")

        
        

