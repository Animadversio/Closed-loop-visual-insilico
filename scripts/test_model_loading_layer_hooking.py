import torch
import sys
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from core.model_load_utils import load_model_transform, MODEL_LAYER_FILTERS, LAYER_ABBREVIATION_MAPS
from circuit_toolkit.layer_hook_utils import featureFetcher

def test_model_loading():
    # List of all supported models
    model_names = [
        "resnet50_robust",
        "resnet50",
        "resnet50_clip",
        "resnet50_dino",
        "clipag_vitb32",
        "siglip2_vitb16",
        "dinov2_vitb14_reg",
        "radio_v2.5-b"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for modelname in model_names:
        print(f"\nTesting model: {modelname}")
        try:
            # Load model and transforms
            model, transforms = load_model_transform(modelname, device)
            print(f"Successfully loaded model: {modelname}")
            
            # Create feature fetcher
            fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
            
            # Get all module names
            all_module_names = list(fetcher.module_names.values())
            print(f"Total number of modules: {len(all_module_names)}")
            
            # Get layer filter for this model
            layer_filter = MODEL_LAYER_FILTERS[modelname]
            layer_abbrev = LAYER_ABBREVIATION_MAPS[modelname]
            # Count layers that pass the filter
            filtered_module_names = [name for name in all_module_names if layer_filter(name)]
            print(f"Number of layers passing filter: {len(filtered_module_names)}")
            print(f"Filtered layer names: {filtered_module_names}")
            filtered_module_names_abbrev = [layer_abbrev(name) for name in filtered_module_names]
            print(f"Filtered layer names (abbreviated): {filtered_module_names_abbrev}")
            
            # Cleanup
            fetcher.cleanup()
            
        except Exception as e:
            print(f"Error loading model {modelname}: {str(e)}")

def test_activation_recording(datapath=None):
    if datapath is None:
        datapath = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data/vvs-accentuate-day1_normalize_red_20241212-20241220.h5"
    data = load_from_hdf5(datapath)
    
    # List of all supported models
    model_names = [
        "resnet50_robust",
        "resnet50",
        "resnet50_clip",
        "resnet50_dino",
        "clipag_vitb32",
        "siglip2_vitb16",
        "dinov2_vitb14_reg",
        "radio_v2.5-b"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    for modelname in model_names:
        print(f"\nTesting model: {modelname}")
        try:
            # Load model and transforms
            model, transforms = load_model_transform(modelname, device)
            print(f"Successfully loaded model: {modelname}")
            
            # Create feature fetcher
            fetcher = featureFetcher(model, input_size=(3, 224, 224), print_module=False)
            
            # Get all module names
            all_module_names = list(fetcher.module_names.values())
            print(f"Total number of modules: {len(all_module_names)}")
            
            # Get layer filter for this model
            layer_filter = MODEL_LAYER_FILTERS[modelname]
            layer_abbrev = LAYER_ABBREVIATION_MAPS[modelname]
            # Count layers that pass the filter
            filtered_module_names = [name for name in all_module_names if layer_filter(name)]
            print(f"Number of layers passing filter: {len(filtered_module_names)}")
            print(f"Filtered layer names: {filtered_module_names}")
            filtered_module_names_abbrev = [layer_abbrev(name) for name in filtered_module_names]
            print(f"Filtered layer names (abbreviated): {filtered_module_names_abbrev}")
            
            # Cleanup
            fetcher.cleanup()
            
        except Exception as e:
            print(f"Error loading model {modelname}: {str(e)}")


if __name__ == "__main__":
    test_model_loading()
