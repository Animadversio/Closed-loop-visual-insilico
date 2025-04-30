import torch
import glob
from os.path import join
import sys
sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
from core.model_load_utils import load_model_transform, MODEL_LAYER_FILTERS, LAYER_ABBREVIATION_MAPS
from neural_regress.regress_lib import record_features, perform_regression_sweeplayer, perform_regression_sweeplayer_RidgeCV
from circuit_toolkit.layer_hook_utils import featureFetcher
from circuit_toolkit.dataset_utils import ImagePathDataset

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



def test_activation_recording(image_fps=None, batch_size=96):
    
    if image_fps is None:
        # stimuli_root = "/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Stimuli"
        stimuli_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/VVS_Accentuation/Stimuli/"
        imgdir_shared = join(stimuli_root, "shared1000")
        imgdir = imgdir_shared
        # find all image files in the shared1000 directory png jpg jpeg
        image_fps = glob.glob(join(imgdir, "shared*.png")) + glob.glob(join(imgdir, "fLoc*.jpg")) + glob.glob(join(imgdir, "*.jpeg"))
        # sort the image files
        image_fps.sort()
    print(f"Found {len(image_fps)} images in {imgdir}")
    print(f"First 10 images: {image_fps[:10]}")
    # List of all supported models
    
    model_names = [
        "resnet50_robust",
        "resnet50",
        "resnet50_clip",
        "resnet50_dino",
        "clipag_vitb32",
        "siglip2_vitb16",
        "dinov2_vitb14_reg",
        "radio_v2.5-b",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
        try:
            # hook the layers
            for name in module_names2record: 
                fetcher.record(name, store_device='cpu', ingraph=False, )
            # Record features
            feat_dict_lyrswp = record_features(model, fetcher, dataset, batch_size=batch_size, device=device)
            # Cleanup
            print(f"{modelname} done!!!")
            fetcher.cleanup()
        except Exception as e:
            print(f"Error recording features for {modelname}: {str(e)}")
            


if __name__ == "__main__":
    # test_model_loading()
    test_activation_recording()