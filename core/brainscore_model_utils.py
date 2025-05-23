import os
import torch
from pathlib import Path
from torchvision import transforms
import boto3

save_root = f"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/brain_score_cache"

def download_from_s3(
    bucket: str,
    key: str,
    local_path: Path,
    version_id: str = None,
):
    """
    Downloads an S3 object to local_path (creating parents if needed),
    optionally at a specific version.
    """
    s3 = boto3.client("s3")
    # ensure local directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    extra_args = {}
    if version_id:
        extra_args["VersionId"] = version_id

    s3.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(local_path),
        ExtraArgs=extra_args  # only passes VersionId if given
    )
    return local_path



def build_ReAlnet_model(identifier: str = "ReAlnet01",
                        save_root: str = save_root):
    """
    Build a model from a given identifier.
    """
    import sys
    sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
    from core.ReAlNet import CORnet_S, Encoder
    
    local_weights_path = f"{save_root}/{identifier}_best_model_params.pt"
    realnet = CORnet_S()
    # (Optional) remove DataParallel if not needed for CPU
    # realnet = torch.nn.DataParallel(realnet)
    # Build encoder model
    encoder = Encoder(realnet, 340)
    
    weights_info = {
        "version_ids": {
        "ReAlnet01": "75oY3CnI17U5S1f_yrZxl1XGhRfJEG9N",
        "ReAlnet02": "TfGdm1CphJJ1vvkJGcm3n266PHvTuOaV",
        "ReAlnet03": "dmohrH_AHZzgL_o8Xd2SDp6XCnjPOdAu",
        "ReAlnet04": "45qJFXHihmIHdpHbjKWZco6STH1eh49p",
        "ReAlnet05": "nqvoYgiBTyWSskjnpF9YOK4yYQfOnc_H",
        "ReAlnet06": "6.cloFvnMihiicwQ0jkag8reEe4bVlxZ",
        "ReAlnet07": "WKJaiN4b1ttpbGYNn8yVjng4LjCqWdk.",
        "ReAlnet08": "vmouew6ePkPnKP.We8VnVxU7TifuhL.x",
        "ReAlnet09": "53gqQ2tgS.5MEoncipy9mrBEqCc5izw5",
        "ReAlnet10": "ZZFMhTm9KQYEXl8OGwKmnTr0S.pxkU0J"
        },
        "sha1s": {
        "ReAlnet01": "05e4e401e8734b97e561aad306fc584b7e027225",
        "ReAlnet02": "e85769fadb3c09ff88a7d73b01451b6bcccefd77",
        "ReAlnet03": "f32d01d73380374ae501a1504e9c8cd219e9f0bf",
        "ReAlnet04": "8062373fd6a74c52360420619235590d3688b4df",
        "ReAlnet05": "88ca110f6b6d225b7b4e7dca02d2e7a906f5a8ed",
        "ReAlnet06": "a1658c15a3c9d61262f87349c9fb7aa63854ac5b",
        "ReAlnet07": "6a1c260839c75f6e6c018e06830562cdcda877e5",
        "ReAlnet08": "1772211b27dd3a7d9255ac59d5f9b7e7cb6c3314",
        "ReAlnet09": "159d96f0433a87c7063259dac4527325a3c7b79a",
        "ReAlnet10": "dbdeaee9280267613ebce92dd5d515d89b544352"
        }
    }
    version_id = weights_info['version_ids'][identifier]
    sha1 = weights_info['sha1s'][identifier]
    if not os.path.exists(local_weights_path):
        key = f"brainscore-vision/models/ReAlnet/{identifier}_best_model_params.pt"
        download_from_s3(bucket="brainscore-storage", key=key, 
                         local_path=Path(local_weights_path), version_id=version_id)
    
    weights_data = torch.load(local_weights_path, map_location="cpu")
    new_state_dict = {}
    for key, val in weights_data.items():
        # remove "module." (if it exists) from the key
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = val

    encoder.load_state_dict(new_state_dict)
    # Retrieve the realnet portion from the encoder
    realnet = encoder.realnet
    realnet.eval()
    realnet.requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return realnet, preprocess


def build_alexnet_brainscore(identifier: str = "training_seed_01",
                             save_root: str = save_root):
    """
    Build a model from a given identifier.
    """
    import sys
    sys.path.append("/n/home12/binxuwang/Github/Closed-loop-visual-insilico")
    from core.alexnet_torch import alexnet_v2_pytorch
    
    model = alexnet_v2_pytorch()
    local_weights_path = f"{save_root}/alexnet_weights_{identifier}.pth"
    if not os.path.exists(local_weights_path):
        # sha1 = "4b1bb7810d5288631c04cf4cde882540d6ebee77"
        key = f"models/model_weights/{identifier}.pth"
        download_from_s3(bucket="brainscorevariability", key=key, 
                        local_path=Path(local_weights_path), version_id=None)
    model_weights = torch.load(local_weights_path, map_location='cpu') 
    model.load_state_dict(model_weights)
    model.eval()
    model.requires_grad_(False)
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return model, preprocess

if __name__ == "__main__":
    model, preprocess = build_ReAlnet_model(identifier="ReAlnet01")
    print(model)
    print(preprocess)
    model, preprocess = build_alexnet_brainscore(identifier="training_seed_01")
    print(model)
    print(preprocess)