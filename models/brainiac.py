import torch
import numpy as np
import pandas as pd
import random
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# MONAI imports for image processing
try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Resized,
        NormalizeIntensityd,
        ToTensord,
    )
    from monai.networks.nets import ViT
except ImportError:
    print("Error: MONAI is required but not installed.")
    print("Please install MONAI: pip install monai==1.3.2")
    exit(1)

# Fix random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =========================
# Dataset Classes and Transforms
# =========================


def get_validation_transform(image_size=(96, 96, 96)):
    """Get validation transforms for feature extraction (no augmentation)"""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )


class PDDataset(Dataset):
    """Dataset class for Parkinson's Disease classification and feature extraction"""

    def __init__(self, csv_path, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_path, dtype={"Subject": str})
        self.root_dir = root_dir
        self.transform = (
            transform if transform is not None else get_validation_transform()
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        subject_id = str(self.dataframe.loc[idx, "Subject"])
        disease_status = self.dataframe.loc[idx, "Group"]  # 1=PD, 0=HC
        sex = self.dataframe.loc[idx, "Sex"]  # 1=M, 0=F
        study = self.dataframe.loc[idx, "Site"]  # Study ID
        scanner_type = self.dataframe.loc[idx, "Scanner"]  # Scanner type ID

        # Construct image path for PD dataset format
        img_path = os.path.join(self.root_dir, subject_id + ".nii.gz")
        sample = {"image": img_path}
        sample = self.transform(sample)
        return {
            "image": sample["image"],
            "disease_status": torch.tensor(disease_status, dtype=torch.int32),
            "sex": torch.tensor(sex, dtype=torch.int32),
            "study": torch.tensor(study, dtype=torch.int32),
            "scanner_type": torch.tensor(scanner_type, dtype=torch.int32),
        }


# =========================
# Model Classes
# =========================


class ViTBackboneNet(nn.Module):
    """ViT backbone network for feature extraction"""

    def __init__(self, simclr_ckpt_path):
        super(ViTBackboneNet, self).__init__()

        # Create ViT backbone with same architecture as SimCLR
        self.backbone = ViT(
            in_channels=1,  # For single channel input
            img_size=(96, 96, 96),  # Adjust this to your input dimensions
            patch_size=(16, 16, 16),
            hidden_size=768,  # Standard for ViT-B
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=True,
        )

        # Load pretrained weights from SimCLR checkpoint
        ckpt = torch.load(simclr_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Extract only backbone weights from SimCLR checkpoint
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # Remove "backbone." prefix
                new_key = key[9:]  # len("backbone.") = 9
                backbone_state_dict[new_key] = value

        # Load the backbone weights
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        print("Backbone weights loaded!!")

    def forward(self, x):
        # Get features from ViT backbone
        features = self.backbone(x)

        # Use CLS token (first token) as global representation
        # features[0] shape: [batch_size, num_tokens, hidden_dim]
        # features[0][:, 0] gets CLS token: [batch_size, hidden_dim]
        cls_token = features[0][:, 0]  # Shape: [batch_size, 768]

        return cls_token


# =========================
# Model Loading Function
# =========================


def load_brainiac(checkpoint_path, device="cuda"):
    """
    Load the ViT backbone model and BrainIAC checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded ViT backbone model with checkpoint weights
    """
    # Create ViT backbone model - the constructor handles checkpoint loading
    model = ViTBackboneNet(checkpoint_path)

    # Move model to specified device
    model = model.to(device)

    return model


def infer(model, test_loader):
    """Extract features from the model with multiple metadata keys"""
    features_df = None  # Placeholder for feature DataFrame
    model.eval()

    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Extracting ViT features", unit="batch"):
            inputs = sample["image"].to(device)
            disease_status = sample["disease_status"].int().to(device)
            sex = sample["sex"].int().to(device)
            study = sample["study"].int().to(device)
            scanner_type = sample["scanner_type"].int().to(device)

            # Get features from the ViT backbone model
            features = model(inputs)
            features_numpy = features.cpu().numpy()

            # Expand features into separate columns
            feature_columns = [f"Feature_{i}" for i in range(features_numpy.shape[1])]
            batch_features = pd.DataFrame(features_numpy, columns=feature_columns)

            # Add metadata columns
            batch_features["disease_status"] = (
                disease_status.cpu().numpy().flatten().astype(int)
            )
            batch_features["sex"] = sex.cpu().numpy().flatten().astype(int)
            batch_features["study"] = study.cpu().numpy().flatten().astype(int)
            batch_features["scanner_type"] = (
                scanner_type.cpu().numpy().flatten().astype(int)
            )

            # Append batch features to features_df
            if features_df is None:
                features_df = batch_features
            else:
                features_df = pd.concat(
                    [features_df, batch_features], ignore_index=True
                )

    return features_df
