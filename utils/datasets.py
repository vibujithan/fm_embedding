import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

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
except ImportError:
    print("Error: MONAI is required but not installed.")
    print("Please install MONAI: pip install monai==1.3.2")


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


class TorchDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, index):
        file_path = os.path.join(self.data_path, str(index))
        return torch.load(file_path)


def create_pd_data_loaders(data_dir, batch_size=8, num_workers=4):
    """Create data loaders for PD dataset splits"""

    # Define paths
    train_csv = os.path.join(data_dir, "splits", "train.csv")
    val_csv = os.path.join(data_dir, "splits", "val.csv")
    test_csv = os.path.join(data_dir, "splits", "test.csv")
    images_dir = os.path.join(data_dir, "raw", "images")

    # Create datasets
    train_dataset = PDDataset(train_csv, images_dir)
    val_dataset = PDDataset(val_csv, images_dir)
    test_dataset = PDDataset(test_csv, images_dir)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
