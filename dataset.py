import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

from transforms import Compose, get_train_transforms, get_val_transforms


class DrivingDataset(Dataset):
    """
    A PyTorch Dataset that loads driving images and their steering angles.

    Parameters
    ----------
    image_indices : np.ndarray
        Array of image index numbers (filenames without extension).
    steering_angles : np.ndarray
        Corresponding steering command for each image.
    image_dir : str
        Directory containing JPEG images named "<index>.jpg".
    transform : Compose or None
        Preprocessing / augmentation pipeline to apply.
    """

    def __init__(
        self,
        image_paths: np.ndarray,
        steering_angles: np.ndarray,
        transform: Compose | None = None,
    ):
        # Store as plain Python/NumPy — no GPU memory needed yet.
        #   image_paths:     full paths to each JPEG file.
        #   steering_angles: the corresponding steering command.
        self.image_paths = image_paths
        self.steering_angles = steering_angles
        self.transform = transform

    # ------------------------------------------------------------------
    # __len__  — REQUIRED.  How many samples in this dataset?
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.steering_angles)

    # ------------------------------------------------------------------
    # __getitem__ — REQUIRED.  Load and return ONE sample.
    #
    # This is called by DataLoader.  It can be called from multiple
    # worker processes in parallel, so avoid shared mutable state.
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        # 1) Get the full file path
        image_path = self.image_paths[idx]

        # 2) Load the image using OpenCV (returns BGR uint8)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # 3) Get the steering angle (our regression target)
        angle = float(self.steering_angles[idx])

        # 4) Apply transforms (augmentation + preprocessing)
        if self.transform is not None:
            image, angle = self.transform(image, angle)

        # 5) Convert to PyTorch tensors
        #    image is already (C, H, W) float32 from ToFloatTensor.
        #    angle needs to be a 1-D tensor so it matches model output (batch, 1).
        image_tensor = torch.from_numpy(image)               # shape: (3, H, W)
        angle_tensor = torch.tensor([angle], dtype=torch.float32)  # shape: (1,)

        return image_tensor, angle_tensor


# ======================================================================
#  FACTORY: build DataLoaders from the CSV file
# ======================================================================

def _load_all_sessions(cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine data from ALL driving sessions.

    Returns
    -------
    image_paths     : np.ndarray of str   — full path to each JPEG
    steering_angles : np.ndarray of float  — corresponding steering cmd

    WHY COMBINE?
    Each .avi is a different driving session (different day, route, or
    conditions).  Combining them gives the model more variety, and
    letting sklearn split across sessions ensures that train/val/test
    contain a mix of all conditions.
    """
    all_paths = []
    all_steers = []

    for name in cfg.dataset_names:
        csv_path = str(Path(cfg.root_path) / f"{name}_keras.csv")
        image_dir = str(Path(cfg.root_path) / name)

        df = pd.read_csv(csv_path)
        paths = [str(Path(image_dir) / f"{idx}.jpg") for idx in df["image_idx"].values]
        steers = df["steer_cmd"].values.tolist()

        all_paths.extend(paths)
        all_steers.extend(steers)
        print(f"  Loaded {name}: {len(paths):,} samples")

    return np.array(all_paths), np.array(all_steers, dtype=np.float32)


def create_dataloaders(cfg):
    """
    Load ALL driving sessions, split into train / val / test, and
    return DataLoaders.

    Split strategy (configurable in config.py):
        train  70%   — model learns from these
        val    15%   — used to tune hyper-parameters / early stopping
        test   15%   — held out; evaluated ONLY after training is done

    WHY THREE SPLITS?
    • Validation tells you when to STOP training (prevents overfitting).
    • Test gives an UNBIASED estimate of real-world performance.
      If you tune on the test set, you leak information and your
      reported accuracy is overly optimistic.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    """
    # ---- 1. Load all sessions ----
    print("Loading datasets:")
    image_paths, steering_angles = _load_all_sessions(cfg)
    print(f"  Total combined: {len(image_paths):,} samples")

    # ---- 2. Train / (val+test) split ----
    #   First split off (val_size + test_size) from training.
    #   Then split that remainder into val and test.
    val_test_size = cfg.val_size + cfg.test_size  # e.g. 0.30
    X_train, X_rest, y_train, y_rest = train_test_split(
        image_paths,
        steering_angles,
        test_size=val_test_size,
        random_state=cfg.seed,
    )

    # ---- 3. Val / test split ----
    #   test_fraction of the remainder = test_size / (val_size + test_size)
    test_fraction = cfg.test_size / val_test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=test_fraction,
        random_state=cfg.seed,
    )

    print(f"  Train : {len(X_train):,}")
    print(f"  Val   : {len(X_val):,}")
    print(f"  Test  : {len(X_test):,}")

    # ---- 4. Build transforms ----
    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    # ---- 5. Build Datasets ----
    #   NOTE: test set uses val_transforms (no augmentation).
    train_ds = DrivingDataset(X_train, y_train, train_transforms)
    val_ds   = DrivingDataset(X_val,   y_val,   val_transforms)
    test_ds  = DrivingDataset(X_test,  y_test,  val_transforms)

    # ---- 6. Build DataLoaders ----
    #
    #   shuffle=True for training (randomise order each epoch).
    #   shuffle=False for val & test (deterministic evaluation).
    #   num_workers — background processes for data loading (2-4 is good).
    #   pin_memory=True — faster CPU→GPU transfer.
    #   drop_last=True — skip incomplete final batch during training.

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
