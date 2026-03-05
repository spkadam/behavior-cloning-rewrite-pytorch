from dataclasses import dataclass, field
from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    """All tuneable knobs for the project live here."""

    # ---- Paths ----
    root_path: str = "/home/administrator/Behavior Cloning/bc_dataset/"

    #   All three driving sessions.  Each entry is the basename (no extension).
    #   The extracted frames live in  <root_path>/<basename>/
    #   The training CSV lives at     <root_path>/<basename>_keras.csv
    dataset_names: list = field(default_factory=lambda: [
        "data1_2018_06_14_15_50_42",
        "data1_2018_07_05_15_01_28",
        "data1_2018_07_17_21_07_10",
    ])

    model_save_dir: str = "" # set in __post_init__

    # ---- Image dimensions (after preprocessing) ----
    #   The original code uses 640×480 raw → crop to 630×300 → convert to HSV.
    #   We keep the same sizes so the NVIDIA model architecture is compatible.
    raw_height: int = 480
    raw_width: int = 640
    crop_top: int = 180       # pixels to cut from the top (sky)
    crop_bottom: int = 480    # keep up to this row
    crop_left: int = 10
    crop_right: int = 640
    image_height: int = 300   # final height after crop
    image_width: int = 630    # final width after crop
    image_channels: int = 3

    # ---- Training hyper-parameters ----
    #   These are the knobs that control HOW the model learns.
    #
    #   epochs:        How many full passes through the training data.
    #   batch_size:    Number of samples processed before updating weights.
    #   learning_rate: Step size for the optimizer (too big → unstable,
    #                  too small → slow).
    #   val_size:      Fraction of data held out for validation.
    #   test_size:     Fraction of data held out for final testing.
    #   dropout_prob:  Probability of dropping a neuron during training
    #                  (regularisation to prevent overfitting).
    #
    #   Split ratios:  train 70% / val 15% / test 15%
    epochs: int = 10
    batch_size: int = 40
    learning_rate: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    dropout_prob: float = 0.5

    # ---- Data augmentation probabilities ----
    augment_prob: float = 0.6       # chance to apply augmentation per sample
    flip_prob: float = 0.5          # chance to horizontally flip
    brightness_range: float = 0.4   # max brightness jitter (±20 %)
    translate_range_x: float = 30.0 # max horizontal shift in pixels
    translate_range_y: float = 10.0 # max vertical shift in pixels
    translate_steer_factor: float = 0.006  # steering correction per pixel shift

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Checkpoint ----
    save_best_only: bool = True

    def __post_init__(self):
        """Derived paths — computed from the base settings."""
        self.model_save_dir = str(Path(self.root_path) / "models_pytorch")
