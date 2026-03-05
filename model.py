"""
model.py — The NVIDIA Self-Driving Car CNN, written from scratch in PyTorch.

ARCHITECTURE OVERVIEW (from the 2016 NVIDIA paper "End to End Learning
for Self-Driving Cars"):

    Input image  (3 × 300 × 630)   ← after preprocessing
          │
    ┌─────▼──────────────────────┐
    │  Normalisation  (x/127.5-1)│   Maps pixel values [0,255] → [-1,1]
    └─────┬──────────────────────┘
    ┌─────▼──────────────────────┐
    │  Conv2D  24 filters 5×5 s2 │   Feature extraction begins
    │  ELU activation            │
    ├────────────────────────────┤
    │  Conv2D  36 filters 5×5 s2 │
    │  ELU                       │
    ├────────────────────────────┤
    │  Conv2D  48 filters 5×5 s2 │
    │  ELU                       │
    ├────────────────────────────┤
    │  Conv2D  64 filters 3×3 s1 │
    │  ELU                       │
    ├────────────────────────────┤
    │  Conv2D  64 filters 3×3 s1 │
    │  ELU                       │
    └─────┬──────────────────────┘
    ┌─────▼──────────────────────┐
    │  Dropout (p=0.5)           │   Regularisation
    ├────────────────────────────┤
    │  Flatten                   │   3-D feature maps → 1-D vector
    ├────────────────────────────┤
    │  FC 100 + ELU              │
    │  FC  50 + ELU              │   Regression head
    │  FC  10 + ELU              │
    │  FC   1                    │   ← predicted steering angle
    └────────────────────────────┘

  • nn.Module — the base class for ALL PyTorch models.
  • nn.Sequential — stacking layers (like Keras Sequential).
  • nn.Conv2d — 2-D convolution; note PyTorch uses (C, H, W) not (H, W, C).
  • nn.ELU — Exponential Linear Unit activation.
  • forward() — defines how data flows through the network.
  • The difference between __init__ (build layers) and forward (connect them).
"""

import torch
import torch.nn as nn


class NvidiaDriveNet(nn.Module):
    """
    A PyTorch re-implementation of the NVIDIA end-to-end driving model.

    Parameters
    ----------
    dropout_prob : float
        Probability of zeroing a neuron during training (default 0.5).
    """

    def __init__(self, dropout_prob: float = 0.5):
        # ──────────────────────────────────────────────────────────────
        # super().__init__() is REQUIRED.  It tells Python to initialise
        # the nn.Module internals (parameter tracking, hooks, etc.).
        # ──────────────────────────────────────────────────────────────
        super().__init__()

        # ===================== CONVOLUTIONAL BACKBONE ==================
        #
        # nn.Sequential groups layers so they run in order.
        # Each nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # creates a learnable convolutional filter bank.
        #
        # KEY CONCEPT — channels:
        #   An RGB image has 3 channels.  The first conv layer takes 3 in
        #   and outputs 24 "feature maps".  Each subsequent layer deepens
        #   the representation (36 → 48 → 64 → 64).
        #
        # KEY CONCEPT — stride:
        #   stride=2 means the filter jumps 2 pixels at a time, halving
        #   the spatial dimensions.  This is an alternative to max-pooling
        #   and is what NVIDIA chose.
        #
        # KEY CONCEPT — ELU:
        #   f(x) = x            if x ≥ 0
        #   f(x) = α(eˣ − 1)   if x < 0
        #   Smoother than ReLU for x < 0, which helps gradient flow.
        # ================================================================

        self.conv_layers = nn.Sequential(
            # --- Layer 1: 5×5 conv, 24 filters, stride 2 ---
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ELU(),

            # --- Layer 2: 5×5 conv, 36 filters, stride 2 ---
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ELU(),

            # --- Layer 3: 5×5 conv, 48 filters, stride 2 ---
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ELU(),

            # --- Layer 4: 3×3 conv, 64 filters, stride 1 ---
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),

            # --- Layer 5: 3×3 conv, 64 filters, stride 1 ---
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        # ===================== DROPOUT ================================
        # During training, randomly zero out neurons with probability p.
        # This forces the network to learn redundant representations and
        # prevents over-fitting.  At test time dropout is automatically
        # disabled (by calling model.eval()).
        # ================================================================
        self.dropout = nn.Dropout(p=dropout_prob)

        # ===================== FULLY-CONNECTED HEAD ====================
        # After convolutions we flatten the feature maps into a 1-D
        # vector and pass through dense (linear) layers to regress the
        # steering angle.
        #
        # We do NOT hardcode the flatten size here.  Instead we compute
        # it dynamically in _get_conv_output_size() so the model works
        # even if you change the input resolution.
        # ================================================================
        self._flat_size: int | None = None  # will be set on first forward

        # Placeholder — we build the FC layers lazily so we don't have
        # to manually calculate the flattened size.
        self.fc_layers: nn.Sequential | None = None

    # ------------------------------------------------------------------
    #  Helper: figure out the size after all conv layers.
    # ------------------------------------------------------------------
    def _build_fc_layers(self, flat_size: int):
        """Construct the FC head once we know the flattened dim."""
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1),   # single output = steering angle
        )
        # Move newly created layers to the same device as the conv layers
        device = next(self.conv_layers.parameters()).device
        self.fc_layers = self.fc_layers.to(device)

    # ------------------------------------------------------------------
    #  FORWARD PASS — this IS the model.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 3, H, W)
            A batch of preprocessed images.  Pixel values in [0, 255].

        Returns
        -------
        Tensor of shape (batch, 1)
            Predicted steering angles.
        """
        # ---------- 1) Normalise: [0, 255] → [-1, 1] ----------
        #   The old Keras code did:  Lambda(lambda x: x/127.5 - 1.0)
        #   In PyTorch we just write the math directly.
        x = x / 127.5 - 1.0

        # ---------- 2) Convolutional feature extraction ----------
        x = self.conv_layers(x)

        # ---------- 3) Dropout ----------
        x = self.dropout(x)

        # ---------- 4) Flatten ----------
        #   x.shape is (batch, channels, h, w) after the convolutions.
        #   We reshape to (batch, channels * h * w).
        x = x.flatten(start_dim=1)

        # ---------- 5) Build FC layers on first pass ----------
        if self.fc_layers is None:
            self._flat_size = x.shape[1]
            self._build_fc_layers(self._flat_size)

        # ---------- 6) Fully-connected regression head ----------
        x = self.fc_layers(x)

        return x


# ======================================================================
# Quick sanity-check / demo  (run:  python model.py)
# ======================================================================
if __name__ == "__main__":
    from config import Config

    cfg = Config()

    # Build the model
    model = NvidiaDriveNet(dropout_prob=cfg.dropout_prob)
    model.eval()  # disable dropout for this test

    # Create a random dummy image batch  (batch=2, channels=3, H=300, W=630)
    dummy = torch.randn(2, 3, cfg.image_height, cfg.image_width) * 255.0

    # Forward pass
    out = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")       # expect (2, 1)
    print(f"Predictions : {out.detach().numpy().flatten()}")

    # Print model summary (number of parameters)
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters : {total_params:,}")
    print(f"Trainable params : {trainable:,}")
