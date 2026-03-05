"""
train.py — The complete training loop, written from scratch.

WHAT YOU'LL LEARN:
  • The anatomy of a PyTorch training loop (the "boilerplate" every
    project needs).
  • How loss.backward() computes gradients via automatic differentiation.
  • How optimizer.step() updates weights using those gradients.
  • Why you must call optimizer.zero_grad() before each backward pass.
  • model.train() vs model.eval() and why it matters (dropout, batchnorm).
  • Saving / loading checkpoints with torch.save / torch.load.
  • Tracking metrics (loss, optional accuracy) per epoch.
  • Plotting learning curves.

THE TRAINING LOOP — step by step
---------------------------------
  for each epoch:
      for each batch:
          1.  Forward pass:   predictions = model(images)
          2.  Compute loss:   loss = criterion(predictions, targets)
          3.  Backward pass:  loss.backward()        ← compute gradients
          4.  Update weights: optimizer.step()        ← gradient descent
          5.  Clear grads:    optimizer.zero_grad()   ← reset for next batch

      Evaluate on validation set (no gradients needed).
      Save checkpoint if validation loss improved.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import Config, DEVICE
from model import NvidiaDriveNet
from dataset import create_dataloaders


# ======================================================================
#  1.  TRAINING ONE EPOCH
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training data.

    Returns
    -------
    avg_loss : float
        Mean training loss across all batches.
    """
    model.train()  # <-- enable dropout / batchnorm training behaviour

    running_loss = 0.0
    num_batches = 0

    for images, angles in loader:
        # ---- Move data to GPU (or CPU) ----
        images = images.to(device)       # (batch, 3, H, W)
        angles = angles.to(device)       # (batch, 1)

        # ---- Forward pass ----
        predictions = model(images)      # (batch, 1)

        # ---- Compute loss ----
        #   MSE = mean of (prediction - target)² over the batch.
        loss = criterion(predictions, angles)

        # ---- Backward pass ----
        #   loss.backward() walks backwards through the computation graph
        #   and computes ∂loss/∂param for every trainable parameter.
        #   These gradients accumulate in param.grad.
        optimizer.zero_grad()  # clear old gradients FIRST
        loss.backward()

        # ---- Update weights ----
        #   optimizer.step() applies the update rule  (e.g. Adam):
        #     param = param - lr * f(gradient, momentum, …)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


# ======================================================================
#  2.  VALIDATION
# ======================================================================

@torch.no_grad()   # <-- decorator that disables gradient computation
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set.

    torch.no_grad() is critical here:
      • Saves memory (no gradient tensors stored).
      • Faster computation.
      • Does NOT affect dropout / batchnorm — that's handled by
        model.eval().
    """
    model.eval()  # <-- disable dropout, use running stats for batchnorm

    running_loss = 0.0
    num_batches = 0

    for images, angles in loader:
        images = images.to(device)
        angles = angles.to(device)

        predictions = model(images)
        loss = criterion(predictions, angles)

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


# ======================================================================
#  3.  FULL TRAINING RUN
# ======================================================================

def train(cfg: Config):
    """
    End-to-end training procedure:
      1. Load data
      2. Build model
      3. Loop over epochs
      4. Save best checkpoint
      5. Plot learning curves
    """
    print("=" * 60)
    print("  Behavior Cloning — PyTorch Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ---- Reproducibility ----
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- Data ----
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # ---- Model ----
    model = NvidiaDriveNet(dropout_prob=cfg.dropout_prob).to(DEVICE)
    print(model)

    # ---- Loss function ----
    #   Mean Squared Error — the standard choice for regression.
    #   Same as the Keras model.compile(loss='mean_squared_error').
    criterion = nn.MSELoss()

    # ---- Optimizer ----
    #   Adam (Adaptive Moment Estimation) adjusts the learning rate per
    #   parameter using running averages of gradients and their squares.
    #   It converges faster than plain SGD for most tasks.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # ---- Learning-rate scheduler (optional but recommended) ----
    #   Reduces LR when validation loss plateaus — helps fine-tune.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ---- Tracking ----
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    os.makedirs(cfg.model_save_dir, exist_ok=True)

    # ==================================================================
    #  THE EPOCH LOOP
    # ==================================================================
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Current learning rate (may have been reduced by scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  |  "
            f"train_loss: {train_loss:.6f}  |  "
            f"val_loss: {val_loss:.6f}  |  "
            f"lr: {current_lr:.2e}  |  "
            f"time: {elapsed:.1f}s"
        )

        # Step the scheduler (it looks at val_loss to decide)
        scheduler.step(val_loss)

        # ---- Save checkpoint if this is the best so far ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(cfg.model_save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model  (val_loss={val_loss:.6f}) → {ckpt_path}")

        # Save every-epoch checkpoint as well
        epoch_path = os.path.join(cfg.model_save_dir, f"model-{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            epoch_path,
        )

    # ---- Plot learning curves ----
    _plot_learning_curves(history, cfg.model_save_dir)
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # ==================================================================
    #  FINAL TEST EVALUATION
    # ==================================================================
    #   Load the best checkpoint and evaluate on the held-out test set.
    #   This is the ONLY time we touch test data — no tuning allowed.
    print("\n" + "=" * 60)
    print("  Evaluating on TEST set (held-out, never seen during training)")
    print("=" * 60)
    best_ckpt = os.path.join(cfg.model_save_dir, "best_model.pth")
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = validate(model, test_loader, criterion, DEVICE)
    print(f"  Test MSE loss: {test_loss:.6f}")
    print(f"  (for reference — val loss was {best_val_loss:.6f})")


# ======================================================================
#  4.  PLOT LEARNING CURVES
# ======================================================================

def _plot_learning_curves(history: dict, save_dir: str):
    """Save a train-vs-validation loss plot."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curves — Behavior Cloning")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved learning curves → {plot_path}")


# ======================================================================
#  5.  RESUME TRAINING FROM A CHECKPOINT
# ======================================================================

def load_checkpoint(model: nn.Module, optimizer, checkpoint_path: str):
    """
    Restore model weights and optimizer state from a saved checkpoint.

    WHY save optimizer state?
      Adam keeps running averages of gradients (momentum buffers).
      If you only save model weights, the optimizer "forgets" its history
      when you resume and training quality degrades.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.6f})")
    return epoch, val_loss


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the NVIDIA driving model (PyTorch)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--csv_filename", type=str, default=None, help="CSV base name")
    parser.add_argument("--root_path", type=str, default=None, help="Root data folder")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = Config()

    # Apply command-line overrides
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.csv_filename is not None:
        cfg.csv_filename = args.csv_filename
        cfg.__post_init__()
    if args.root_path is not None:
        cfg.root_path = args.root_path
        cfg.__post_init__()

    # Print config
    print("\nConfiguration:")
    print("-" * 40)
    for k, v in vars(cfg).items():
        print(f"  {k:25s} = {v}")
    print("-" * 40)

    train(cfg)
