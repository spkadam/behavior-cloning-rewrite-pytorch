"""
predict.py — Inference and visualisation.

  • Loading a saved PyTorch model for inference.
  • Running a forward pass on a single image.
  • Using model.eval() and torch.no_grad() for inference.
  • Visualising predicted vs actual steering on video frames.

USAGE:
  # Predict on random validation samples (with display)
  python predict.py --checkpoint models_pytorch/best_model.pth

  # Overlay predictions on a video file
  python predict.py --checkpoint models_pytorch/best_model.pth --video data.avi --csv data.csv
"""

import os
import cv2
import numpy as np
import torch

from config import Config, DEVICE
from model import NvidiaDriveNet
from transforms import Resize, Crop, BGR2HSV, ToFloatTensor, Compose


# ======================================================================
#  1.  LOAD A TRAINED MODEL
# ======================================================================

def load_trained_model(checkpoint_path: str, cfg: Config) -> NvidiaDriveNet:
    """
    Instantiate the model and load saved weights.

    KEY CONCEPT — state_dict:
      A state_dict is a Python dict that maps each layer name to its
      weight tensor.  torch.save serialises this dict; torch.load
      deserialises it.  model.load_state_dict() copies the weights into
      the live model.
    """
    model = NvidiaDriveNet(dropout_prob=cfg.dropout_prob).to(DEVICE)

    # We need to do one dummy forward pass to build the FC layers
    # (because we used lazy initialisation in the model).
    dummy = torch.zeros(1, 3, cfg.image_height, cfg.image_width, device=DEVICE)
    _ = model(dummy)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # <-- CRITICAL: disable dropout for inference
    print(f"Loaded model from: {checkpoint_path}")
    print(f"  (trained for {checkpoint.get('epoch', '?')} epochs, "
          f"val_loss={checkpoint.get('val_loss', '?'):.6f})")
    return model


# ======================================================================
#  2.  PREPROCESS & PREDICT A SINGLE IMAGE
# ======================================================================

def preprocess_image(image_bgr: np.ndarray, cfg: Config) -> torch.Tensor:
    """
    Apply the same preprocessing pipeline used during validation
    and return a batched tensor ready for the model.
    """
    transform = Compose([
        Resize(cfg.raw_width, cfg.raw_height),
        Crop(cfg.crop_top, cfg.crop_bottom, cfg.crop_left, cfg.crop_right),
        BGR2HSV(),
        ToFloatTensor(),
    ])
    image, _ = transform(image_bgr, 0.0)  # angle is unused
    # Add batch dimension:  (C, H, W) → (1, C, H, W)
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    return tensor


@torch.no_grad()
def predict_steering(model: NvidiaDriveNet, image_bgr: np.ndarray, cfg: Config) -> float:
    """
    Given a BGR image, return the predicted steering angle.

    The `@torch.no_grad()` decorator ensures we don't waste memory
    recording the computation graph — we only need the forward pass.
    """
    tensor = preprocess_image(image_bgr, cfg)
    prediction = model(tensor)              # shape: (1, 1)
    steering = float(prediction.item())     # scalar
    return steering


# ======================================================================
#  3.  PREDICT ON RANDOM SAMPLES (like old model_predict)
# ======================================================================

def predict_random_samples(checkpoint_path: str, cfg: Config, num_samples: int = 20):
    """
    Load the model, pick random images from the CSV, and compare
    predicted vs actual steering angles.
    """
    import pandas as pd

    model = load_trained_model(checkpoint_path, cfg)

    df = pd.read_csv(cfg.csv_path)
    indices = df["image_idx"].values
    steerings = df["steer_cmd"].values

    cv2.namedWindow("Prediction", cv2.WINDOW_AUTOSIZE)

    for _ in range(num_samples):
        rand_idx = np.random.randint(len(df))
        img_idx = indices[rand_idx]
        true_steer = steerings[rand_idx]

        image_path = os.path.join(cfg.image_dir, f"{img_idx}.jpg")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load: {image_path}")
            continue

        pred_steer = predict_steering(model, image, cfg)

        # Draw steering indicators on the image
        h, w = image.shape[:2]
        centre_x = w // 2

        # True steering — RED dot
        true_x = int(centre_x + true_steer * centre_x)
        cv2.circle(image, (true_x, 15), 10, (0, 0, 255), -1)

        # Predicted steering — BLUE dot
        pred_x = int(centre_x + pred_steer * centre_x)
        cv2.circle(image, (pred_x, 15), 10, (255, 0, 0), -1)

        # Centre line
        cv2.line(image, (centre_x, 0), (centre_x, 30), (0, 255, 0), 1)

        # Text overlay
        cv2.putText(
            image,
            f"TRUE: {true_steer:+.4f}  PRED: {pred_steer:+.4f}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow("Prediction", image)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


# ======================================================================
#  4.  OVERLAY ON VIDEO (like driveai_CUT_vizsteering_red_blue.py)
# ======================================================================

def predict_on_video(
    checkpoint_path: str,
    video_path: str,
    csv_path: str,
    cfg: Config,
    output_path: str | None = None,
):
    """
    Overlay predicted (blue) and actual (red) steering on every frame
    of a video file.  Optionally save the result to a new video.
    """
    import pandas as pd

    model = load_trained_model(checkpoint_path, cfg)

    # Read steering data
    df = pd.read_csv(csv_path)
    np_data = df.values
    steer_col = df.columns.get_loc("steer_cmd")
    time_col = df.columns.get_loc("time") if "time" in df.columns else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    # Optional: set up video writer
    writer = None
    if output_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cv2.namedWindow("Steering", cv2.WINDOW_AUTOSIZE)
    row_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if row_idx >= len(np_data):
            break

        # Actual steering from CSV
        true_steer = float(np_data[row_idx, steer_col])

        # Predicted steering from CNN
        pred_steer = predict_steering(model, frame, cfg)

        h, w = frame.shape[:2]
        cx = w // 2

        # True = RED, Predicted = BLUE
        true_x = int(cx + true_steer * cx)
        pred_x = int(cx + pred_steer * cx)
        cv2.circle(frame, (true_x, 10), 10, (0, 0, 255), -1)
        cv2.circle(frame, (pred_x, 10), 10, (255, 0, 0), -1)
        cv2.line(frame, (cx, 0), (cx, 25), (0, 255, 0), 1)

        # Timestamp
        if time_col is not None:
            time_val = str(np_data[row_idx, time_col])
            cv2.putText(frame, time_val, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Steering", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

        row_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {row_idx} frames.")


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict steering angles from images or video")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--video", default=None, help="Path to .avi video (for overlay mode)")
    parser.add_argument("--csv", default=None, help="Path to CSV with actual steering (for overlay)")
    parser.add_argument("--output", default=None, help="Save annotated video to this path")
    parser.add_argument("--samples", type=int, default=20, help="Random samples to show (non-video mode)")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--csv_filename", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.root_path:
        cfg.root_path = args.root_path
        cfg.__post_init__()
    if args.csv_filename:
        cfg.csv_filename = args.csv_filename
        cfg.__post_init__()

    if args.video:
        csv = args.csv or cfg.csv_path
        predict_on_video(args.checkpoint, args.video, csv, cfg, args.output)
    else:
        predict_random_samples(args.checkpoint, cfg, args.samples)
