# Behavior Cloning — PyTorch from Scratch

A complete rewrite of the 2018 Keras-based behavior cloning project using **modern PyTorch**. Every file is heavily commented so you can learn how to build a deep learning model from the ground up.

## Demo

![Steering prediction overlay — red dot is ground truth, blue dot is model prediction](output.gif)

---

## Project Structure

```
pytorch_version/
├── config.py           # All hyperparameters and paths in one place
├── model.py            # NVIDIA self-driving CNN (nn.Module)
├── transforms.py       # Image preprocessing & data augmentation
├── dataset.py          # Custom PyTorch Dataset + DataLoader
├── train.py            # Training loop from scratch
├── predict.py          # Inference & visualisation
├── extract_frames.py   # Video → JPEG frames + CSV
└── README.md           # This file
```

## Old Code → New Code Mapping

| Old file (Keras, 2018) | New file (PyTorch) | What changed |
|---|---|---|
| `utils_CUT.py` — preprocessing functions | `transforms.py` | Callable transform classes instead of loose functions |
| `utils_CUT.py` — `batch_generator()` | `dataset.py` | `torch.utils.data.Dataset` + `DataLoader` replace the generator |
| `train_driveai_keras.py` — `build_model()` | `model.py` | `nn.Module` subclass instead of Keras `Sequential` |
| `train_driveai_keras.py` — `train_model()` | `train.py` | Explicit training loop instead of `model.fit_generator()` |
| `train_driveai_keras.py` — `model_predict()` | `predict.py` | Same visualisation, modernised |
| `driveai_CUT_vizsteering_red_blue.py` | `predict.py` — `predict_on_video()` | Combined into one script |
| `video_2_keras_csv.py` | `extract_frames.py` | Cleaner argument handling |
| Hard-coded paths at top of every file | `config.py` | Single source of truth |

---

## Key PyTorch Concepts You'll Learn

### 1. `nn.Module` — The Model Base Class
Every neural network in PyTorch inherits from `nn.Module`. You define layers in `__init__()` and wire them together in `forward()`. See [model.py](model.py).

### 2. The Training Loop
Unlike Keras's `model.fit()`, PyTorch gives you full control:
```python
for epoch in range(num_epochs):
    for images, targets in train_loader:
        predictions = model(images)         # forward pass
        loss = criterion(predictions, targets)
        optimizer.zero_grad()               # clear old gradients
        loss.backward()                     # compute new gradients
        optimizer.step()                    # update weights
```
See [train.py](train.py) for the complete implementation.

### 3. Dataset & DataLoader
Instead of writing a `while True` generator, you subclass `Dataset` with `__len__` and `__getitem__`, then `DataLoader` handles batching, shuffling, and parallel loading. See [dataset.py](dataset.py).

### 4. Transforms
Each preprocessing step is a callable class. Chain them with `Compose`. Both the image AND the steering angle flow through the pipeline. See [transforms.py](transforms.py).

### 5. Saving & Loading
```python
# Save
torch.save({"model_state_dict": model.state_dict(), ...}, "model.pth")

# Load
checkpoint = torch.load("model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
```
PyTorch saves the `state_dict` (a dict of layer → weight tensor), not the entire model object.

### 6. GPU Usage
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tensor = tensor.to(device)
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python matplotlib
```

### 2. Extract Frames (if starting from video)
```bash
python extract_frames.py \
    --video /path/to/data.avi \
    --csv   /path/to/data.csv \
    --outdir /path/to/frames/
```

### 3. Train
```bash
# Using defaults from config.py
python train.py

# Or with overrides
python train.py --epochs 20 --lr 5e-5 --batch_size 32
```

### 4. Predict on Random Samples
```bash
python predict.py --checkpoint models_pytorch/best_model.pth
```

### 5. Overlay on Video
```bash
python predict.py \
    --checkpoint models_pytorch/best_model.pth \
    --video /path/to/data.avi \
    --csv   /path/to/data.csv \
    --output annotated_output.avi
```

### 6. Inspect the Model Architecture
```bash
python model.py
# Prints input/output shapes, parameter count
```

---

## The NVIDIA Architecture

This is the CNN from NVIDIA's 2016 paper *"End to End Learning for Self-Driving Cars"*:

```
Input (3×300×630)
    │
    ├─ Normalize: x/127.5 − 1
    ├─ Conv2D(24, 5×5, stride=2) + ELU
    ├─ Conv2D(36, 5×5, stride=2) + ELU
    ├─ Conv2D(48, 5×5, stride=2) + ELU
    ├─ Conv2D(64, 3×3)           + ELU
    ├─ Conv2D(64, 3×3)           + ELU
    ├─ Dropout(0.5)
    ├─ Flatten
    ├─ Dense(100) + ELU
    ├─ Dense(50)  + ELU
    ├─ Dense(10)  + ELU
    └─ Dense(1)                    ← steering angle
```

**Loss**: Mean Squared Error  
**Optimizer**: Adam (lr = 1e-4)

---

## Hyperparameter Cheat Sheet

| Parameter | Default | Where | Notes |
|---|---|---|---|
| `epochs` | 10 | `config.py` | More epochs = more passes over data |
| `batch_size` | 40 | `config.py` | Larger = faster but needs more GPU RAM |
| `learning_rate` | 1e-4 | `config.py` | Start here, reduce if loss is noisy |
| `dropout_prob` | 0.5 | `config.py` | Lower = less regularisation |
| `test_size` | 0.2 | `config.py` | 80/20 train/val split |
| `augment_prob` | 0.6 | `config.py` | Chance to augment each training sample |
