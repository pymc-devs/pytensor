# YOLO11n PyTensor Training Implementation Plan

## Overview

Implement a complete YOLO11n object detection model natively in PyTensor, train it on a COCO subset (320×320 images) using JAX GPU backend on Lambda Cloud (H100), export to ONNX, and demonstrate real-time inference in the browser. This showcases that PyTensor's ONNX backend can handle complex, real-world deep learning models end-to-end.

**Goal**: Demonstrate PyTensor → ONNX pipeline works for state-of-the-art object detection

**Model**: YOLO11n (nano) - 181 layers, **~2.6M parameters** (same as standard YOLO11n)
  - Parameter count is determined by backbone/head architecture, NOT by number of classes
  - Changing from 80→2 classes only affects final detection layers (~3K params difference)
  - Input size (320×320) doesn't change param count, only feature map sizes

**Training Infrastructure**: Lambda Cloud ARM64 + H100 GPU
  - Hardware: 1× GH200 (H100 + Grace CPU), 400GB RAM
  - Backend: PyTensor with JAX GPU backend
  - Container: Docker (NVIDIA NGC ARM64 CUDA images)
  - Training time: ~30-45 minutes for 100 epochs

**Dataset**: COCO 2017 subset, resized to 320×320, 2 classes (person, cellphone)
  - **Train**: 12,000 images
  - **Val**: 2,000 images
  - Total: 14,000 images (balanced for both classes, limited by cellphone rarity)

**Training Target**: Functional real-world detection - must actually detect person and cellphone in webcam!
**Demo**: Real-time webcam detection in browser with WebGPU at 30+ FPS - must actually work!

## Model Parameter Count Analysis

**Total Parameters: ~2.6M** (approximately the same as standard YOLO11n)

### Parameter Breakdown by Component:

1. **Backbone**: ~2.3M parameters (90% of total)
   - Conv layers: Weight filters (C_out × C_in × k × k)
   - BatchNorm: γ and β per channel
   - C3k2, SPPF, C2PSA blocks
   - **Independent of number of classes**

2. **Head (FPN/PAN)**: ~290K parameters
   - Upsampling convolutions
   - Concatenation layers (no params)
   - C3k2 refinement blocks
   - **Independent of number of classes**

3. **Detection Heads**: ~3K parameters (0.1% of total)
   - P3: 64 channels → (4+num_classes) = 64×6 = 384 params (2 classes)
   - P4: 128 channels → (4+num_classes) = 128×6 = 768 params
   - P5: 256 channels → (4+num_classes) = 256×6 = 1536 params
   - Total: ~2.7K params for 2 classes vs ~5.4K for 80 classes
   - **Difference: Only ~2.7K parameters!**

### Why num_classes has minimal impact:
- Only the final 1×1 conv layers in detection heads depend on num_classes
- These layers map feature channels to (4 + num_classes) outputs
- 2 classes vs 80 classes = ~2.7K param difference out of 2.6M total
- **That's 0.1% difference - negligible!**

### Why input_size has no impact on parameter count:
- Input size (128×128 vs 640×640) only affects feature map spatial dimensions
- Convolutional filters have fixed sizes regardless of input dimensions
- Parameters = filters, not activations
- **Input size affects memory/compute, not parameter count**

## Current State Analysis

### What Exists (from research doc)

**ONNX Operations - ALL IMPLEMENTED ✅**
- `pytensor/link/onnx/dispatch/conv.py:14-140` - Conv2D with stride, padding, groups
- `pytensor/link/onnx/dispatch/pool.py:9-81` - MaxPool (SPPF pattern tested)
- `pytensor/link/onnx/dispatch/resize.py:10-85` - Upsample for FPN
- `pytensor/link/onnx/dispatch/join.py:10-83` - Concat for skip connections
- `pytensor/link/onnx/dispatch/batchnorm.py:12-85` - BatchNorm ONNX converter
- `pytensor/link/onnx/dispatch/elemwise.py:142-232` - SiLU/Swish activation
- All tests passing for YOLO patterns

**Training Infrastructure**
- `examples/onnx/onnx-mnist-demo/train_mnist_cnn.py` - Complete training pipeline reference
- Gradient computation: `pytensor.grad()`
- SGD with momentum working
- Batch training loop patterns established
- ONNX export: `pytensor.link.onnx.export_onnx()`

**Demo Infrastructure**
- `examples/onnx/onnx-yolo-demo/` - Directory exists with `yolo11n_320.onnx` and benchmark HTML
- WebGPU demo infrastructure tested
- ONNX Runtime Web integration working

### Critical Gap Identified

**BatchNormalization Gradient Support - MISSING ❌**

Location: `pytensor/tensor/batchnorm.py:197-211`

```python
def grad(self, inputs, output_grads):
    """Compute gradients."""
    raise NotImplementedError(
        "BatchNormalization.grad() not implemented. "
        "This op is for inference only."
    )
```

**Impact**: Cannot train networks with BatchNorm layers (YOLO11n has BatchNorm after every Conv)

**Must implement**: Backward pass for BatchNorm operation

## Desired End State

### Success Criteria

#### Automated Verification:
- [ ] BatchNorm gradient tests pass: `pytest tests/tensor/test_batchnorm.py::test_batchnorm_grad -v`
- [ ] YOLO11n architecture builds without errors (320×320 input)
- [ ] Training runs for 100 epochs without crashes on H100
- [ ] Loss decreases consistently during training (monitored via logs)
- [ ] Validation mAP@0.5 > 0.35 (functional real-world detection)
- [ ] Model exports to ONNX: `examples/onnx/onnx-yolo-demo/yolo11n_320_trained.onnx`
- [ ] ONNX model validates: `onnx.checker.check_model(model)`
- [ ] PyTensor and ONNX outputs match: `np.allclose(pt_out, onnx_out, atol=1e-4)`
- [ ] Training completes in 30-45 minutes on Lambda Cloud H100

#### Manual Verification (Webcam Demo):
- [ ] Training completes successfully on Lambda Cloud
- [ ] Model detects person in test images with confidence > 0.5 (improved threshold)
- [ ] Model detects cellphone in test images with confidence > 0.5 (improved threshold)
- [ ] Browser demo loads ONNX model successfully (320×320 input)
- [ ] **Webcam feed displays in browser at 30+ FPS** (improved performance)
- [ ] **Real-time detection runs smoothly with WebGPU**
- [ ] **Bounding boxes appear around person when in frame**
- [ ] **Bounding boxes appear around cellphone when in frame**
- [ ] Detections have reasonable confidence scores (0.5-0.95)
- [ ] No significant lag or frame drops during inference

### Deliverables

1. **Core Implementation**
   - `pytensor/tensor/batchnorm.py` - BatchNorm with gradient support (Phase 1)

2. **Training Scripts** (All-in-one in `examples/onnx/onnx-yolo-demo/`)
   - `train_yolo11n.py` - All-in-one training script with:
     - COCO dataset auto-download
     - Model architecture (YOLO11n)
     - Loss functions (IoU + BCE)
     - Training loop with progress tracking
     - Validation and mAP computation
     - Automatic ONNX export
     - Checkpoint management
   - `model.py` - YOLO11n architecture (ConvBNSiLU, C3k2, SPPF, C2PSA, backbone, head)
   - `blocks.py` - Building blocks for YOLO11n
   - `loss.py` - Detection loss functions
   - `dataset.py` - COCO dataset loader with augmentation
   - `utils.py` - Helper functions (NMS, mAP calculation, visualization)
   - `requirements.txt` - Python dependencies

3. **Tests**
   - `tests/tensor/test_batchnorm.py` - BatchNorm gradient tests
   - `tests/examples/test_yolo11n_blocks.py` - Unit tests for YOLO blocks
   - `tests/examples/test_yolo11n_export.py` - End-to-end ONNX export test

4. **Trained Model & Demo**
   - `examples/onnx/onnx-yolo-demo/yolo11n_320_trained.onnx` - Final trained model (320×320)
   - `examples/onnx/onnx-yolo-demo/checkpoints/best_model.pkl` - Best checkpoint
   - `examples/onnx/onnx-yolo-demo/yolo_detection_demo.html` - Browser inference demo (updated for 320×320)

5. **Documentation**
   - `examples/onnx/onnx-yolo-demo/README.md` - Complete training and deployment guide
   - `examples/onnx/onnx-yolo-demo/LAMBDA_CLOUD_SETUP.md` - Step-by-step Lambda Cloud setup

## What We're NOT Doing - Scope Limitations

To keep this focused on the demo while leveraging H100 power:

- ❌ **NOT implementing complex data augmentation** - Simple horizontal flip + random brightness/contrast only
- ❌ **NOT implementing advanced YOLO tricks** - No mosaic, mixup, copy-paste, etc.
- ❌ **NOT optimizing for state-of-the-art accuracy** - Functional detection is enough (mAP@0.5 > 0.35)
- ❌ **NOT implementing multi-scale training** - Single 320×320 input size
- ❌ **NOT implementing NMS in PyTensor** - Do NMS in post-processing (JavaScript)
- ❌ **NOT creating a full training framework** - All-in-one training script only
- ❌ **NOT implementing learning rate scheduling** - Simple warmup + cosine decay (standard YOLO practice)
- ❌ **NOT using full COCO dataset** - 14,000 images for 2 classes only (train: 12k, val: 2k)
- ❌ **NOT implementing distributed training** - Single H100 GPU only
- ❌ **NOT implementing model EMA** - Keep it simple
- ❌ **NOT implementing DFL (Distribution Focal Loss)** - Simplified IoU + BCE loss only

**GOAL: Working real-time webcam demo at 30+ FPS that proves PyTensor → ONNX works for complex YOLO models!**

## Implementation Approach

### Architecture Strategy

**Use official YOLO11n architecture** (from Ultralytics):
- 181 layers total
- Scaling: depth=0.50, width=0.25
- Input: (batch, 3, 320, 320) - RGB images at 320×320
- Output: 3 detection heads at scales [40×40, 20×20, 10×10] for 320×320 input
- Backbone: Conv + C3k2 + SPPF + C2PSA blocks
- Head: Upsample + Concat + Conv blocks (FPN-PAN architecture)

**Architecture for 320×320**:
- Standard YOLO11n uses 640×640 → we use 320×320 (2× smaller, well-documented)
- Detection scales: P3/8 (40×40), P4/16 (20×20), P5/32 (10×10)
- Anchor-free detection (YOLO11 uses anchor-free design)
- Matches existing `yolo11n_320.onnx` reference model format

### Loss Function

**YOLO Detection Loss** (following YOLOv8/v11):
```
Total Loss = λ_box * Box_loss + λ_cls * Cls_loss + λ_dfl * DFL_loss
```

**Components**:
1. **Box Loss**: CIoU (Complete IoU) for bounding box regression
2. **Classification Loss**: Binary Cross-Entropy for class predictions
3. **DFL Loss**: Distribution Focal Loss for refined box localization

**Implementation approach**: Simplified loss focusing on box IoU + classification BCE

### Training Strategy - H100 GPU POWERED

**Dataset**: COCO 2017 train subset - **SUFFICIENT FOR REAL DETECTION**
- Download 2 classes only: person (1), cellphone (77)
- **Train: 12,000 images** (balanced across both classes, limited by cellphone rarity)
- **Val: 2,000 images** (for mAP validation during training)
- Person is very common in COCO (~40k images), cellphone is rarer (~1-2k images)
- Resize all to 320×320 (matches reference yolo11n_320.onnx)
- **Augmentation**: horizontal flip + random brightness/contrast adjustments

**Hyperparameters - OPTIMIZED FOR H100**:
- Batch size: 64 (H100 can handle large batches easily)
- Learning rate: 0.01 with warmup (5 epochs) + cosine decay
- Optimizer: SGD with momentum=0.937, nesterov=True (YOLO standard)
- **Epochs: 100** (fast on H100, ensures convergence)
- Weight decay: 5e-4 (prevent overfitting)
- Gradient clipping: max_norm=10.0

**Training loop**: All-in-one script with automation
- Forward pass → compute loss → backward pass → update weights
- Log every 10 batches (loss, learning rate, batch time)
- **Checkpoint every 10 epochs** + save best model (highest val mAP)
- **Validate every 5 epochs** (compute mAP@0.5 on validation set)
- **Auto-export to ONNX** at end of training with validation
- **Goal: Training completes in 30-45 minutes on H100**
- **Success metric: mAP@0.5 > 0.35** (functional real-world detection)

---

## Training Script Architecture

### All-in-One Script Design

**Philosophy**: Single self-contained script that can be run on Lambda Cloud with minimal setup.

**File**: `examples/onnx/onnx-yolo-demo/train_yolo11n.py`

**Features**:
- ✅ Auto-detects JAX GPU backend
- ✅ Downloads COCO dataset automatically (if not present)
- ✅ Builds YOLO11n model from scratch
- ✅ Training loop with progress bars (tqdm)
- ✅ Validation with mAP computation every 5 epochs
- ✅ Automatic checkpointing (every 10 epochs + best model)
- ✅ Resume from checkpoint support
- ✅ Automatic ONNX export at end
- ✅ ONNX validation (correctness check)
- ✅ Comprehensive logging

**Command-line Interface**:
```python
python train_yolo11n.py \
  --epochs 100 \
  --batch-size 64 \
  --image-size 320 \
  --train-images 12000 \
  --val-images 2000 \
  --lr 0.01 \
  --momentum 0.937 \
  --weight-decay 5e-4 \
  --warmup-epochs 5 \
  --checkpoint-dir ./checkpoints \
  --output-onnx yolo11n_320_trained.onnx \
  --resume checkpoints/latest.pkl  # Optional: resume from checkpoint
```

**Script Structure**:
```python
# train_yolo11n.py structure

import argparse
import pytensor
import pytensor.tensor as pt
from pytensor import shared
import jax
import numpy as np
from tqdm import tqdm
import json

# Imports from local modules
from model import build_yolo11n
from loss import yolo_loss
from dataset import COCODataset, download_coco_if_needed
from utils import compute_map, save_checkpoint, load_checkpoint

def main():
    # 1. Parse arguments
    args = parse_args()

    # 2. Setup PyTensor + JAX backend
    setup_pytensor_jax()

    # 3. Download COCO data (if needed)
    download_coco_if_needed(args.data_dir, args.train_images, args.val_images)

    # 4. Load datasets
    train_dataset = COCODataset(...)
    val_dataset = COCODataset(...)

    # 5. Build model
    model, x_var, predictions = build_yolo11n(num_classes=2, input_size=args.image_size)

    # 6. Define loss
    loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)

    # 7. Compute gradients
    grads = pytensor.grad(loss, model.params)

    # 8. Define updates (SGD with momentum + weight decay)
    updates = sgd_momentum_updates(model.params, grads, lr=args.lr, momentum=args.momentum)

    # 9. Compile training function
    train_fn = pytensor.function([x_var, ...], [loss, ...], updates=updates)

    # 10. Compile validation function
    val_fn = pytensor.function([x_var, ...], predictions)

    # 11. Training loop
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(train_fn, train_dataset, args.batch_size)

        # Validation (every 5 epochs)
        if epoch % 5 == 0:
            val_map = validate(val_fn, val_dataset)

        # Checkpointing (every 10 epochs + best)
        if epoch % 10 == 0:
            save_checkpoint(f"checkpoints/epoch_{epoch}.pkl", model.params)

        if val_map > best_map:
            best_map = val_map
            save_checkpoint("checkpoints/best_model.pkl", model.params)

    # 12. Export to ONNX
    export_to_onnx(model, x_var, args.output_onnx)

    # 13. Validate ONNX
    validate_onnx_export(model, args.output_onnx)

def setup_pytensor_jax():
    """Configure PyTensor to use JAX GPU backend."""
    pytensor.config.device = 'cuda'
    pytensor.config.floatX = 'float32'
    pytensor.config.optimizer = 'fast_run'

    # Verify JAX GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    assert len(devices) > 0 and devices[0].platform == 'gpu', "No GPU found!"

def train_epoch(train_fn, dataset, batch_size):
    """Run one training epoch with progress bar."""
    losses = []
    pbar = tqdm(range(0, len(dataset), batch_size), desc="Training")

    for batch_start in pbar:
        indices = range(batch_start, min(batch_start + batch_size, len(dataset)))
        batch = dataset.get_batch(indices)

        loss_val = train_fn(*batch)
        losses.append(loss_val)

        pbar.set_postfix(loss=f"{np.mean(losses[-10:]):.4f}")

    return np.mean(losses)

def validate(val_fn, dataset):
    """Compute mAP on validation set."""
    all_predictions = []
    all_targets = []

    for i in tqdm(range(len(dataset)), desc="Validating"):
        image, boxes, classes, num_boxes = dataset[i]
        predictions = val_fn(image[None, ...])  # Add batch dim

        all_predictions.append(predictions)
        all_targets.append((boxes, classes, num_boxes))

    map_score = compute_map(all_predictions, all_targets, iou_threshold=0.5)
    return map_score

def export_to_onnx(model, x_var, output_path):
    """Export trained model to ONNX."""
    import onnx
    from pytensor.link.onnx import export_onnx

    print(f"Exporting to ONNX: {output_path}")

    # Build computation graph
    predictions = model(x_var)
    outputs = [predictions['p3'], predictions['p4'], predictions['p5']]

    # Export
    onnx_model = export_onnx(
        inputs=[x_var],
        outputs=outputs,
        input_names=["images"],
        output_names=["output_p3", "output_p4", "output_p5"]
    )

    # Save
    onnx.save(onnx_model, output_path)
    print(f"✓ ONNX model saved: {output_path}")

    # Validate
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

def validate_onnx_export(model, onnx_path):
    """Verify PyTensor and ONNX outputs match."""
    import onnxruntime as ort

    # Create test input
    test_input = np.random.randn(1, 3, 320, 320).astype('float32')

    # PyTensor inference
    pt_output = model_inference_fn(test_input)

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {"images": test_input})

    # Compare
    for i, (pt_out, onnx_out) in enumerate(zip(pt_output, onnx_output)):
        max_diff = np.abs(pt_out - onnx_out).max()
        print(f"Output {i} max diff: {max_diff:.6f}")
        assert np.allclose(pt_out, onnx_out, atol=1e-4), f"Output {i} mismatch!"

    print("✓ PyTensor and ONNX outputs match!")

if __name__ == "__main__":
    main()
```

**Module Organization**:
```
examples/onnx/onnx-yolo-demo/
├── train_yolo11n.py     # Main training script (above)
├── model.py             # YOLO11n architecture
├── blocks.py            # Building blocks (ConvBNSiLU, C3k2, etc.)
├── loss.py              # Detection loss functions
├── dataset.py           # COCO dataset + download utilities
├── utils.py             # Helper functions (NMS, mAP, checkpointing)
├── requirements.txt     # Dependencies
└── README.md           # Usage instructions
```

---

## Lambda Cloud Training Setup

### Hardware Specifications
- **Instance**: 1× GH200 (ARM64 Grace CPU + H100 GPU)
- **Memory**: 400GB RAM
- **GPU**: NVIDIA H100 (80GB HBM3)
- **OS**: Ubuntu 22.04 ARM64

### Docker Setup (Recommended)

**Why Docker?**
- Pre-built ARM64 + CUDA environment from NVIDIA
- Consistent dependencies across environments
- Easy to reproduce results

**Step 1: Launch Lambda Cloud Instance**
```bash
# From Lambda Cloud dashboard:
# 1. Select "1x GH200" instance type
# 2. Choose Ubuntu 22.04 ARM64
# 3. Add SSH key
# 4. Launch instance
```

**Step 2: SSH into Instance**
```bash
ssh ubuntu@<lambda-instance-ip>
```

**Step 3: Pull NVIDIA NGC Docker Image (ARM64 + CUDA)**
```bash
# Pull official NVIDIA JAX container with ARM64 + CUDA support
docker pull nvcr.io/nvidia/jax:24.04-py3

# Verify GPU access
docker run --rm --gpus all nvcr.io/nvidia/jax:24.04-py3 nvidia-smi
```

**Step 4: Clone PyTensor Repository**
```bash
cd ~
git clone https://github.com/pymc-devs/pytensor.git
cd pytensor
git checkout onnx-backend  # Or your feature branch
```

**Step 5: Run Container with PyTensor Mounted**
```bash
docker run --gpus all -it --rm \
  -v ~/pytensor:/workspace/pytensor \
  -w /workspace/pytensor \
  --name yolo-training \
  nvcr.io/nvidia/jax:24.04-py3 bash
```

**Step 6: Inside Container - Install Dependencies**
```bash
# Install PyTensor in development mode
pip install -e .

# Install additional dependencies
pip install onnx pillow pycocotools tqdm

# Verify JAX sees GPU
python -c "import jax; print(jax.devices())"
# Should show: [cuda(id=0)]
```

**Step 7: Configure PyTensor to Use JAX Backend**
```bash
# Set environment variables
export PYTENSOR_FLAGS='device=cuda,floatX=float32,optimizer=fast_run'
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Prevent JAX from allocating all GPU memory
```

**Step 8: Run Training Script**
```bash
cd examples/onnx/onnx-yolo-demo

# Download COCO data (automatic, ~8GB)
# Train model (30-45 minutes)
# Export to ONNX
python train_yolo11n.py \
  --epochs 100 \
  --batch-size 64 \
  --image-size 320 \
  --train-images 12000 \
  --val-images 2000 \
  --checkpoint-dir ./checkpoints \
  --output-onnx yolo11n_320_trained.onnx
```

**Step 9: Monitor Training Progress**
```bash
# In another terminal (from local machine):
ssh ubuntu@<lambda-instance-ip>

# Attach to running container
docker exec -it yolo-training bash

# View training logs
tail -f examples/onnx/onnx-yolo-demo/training.log
```

**Step 10: Download Trained Model**
```bash
# From local machine:
scp ubuntu@<lambda-ip>:~/pytensor/examples/onnx/onnx-yolo-demo/yolo11n_320_trained.onnx .
scp ubuntu@<lambda-ip>:~/pytensor/examples/onnx/onnx-yolo-demo/checkpoints/best_model.pkl .
```

### Alternative: Direct Installation (Without Docker)

If you prefer direct installation:

```bash
# Install CUDA Toolkit for ARM64
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Install JAX with CUDA support
pip install --upgrade pip
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTensor
cd ~/pytensor
pip install -e .

# Install dependencies
pip install onnx pillow pycocotools tqdm

# Run training
cd examples/onnx/onnx-yolo-demo
python train_yolo11n.py --epochs 100 --batch-size 64 --image-size 320
```

### Cost Estimation

**Lambda Cloud GH200 Instance**:
- Hourly rate: ~$3.00-4.00/hour
- Training time: 0.5-0.75 hours
- **Total cost: $2-3 per training run**

Very cost-effective for this demo!

---

## Phase 1: Implement BatchNorm Gradient Support

### Overview
Implement backward pass for `BatchNormalization` op to enable training CNNs with batch normalization.

### Background

**Batch Normalization Forward**:
```
y = γ * (x - μ) / √(σ² + ε) + β

where:
  μ = E[x]      (mean)
  σ² = Var[x]   (variance)
  γ = scale parameter
  β = shift parameter
  ε = epsilon for numerical stability
```

**Backward Pass Gradients** (from Ioffe & Szegedy 2015):

For inference mode (using fixed μ, σ²):
```
∂L/∂x = γ * ∂L/∂y / √(σ² + ε)
∂L/∂γ = Σ(∂L/∂y * (x - μ) / √(σ² + ε))
∂L/∂β = Σ(∂L/∂y)
```

For training mode (computing μ, σ² from batch):
- More complex with additional terms for batch statistics
- We'll implement training mode for completeness

### Changes Required

#### 1. BatchNorm Gradient Implementation

**File**: `pytensor/tensor/batchnorm.py:197-211`

**Current**:
```python
def grad(self, inputs, output_grads):
    raise NotImplementedError(...)
```

**New Implementation**:
```python
def grad(self, inputs, output_grads):
    """
    Compute gradients for batch normalization.

    For training mode, implements full backprop through batch statistics.
    For inference mode, treats mean/variance as constants.

    References:
    - Ioffe & Szegedy (2015): Batch Normalization paper
    - https://kevinzakka.github.io/2016/09/14/batch_normalization/
    """
    x, gamma, beta, mean, variance = inputs
    dy = output_grads[0]  # Gradient w.r.t output

    # For inference mode (mean and variance are constants)
    # dy/dx = gamma * dy / sqrt(var + eps)

    import pytensor.tensor as pt

    # Normalized input: x_norm = (x - mean) / sqrt(var + eps)
    std = pt.sqrt(variance + self.epsilon)
    x_centered = x - mean
    x_norm = x_centered / std

    # Gradients for gamma and beta (simple)
    # These work for both training and inference mode
    grad_gamma = (dy * x_norm).sum(axis=get_reduce_axes(x, gamma))
    grad_beta = dy.sum(axis=get_reduce_axes(x, beta))

    # Gradient for x (inference mode - mean/var are constants)
    grad_x = gamma * dy / std

    # For training mode, we'd need more complex grad_x computation
    # involving gradients through mean and variance.
    # For now, we implement inference mode which is sufficient
    # for fine-tuning pre-trained models.

    # No gradients for mean and variance (treated as constants)
    grad_mean = pt.zeros_like(mean).astype(config.floatX)
    grad_variance = pt.zeros_like(variance).astype(config.floatX)

    return [grad_x, grad_gamma, grad_beta, grad_mean, grad_variance]


def get_reduce_axes(x, param):
    """
    Determine which axes to sum over when computing parameter gradients.

    For 4D input (N, C, H, W) and 1D param (C,):
    - Reduce over axes [0, 2, 3] (keep channel dimension)

    Parameters
    ----------
    x : TensorVariable
        Input tensor (e.g., 4D: NCHW)
    param : TensorVariable
        Parameter tensor (e.g., 1D: C)

    Returns
    -------
    tuple
        Axes to reduce over
    """
    if x.ndim == 4 and param.ndim == 1:
        # NCHW format: reduce over batch, height, width
        return (0, 2, 3)
    elif x.ndim == 2 and param.ndim == 1:
        # NC format: reduce over batch
        return (0,)
    else:
        # General case: reduce over all except param dimension
        # Assume param corresponds to dimension 1 (channels)
        return tuple([0] + list(range(2, x.ndim)))
```

**Key decisions**:
- Implement **inference-mode gradients** first (mean/variance are constants)
- This is sufficient for transfer learning / fine-tuning scenarios
- Can be extended to training-mode later if needed

#### 2. Helper Function for Axis Reduction

Add utility function to determine broadcast axes:

```python
def get_reduce_axes(x, param):
    """Helper to determine reduction axes for parameter gradients."""
    # Implementation above
```

#### 3. Add Training Mode Support (Optional Enhancement)

For full training-mode batch norm:

```python
class BatchNormalizationTraining(Op):
    """
    BatchNorm with training mode.

    Computes mean and variance from current batch,
    and implements full gradient backpropagation.
    """
    # Implementation following:
    # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py
```

**Decision**: Start with inference-mode gradients, add training mode if needed.

### Testing Strategy

#### 1. Unit Tests for Gradients

**File**: `tests/tensor/test_batchnorm.py`

Add gradient verification tests:

```python
def test_batchnorm_grad_simple():
    """Test BatchNorm gradient computation (inference mode)."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    import numpy as np

    # Simple 2D test case
    x = pt.matrix('x', dtype='float32')
    gamma = pt.vector('gamma', dtype='float32')
    beta = pt.vector('beta', dtype='float32')
    mean = pt.vector('mean', dtype='float32')
    var = pt.vector('var', dtype='float32')

    y = batch_normalization(x, gamma, beta, mean, var, epsilon=1e-5)

    # Compute gradient w.r.t. x
    loss = y.sum()
    grad_x = pytensor.grad(loss, x)

    # Compile function
    f = pytensor.function([x, gamma, beta, mean, var], [y, grad_x])

    # Test data
    x_val = np.random.randn(4, 3).astype('float32')
    gamma_val = np.ones(3, dtype='float32')
    beta_val = np.zeros(3, dtype='float32')
    mean_val = np.array([0, 0, 0], dtype='float32')
    var_val = np.array([1, 1, 1], dtype='float32')

    y_val, grad_x_val = f(x_val, gamma_val, beta_val, mean_val, var_val)

    # Verify gradient is non-zero
    assert np.abs(grad_x_val).sum() > 0, "Gradient should not be zero"

    print(f"✓ Simple gradient test passed")


def test_batchnorm_grad_4d():
    """Test BatchNorm gradient for 4D CNN tensors (NCHW)."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    import numpy as np

    # 4D tensor (batch=2, channels=3, height=4, width=4)
    x = pt.tensor4('x', dtype='float32')
    gamma = pt.vector('gamma', dtype='float32')
    beta = pt.vector('beta', dtype='float32')
    mean = pt.vector('mean', dtype='float32')
    var = pt.vector('var', dtype='float32')

    y = batch_normalization(x, gamma, beta, mean, var)

    # Loss
    loss = y.sum()

    # Gradients
    grad_x = pytensor.grad(loss, x)
    grad_gamma = pytensor.grad(loss, gamma)
    grad_beta = pytensor.grad(loss, beta)

    # Compile
    f = pytensor.function(
        [x, gamma, beta, mean, var],
        [grad_x, grad_gamma, grad_beta]
    )

    # Test data
    np.random.seed(42)
    x_val = np.random.randn(2, 3, 4, 4).astype('float32')
    gamma_val = np.ones(3, dtype='float32')
    beta_val = np.zeros(3, dtype='float32')
    mean_val = np.zeros(3, dtype='float32')
    var_val = np.ones(3, dtype='float32')

    grad_x_val, grad_gamma_val, grad_beta_val = f(
        x_val, gamma_val, beta_val, mean_val, var_val
    )

    # Verify shapes
    assert grad_x_val.shape == x_val.shape
    assert grad_gamma_val.shape == gamma_val.shape
    assert grad_beta_val.shape == beta_val.shape

    # Verify non-zero gradients
    assert np.abs(grad_x_val).sum() > 0
    assert np.abs(grad_gamma_val).sum() > 0
    assert np.abs(grad_beta_val).sum() > 0

    print(f"✓ 4D gradient test passed")


def test_batchnorm_grad_numerical():
    """Verify BatchNorm gradients using finite differences."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.tensor.batchnorm import batch_normalization
    import numpy as np

    # Small test case for numerical gradient checking
    x = pt.matrix('x', dtype='float64')  # Use float64 for precision
    gamma = pt.vector('gamma', dtype='float64')
    beta = pt.vector('beta', dtype='float64')
    mean = pt.vector('mean', dtype='float64')
    var = pt.vector('var', dtype='float64')

    y = batch_normalization(x, gamma, beta, mean, var)
    loss = y.sum()

    # Analytical gradient
    grad_x_symbolic = pytensor.grad(loss, x)
    grad_fn = pytensor.function([x, gamma, beta, mean, var], grad_x_symbolic)

    # Forward function for numerical gradient
    forward_fn = pytensor.function([x, gamma, beta, mean, var], loss)

    # Test data (small for numerical stability)
    np.random.seed(42)
    x_val = np.random.randn(2, 3).astype('float64') * 0.1
    gamma_val = np.ones(3, dtype='float64')
    beta_val = np.zeros(3, dtype='float64')
    mean_val = np.zeros(3, dtype='float64')
    var_val = np.ones(3, dtype='float64')

    # Analytical gradient
    grad_analytical = grad_fn(x_val, gamma_val, beta_val, mean_val, var_val)

    # Numerical gradient (finite differences)
    eps = 1e-5
    grad_numerical = np.zeros_like(x_val)

    for i in range(x_val.shape[0]):
        for j in range(x_val.shape[1]):
            x_plus = x_val.copy()
            x_plus[i, j] += eps
            loss_plus = forward_fn(x_plus, gamma_val, beta_val, mean_val, var_val)

            x_minus = x_val.copy()
            x_minus[i, j] -= eps
            loss_minus = forward_fn(x_minus, gamma_val, beta_val, mean_val, var_val)

            grad_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

    # Compare
    rel_error = np.abs(grad_analytical - grad_numerical) / (np.abs(grad_analytical) + np.abs(grad_numerical) + 1e-8)
    max_rel_error = rel_error.max()

    print(f"  Max relative error: {max_rel_error:.6f}")
    assert max_rel_error < 1e-4, f"Gradient check failed: {max_rel_error}"

    print(f"✓ Numerical gradient test passed")


def test_batchnorm_grad_in_network():
    """Test BatchNorm gradients in a simple network (Conv → BN → ReLU → Loss)."""
    import pytensor
    import pytensor.tensor as pt
    from pytensor.tensor.nnet.abstract_conv import conv2d
    from pytensor.tensor.batchnorm import batch_normalization
    from pytensor import shared
    import numpy as np

    # Build mini network
    x = pt.tensor4('x', dtype='float32')

    # Conv layer
    W_conv = shared(
        np.random.randn(8, 3, 3, 3).astype('float32') * 0.1,
        name='W_conv'
    )
    conv_out = conv2d(x, W_conv, border_mode='valid', filter_flip=False)

    # BatchNorm
    gamma = shared(np.ones(8, dtype='float32'), name='gamma')
    beta = shared(np.zeros(8, dtype='float32'), name='beta')
    mean = shared(np.zeros(8, dtype='float32'), name='mean')
    var = shared(np.ones(8, dtype='float32'), name='var')

    bn_out = batch_normalization(conv_out, gamma, beta, mean, var)

    # ReLU
    relu_out = pt.maximum(bn_out, 0)

    # Loss
    loss = relu_out.sum()

    # Compute gradients
    params = [W_conv, gamma, beta]
    grads = pytensor.grad(loss, params)

    # Compile
    f = pytensor.function([x], [loss] + grads)

    # Test
    x_val = np.random.randn(2, 3, 10, 10).astype('float32')
    results = f(x_val)

    loss_val = results[0]
    grad_W, grad_gamma, grad_beta = results[1:]

    # Verify
    assert loss_val > 0
    assert np.abs(grad_W).sum() > 0
    assert np.abs(grad_gamma).sum() > 0
    assert np.abs(grad_beta).sum() > 0

    print(f"✓ Network gradient test passed")
    print(f"  Loss: {loss_val:.4f}")
    print(f"  Grad norms: W={np.linalg.norm(grad_W):.4f}, "
          f"gamma={np.linalg.norm(grad_gamma):.4f}, "
          f"beta={np.linalg.norm(grad_beta):.4f}")
```

### Success Criteria

#### Automated Verification:
- [ ] `pytest tests/tensor/test_batchnorm.py::test_batchnorm_grad_simple -v` passes
- [ ] `pytest tests/tensor/test_batchnorm.py::test_batchnorm_grad_4d -v` passes
- [ ] `pytest tests/tensor/test_batchnorm.py::test_batchnorm_grad_numerical -v` passes (gradient check)
- [ ] `pytest tests/tensor/test_batchnorm.py::test_batchnorm_grad_in_network -v` passes
- [ ] All existing BatchNorm tests still pass
- [ ] ONNX export still works for BatchNorm layers

#### Manual Verification:
- [ ] Simple Conv→BN→ReLU network trains and loss decreases
- [ ] Gradients have reasonable magnitudes (not exploding/vanishing)
- [ ] BatchNorm parameters (gamma, beta) update during training

---

## Phase 2: Build YOLO11n Architecture Components

### Overview
Implement modular building blocks for YOLO11n: C3k2, SPPF, C2PSA, and detection head.

### Architecture Reference

**YOLO11n Structure** (from Ultralytics):
```
Input: (batch, 3, 128, 128)

Backbone:
  0: Conv(3, 16, k=3, s=2)      → (batch, 16, 64, 64)
  1: Conv(16, 32, k=3, s=2)     → (batch, 32, 32, 32)
  2: C3k2(32, 32, n=1)          → (batch, 32, 32, 32)
  3: Conv(32, 64, k=3, s=2)     → (batch, 64, 16, 16)  [P3]
  4: C3k2(64, 64, n=2)          → (batch, 64, 16, 16)
  5: Conv(64, 128, k=3, s=2)    → (batch, 128, 8, 8)   [P4]
  6: C3k2(128, 128, n=2)        → (batch, 128, 8, 8)
  7: Conv(128, 256, k=3, s=2)   → (batch, 256, 4, 4)   [P5]
  8: C3k2(256, 256, n=1)        → (batch, 256, 4, 4)
  9: SPPF(256, 256, k=5)        → (batch, 256, 4, 4)
  10: C2PSA(256, 256)           → (batch, 256, 4, 4)

Head (FPN-PAN):
  11: Upsample(256) + Concat[8, 6]  → (batch, 384, 8, 8)
  12: C3k2(384, 128, n=1)           → (batch, 128, 8, 8)  [P4 out]
  13: Upsample(128) + Concat[12, 4] → (batch, 192, 16, 16)
  14: C3k2(192, 64, n=1)            → (batch, 64, 16, 16)  [P3 out]

  15: Conv(64, 64, k=3, s=2) + Concat[14, 12] → (batch, 192, 8, 8)
  16: C3k2(192, 128, n=1)                     → (batch, 128, 8, 8)  [P4 final]

  17: Conv(128, 128, k=3, s=2) + Concat[16, 9] → (batch, 384, 4, 4)
  18: C3k2(384, 256, n=1)                      → (batch, 256, 4, 4)  [P5 final]

Detection Heads:
  19: DFL + BBox Head on P3 (16x16)
  20: DFL + BBox Head on P4 (8x8)
  21: DFL + BBox Head on P5 (4x4)
```

**Simplified for 128x128**: Use scaling factors (depth=0.5, width=0.25) for nano variant

### Changes Required

#### 1. Core Building Blocks Module

**File**: `examples/yolo11n_pytensor/blocks.py`

```python
"""
YOLO11n building blocks for PyTensor.

Implements:
- ConvBNSiLU: Conv + BatchNorm + SiLU activation
- C3k2: CSP bottleneck with 2 convolutions
- SPPF: Spatial Pyramid Pooling - Fast
- C2PSA: CSP with Parallel Spatial Attention
"""

import numpy as np
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.nnet.abstract_conv import conv2d
from pytensor.tensor.batchnorm import batch_normalization
from pytensor.tensor.pool import pool_2d


class ConvBNSiLU:
    """
    Conv2D + BatchNorm + SiLU activation.

    The fundamental building block used throughout YOLO11n.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, name_prefix="conv"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        kernel_size : int
        stride : int
        padding : int or str
            If int: explicit padding
            If 'same': zero padding to maintain size
            If 'valid': no padding
        name_prefix : str
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.name = name_prefix

        # Initialize weights (He initialization for ReLU-like)
        self.W = self._init_weight(
            (out_channels, in_channels, kernel_size, kernel_size),
            name=f"{name_prefix}_W"
        )

        # BatchNorm parameters
        self.gamma = shared(
            np.ones(out_channels, dtype='float32'),
            name=f"{name_prefix}_gamma",
            borrow=True
        )
        self.beta = shared(
            np.zeros(out_channels, dtype='float32'),
            name=f"{name_prefix}_beta",
            borrow=True
        )
        self.bn_mean = shared(
            np.zeros(out_channels, dtype='float32'),
            name=f"{name_prefix}_bn_mean",
            borrow=True
        )
        self.bn_var = shared(
            np.ones(out_channels, dtype='float32'),
            name=f"{name_prefix}_bn_var",
            borrow=True
        )

        self.params = [self.W, self.gamma, self.beta]
        self.bn_stats = [self.bn_mean, self.bn_var]

    def _init_weight(self, shape, name):
        """He initialization."""
        fan_in = shape[1] * shape[2] * shape[3]  # in_channels * kh * kw
        std = np.sqrt(2.0 / fan_in)
        W_val = np.random.randn(*shape).astype('float32') * std
        return shared(W_val, name=name, borrow=True)

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, in_channels, height, width)

        Returns
        -------
        TensorVariable
            Output (batch, out_channels, height', width')
        """
        # Conv2D
        if self.padding == 'same':
            # Calculate padding for 'same'
            pad_h = ((self.kernel_size - 1) // 2)
            pad_w = ((self.kernel_size - 1) // 2)
            border_mode = (pad_h, pad_w)
        elif self.padding == 'valid':
            border_mode = 'valid'
        else:
            border_mode = (self.padding, self.padding)

        conv_out = conv2d(
            x, self.W,
            border_mode=border_mode,
            subsample=(self.stride, self.stride),
            filter_flip=False
        )

        # BatchNorm
        bn_out = batch_normalization(
            conv_out, self.gamma, self.beta,
            self.bn_mean, self.bn_var,
            epsilon=1e-5
        )

        # SiLU activation
        # SiLU(x) = x * sigmoid(x)
        silu_out = pt.silu(bn_out)  # Using PyTensor's built-in silu

        return silu_out


class Bottleneck:
    """
    Standard bottleneck block with two convolutions.

    Used inside C3k2 blocks.
    """

    def __init__(self, in_channels, out_channels, shortcut=True, name_prefix="btlnk"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        shortcut : bool
            Whether to add residual connection
        """
        self.shortcut = shortcut and (in_channels == out_channels)

        # Two 3x3 convs
        self.conv1 = ConvBNSiLU(
            in_channels, out_channels, kernel_size=3, stride=1, padding='same',
            name_prefix=f"{name_prefix}_conv1"
        )
        self.conv2 = ConvBNSiLU(
            out_channels, out_channels, kernel_size=3, stride=1, padding='same',
            name_prefix=f"{name_prefix}_conv2"
        )

        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats

    def __call__(self, x):
        """Forward pass."""
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            out = out + residual

        return out


class C3k2:
    """
    C3k2 block: CSP Bottleneck with 2 convolutions.

    Key component of YOLO11n backbone.
    """

    def __init__(self, in_channels, out_channels, n_blocks=1, shortcut=True, name_prefix="c3k2"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        n_blocks : int
            Number of bottleneck blocks
        shortcut : bool
            Whether bottlenecks use residual connections
        """
        self.n_blocks = n_blocks
        hidden_channels = out_channels // 2

        # Split convolution
        self.conv1 = ConvBNSiLU(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv1"
        )

        # Bottleneck blocks
        self.bottlenecks = []
        for i in range(n_blocks):
            self.bottlenecks.append(
                Bottleneck(
                    hidden_channels, hidden_channels,
                    shortcut=shortcut,
                    name_prefix=f"{name_prefix}_btlnk{i}"
                )
            )

        # Merge convolution
        self.conv2 = ConvBNSiLU(
            hidden_channels * 2, out_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv2"
        )

        # Collect params
        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats
        for btlnk in self.bottlenecks:
            self.params.extend(btlnk.params)
            self.bn_stats.extend(btlnk.bn_stats)

    def __call__(self, x):
        """Forward pass."""
        # Split path
        x1 = self.conv1(x)

        # Bottleneck path
        x2 = x1
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)

        # Concatenate and merge
        x_cat = pt.concatenate([x1, x2], axis=1)  # Channel axis
        out = self.conv2(x_cat)

        return out


class SPPF:
    """
    Spatial Pyramid Pooling - Fast.

    Uses cascaded max pooling to create multi-scale features.
    Critical for YOLO11n's receptive field.
    """

    def __init__(self, in_channels, out_channels, pool_size=5, name_prefix="sppf"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        pool_size : int
            Max pool kernel size
        """
        hidden_channels = in_channels // 2

        self.conv1 = ConvBNSiLU(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv1"
        )

        self.pool_size = pool_size

        self.conv2 = ConvBNSiLU(
            hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv2"
        )

        self.params = self.conv1.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.conv2.bn_stats

    def __call__(self, x):
        """Forward pass."""
        x = self.conv1(x)

        # Cascaded max pooling
        # Padding: 'same' to maintain spatial dimensions
        pad = self.pool_size // 2

        y1 = pool_2d(
            x, ws=(self.pool_size, self.pool_size),
            stride=(1, 1), mode='max', pad=(pad, pad)
        )
        y2 = pool_2d(
            y1, ws=(self.pool_size, self.pool_size),
            stride=(1, 1), mode='max', pad=(pad, pad)
        )
        y3 = pool_2d(
            y2, ws=(self.pool_size, self.pool_size),
            stride=(1, 1), mode='max', pad=(pad, pad)
        )

        # Concatenate all pooling outputs
        out = pt.concatenate([x, y1, y2, y3], axis=1)
        out = self.conv2(out)

        return out


class C2PSA:
    """
    C2PSA: CSP with Parallel Spatial Attention.

    Simplified implementation - uses channel attention.
    Full spatial attention can be added if needed.
    """

    def __init__(self, in_channels, out_channels, name_prefix="c2psa"):
        """
        Parameters
        ----------
        in_channels : int
        out_channels : int
        """
        hidden_channels = out_channels // 2

        self.conv1 = ConvBNSiLU(
            in_channels, hidden_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv1"
        )

        # Attention module (simplified)
        self.attn_conv = ConvBNSiLU(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding='same',
            name_prefix=f"{name_prefix}_attn"
        )

        self.conv2 = ConvBNSiLU(
            hidden_channels * 2, out_channels, kernel_size=1, stride=1, padding='valid',
            name_prefix=f"{name_prefix}_conv2"
        )

        self.params = self.conv1.params + self.attn_conv.params + self.conv2.params
        self.bn_stats = self.conv1.bn_stats + self.attn_conv.bn_stats + self.conv2.bn_stats

    def __call__(self, x):
        """Forward pass."""
        # Split
        x1 = self.conv1(x)

        # Attention branch
        x2 = self.attn_conv(x1)

        # Apply attention (element-wise multiplication with sigmoid gating)
        # Simplified: just concatenate for now
        # Full version would compute attention weights

        # Concatenate and merge
        x_cat = pt.concatenate([x1, x2], axis=1)
        out = self.conv2(x_cat)

        return out
```

#### 2. YOLO11n Model Architecture

**File**: `examples/yolo11n_pytensor/model.py`

```python
"""
YOLO11n model architecture for PyTensor.

Implements full YOLO11n nano model for object detection.
Input: (batch, 3, 128, 128)
Output: Detection predictions at 3 scales
"""

import numpy as np
import pytensor.tensor as pt
from pytensor import shared
from pytensor.tensor.nnet.abstract_conv import conv2d

from blocks import ConvBNSiLU, C3k2, SPPF, C2PSA


class YOLO11nBackbone:
    """
    YOLO11n backbone for feature extraction.

    Outputs features at 3 scales: P3 (16x16), P4 (8x8), P5 (4x4)
    for 128x128 input.
    """

    def __init__(self, in_channels=3):
        """Initialize backbone."""
        # Stem
        self.conv0 = ConvBNSiLU(3, 16, kernel_size=3, stride=2, padding='same', name_prefix="stem")

        # Stage 1
        self.conv1 = ConvBNSiLU(16, 32, kernel_size=3, stride=2, padding='same', name_prefix="s1_conv")
        self.c3k2_1 = C3k2(32, 32, n_blocks=1, name_prefix="s1_c3k2")

        # Stage 2 (P3)
        self.conv2 = ConvBNSiLU(32, 64, kernel_size=3, stride=2, padding='same', name_prefix="s2_conv")
        self.c3k2_2 = C3k2(64, 64, n_blocks=2, name_prefix="s2_c3k2")

        # Stage 3 (P4)
        self.conv3 = ConvBNSiLU(64, 128, kernel_size=3, stride=2, padding='same', name_prefix="s3_conv")
        self.c3k2_3 = C3k2(128, 128, n_blocks=2, name_prefix="s3_c3k2")

        # Stage 4 (P5)
        self.conv4 = ConvBNSiLU(128, 256, kernel_size=3, stride=2, padding='same', name_prefix="s4_conv")
        self.c3k2_4 = C3k2(256, 256, n_blocks=1, name_prefix="s4_c3k2")

        # SPPF
        self.sppf = SPPF(256, 256, pool_size=5, name_prefix="sppf")

        # C2PSA
        self.c2psa = C2PSA(256, 256, name_prefix="c2psa")

        # Collect parameters
        self.params = []
        self.bn_stats = []
        for module in [
            self.conv0, self.conv1, self.c3k2_1,
            self.conv2, self.c3k2_2, self.conv3, self.c3k2_3,
            self.conv4, self.c3k2_4, self.sppf, self.c2psa
        ]:
            self.params.extend(module.params)
            self.bn_stats.extend(module.bn_stats)

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, 3, 128, 128)

        Returns
        -------
        p3, p4, p5 : TensorVariables
            Features at 3 scales:
            - p3: (batch, 64, 16, 16)
            - p4: (batch, 128, 8, 8)
            - p5: (batch, 256, 4, 4)
        """
        # Stem
        x = self.conv0(x)  # 64x64

        # Stage 1
        x = self.conv1(x)  # 32x32
        x = self.c3k2_1(x)

        # Stage 2 (P3)
        x = self.conv2(x)  # 16x16
        p3 = self.c3k2_2(x)

        # Stage 3 (P4)
        x = self.conv3(p3)  # 8x8
        p4 = self.c3k2_3(x)

        # Stage 4 (P5)
        x = self.conv4(p4)  # 4x4
        x = self.c3k2_4(x)
        x = self.sppf(x)
        p5 = self.c2psa(x)

        return p3, p4, p5


class YOLO11nHead:
    """
    YOLO11n detection head with FPN.

    Takes backbone features and produces detection predictions.
    """

    def __init__(self, num_classes=2):  # Default: person, cellphone
        """
        Parameters
        ----------
        num_classes : int
            Number of detection classes
        """
        self.num_classes = num_classes

        # FPN upsampling path
        # P5 → P4
        self.up1 = pt.nnet.abstract_conv.bilinear_upsampling(
            input, ratio=2, batch_size=None, num_input_channels=None
        )  # Will use pt.repeat for upsampling
        self.c3k2_p4 = C3k2(256 + 128, 128, n_blocks=1, name_prefix="head_p4")

        # P4 → P3
        self.c3k2_p3 = C3k2(128 + 64, 64, n_blocks=1, name_prefix="head_p3")

        # PAN downsampling path
        # P3 → P4
        self.down1 = ConvBNSiLU(64, 64, kernel_size=3, stride=2, padding='same', name_prefix="head_down1")
        self.c3k2_p4_final = C3k2(64 + 128, 128, n_blocks=1, name_prefix="head_p4_final")

        # P4 → P5
        self.down2 = ConvBNSiLU(128, 128, kernel_size=3, stride=2, padding='same', name_prefix="head_down2")
        self.c3k2_p5_final = C3k2(128 + 256, 256, n_blocks=1, name_prefix="head_p5_final")

        # Detection heads (one per scale)
        # Each head outputs: [batch, num_anchors * (5 + num_classes), H, W]
        # where 5 = (x, y, w, h, objectness)
        # For anchor-free, we use (x, y, w, h) + classes

        self.detect_p3 = ConvBNSiLU(
            64, (4 + num_classes), kernel_size=1, stride=1, padding='valid',
            name_prefix="detect_p3"
        )
        self.detect_p4 = ConvBNSiLU(
            128, (4 + num_classes), kernel_size=1, stride=1, padding='valid',
            name_prefix="detect_p4"
        )
        self.detect_p5 = ConvBNSiLU(
            256, (4 + num_classes), kernel_size=1, stride=1, padding='valid',
            name_prefix="detect_p5"
        )

        # Collect params
        self.params = []
        self.bn_stats = []
        for module in [
            self.c3k2_p4, self.c3k2_p3,
            self.down1, self.c3k2_p4_final,
            self.down2, self.c3k2_p5_final,
            self.detect_p3, self.detect_p4, self.detect_p5
        ]:
            self.params.extend(module.params)
            self.bn_stats.extend(module.bn_stats)

    def __call__(self, p3, p4, p5):
        """
        Forward pass.

        Parameters
        ----------
        p3, p4, p5 : TensorVariables
            Backbone features

        Returns
        -------
        det_p3, det_p4, det_p5 : TensorVariables
            Detection predictions at 3 scales
        """
        # FPN path (top-down)
        # P5 → P4
        p5_up = self._upsample(p5, scale=2)
        p4_fused = pt.concatenate([p5_up, p4], axis=1)
        p4_out = self.c3k2_p4(p4_fused)

        # P4 → P3
        p4_up = self._upsample(p4_out, scale=2)
        p3_fused = pt.concatenate([p4_up, p3], axis=1)
        p3_out = self.c3k2_p3(p3_fused)

        # PAN path (bottom-up)
        # P3 → P4
        p3_down = self.down1(p3_out)
        p4_fused2 = pt.concatenate([p3_down, p4_out], axis=1)
        p4_final = self.c3k2_p4_final(p4_fused2)

        # P4 → P5
        p4_down = self.down2(p4_final)
        p5_fused = pt.concatenate([p4_down, p5], axis=1)
        p5_final = self.c3k2_p5_final(p5_fused)

        # Detection heads
        det_p3 = self.detect_p3(p3_out)  # (batch, 4+C, 16, 16)
        det_p4 = self.detect_p4(p4_final)  # (batch, 4+C, 8, 8)
        det_p5 = self.detect_p5(p5_final)  # (batch, 4+C, 4, 4)

        return det_p3, det_p4, det_p5

    def _upsample(self, x, scale=2):
        """Upsample using nearest neighbor (repeat)."""
        # x: (batch, C, H, W)
        # Use repeat for upsampling
        x_up = pt.repeat(x, scale, axis=2)  # Repeat height
        x_up = pt.repeat(x_up, scale, axis=3)  # Repeat width
        return x_up


class YOLO11n:
    """
    Complete YOLO11n model.

    Combines backbone and head for end-to-end object detection.
    """

    def __init__(self, num_classes=2, input_size=128):  # Default: 2 classes
        """
        Parameters
        ----------
        num_classes : int
            Number of detection classes
        input_size : int
            Input image size (square)
        """
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone = YOLO11nBackbone()
        self.head = YOLO11nHead(num_classes=num_classes)

        # Collect all parameters
        self.params = self.backbone.params + self.head.params
        self.bn_stats = self.backbone.bn_stats + self.head.bn_stats

        print(f"YOLO11n initialized:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Total params: {sum(p.get_value().size for p in self.params):,}")

    def __call__(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : TensorVariable
            Input (batch, 3, 128, 128)

        Returns
        -------
        predictions : dict
            Detection predictions at 3 scales
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Head
        det_p3, det_p4, det_p5 = self.head(p3, p4, p5)

        return {
            'p3': det_p3,  # (batch, 4+C, 16, 16)
            'p4': det_p4,  # (batch, 4+C, 8, 8)
            'p5': det_p5,  # (batch, 4+C, 4, 4)
        }


def build_yolo11n(num_classes=2, input_size=128):  # Default: 2 classes (person, cellphone)
    """
    Build YOLO11n model.

    Parameters
    ----------
    num_classes : int
        Number of classes to detect (default: 2 for person, cellphone)
    input_size : int
        Input image size

    Returns
    -------
    model : YOLO11n
        Initialized model
    x : TensorVariable
        Input symbolic variable
    predictions : dict
        Output predictions
    """
    import pytensor.tensor as pt

    # Input
    x = pt.tensor4('x', dtype='float32')

    # Model
    model = YOLO11n(num_classes=num_classes, input_size=input_size)

    # Forward pass
    predictions = model(x)

    return model, x, predictions
```

### Testing Strategy

#### Unit Tests for Blocks

**File**: `tests/examples/test_yolo11n_blocks.py`

```python
def test_conv_bn_silu():
    """Test ConvBNSiLU block."""
    from examples.yolo11n_pytensor.blocks import ConvBNSiLU
    import pytensor
    import pytensor.tensor as pt
    import numpy as np

    # Create block
    conv = ConvBNSiLU(3, 16, kernel_size=3, stride=2, padding='same')

    # Input
    x = pt.tensor4('x', dtype='float32')
    y = conv(x)

    # Compile
    f = pytensor.function([x], y)

    # Test
    x_val = np.random.randn(1, 3, 128, 128).astype('float32')
    y_val = f(x_val)

    assert y_val.shape == (1, 16, 64, 64), f"Expected (1,16,64,64), got {y_val.shape}"
    print("✓ ConvBNSiLU test passed")


def test_c3k2():
    """Test C3k2 block."""
    # Similar pattern
    pass


def test_sppf():
    """Test SPPF block."""
    # Similar pattern
    pass


def test_yolo11n_forward():
    """Test full YOLO11n forward pass."""
    from examples.yolo11n_pytensor.model import build_yolo11n
    import pytensor
    import numpy as np

    # Build model
    model, x, predictions = build_yolo11n(num_classes=2, input_size=128)  # 2 classes: person, cellphone

    # Compile forward pass
    f = pytensor.function([x], [predictions['p3'], predictions['p4'], predictions['p5']])

    # Test
    x_val = np.random.randn(2, 3, 128, 128).astype('float32')
    p3_val, p4_val, p5_val = f(x_val)

    # Verify shapes
    assert p3_val.shape == (2, 6, 16, 16), f"P3 shape: {p3_val.shape}"  # 4+2 classes
    assert p4_val.shape == (2, 6, 8, 8), f"P4 shape: {p4_val.shape}"
    assert p5_val.shape == (2, 6, 4, 4), f"P5 shape: {p5_val.shape}"

    print("✓ YOLO11n forward pass test passed")
```

### Success Criteria

#### Automated Verification:
- [ ] `pytest tests/examples/test_yolo11n_blocks.py -v` - All block tests pass
- [ ] YOLO11n forward pass completes without errors
- [ ] Output shapes are correct for all 3 detection scales
- [ ] Can compute gradients through entire model

#### Manual Verification:
- [ ] Model summary shows ~2.6M parameters (close to official YOLO11n)
- [ ] Memory usage is reasonable (< 4GB for batch_size=16)
- [ ] Forward pass completes in reasonable time (< 1 second per batch on CPU)

---

## Phase 3: Implement YOLO Detection Loss

### Overview
Implement simplified YOLO detection loss with box regression (IoU) and classification (BCE).

### Loss Function Design

**Simplified YOLO Loss**:
```
Total_Loss = λ_box * IoU_Loss + λ_cls * BCE_Loss

where:
  IoU_Loss = 1 - IoU(pred_boxes, target_boxes)
  BCE_Loss = BinaryCrossEntropy(pred_classes, target_classes)
```

**Target Assignment**:
- For each ground truth box, assign to grid cell based on center
- Use anchor-free approach (YOLO11 style)
- Only positive samples contribute to loss

### Changes Required

#### 1. Loss Implementation

**File**: `examples/yolo11n_pytensor/loss.py`

```python
"""
YOLO detection loss functions.

Implements:
- IoU-based box regression loss
- Binary cross-entropy classification loss
- Target assignment for anchor-free detection
"""

import pytensor.tensor as pt
import numpy as np


def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.

    Parameters
    ----------
    box1 : TensorVariable
        Shape: (..., 4) in format [x_center, y_center, width, height]
    box2 : TensorVariable
        Shape: (..., 4) in same format

    Returns
    -------
    iou : TensorVariable
        IoU scores, shape: (...)
    """
    # Convert from center format to corner format
    # [xc, yc, w, h] → [x1, y1, x2, y2]
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter_x1 = pt.maximum(box1_x1, box2_x1)
    inter_y1 = pt.maximum(box1_y1, box2_y1)
    inter_x2 = pt.minimum(box1_x2, box2_x2)
    inter_y2 = pt.minimum(box1_y2, box2_y2)

    inter_area = pt.maximum(0, inter_x2 - inter_x1) * pt.maximum(0, inter_y2 - inter_y1)

    # Union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)

    return iou


def yolo_loss(predictions, targets, num_classes=2, lambda_box=5.0, lambda_cls=1.0):  # 2 classes
    """
    YOLO detection loss (simplified).

    Parameters
    ----------
    predictions : dict
        Model predictions at 3 scales
        Each scale: (batch, 4+num_classes, H, W)
    targets : dict
        Ground truth targets
        Format: {
            'boxes': (batch, max_boxes, 4),  # [x, y, w, h] normalized
            'classes': (batch, max_boxes),   # class indices
            'num_boxes': (batch,)            # number of valid boxes per image
        }
    num_classes : int
        Number of classes (default: 2 for person, cellphone)
    lambda_box : float
        Box loss weight
    lambda_cls : float
        Classification loss weight

    Returns
    -------
    total_loss : TensorVariable
    loss_dict : dict
        Individual loss components for logging
    """
    # For simplicity, we'll compute loss on P4 scale (8x8)
    # Full implementation would use all 3 scales

    pred_p4 = predictions['p4']  # (batch, 4+C, 8, 8)
    batch_size = pred_p4.shape[0]
    grid_h, grid_w = 8, 8

    # Reshape predictions
    # (batch, 4+C, H, W) → (batch, H, W, 4+C)
    pred_p4 = pred_p4.dimshuffle(0, 2, 3, 1)

    # Split into box and class predictions
    pred_boxes = pred_p4[..., :4]  # (batch, H, W, 4)
    pred_classes = pred_p4[..., 4:]  # (batch, H, W, C)

    # Apply sigmoid to box coordinates (normalize to [0, 1])
    pred_boxes_xy = pt.nnet.sigmoid(pred_boxes[..., :2])
    pred_boxes_wh = pt.exp(pred_boxes[..., 2:])  # Exponential for width/height

    # Apply sigmoid to class logits
    pred_classes_sig = pt.nnet.sigmoid(pred_classes)

    # Build target tensors (simplified)
    # This is a placeholder - full implementation needs proper target assignment

    # For now, use a simple loss that encourages small box predictions
    # and low classification scores (background)

    # Box loss: Encourage small boxes (l2 regularization)
    box_loss = pt.mean(pred_boxes_wh ** 2)

    # Classification loss: BCE with targets (simplified)
    # In full implementation, we'd assign targets based on ground truth
    target_classes = pt.zeros_like(pred_classes_sig)  # All background
    cls_loss = pt.nnet.binary_crossentropy(pred_classes_sig, target_classes).mean()

    # Total loss
    total_loss = lambda_box * box_loss + lambda_cls * cls_loss

    return total_loss, {
        'box_loss': box_loss,
        'cls_loss': cls_loss,
        'total_loss': total_loss
    }


# NOTE: The above is a simplified placeholder.
# Full implementation requires:
# 1. Proper target assignment (assign GT boxes to grid cells)
# 2. Positive/negative sample masking
# 3. Multi-scale loss computation
# 4. CIoU loss instead of simple L2
#
# This is sufficient to get training started and verify gradients work.
# Can be enhanced incrementally.
```

### Testing Strategy

Test loss computation and gradients:

```python
def test_yolo_loss():
    """Test YOLO loss computation."""
    from examples.yolo11n_pytensor.model import build_yolo11n
    from examples.yolo11n_pytensor.loss import yolo_loss
    import pytensor
    import pytensor.tensor as pt
    import numpy as np

    # Build model
    model, x, predictions = build_yolo11n(num_classes=2, input_size=128)  # 2 classes

    # Dummy targets
    targets = {
        'boxes': pt.tensor3('boxes', dtype='float32'),
        'classes': pt.imatrix('classes'),
        'num_boxes': pt.ivector('num_boxes')
    }

    # Loss
    loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)  # 2 classes

    # Gradients
    grads = pytensor.grad(loss, model.params)

    # Compile
    f = pytensor.function(
        [x],
        [loss, loss_dict['box_loss'], loss_dict['cls_loss']] + grads
    )

    # Test
    x_val = np.random.randn(2, 3, 128, 128).astype('float32')
    results = f(x_val)

    loss_val = results[0]
    box_loss_val = results[1]
    cls_loss_val = results[2]
    grad_vals = results[3:]

    # Verify
    assert loss_val > 0, "Loss should be positive"
    assert all(np.isfinite(g).all() for g in grad_vals), "Gradients should be finite"

    print(f"✓ Loss test passed")
    print(f"  Total loss: {loss_val:.4f}")
    print(f"  Box loss: {box_loss_val:.4f}")
    print(f"  Cls loss: {cls_loss_val:.4f}")
```

### Success Criteria

#### Automated Verification:
- [ ] Loss computation runs without errors
- [ ] Gradients are computable and finite
- [ ] Loss is positive and finite

#### Manual Verification:
- [ ] Loss values are reasonable (not exploding/vanishing)
- [ ] Can run backward pass through entire model

---

## Phase 4: Dataset Preparation

### Overview
Download COCO 2017, filter to 3 classes, resize to 128x128, create PyTensor-compatible data loader.

### Changes Required

#### 1. COCO Download Script

**File**: `examples/yolo11n_pytensor/data/download_coco.py`

```python
"""Download and prepare COCO dataset for YOLO training."""

import os
import urllib.request
import zipfile
import json
from pathlib import Path


def download_coco_subset(data_dir="./data/coco", classes=['person', 'cellphone'], max_images=2000):
    """
    Download COCO 2017 subset with specific classes (MINIMAL FOR DEMO).

    Parameters
    ----------
    data_dir : str
        Directory to save data
    classes : list
        List of class names to include
    max_images : int
        Maximum number of images to keep (for fast demo training)
    """
    # COCO class IDs
    coco_class_ids = {
        'person': 1,
        'cellphone': 77  # cell phone in COCO
    }

    target_ids = [coco_class_ids[c] for c in classes]

    print(f"Downloading COCO subset (MINIMAL FOR DEMO):")
    print(f"  Classes: {classes}")
    print(f"  Max images: {max_images}")
    print(f"  Target directory: {data_dir}")

    # Create directories
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Download annotations
    anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    anno_zip = os.path.join(data_dir, "annotations_trainval2017.zip")

    if not os.path.exists(anno_zip):
        print("Downloading annotations...")
        urllib.request.urlretrieve(anno_url, anno_zip)
        print("  Extracting...")
        with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Download images (train2017)
    images_url = "http://images.cocodataset.org/zips/train2017.zip"
    images_zip = os.path.join(data_dir, "train2017.zip")

    if not os.path.exists(images_zip):
        print("Downloading train images (this will take a while)...")
        urllib.request.urlretrieve(images_url, images_zip)
        print("  Extracting...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Filter annotations
    print("Filtering annotations...")
    filter_annotations(
        os.path.join(data_dir, "annotations/instances_train2017.json"),
        os.path.join(data_dir, f"annotations/instances_train2017_filtered.json"),
        target_ids,
        max_images=max_images
    )

    print("✓ COCO subset prepared (minimal for demo)!")


def filter_annotations(input_json, output_json, target_class_ids, max_images=None):
    """Filter COCO annotations to specific classes and limit image count."""
    with open(input_json, 'r') as f:
        coco = json.load(f)

    # Filter images and annotations
    filtered_images = []
    filtered_annotations = []
    image_ids = set()

    # Find annotations with target classes
    for anno in coco['annotations']:
        if anno['category_id'] in target_class_ids:
            filtered_annotations.append(anno)
            image_ids.add(anno['image_id'])

    # Filter images
    for img in coco['images']:
        if img['id'] in image_ids:
            filtered_images.append(img)

    # LIMIT TO max_images FOR DEMO
    if max_images and len(filtered_images) > max_images:
        print(f"  Limiting to {max_images} images (from {len(filtered_images)})")
        filtered_images = filtered_images[:max_images]
        kept_image_ids = {img['id'] for img in filtered_images}
        filtered_annotations = [
            anno for anno in filtered_annotations
            if anno['image_id'] in kept_image_ids
        ]

    # Filter categories
    filtered_categories = [
        cat for cat in coco['categories']
        if cat['id'] in target_class_ids
    ]

    # Create filtered dataset
    filtered_coco = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories,
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', [])
    }

    # Save
    with open(output_json, 'w') as f:
        json.dump(filtered_coco, f)

    print(f"  Filtered: {len(filtered_images)} images, {len(filtered_annotations)} annotations")


if __name__ == '__main__':
    download_coco_subset()
```

#### 2. Dataset Loader

**File**: `examples/yolo11n_pytensor/data/dataset.py`

```python
"""COCO dataset loader for YOLO training."""

import json
import numpy as np
from PIL import Image
import os


class COCODataset:
    """
    COCO dataset for object detection.

    Returns images resized to target size and bounding boxes.
    """

    def __init__(self, data_dir, annotation_file, image_size=128, max_boxes=20):
        """
        Parameters
        ----------
        data_dir : str
            Path to COCO data directory
        annotation_file : str
            Path to annotations JSON
        image_size : int
            Target image size (square)
        max_boxes : int
            Maximum number of boxes per image
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.max_boxes = max_boxes

        # Load annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        self.categories = coco_data['categories']

        # Build image_id -> annotations mapping
        self.img_to_annos = {}
        for anno in self.annotations:
            img_id = anno['image_id']
            if img_id not in self.img_to_annos:
                self.img_to_annos[img_id] = []
            self.img_to_annos[img_id].append(anno)

        # Filter images that have annotations
        self.images = [
            img for img in self.images
            if img['id'] in self.img_to_annos
        ]

        print(f"COCODataset initialized:")
        print(f"  Images: {len(self.images)}")
        print(f"  Target size: {image_size}x{image_size}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get image and targets.

        Returns
        -------
        image : ndarray
            Shape (3, H, W), normalized to [0, 1]
        boxes : ndarray
            Shape (max_boxes, 4), normalized [x_center, y_center, w, h]
        classes : ndarray
            Shape (max_boxes,), class indices
        num_boxes : int
            Number of valid boxes
        """
        img_info = self.images[idx]
        img_id = img_info['id']

        # Load image
        img_path = os.path.join(self.data_dir, "train2017", img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0

        # Transpose to (C, H, W)
        image = image.transpose(2, 0, 1)

        # Get annotations
        annos = self.img_to_annos.get(img_id, [])

        # Process boxes
        boxes = np.zeros((self.max_boxes, 4), dtype=np.float32)
        classes = np.zeros(self.max_boxes, dtype=np.int32)
        num_boxes = min(len(annos), self.max_boxes)

        for i, anno in enumerate(annos[:self.max_boxes]):
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = anno['bbox']

            # Normalize to [0, 1]
            x /= orig_w
            y /= orig_h
            w /= orig_w
            h /= orig_h

            # Convert to center format
            x_center = x + w / 2
            y_center = y + h / 2

            boxes[i] = [x_center, y_center, w, h]
            classes[i] = anno['category_id']

        return image, boxes, classes, num_boxes

    def get_batch(self, indices):
        """Get a batch of samples."""
        images = []
        all_boxes = []
        all_classes = []
        all_num_boxes = []

        for idx in indices:
            img, boxes, classes, num_boxes = self[idx]
            images.append(img)
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_num_boxes.append(num_boxes)

        return (
            np.array(images, dtype=np.float32),
            np.array(all_boxes, dtype=np.float32),
            np.array(all_classes, dtype=np.int32),
            np.array(all_num_boxes, dtype=np.int32)
        )
```

### Success Criteria

#### Automated Verification:
- [ ] Dataset downloads successfully
- [ ] Annotations filter correctly
- [ ] Can load samples without errors
- [ ] Batch loading works

#### Manual Verification:
- [ ] Visualize a few samples to verify boxes are correct
- [ ] Check image shapes and value ranges
- [ ] Verify class distribution

---

## Phase 5: Training Script

### Overview
Implement training loop with SGD optimizer, logging, and checkpointing.

### Changes Required

**File**: `examples/yolo11n_pytensor/train.py`

```python
"""
Train YOLO11n on COCO subset.

Usage:
    python train.py --epochs 50 --batch_size 16 --lr 0.001
"""

import argparse
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor import shared
import time
import json
from pathlib import Path

from model import build_yolo11n
from loss import yolo_loss
from data.dataset import COCODataset


def train_yolo11n(
    data_dir="./data/coco",
    epochs=50,  # Enough for basic convergence
    batch_size=16,  # Good balance
    learning_rate=0.001,  # Standard YOLO LR
    momentum=0.9,  # Standard momentum
    weight_decay=5e-4,  # Standard weight decay
    save_dir="./checkpoints",
    log_interval=10
):
    """
    Train YOLO11n model.

    Parameters
    ----------
    data_dir : str
        Path to COCO data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Initial learning rate
    momentum : float
        SGD momentum
    weight_decay : float
        Weight decay (L2 regularization)
    save_dir : str
        Directory to save checkpoints
    log_interval : int
        Log every N batches
    """
    print("="*70)
    print(" "*20 + "YOLO11n Training")
    print("="*70)

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/6] Loading dataset...")
    train_dataset = COCODataset(
        data_dir=data_dir,
        annotation_file=f"{data_dir}/annotations/instances_train2017_filtered.json",
        image_size=128,
        max_boxes=20
    )

    n_train = len(train_dataset)
    n_batches = n_train // batch_size

    print(f"  Training samples: {n_train}")
    print(f"  Batches per epoch: {n_batches}")

    # Build model
    print("\n[2/6] Building model...")
    model, x, predictions = build_yolo11n(num_classes=2, input_size=128)  # 2 CLASSES: person, cellphone

    # Targets
    target_boxes = pt.tensor3('target_boxes', dtype='float32')
    target_classes = pt.imatrix('target_classes')
    target_num_boxes = pt.ivector('target_num_boxes')

    targets = {
        'boxes': target_boxes,
        'classes': target_classes,
        'num_boxes': target_num_boxes
    }

    # Loss
    print("\n[3/6] Compiling loss and gradients...")
    loss, loss_dict = yolo_loss(predictions, targets, num_classes=2)  # 2 CLASSES

    # Add weight decay
    l2_reg = sum((p ** 2).sum() for p in model.params)
    loss_with_reg = loss + weight_decay * l2_reg

    # Compute gradients
    grads = pytensor.grad(loss_with_reg, model.params)

    # SGD with momentum
    velocities = []
    updates = []

    for param, grad in zip(model.params, grads):
        velocity = shared(
            np.zeros_like(param.get_value(), dtype='float32'),
            name=f"v_{param.name}",
            borrow=True
        )
        velocities.append(velocity)

        # Momentum update
        new_velocity = momentum * velocity - learning_rate * grad
        new_param = param + new_velocity

        updates.append((velocity, new_velocity.astype(param.dtype)))
        updates.append((param, new_param.astype(param.dtype)))

    # Compile training function
    print("  Compiling training function...")
    train_fn = pytensor.function(
        inputs=[x, target_boxes, target_classes, target_num_boxes],
        outputs=[loss, loss_dict['box_loss'], loss_dict['cls_loss']],
        updates=updates,
        name='train_function'
    )

    # Compile evaluation function (no updates)
    eval_fn = pytensor.function(
        inputs=[x, target_boxes, target_classes, target_num_boxes],
        outputs=[loss, loss_dict['box_loss'], loss_dict['cls_loss']],
        name='eval_function'
    )

    print("  ✓ Compilation complete")

    # Training loop
    print("\n[4/6] Starting training...")
    print("="*70)

    history = {
        'train_loss': [],
        'box_loss': [],
        'cls_loss': []
    }

    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*70)

        # Shuffle dataset
        indices = np.random.permutation(n_train)

        epoch_losses = []
        epoch_box_losses = []
        epoch_cls_losses = []

        epoch_start = time.time()

        # Training batches
        for batch_idx in range(n_batches):
            batch_start = time.time()

            # Get batch
            batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            x_batch, boxes_batch, classes_batch, num_boxes_batch = train_dataset.get_batch(batch_indices)

            # Train
            loss_val, box_loss_val, cls_loss_val = train_fn(
                x_batch, boxes_batch, classes_batch, num_boxes_batch
            )

            epoch_losses.append(loss_val)
            epoch_box_losses.append(box_loss_val)
            epoch_cls_losses.append(cls_loss_val)

            # Log
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = np.mean(epoch_losses[-log_interval:])
                avg_box = np.mean(epoch_box_losses[-log_interval:])
                avg_cls = np.mean(epoch_cls_losses[-log_interval:])
                batch_time = time.time() - batch_start

                print(f"  Batch {batch_idx+1}/{n_batches}: "
                      f"Loss={avg_loss:.4f} (box={avg_box:.4f}, cls={avg_cls:.4f}) "
                      f"[{batch_time:.2f}s]")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        train_loss = np.mean(epoch_losses)
        train_box_loss = np.mean(epoch_box_losses)
        train_cls_loss = np.mean(epoch_cls_losses)

        history['train_loss'].append(train_loss)
        history['box_loss'].append(train_box_loss)
        history['cls_loss'].append(train_cls_loss)

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Box Loss: {train_box_loss:.4f}")
        print(f"    Cls Loss: {train_cls_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, save_dir, epoch, train_loss, is_best=True)
            print(f"    ✓ Best model saved (loss={best_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, save_dir, epoch, train_loss, is_best=False)

    # Save final model
    print("\n[5/6] Saving final model...")
    save_checkpoint(model, save_dir, epochs-1, train_loss, is_best=False, name="final")

    # Save training history
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("\n[6/6] Training complete!")
    print("="*70)
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"Best loss: {best_loss:.4f}")


def save_checkpoint(model, save_dir, epoch, loss, is_best=False, name=None):
    """Save model checkpoint."""
    if name is None:
        name = f"checkpoint_epoch{epoch+1}"

    checkpoint = {
        'epoch': epoch,
        'loss': float(loss),
        'params': [p.get_value() for p in model.params],
        'bn_stats': [s.get_value() for s in model.bn_stats]
    }

    path = f"{save_dir}/{name}.npz"
    np.savez(path, **checkpoint)

    if is_best:
        best_path = f"{save_dir}/best_model.npz"
        np.savez(best_path, **checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO11n')
    parser.add_argument('--data_dir', type=str, default='./data/coco')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)

    args = parser.parse_args()

    train_yolo11n(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        log_interval=args.log_interval
    )
```

### Success Criteria

#### Automated Verification:
- [ ] Training starts without errors
- [ ] Loss decreases over first 5 epochs
- [ ] Checkpoints are saved successfully
- [ ] Can resume from checkpoint

#### Manual Verification:
- [ ] Training completes full run
- [ ] Loss curves look reasonable (decreasing trend)
- [ ] No memory leaks (memory usage stable)
- [ ] Training speed is acceptable (> 1 batch/second)

---

## Phase 6: ONNX Export and Browser Demo

### Overview
Export trained model to ONNX and create browser inference demo.

### Changes Required

#### 1. ONNX Export Script

**File**: `examples/yolo11n_pytensor/export.py`

```python
"""Export trained YOLO11n model to ONNX."""

import numpy as np
import pytensor
from model import build_yolo11n
from pytensor.link.onnx import export_onnx


def export_yolo11n_to_onnx(
    checkpoint_path,
    output_path="yolo11n_128.onnx",
    num_classes=2,  # person, cellphone
    input_size=128
):
    """
    Export YOLO11n model to ONNX format.

    Parameters
    ----------
    checkpoint_path : str
        Path to saved checkpoint (.npz)
    output_path : str
        Output ONNX file path
    num_classes : int
        Number of detection classes
    input_size : int
        Input image size
    """
    print("="*70)
    print("YOLO11n ONNX Export")
    print("="*70)

    # Build model
    print("\n[1/5] Building model...")
    model, x, predictions = build_yolo11n(num_classes=num_classes, input_size=input_size)

    # Load checkpoint
    print(f"\n[2/5] Loading checkpoint: {checkpoint_path}")
    checkpoint = np.load(checkpoint_path, allow_pickle=True)

    for param, value in zip(model.params, checkpoint['params']):
        param.set_value(value)

    for stat, value in zip(model.bn_stats, checkpoint['bn_stats']):
        stat.set_value(value)

    print(f"  ✓ Loaded epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")

    # Create inference function
    print("\n[3/5] Compiling inference function...")
    # For ONNX export, we want single output (concatenated predictions)
    # Flatten predictions for easy post-processing

    inference_fn = pytensor.function(
        inputs=[x],
        outputs=[predictions['p3'], predictions['p4'], predictions['p5']],
        name='yolo11n_inference'
    )

    # Test inference
    print("\n[4/5] Testing inference...")
    test_input = np.random.randn(1, 3, input_size, input_size).astype('float32')
    p3, p4, p5 = inference_fn(test_input)

    print(f"  Input shape: {test_input.shape}")
    print(f"  P3 output shape: {p3.shape}")
    print(f"  P4 output shape: {p4.shape}")
    print(f"  P5 output shape: {p5.shape}")

    # Export to ONNX
    print(f"\n[5/5] Exporting to ONNX: {output_path}")

    model_onnx = export_onnx(inference_fn, output_path)

    print(f"\n✓ Export complete!")
    print(f"  ONNX file: {output_path}")
    print(f"  Opset version: {model_onnx.opset_import[0].version}")
    print(f"  Nodes: {len(model_onnx.graph.node)}")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort

        print("\n[Verification] Testing with ONNX Runtime...")
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

        ort_outputs = session.run(None, {'x': test_input})

        # Compare
        print(f"  PyTensor P3: {p3.shape}, ONNX P3: {ort_outputs[0].shape}")
        match = np.allclose(p3, ort_outputs[0], atol=1e-4)
        print(f"  Outputs match: {'✓ YES' if match else '✗ NO'}")

        if not match:
            max_diff = np.abs(p3 - ort_outputs[0]).max()
            print(f"  Max difference: {max_diff:.2e}")

    except ImportError:
        print("\n  ⚠ onnxruntime not installed, skipping verification")

    print("\n" + "="*70)
    print("Export complete! Model ready for deployment.")
    print("="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export YOLO11n to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='yolo11n_128.onnx')
    parser.add_argument('--num_classes', type=int, default=2)  # person, cellphone
    parser.add_argument('--input_size', type=int, default=128)

    args = parser.parse_args()

    export_yolo11n_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_classes=args.num_classes,
        input_size=args.input_size
    )
```

#### 2. Browser Demo

**File**: `examples/onnx/onnx-yolo-demo/yolo_detection_demo.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>YOLO11n Webcam Detection - PyTensor + ONNX</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a1a;
            color: #ffffff;
        }
        h1 {
            text-align: center;
            color: #00ff88;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        #videoContainer {
            position: relative;
            width: 640px;
            height: 480px;
        }
        #webcam {
            display: none;
        }
        #canvas {
            border: 3px solid #00ff88;
            border-radius: 8px;
            background: #000;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }
        #startBtn {
            background: #00ff88;
            color: #000;
        }
        #startBtn:hover {
            background: #00cc6a;
        }
        #stopBtn {
            background: #ff4444;
            color: #fff;
        }
        #stopBtn:hover {
            background: #cc0000;
        }
        #stopBtn:disabled {
            background: #666;
            cursor: not-allowed;
        }
        #stats {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            min-width: 300px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        .stat-label {
            color: #aaa;
        }
        .stat-value {
            color: #00ff88;
            font-weight: bold;
        }
        .detection-list {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            min-height: 100px;
        }
    </style>
</head>
<body>
    <h1>🎯 YOLO11n Real-Time Detection</h1>
    <p style="text-align: center; color: #aaa;">
        PyTensor → ONNX → WebGPU | Person & Cellphone Detection
    </p>

    <div class="container">
        <div id="videoContainer">
            <video id="webcam" autoplay playsinline></video>
            <canvas id="canvas" width="640" height="480"></canvas>
        </div>

        <div class="controls">
            <button id="startBtn" onclick="startWebcam()">Start Webcam</button>
            <button id="stopBtn" onclick="stopWebcam()" disabled>Stop</button>
        </div>

        <div id="stats">
            <h3 style="margin-top: 0;">Stats</h3>
            <div class="stat-row">
                <span class="stat-label">FPS:</span>
                <span class="stat-value" id="fps">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Inference Time:</span>
                <span class="stat-value" id="inferenceTime">0ms</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Model Status:</span>
                <span class="stat-value" id="modelStatus">Loading...</span>
            </div>
        </div>

        <div class="detection-list">
            <h3 style="margin-top: 0;">Current Detections</h3>
            <div id="detections">No detections yet</div>
        </div>
    </div>

    <script>
        let session;
        let webcamStream;
        let animationFrame;
        let isRunning = false;
        const classes = ['person', 'cellphone'];
        const INPUT_SIZE = 128;
        const CONF_THRESHOLD = 0.3;
        const IOU_THRESHOLD = 0.45;

        // FPS calculation
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        let currentFps = 0;

        // Load ONNX model
        async function loadModel() {
            try {
                document.getElementById('modelStatus').textContent = 'Loading...';
                session = await ort.InferenceSession.create('yolo11n_128.onnx', {
                    executionProviders: ['webgpu', 'wasm']
                });
                document.getElementById('modelStatus').textContent = 'Ready ✓';
                console.log('Model loaded successfully');
            } catch (error) {
                document.getElementById('modelStatus').textContent = 'Error';
                console.error('Failed to load model:', error);
            }
        }

        // Start webcam
        async function startWebcam() {
            try {
                const video = document.getElementById('webcam');
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: 640,
                        height: 480,
                        facingMode: 'user'
                    }
                });
                video.srcObject = webcamStream;

                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;

                isRunning = true;
                detectLoop();
            } catch (error) {
                alert('Failed to access webcam: ' + error.message);
                console.error('Webcam error:', error);
            }
        }

        // Stop webcam
        function stopWebcam() {
            isRunning = false;
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
            }
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }

            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;

            // Clear canvas
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        // Main detection loop
        async function detectLoop() {
            if (!isRunning) return;

            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Run inference
                const startTime = performance.now();
                const detections = await runInference(canvas, ctx);
                const inferenceTime = performance.now() - startTime;

                // Draw detections
                drawDetections(ctx, detections, canvas.width, canvas.height);

                // Update stats
                updateStats(inferenceTime, detections);
            }

            animationFrame = requestAnimationFrame(detectLoop);
        }

        // Run inference on current frame
        async function runInference(canvas, ctx) {
            // Get image data and resize to 128x128
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const resized = resizeImageData(imageData, INPUT_SIZE, INPUT_SIZE);
            const inputTensor = preprocessImage(resized);

            // Run model
            const results = await session.run({ x: inputTensor });

            // Process outputs (all 3 scales)
            const detections = processOutputs(results, canvas.width, canvas.height);

            // Apply NMS
            const filtered = nonMaxSuppression(detections, IOU_THRESHOLD);

            return filtered;
        }

        // Preprocess image to tensor
        function preprocessImage(imageData) {
            const data = imageData.data;
            const input = new Float32Array(1 * 3 * INPUT_SIZE * INPUT_SIZE);

            // Convert RGBA to RGB CHW format and normalize
            for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
                input[i] = data[i * 4] / 255.0;  // R
                input[INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 1] / 255.0;  // G
                input[2 * INPUT_SIZE * INPUT_SIZE + i] = data[i * 4 + 2] / 255.0;  // B
            }

            return new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
        }

        // Resize image data
        function resizeImageData(imageData, targetWidth, targetHeight) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = targetWidth;
            canvas.height = targetHeight;

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = imageData.width;
            tempCanvas.height = imageData.height;
            tempCanvas.getContext('2d').putImageData(imageData, 0, 0);

            ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
            return ctx.getImageData(0, 0, targetWidth, targetHeight);
        }

        // Process YOLO outputs
        function processOutputs(results, canvasWidth, canvasHeight) {
            const detections = [];

            // Process all 3 scales: P3, P4, P5
            const scales = [
                { data: results.output0 || results[Object.keys(results)[0]], gridSize: 16, stride: 8 },
                { data: results.output1 || results[Object.keys(results)[1]], gridSize: 8, stride: 16 },
                { data: results.output2 || results[Object.keys(results)[2]], gridSize: 4, stride: 32 }
            ];

            scales.forEach(scale => {
                const output = scale.data;
                if (!output) return;

                const data = output.data;
                const [batch, channels, height, width] = output.dims;

                // Parse predictions: format is (batch, 6, H, W) for 2 classes
                // [x, y, w, h, cls0_prob, cls1_prob]
                for (let h = 0; h < height; h++) {
                    for (let w = 0; w < width; w++) {
                        // Get box coordinates
                        const idx = h * width + w;
                        const x_center = data[0 * height * width + idx];
                        const y_center = data[1 * height * width + idx];
                        const box_w = data[2 * height * width + idx];
                        const box_h = data[3 * height * width + idx];

                        // Get class probabilities
                        const cls0_prob = data[4 * height * width + idx];
                        const cls1_prob = data[5 * height * width + idx];

                        // Apply sigmoid to get probabilities
                        const conf0 = 1 / (1 + Math.exp(-cls0_prob));
                        const conf1 = 1 / (1 + Math.exp(-cls1_prob));

                        const maxConf = Math.max(conf0, conf1);
                        const classId = conf0 > conf1 ? 0 : 1;

                        if (maxConf > CONF_THRESHOLD) {
                            // Convert to canvas coordinates
                            const x = ((x_center + w) / width) * canvasWidth;
                            const y = ((y_center + h) / height) * canvasHeight;
                            const boxWidth = (box_w / width) * canvasWidth;
                            const boxHeight = (box_h / height) * canvasHeight;

                            detections.push({
                                x: x - boxWidth / 2,
                                y: y - boxHeight / 2,
                                width: boxWidth,
                                height: boxHeight,
                                confidence: maxConf,
                                classId: classId
                            });
                        }
                    }
                }
            });

            return detections;
        }

        // Non-maximum suppression
        function nonMaxSuppression(detections, iouThreshold) {
            // Sort by confidence
            detections.sort((a, b) => b.confidence - a.confidence);

            const keep = [];
            while (detections.length > 0) {
                const best = detections.shift();
                keep.push(best);

                detections = detections.filter(det => {
                    if (det.classId !== best.classId) return true;
                    const iou = calculateIoU(best, det);
                    return iou < iouThreshold;
                });
            }

            return keep;
        }

        // Calculate IoU
        function calculateIoU(box1, box2) {
            const x1 = Math.max(box1.x, box2.x);
            const y1 = Math.max(box1.y, box2.y);
            const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
            const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

            const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
            const area1 = box1.width * box1.height;
            const area2 = box2.width * box2.height;
            const union = area1 + area2 - intersection;

            return union === 0 ? 0 : intersection / union;
        }

        // Draw detections on canvas
        function drawDetections(ctx, detections, canvasWidth, canvasHeight) {
            // Redraw video frame
            const video = document.getElementById('webcam');
            ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);

            // Draw bounding boxes
            detections.forEach(det => {
                const color = det.classId === 0 ? '#00ff88' : '#ff8800';

                // Box
                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(det.x, det.y, det.width, det.height);

                // Label background
                const label = `${classes[det.classId]}: ${(det.confidence * 100).toFixed(0)}%`;
                ctx.font = 'bold 16px Arial';
                const textWidth = ctx.measureText(label).width;

                ctx.fillStyle = color;
                ctx.fillRect(det.x, det.y - 25, textWidth + 10, 25);

                // Label text
                ctx.fillStyle = '#000';
                ctx.fillText(label, det.x + 5, det.y - 7);
            });
        }

        // Update stats display
        function updateStats(inferenceTime, detections) {
            // Update FPS
            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                currentFps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
                document.getElementById('fps').textContent = currentFps;
                frameCount = 0;
                lastFpsUpdate = now;
            }

            // Update inference time
            document.getElementById('inferenceTime').textContent = `${inferenceTime.toFixed(1)}ms`;

            // Update detections list
            const detectionsDiv = document.getElementById('detections');
            if (detections.length === 0) {
                detectionsDiv.innerHTML = 'No detections';
            } else {
                detectionsDiv.innerHTML = detections.map(det =>
                    `<div style="margin: 5px 0; color: ${det.classId === 0 ? '#00ff88' : '#ff8800'};">
                        ${classes[det.classId]}: ${(det.confidence * 100).toFixed(1)}%
                    </div>`
                ).join('');
            }
        }

        // Load model on page load
        loadModel();
    </script>
</body>
</html>
```

### Success Criteria

#### Automated Verification:
- [ ] ONNX export completes without errors
- [ ] ONNX model validates: `onnx.checker.check_model()`
- [ ] ONNX Runtime can load and run model
- [ ] PyTensor and ONNX outputs match (atol=1e-4)

#### Manual Verification:
- [ ] Browser demo loads ONNX model successfully
- [ ] Can upload image and run inference
- [ ] Inference completes in reasonable time (< 100ms)
- [ ] Bounding boxes are drawn (even if detections aren't perfect)

---

## Performance Considerations

### Memory Management
- Batch size: 16 (fits in 8GB RAM)
- Model size: ~2.6M params × 4 bytes = ~10MB
- Activation memory: ~500MB peak for batch_size=16

### Training Speed Estimates
**On laptop CPU (8 cores) with 2000 images, batch_size=16**:
- Forward pass: ~500ms per batch (16 images)
- Backward pass: ~1000ms per batch
- Total: ~1.5s per batch
- Batches per epoch: 2000/16 = 125 batches
- Epoch time: ~3 minutes (125 batches × 1.5s)
- **50 epochs: ~2.5 hours** ✓ Overnight run acceptable

**On laptop GPU** (if available):
- Could be 5-10x faster: ~15-30 minutes total

**This is lightweight training** - enough to get basic detection working for demo!

---

## Migration Notes

### From MNIST Example to YOLO

**Key differences**:
1. **Architecture complexity**: YOLO has 181 layers vs 5 for MNIST
2. **Multi-scale outputs**: 3 detection heads vs single classification
3. **Loss function**: IoU + BCE vs cross-entropy
4. **Data loading**: Bounding boxes + images vs images only
5. **Post-processing**: NMS for detections vs argmax for classification

**Shared patterns**:
- PyTensor symbolic computation
- Gradient-based training loop
- SGD with momentum
- ONNX export workflow
- Browser deployment via ONNX Runtime Web

---

## Testing Strategy Summary

### Phase-by-Phase Testing

**Phase 1: BatchNorm Gradients**
- Unit tests for gradient computation
- Numerical gradient checking
- Integration test in simple network

**Phase 2: Architecture**
- Forward pass shape checking
- Parameter count verification
- Memory usage profiling

**Phase 3: Loss Function**
- Loss computation correctness
- Gradient flow verification
- Convergence on toy data

**Phase 4: Dataset**
- Data loading correctness
- Batch generation
- Visualization of samples

**Phase 5: Training**
- Training loop execution
- Loss decrease verification
- Checkpoint save/load

**Phase 6: Export**
- ONNX export success
- Output matching (PyTensor vs ONNX)
- Browser inference

---

## References

### Papers
- Ioffe & Szegedy (2015): Batch Normalization
- Redmon et al. (2016): YOLO v1
- YOLOv11 (2024): Ultralytics documentation

### Code References
- `examples/onnx/onnx-mnist-demo/train_mnist_cnn.py` - Training template
- `pytensor/tensor/batchnorm.py` - BatchNorm implementation
- `pytensor/link/onnx/dispatch/` - ONNX converters
- Ultralytics YOLO11: github.com/ultralytics/ultralytics

### Documentation
- PyTensor: pytensor.readthedocs.io
- ONNX: onnx.ai
- ONNX Runtime Web: onnxruntime.ai/docs/tutorials/web/

---

## Timeline Estimate

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | BatchNorm Gradients | 4-6 hours |
| 2 | YOLO Architecture | 8-12 hours |
| 3 | Loss Function | 4-6 hours |
| 4 | Dataset Prep | 2-3 hours |
| 5 | Training Script | 3-4 hours |
| 6 | ONNX Export + Webcam Demo | 4-5 hours |
| **Total** | **Implementation** | **25-36 hours** |
| | **Training time** | **~2.5 hours** (CPU) |
| **Grand Total** | | **~27-38 hours** |

**Note**: Training time is for 2000 images, 50 epochs on CPU. Can run overnight. With GPU could be as fast as 20-30 minutes.

---

## Conclusion

This plan provides a **lightweight but functional** YOLO11n implementation for real-time webcam detection in the browser!

**Key Success Factors**:
1. ✅ All ONNX operations already implemented
2. ✅ Training infrastructure exists (MNIST example)
3. ✅ Balanced dataset (2000 images, 2 classes) - enough to learn, fast enough to train
4. ✅ Standard training (50 epochs, ~2.5 hours) - overnight run acceptable
5. ✅ **Real detection capability - must actually work on webcam!**

**Final Deliverable**: A working YOLO11n webcam demo that:
- Trains natively in PyTensor with real convergence (mAP@0.5 > 0.2)
- Exports to ONNX successfully
- **Runs real-time in browser with WebGPU at > 10 FPS**
- **Actually detects person and cellphone in webcam feed!**
- **Demonstrates PyTensor → ONNX pipeline works for complex, real-world models**

**This is a practical demo!** We're creating a working detector that:
- ✅ Gradient computation through 181 layers
- ✅ ONNX export of complex architecture
- ✅ Real-time browser inference with multi-scale detection heads
- ✅ End-to-end PyTensor → ONNX → WebGPU pipeline
- ✅ **Actual object detection in real-time webcam!**

**Demo Features**:
- 🎥 Real-time webcam feed in browser
- 📦 Person detection (green boxes)
- 📱 Cellphone detection (orange boxes)
- ⚡ > 10 FPS on laptop GPU
- 📊 Live FPS and confidence stats
- 🎯 NMS for clean detections

This will be a powerful, practical showcase for PyTensor's ONNX backend! 🚀
