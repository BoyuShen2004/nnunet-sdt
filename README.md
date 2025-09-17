# nnUNet-SDT

# nnUNet-SDT: Signed Distance Transform Regression in nnU-Net v2

This repository documents how **vanilla nnU-Net v2** was extended into **nnUNet-SDT**, a model that learns to regress **Signed Distance Transforms (SDTs)** of segmentation masks instead of (or in addition to) predicting categorical labels.  

---

## 🔎 Motivation

- **Regular nnU-Net:** predicts per-voxel class probabilities with Cross-Entropy + Dice loss.  
- **nnUNet-SDT:** predicts continuous **Signed Distance Transform** maps, trained with regression losses (MSE).  
- **Why:** SDT provides smooth gradients and boundary-aware supervision, which can improve fine structures and topology.

---

## 📂 Added & Modified Files

Your modifications introduced three new files:

You added three main files to support SDT-based training:

1. **SDT Transform**  
   - Location:  
     ```
     /orcd/data/edboyden/002/shenb/nnUNet/nnunetv2/training/data_augmentation/custom_transforms/sdt.py
     ```
   - Purpose: Converts segmentation masks into Signed Distance Transform maps during data augmentation.  

2. **Custom Trainer**  
   - Location:  
     ```
     /orcd/data/edboyden/002/shenb/nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerSDT.py
     ```
   - Purpose: Defines the `nnUNetTrainerSDT` class, which switches loss to MSE and aligns output channels with SDT targets.  

3. **SDT to Mask Post-Processor**  
   - Location:  
     ```
     scripts/sdt_to_mask.py  (or any folder, since it is standalone)
     ```
   - Purpose: Converts predicted SDT maps (`.npz`) into binary masks (`.tif`). Can be placed anywhere on your system because it is run manually after prediction.

---

## 📝 File-by-File Documentation

### 1. `sdt.py`

**Purpose:**  
Transforms binary masks into continuous **Signed Distance Transform** maps on-the-fly during data loading.  

**Key Features:**
- Computes distance inside and outside objects.  
- Negative values inside, positive outside.  
- Supports smoothing, quantization, and different background handling.  

---

### 2. `nnUNetTrainerSDT.py`

**Purpose:**  
Defines a custom trainer `nnUNetTrainerSDT` that extends `nnUNetTrainer`.  

**Main Modifications:**
- **Loss Function:** replaced `Dice + CrossEntropy` with **MSE** loss for regression.  
- **Output Channels:**  
  - Vanilla nnU-Net: 1 channel for binary, N channels for N-class classification.  
  - nnUNet-SDT: typically **1 channel** (continuous SDT values).  
- **Training Pipeline:** integrates the `SDT` transform from `sdt.py` into `get_training_transforms`.  

---

### 3. `sdt_to_mask.py`

**Purpose:**  
Converts **predicted SDT maps** (stored as `.npz` files by `nnUNetv2_predict`) back into **binary masks (.tif)**.  

**Functionality:**
- Loads `.npz` prediction files.  
- Thresholds SDT: `mask = (SDT < -eps)` (inside region).  
- Writes masks as `.tif`.  
- Optionally saves raw SDT maps as `.tif` for inspection.  

**Location Note:**  
This script is not tied to the nnU-Net framework. You can place it in `scripts/`, your project root, or any utilities folder. It only needs correct input/output paths.

---

## 🔄 How Training Changes

### Regular nnU-Net
- **Inputs:** images  
- **Targets:** discrete masks (e.g., 0 background, 1 foreground)  
- **Outputs:** per-voxel class probabilities (softmax)  
- **Loss:** Cross-Entropy + Dice  

### nnUNet-SDT
- **Inputs:** images  
- **Targets:** **continuous SDT maps** (generated on-the-fly by `sdt.py`)  
- **Outputs:** per-voxel regression map (float)  
- **Loss:** Mean Squared Error (MSE)  

---

## ⚙️ Usage Guide

### 1. Preprocess Data
Dataset stays the same as nnU-Net (images, labels as masks). No SDT labels required.

```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity
```

### 2. Train nnUNet-SDT
Use the custom trainer:

```bash
srun nnUNetv2_train <DATASET_ID> <CONFIG> <FOLD> -tr nnUNetTrainerSDT
```

### 3. Predict
Run prediction with your trainer:

```bash
nnUNetv2_predict \
  -i <IMAGES_TS> \
  -o <PRED_DIR> \
  -d <DATASET> \
  -c <CONFIG> \
  -tr nnUNetTrainerSDT \
  -f <FOLD> \
  --disable_tta \
  --save_probabilities
```

This writes `.npz` files with predicted SDT arrays.

### 4. Convert SDT → Mask
Post-process with `sdt_to_mask.py`:

```bash
python sdt_to_mask.py \
  --pred_dir <PRED_DIR> \
  --out_mask_dir <MASK_DIR> \
  --out_sdt_dir <SDT_TIF_DIR>
```

Outputs:
- `*_mask.tif` (binary mask)
- `*_sdt.tif` (optional raw SDT)

---

## 🔍 Key Differences Recap

| Aspect           | nnU-Net (vanilla)       | nnUNet-SDT              |
|------------------|-------------------------|--------------------------|
| **Target**       | Discrete masks          | Continuous SDT maps      |
| **Loss**         | Dice + CE               | MSE (regression)         |
| **Output channels** | 1 (binary) / N (classes) | 1 (regressed SDT)        |
| **Post-processing** | Argmax / Softmax      | Threshold SDT → mask      |
| **Added files**  | None                    | `sdt.py`, `nnUNetTrainerSDT.py`, `sdt_to_mask.py` |
