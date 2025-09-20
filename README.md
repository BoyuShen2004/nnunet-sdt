# nnUNet-SDT: Signed Distance Transform Regression in nnU-Net v2

This repository documents how vanilla [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) was extended into **nnUNet-SDT**, a model that learns to regress **Signed Distance Transforms (SDTs)** of segmentation masks instead of (or in addition to) predicting categorical labels.  

---

## 🔎 Motivation

- **Regular nnU-Net:** predicts per-voxel class probabilities with Cross-Entropy + Dice loss.  
- **nnUNet-SDT:** predicts continuous **Signed Distance Transform** maps, trained with regression losses (MSE).  
- **Why:** SDT provides smooth gradients and boundary-aware supervision, which can improve fine structures and topology.

---

## September 16 Update

---

## 📂 Added & Modified Files

Your modifications introduced three new files:

You added three main files to support SDT-based training:

1. **SDT Transform**  
   - Location:  
     ```
     /nnUNet/nnunetv2/training/data_augmentation/custom_transforms/sdt.py
     ```
   - Purpose: Converts segmentation masks into Signed Distance Transform maps during data augmentation.  

2. **Custom Trainer**  
   - Location:  
     ```
     /nnUNet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerSDT.py
     ```
   - Purpose: Defines the `nnUNetTrainerSDT` class, which switches loss to MSE and aligns output channels with SDT targets.  

3. **SDT to Mask Post-Processor**  
   - Location:  
     ```
     sdt_to_mask.py  (or any folder, since it is standalone)
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

---

## 🧪 Sanity Checks

- Ensure **output channels = target channels**.  
- If you see warnings like:

```text
Using a target size (B,1,H,W) different to input size (B,2,H,W)
```

→ you left the network at 2 outputs instead of 1. Fix in nnUNetTrainerSDT.py.

---

## 📖 Citation
If you use this code, please cite:
- The original nnU-Net papers, and  
- Relevant SDT supervision papers if you adapt the approach further.

---

## 📜 License
Same license as nnU-Net (Apache 2.0) unless otherwise stated.

---

## ❓ FAQ

**Q: Why not just use Dice loss?**  
A: Dice compares discrete masks. SDT regression enforces smooth, boundary-aware supervision.

**Q: Do I need new ground truth files?**  
A: No. SDTs are generated dynamically from your binary labels.

**Q: Can I combine SDT with masks?**  
A: Yes — by outputting 2 channels (mask + SDT) and adjusting the trainer & loss accordingly.

---

September 20 Update

Two new utility scripts were added to improve **mask cleaning** and **SDT post-processing**:

1. **remove_unconnected_small_components.py**  
   - Purpose: Cleans `.tif` masks by removing disconnected noise (small blobs).  
   - Keeps only the largest connected component (e.g., main worm body).  
   - Useful for denoising masks after training or manual annotation.  

2. **sdt_to_mask_denoise.py**  
   - Purpose: Extends `sdt_to_mask.py` by not only converting `.npz` SDT predictions into binary masks, but also automatically denoising them.  
   - Ensures that final outputs contain only the biologically relevant structure, without stray specks.  
   - Optionally saves raw SDT maps as `.tif` for inspection.  

---

## 🛠️ Usage

### `remove_unconnected_small_components.py`

Keep only the largest component in all masks from a folder:

```bash
python remove_unconnected_small_components.py \
  --in_dir /path/to/input_masks \
  --out_dir /path/to/output_masks
```

Process just one file inside the input directory:

```bash
python remove_unconnected_small_components.py \
  --in_dir /path/to/input_masks \
  --out_dir /path/to/output_masks \
  --file example_mask.tif
```

Process when you already have a single file path:

```bash
python remove_unconnected_small_components.py \
  --in_dir /path/to/input_masks/example_mask.tif \
  --out_dir /path/to/output_masks
```

### `sdt_to_mask_denoise.py`

Convert SDT predictions (`.npz`) into cleaned binary masks:

```bash
python sdt_to_mask_denoise.py \
  --pred_dir /path/to/npz_predictions \
  --out_mask_dir /path/to/clean_masks \
  --out_sdt_dir /path/to/save_raw_sdt
```

By default, only the largest connected component is preserved.

If you prefer to use a minimum-size filter instead:

```bash
python sdt_to_mask_denoise.py \
  --pred_dir /path/to/npz_predictions \
  --out_mask_dir /path/to/clean_masks \
  --min_size 500 \
  --no_keep_largest
```

This workflow removes disconnected specks while ensuring the main object is retained in every output mask.
