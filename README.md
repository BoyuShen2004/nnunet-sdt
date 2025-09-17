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

