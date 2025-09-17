import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.sdt import SDT, LabelToSDT


class _SDTMSELossAdaptor(nn.Module):
    """
    Wraps MSE so that it can accept nnUNet outputs directly (tensor or list for deep supervision),
    forces prediction to 1 channel for SDT regression, and shape/dtype/device-match with target.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def _match_and_loss(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Force pred to one channel
        if p.dim() >= 3 and p.shape[1] != 1:
            p = p[:, :1, ...]  # keep first channel only
        # Match spatial size
        if p.shape[2:] != t.shape[2:]:
            mode = 'trilinear' if p.ndim == 5 else 'bilinear'
            t = F.interpolate(t, size=p.shape[2:], mode=mode, align_corners=False)
        # Match dtype/device
        t = t.to(p.dtype).to(p.device)
        return self.mse(p, t)

    def forward(self, pred, target):
        # nnU-Net may pass a list of predictions for deep supervision
        if isinstance(pred, (list, tuple)):
            losses = [self._match_and_loss(p, target) for p in pred]
            return sum(losses) / len(losses)
        else:
            return self._match_and_loss(pred, target)


class nnUNetTrainerSDT(nnUNetTrainer):
    """
    nnU-Net v2 trainer for SDT regression:
      - Regress a single-channel signed distance map (continuous)
      - Use MSE loss via adaptor that trims pred to 1 channel
      - Insert LabelToSDT transform so GT masks become SDT floats
      - Disable deep supervision loss accumulation
    """

    def configure_label_manager(self):
        super().configure_label_manager()
        try:
            self.label_manager.num_segmentation_heads = 1
            self.label_manager.ignore_regions = True
            self.label_manager.foreground_regions = []
        except Exception:
            pass

    def initialize(self):
        super().initialize()
        # Disable deep supervision weighting (regression does not use CE/Dice)
        self.enable_deep_supervision = False
        # Critical: make self.loss handle slicing & matching
        self.loss = _SDTMSELossAdaptor()

    def get_training_transforms(self, *args, **kwargs):
        tr = super().get_training_transforms(*args, **kwargs)
        sdt_tf = LabelToSDT(SDT(
            mode='sdt',
            alpha=1.0,
            smooth=False,
            background_value=-1.0
        ))
        if hasattr(tr, "transforms") and isinstance(tr.transforms, list) and len(tr.transforms) > 0:
            try:
                tr.transforms.insert(-1, sdt_tf)
            except Exception:
                tr.transforms.append(sdt_tf)
        else:
            try:
                tr.append(sdt_tf)
            except Exception:
                pass
        return tr

    def get_validation_transforms(self, *args, **kwargs):
        va = super().get_validation_transforms(*args, **kwargs)
        sdt_tf = LabelToSDT(SDT(
            mode='sdt',
            alpha=1.0,
            smooth=False,
            background_value=-1.0
        ))
        if hasattr(va, "transforms") and isinstance(va.transforms, list) and len(va.transforms) > 0:
            try:
                va.transforms.insert(-1, sdt_tf)
            except Exception:
                va.transforms.append(sdt_tf)
        else:
            try:
                va.append(sdt_tf)
            except Exception:
                pass
        return va
