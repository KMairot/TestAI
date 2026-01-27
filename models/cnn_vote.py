from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

class ResNetSliceClassifier(nn.Module):
    """
    Slice-level classifier based on ResNet-50.
    Your CNN checkpoint (cnn_resnet50_2025-10-20_best.pt) uses a 'resnet.*' prefix,
    so we expose a .resnet attribute matching torchvision's ResNet-50.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        Accepts:
          - x: [B, 3, H, W] -> logits [B, C]
          - x: [B, 19, 3, H, W] -> logits [B, 19, C]
        """
        if x.ndim == 4:
            return self.resnet(x)
        b, n, c, h, w = x.shape
        x = x.view(b*n, c, h, w)
        logits = self.resnet(x).view(b, n, -1)
        return logits

    def forward_slices(self, x):
        # explicit alias used by run_inference
        return self.forward(x)
