from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetSliceClassifier(nn.Module):
    """
    Slice-level classifier based on ResNet-50.

    IMPORTANT:
    Your checkpoint stores classifier weights under:
        resnet.fc.1.weight / resnet.fc.1.bias
    which implies: resnet.fc is a nn.Sequential(<something>, nn.Linear).

    To be compatible, we define:
        self.resnet.fc = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(..., num_classes))

    (Dropout has no weights, so it's safe even if training used a different p.)
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet = models.resnet50(weights=None)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - x: [B, 3, H, W] -> logits [B, C]
          - x: [B, 19, 3, H, W] -> logits [B, 19, C]
        """
        if x.ndim == 4:
            return self.resnet(x)

        if x.ndim != 5:
            raise ValueError(f"Expected x with 4 or 5 dims, got shape={tuple(x.shape)}")

        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        logits = self.resnet(x).view(b, n, -1)
        return logits

    def forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        # explicit alias used by run_inference
        return self.forward(x)
