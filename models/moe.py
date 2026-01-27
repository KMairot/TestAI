from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNClassifier(nn.Module):
    """
    Minimal ResNet-50 classifier (used only to define a backbone for MoE init).
    Weights are overwritten by the checkpoint when loading.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class MixtureOfExpertsModel(nn.Module):
    """
    MoE across slices:
      - feature extractor: ResNet trunk (without FC)
      - per-slice classifier -> logits per slice
      - router -> weights per slice (softmax)
      - pooled logits = sum_i w_i * logits_i
    """
    def __init__(self, cnn_model: CNNClassifier, num_classes: int, router_hidden: int = 256):
        super().__init__()
        backbone = cnn_model.resnet
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # [B,2048,1,1]
        self.feature_dim = 2048
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.router = nn.Sequential(
            nn.Linear(self.feature_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, 1)
        )

    def forward(self, x):  # x [B,N,3,H,W]
        b, n, c, h, w = x.shape
        x = x.view(b*n, c, h, w)
        feats = self.feature_extractor(x).flatten(1)             # [B*N,2048]
        logits = self.classifier(feats).view(b, n, -1)           # [B,N,C]
        weights = self.router(feats).view(b, n, 1)               # [B,N,1]
        weights = F.softmax(weights, dim=1)
        pooled = (weights * logits).sum(dim=1)                   # [B,C]
        return pooled, weights.squeeze(-1)                       # weights [B,N]

class MoEWrapper(nn.Module):
    """
    Training code saved checkpoints with 'core.*' prefix (core.feature_extractor, core.router, ...).
    This wrapper reproduces that structure to enable strict loading.
    """
    def __init__(self, num_classes: int = 3, router_hidden: int = 256):
        super().__init__()
        base = CNNClassifier(num_classes=num_classes)
        self.core = MixtureOfExpertsModel(base, num_classes=num_classes, router_hidden=router_hidden)

    def forward(self, x):
        return self.core(x)
