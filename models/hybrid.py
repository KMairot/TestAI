from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

class CNN2DExtractor(nn.Module):
    def __init__(self, resnet_model, out_dim=512):
        super().__init__()
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-1])
        self.projection = nn.Linear(2048, out_dim)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B,1,3,H,W]
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        feat = self.resnet(x).flatten(1)               # [B*S,2048]
        feat = self.projection(feat)                   # [B*S,D]
        return feat.view(b, s, -1)                     # [B,S,D]

class TransformerFusion(nn.Module):
    def __init__(self, input_dim, num_heads, num_classes, num_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, input_dim))   # 19 slices + CLS
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x [B,19,D]
        b, s, d = x.shape
        cls = self.cls_token.expand(b, -1, -1)         # [B,1,D]
        x = torch.cat([cls, x], dim=1)                 # [B,20,D]
        x = x + self.pos_embedding[:, : (s + 1), :]
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)

class HybridModel(nn.Module):
    def __init__(self, resnet_model, num_classes=3, input_dim=512, nhead=8, num_encoder_layers=6):
        super().__init__()
        self.cnn_extractor = CNN2DExtractor(resnet_model, out_dim=input_dim)
        self.transformer_fusion = TransformerFusion(
            input_dim=input_dim,
            num_heads=nhead,
            num_classes=num_classes,
            num_layers=num_encoder_layers,
        )

    def forward(self, x):  # x [B,19,3,H,W]
        feats = self.cnn_extractor(x)                  # [B,19,D]
        return self.transformer_fusion(feats)          # [B,C]

def build_hybrid_model(num_classes: int = 3, input_dim: int = 512, nhead: int = 8, num_encoder_layers: int = 6):
    resnet = models.resnet50(weights=None)
    return HybridModel(resnet, num_classes=num_classes, input_dim=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers)
