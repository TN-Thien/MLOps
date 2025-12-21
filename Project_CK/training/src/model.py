from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

class MoHinhIQA(nn.Module):
    """
    Mô hình IQA: backbone CNN + head hồi quy 1 giá trị (0–1).
    Dùng sigmoid để đảm bảo output nằm trong [0, 1].
    """
    def __init__(self, backbone: str = "efficientnet_b0", dong_bang_backbone: bool = False) -> None:
        super().__init__()
        backbone = backbone.lower()

        if backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_feats = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_feats, 1)
            self.backbone = m

        elif backbone == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_feats = m.fc.in_features
            m.fc = nn.Linear(in_feats, 1)
            self.backbone = m

        elif backbone == "mobilenet_v2":
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_feats = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_feats, 1)
            self.backbone = m

        else:
            raise ValueError("Backbone không hỗ trợ. Chọn: efficientnet_b0 | resnet18 | mobilenet_v2")

        if dong_bang_backbone:
            for name, p in self.backbone.named_parameters():
                if "classifier" in name or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
                    continue
                p.requires_grad = False

        self.kich_hoat = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        y = self.kich_hoat(y)
        return y