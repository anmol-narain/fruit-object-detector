import torch
import torch.nn as nn

class FruitDetector(nn.Module):
    def __init__(self, num_classes):
        super(FruitDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # [B, 3, 224, 224] → [B, 16, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2),                # → [B, 16, 112, 112]

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # → [B, 32, 56, 56]

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # → [B, 64, 28, 28]
        )

        self.flatten = nn.Flatten()
        self.fc_shared = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU()
        )

        self.bbox_head = nn.Linear(512, 4)           # [x, y, w, h]
        self.class_head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc_shared(x)
        bbox = self.bbox_head(x)
        class_logits = self.class_head(x)
        return bbox, class_logits
