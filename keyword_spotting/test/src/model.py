import torch
from torch import nn


class KeywordCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        blocks = []
        previous = 1
        for channels in config["channels"]:
            blocks.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.BatchNorm2d(channels),
                           nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.convolutions = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Linear(previous, config["hidden_size"]), nn.ReLU(),
                                        nn.Dropout(config["dropout"]), nn.Linear(config["hidden_size"], 1))

    def forward(self, features):
        encoded = self.convolutions(features)
        pooled = torch.amax(encoded, dim=(2, 3))
        return self.classifier(pooled).squeeze(-1)  # Raw logits, not sigmoid.
